#!/usr/bin/env python3
"""Rebuild the project page's Fig. 3 and Fig. 4 JSON from the final measurements.

The page used to carry the single-dataset layout: Fig. 3 split the CKA reading
into "observed by both" and "observed by the teacher only", and Fig. 4 carried
six panels in two columns (feature planes, voxel logits).  Both were superseded
when the measurement moved to three datasets and the two support domains were
merged, so this rebuilds them from the shipped numbers:

    docs/handoffs/FIGURES_FINAL_20261112/figure_numbers_final_v8.json
    docs/handoffs/FIGURES_FINAL_20261112/figure_cka_curves_iqr.json
    docs/handoffs/FIGURES_FINAL_20261112/axis_ranges_fig1.json
    docs/handoffs/FIGURES_FINAL_20261112/axis_layout_fig1.json

New layout, matching the published figures:

    Fig. 3a   TPV features   SK | NU | WA   band CKA, with a gap-closed strip
    Fig. 3b   voxel logits   SK | NU | WA   under each
    Fig. 4    a  TPV, real vs pseudo        SK | NU | WA
              b  TPV, paired change
              c  voxel, real vs pseudo
              d  voxel, paired change

Nothing is retyped: every drawn number is read out of those four files.  Style
(colours, fonts, tooltip text) is carried over from the JSON being replaced, so
only the data and the panel layout change.

    python3 LiEQpage/tools/build_page_figs.py
"""
import collections
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, 'docs', 'handoffs', 'FIGURES_FINAL_20261112')
OUT = os.path.join(ROOT, 'LiEQpage', 'docs', 'assets', 'data')

DATASETS = [('SK', 'SemanticKITTI'), ('NU', 'nuScenes'), ('WA', 'Waymo')]
NBANDS = 11


def load(name, where=SRC):
    with io.open(os.path.join(where, name), encoding='utf-8') as fh:
        return json.load(fh)


def dump(obj, name):
    path = os.path.join(OUT, name)
    with io.open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=False)
        fh.write('\n')
    return path


NUM = load('figure_numbers_final_v8.json')
IQR = load('figure_cka_curves_iqr.json')['fig1_band_cka_3ds']
RANGES = load('axis_ranges_fig1.json')['axis_ranges_fig1']
LAYOUT = load('axis_layout_fig1.json')

C = {
    'C_TEACHER': '#52514e',
    'C_LISEER': '#eb6834',
    'C_LISEER_LIGHT': '#f6b79a',
    'C_LIEQ': '#2a78d6',
    'C_LIEQ_LIGHT': '#86b6ef',
    'C_NOISE': '#8f8d88',
    'C_GAP': '#56b89a',
    'SHADE': '#f3f2ef',
    'INK': '#0b0b0b',
    'INK2': '#52514e',
    'MUTED': '#8a8883',
    'GRID': '#e8e7e3',
    'AXIS': '#bdbbb5',
    'SURFACE': '#ffffff',
}

BAND_EDGES = [-0.05, 0.01, 0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, 1.50, 2.02]
EDGE_LABELS = ['0', '.01', '.03', '.08', '.15', '.25', '.40', '.60', '.85', '1.15', '1.50']


def x_axis(label=''):
    """The shared band axis: 0-10 equidistant, five labelled ticks, HF shaded."""
    vis = LAYOUT['visible_tick_indices']
    return {
        'label': label,
        'lim': list(LAYOUT['xlim']),
        'scale': 'linear',
        'ticks': [float(i) for i in vis],
        'tick_labels': [EDGE_LABELS[i] for i in vis],
        'tick_label_side': 'bottom',
        'label_is_shared_figure_text': True,
        'quantity': 'graph-frequency band index (0-10)',
        'band_index': list(range(NBANDS)),
        'band_lambda_edges': list(BAND_EDGES),
        'band_lower_edge_labels_all': list(EDGE_LABELS),
        'band_lower_edge_note':
            'tick labels give the lower lambda edge of each band; band 0 is '
            'labelled 0 (its lower edge is -0.05, below the spectrum). Bands '
            'are drawn at equal width, not at their lambda spacing.',
        'hf_shading': {
            'from_x': LAYOUT['high_frequency_shading_xspan'][0],
            'to_x': LAYOUT['high_frequency_shading_xspan'][1],
            'meaning': 'high frequency, lambda >= 0.60 (bands 7-10)',
        },
    }


def nice_ticks(lo, hi, want=4):
    """Round tick positions inside [lo, hi], chosen from a 1/2/2.5/5 ladder."""
    span = float(hi) - float(lo)
    if span <= 0:
        return [lo]
    raw = span / float(want)
    mag = 10.0 ** math_floor_log10(raw)
    for mult in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = mult * mag
        if span / step <= want + 0.5:
            break
    out, k = [], math_ceil(lo / step)
    while k * step <= hi + 1e-9:
        out.append(round(k * step, 10))
        k += 1
    return out or [lo, hi]


def math_floor_log10(v):
    import math
    return math.floor(math.log10(abs(v))) if v else 0


def math_ceil(v):
    import math
    return int(math.ceil(v - 1e-12))


def fmt_tick(v, unit=None):
    if unit == 'percent':
        return ('%g%%' % round(v, 6))
    s = ('%g' % round(v, 6))
    return s.replace('-', '−')


# ------------------------------------------------------------------ Fig. 3 --

def cka_panel(ds, pretty, layer):
    """One dataset's band-CKA curves, teacher ceiling at 1."""
    cur = IQR[ds][layer]['curves']
    nfr = IQR[ds][layer]['n_frames']
    ylim = RANGES['%s/%s' % (ds, layer)]['ylim']
    ticks = [t for t in (0.0, 0.25, 0.5, 0.75, 1.0) if ylim[0] - 1e-9 <= t <= ylim[1] + 1e-9]
    xs = [float(i) for i in range(NBANDS)]

    def line(variant, name, color, draw):
        c = cur[variant]
        return {
            'name': name, 'variant': variant, 'kind': 'line', 'x': list(xs),
            'q25': list(c['q25']), 'q50': list(c['median']), 'q75': list(c['q75']),
            'n_frames_finite': [nfr] * NBANDS,
            'color': color, 'draw': draw, 'zorder': 3,
            'statistic': 'median over frames of CKA(model, Dense teacher) divided '
                         'by the teacher re-truncation noise ceiling of the same '
                         'frame and band; ribbon is the interquartile range',
        }

    return {
        'id': 'cka_' + ds,
        'dataset': ds,
        'column': pretty,
        'layer': layer,
        'title': pretty,
        'x': x_axis(),
        'y': {
            'label': 'CKA to Dense teacher\n(÷ noise ceiling)',
            'lim': list(ylim), 'scale': 'linear', 'ticks': ticks,
            'tick_labels': [('%g' % t) for t in ticks],
            'tick_label_side': 'left',
        },
        'series': [
            {'name': 'Teacher, re-truncated input (= 1)', 'variant': 'T_union2',
             'kind': 'hline', 'y': float(LAYOUT['noise_ceiling']),
             'color': C['C_NOISE'], 'linewidth_pt': 1.2, 'zorder': 2,
             'style': 'dashed',
             'note': 'noise ceiling; every curve is divided by it per frame and '
                     'band, so it is 1 by construction'},
            line('T_raw', 'Teacher, sparse input', C['C_TEACHER'],
                 {'markers': 'hollow', 'linewidth_pt': 1.2, 'ribbon_alpha': 0.14}),
            line('S_liseer', 'LiSeer-P', C['C_LISEER'],
                 {'linewidth_pt': 1.5, 'ribbon_alpha': 0.14}),
            line('S_lieq', 'LiEqualizer', C['C_LIEQ'],
                 {'linewidth_pt': 1.5, 'ribbon_alpha': 0.14}),
        ],
        'annotations': [
            {'text': 'high frequency\n(λ ≥ .60)', 'x': 1.0, 'y': 1.0,
             'coords': 'axes_fraction', 'offset_pt': [-4.0, -4.0],
             'ha': 'right', 'va': 'top', 'color': C['INK2'], 'fontsize_pt': 6.3},
        ],
        'legend': None,
    }


def gap_panel(ds, pretty, layer):
    """The strip under each CKA panel: share of the LiSeer-P-to-ceiling gap closed."""
    strip = RANGES['%s/%s/strip' % (ds, layer)]
    gap = strip['gap_pct']
    assert len(gap) == NBANDS, (ds, layer, len(gap))
    ylim = strip['ylim']
    ticks = nice_ticks(0.0, ylim[1], 3)
    nfr = IQR[ds][layer]['n_frames']
    return {
        'id': 'gap_' + ds,
        'dataset': ds,
        'column': pretty,
        'layer': layer,
        'title': 'gap closed',
        'height_ratio': 0.62,
        'x': x_axis(),
        'y': {
            'label': 'gap\nclosed', 'lim': [0.0, ylim[1]], 'scale': 'linear',
            'ticks': ticks, 'tick_labels': [fmt_tick(t, 'percent') for t in ticks],
            'tick_label_side': 'left', 'unit': 'percent',
        },
        'series': [{
            'name': 'Gap closed by LiEqualizer', 'kind': 'bar',
            'band_index': list(range(NBANDS)),
            'x': [float(i) for i in range(NBANDS)],
            'offset': 0.0, 'width': 0.72,
            'y': list(gap), 'n_frames_finite': [nfr] * NBANDS,
            'color': C['C_GAP'], 'zorder': 3,
            'statistic': '100 x median over frames of (CKA_LiEqualizer - '
                         'CKA_LiSeer-P) / (1 - CKA_LiSeer-P), the share of the '
                         'remaining distance to the noise ceiling that '
                         'LiEqualizer closes',
        }],
        'annotations': [],
        'legend': None,
    }


def fig3(layer, paper_ref, field_desc, out_name):
    panels = []
    for ds, pretty in DATASETS:
        panels.append(cka_panel(ds, pretty, layer))
    for ds, pretty in DATASETS:
        panels.append(gap_panel(ds, pretty, layer))
    frames = sorted({IQR[ds][layer]['n_frames'] for ds, _ in DATASETS})
    doc = collections.OrderedDict()
    doc['figure'] = 'fig1_band_cka_3ds'
    doc['paper_ref'] = paper_ref
    doc['description'] = (
        'Per-band CKA between each model and the Dense teacher on ' + field_desc +
        ', divided by the teacher’s re-truncation noise ceiling, on '
        'SemanticKITTI, nuScenes and Waymo. The strip under each panel is the '
        'share of the LiSeer-P-to-ceiling gap that LiEqualizer closes in that band.')
    doc['protocol'] = {
        'frames': {'count': frames[0] if len(frames) == 1 else frames,
                   'per_dataset': {ds: IQR[ds][layer]['n_frames'] for ds, _ in DATASETS}},
        'datasets': [p for _, p in DATASETS],
        'field': field_desc,
        'support': 'the teacher support Ω_T, read as one domain',
        'bands': {'count': NBANDS, 'edges': list(BAND_EDGES),
                  'high_frequency': 'λ ≥ 0.60 (bands 7-10)',
                  'low_frequency': 'λ < 0.08 (bands 0-2)'},
        'metric': 'linear CKA between the band-k filtered model field and the '
                  'band-k filtered Dense-teacher field over the same cells',
        'normalisation': 'per frame and band: CKA(model, Dense teacher) / '
                         'CKA(Teacher re-truncated input, Dense teacher). The '
                         'values are already normalised; they are not '
                         'renormalised anywhere downstream.',
        'statistic': 'median over frames, with the interquartile range as the ribbon',
        'gap_closed': '100 x median over frames of (CKA_LiEqualizer - CKA_LiSeer-P) '
                      '/ (1 - CKA_LiSeer-P)',
    }
    doc['units'] = {
        'x': 'graph-frequency band index 0-10 (tick label = lower λ edge of the band)',
        'y_curves': 'CKA ÷ noise ceiling (dimensionless; 1 = Teacher, re-truncated input)',
        'y_strip': 'percent of the LiSeer-P-to-ceiling CKA gap closed by LiEqualizer',
    }
    doc['colors'] = dict(C)
    doc['figure_size_in'] = [7.2, 4.1]
    doc['figure_legends'] = [{
        'entries': [
            {'label': 'Teacher, re-truncated (= 1)', 'kind': 'line',
             'color': C['C_NOISE'], 'linewidth_pt': 1.2, 'style': 'dashed'},
            {'label': 'Teacher, sparse input', 'kind': 'line',
             'color': C['C_TEACHER'], 'linewidth_pt': 1.2, 'style': 'solid',
             'marker': {'shape': 'o', 'face': '#ffffff', 'edge': C['C_TEACHER'], 'size_pt': 3.6}},
            {'label': 'LiSeer-P', 'kind': 'line', 'color': C['C_LISEER'],
             'linewidth_pt': 1.5, 'style': 'solid',
             'marker': {'shape': 'o', 'face': C['C_LISEER'], 'edge': '#ffffff', 'size_pt': 3.6}},
            {'label': 'LiEqualizer', 'kind': 'line', 'color': C['C_LIEQ'],
             'linewidth_pt': 1.5, 'style': 'solid',
             'marker': {'shape': 'o', 'face': C['C_LIEQ'], 'edge': '#ffffff', 'size_pt': 3.6}},
            {'label': 'Gap closed by LiEqualizer', 'kind': 'patch', 'color': C['C_GAP']},
        ],
        'loc': 'upper center', 'ncol': 5,
    }]
    doc['figure_texts'] = [{
        'text': 'Graph-frequency band  (lower edge of λ)',
        'x_fig_fraction': 0.5, 'y_fig_fraction': 0.04,
        'ha': 'center', 'va': 'bottom', 'color': C['INK'],
        'fontsize_pt': 7.5, 'bold': False,
    }]
    doc['grid'] = {'cols': 3, 'rows': [['cka_SK', 'cka_NU', 'cka_WA'],
                                       ['gap_SK', 'gap_NU', 'gap_WA']]}
    doc['panels'] = panels
    return dump(doc, out_name)


# ------------------------------------------------------------------ Fig. 4 --

ROW_META = [
    ('a', 'a', 'TPV features (xy plane): real vs. pseudo energy', 'tpv', 'stack'),
    ('b', 'b', 'TPV features: what LiEqualizer changes vs. LiSeer-P (paired, per frame)',
     'tpv', 'paired'),
    ('c', 'c', 'Voxel logits: real vs. pseudo energy', 'voxel', 'stack'),
    ('d', 'd', 'Voxel logits: what LiEqualizer changes vs. LiSeer-P (paired, per frame)',
     'voxel', 'paired'),
]


def stack_panel(row, ds, pretty, title):
    p = NUM['fig2_energy_3ds'][ds][row]
    cap = p['cap']
    ylim = [0.0, p['ylim'][1]]
    hatch = list(p['hatched_bands_not_supported_by_sparse_scan'])
    ticks = nice_ticks(0.0, cap, 3)
    xs = [float(i) for i in range(NBANDS)]

    def pair(variant, label, dark, light, offset):
        real = p['real'][variant]
        pseudo = p['pseudo'][variant]
        return [
            {'name': label + ', real', 'variant': variant, 'part': 'real',
             'kind': 'bar', 'band_index': list(range(NBANDS)),
             'x': [v + offset for v in xs], 'y': list(real),
             'offset': offset, 'width': 0.38, 'color': dark, 'zorder': 3,
             'hatch_bands': hatch,
             'statistic': 'median over frames of the band energy a ridge fit on '
                          'held-out azimuthal sectors attributes to the teacher, '
                          'divided by the teacher’s own band energy'},
            {'name': label + ', pseudo', 'variant': variant, 'part': 'pseudo',
             'kind': 'bar', 'band_index': list(range(NBANDS)),
             'x': [v + offset for v in xs], 'y': list(pseudo),
             'bottom': list(real), 'stacked_on': label + ', real',
             'offset': offset, 'width': 0.38, 'color': light, 'zorder': 3,
             'hatch_bands': hatch,
             'statistic': 'the remainder of the band energy, which the teacher '
                          'does not explain'},
        ]

    series = [{
        'name': 'Dense teacher (= 1)', 'kind': 'hline', 'y': 1.0,
        'color': C['INK2'], 'linewidth_pt': 1.0, 'zorder': 2, 'style': 'dashed',
    }]
    series += pair('S_liseer', 'LiSeer-P', C['C_LISEER'], C['C_LISEER_LIGHT'], -0.20)
    series += pair('S_lieq', 'LiEqualizer', C['C_LIEQ'], C['C_LIEQ_LIGHT'], 0.20)
    for variant, label, col, off in (('S_liseer', 'LiSeer-P', C['C_LISEER'], -0.20),
                                     ('S_lieq', 'LiEqualizer', C['C_LIEQ'], 0.20)):
        series.append({
            'name': label + ', energy distribution', 'variant': variant,
            'part': 'total', 'kind': 'line', 'x': [v + off for v in xs],
            'q50': list(p['bar_top'][variant]), 'color': col, 'zorder': 4,
            'draw': {'linewidth_pt': 1.2, 'dashed': True},
            'statistic': 'the top of the stacked bar: total band energy relative '
                         'to the teacher’s',
        })

    over = []
    for o in p.get('overflow', []):
        over.append({
            'band': o['band'],
            'variant': o['model'],
            'offset': -0.20 if o['model'] == 'S_liseer' else 0.20,
            'true_value': o['true_value'],
            'break_y': min(o['break_center_y'], cap * 0.93),
            'color': C['C_LISEER'] if o['model'] == 'S_liseer' else C['C_LIEQ'],
        })

    return {
        'id': row + '_' + ds, 'row': row, 'dataset': ds, 'column': pretty,
        'row_title': title, 'title': pretty,
        'x': x_axis(),
        'y': {'label': 'Energy rel. to\nDense teacher', 'lim': ylim,
              'scale': 'linear', 'ticks': ticks,
              'tick_labels': [fmt_tick(t) for t in ticks], 'tick_label_side': 'left'},
        'series': series,
        'overflow': over,
        'hatched_bands': hatch,
        'hatch_meaning': 'the sparse scan does not support this band: the '
                         'structure is absent from the observation rather than '
                         'attenuated in it, so no objective can recover it',
        'annotations': [],
        'legend': None,
    }


def paired_panel(row, ds, pretty, title):
    p = NUM['fig2_energy_3ds'][ds][row]
    sa = p['split_axis']
    hatch = list(p['hatched_bands_not_supported_by_sparse_scan'])
    xs = [float(i) for i in range(NBANDS)]
    ticks = sorted(list(sa['neg_ticks']) + [0.0] + list(sa['pos_ticks']))
    labels = []
    for t in ticks:
        if t == 0:
            labels.append('0')
        elif t > 0:
            labels.append('+' + ('%g' % round(t, 6)))
        else:
            labels.append(('%g' % round(t, 6)).replace('-', '−'))
    return {
        'id': row + '_' + ds, 'row': row, 'dataset': ds, 'column': pretty,
        'row_title': title, 'title': pretty,
        'x': x_axis(),
        'y': {
            'label': 'LiEqualizer − LiSeer-P\n(÷ Dense teacher)',
            'lim': [-sa['N'], sa['P']], 'scale': 'linear',
            'ticks': ticks, 'tick_labels': labels, 'tick_label_side': 'left',
            'split_axis': {'P': sa['P'], 'N': sa['N'], 'h': sa['h'],
                           'note': 'zero sits at the fraction h of the panel '
                                   'height from the bottom; [0, P] is mapped '
                                   'above it and [-N, 0] below it, each side '
                                   'tight to its own data'},
        },
        'series': [
            {'name': 'Δ real energy (LiEqualizer − LiSeer-P)', 'part': 'real',
             'kind': 'bar', 'band_index': list(range(NBANDS)),
             'x': [v - 0.0 for v in xs], 'y': list(p['paired_real_median']),
             'offset': -0.19, 'width': 0.38, 'color': C['C_LIEQ'], 'zorder': 3,
             'hatch_bands': hatch,
             'statistic': 'median over frames of the per-frame difference in the '
                          'energy the teacher explains'},
            {'name': 'Δ pseudo energy (LiEqualizer − LiSeer-P)', 'part': 'pseudo',
             'kind': 'bar', 'band_index': list(range(NBANDS)),
             'x': [v + 0.0 for v in xs], 'y': list(p['paired_pseudo_median']),
             'offset': 0.19, 'width': 0.38, 'color': C['C_LIEQ_LIGHT'], 'zorder': 3,
             'hatch_bands': hatch,
             'statistic': 'median over frames of the per-frame difference in the '
                          'energy the teacher does not explain'},
            {'name': 'zero', 'kind': 'hline', 'y': 0.0, 'color': C['AXIS'],
             'linewidth_pt': 1.0, 'zorder': 2, 'style': 'solid'},
        ],
        'hatched_bands': hatch,
        # y is a fraction of the plot height from the bottom, so the bars that
        # rise go with the label near the top
        'annotations': [
            {'text': 'real ↑', 'x': 0.015, 'y': 0.97, 'coords': 'axes_fraction',
             'ha': 'left', 'va': 'bottom', 'color': C['C_LIEQ'], 'fontsize_pt': 6.3},
            {'text': 'pseudo ↓', 'x': 0.015, 'y': 0.06, 'coords': 'axes_fraction',
             'ha': 'left', 'va': 'top', 'color': C['C_LIEQ_LIGHT'], 'fontsize_pt': 6.3},
        ],
        'legend': None,
    }


def fig4():
    panels, rows = [], []
    for row, _, title, _, kind in ROW_META:
        ids = []
        for ds, pretty in DATASETS:
            panels.append((stack_panel if kind == 'stack' else paired_panel)(
                row, ds, pretty, title))
            ids.append(row + '_' + ds)
        rows.append(ids)
    doc = collections.OrderedDict()
    doc['figure'] = 'fig2_energy_3ds'
    doc['paper_ref'] = 'Fig. 4'
    doc['description'] = (
        'Band energy split into the part a ridge fit on held-out azimuthal '
        'sectors attributes to the teacher (real) and the part it does not '
        '(pseudo), on the TPV feature planes and the voxel logits, for '
        'SemanticKITTI, nuScenes and Waymo. Rows b and d are the paired '
        'per-frame change from LiSeer-P to LiEqualizer.')
    doc['protocol'] = {
        'datasets': [p for _, p in DATASETS],
        'statistic': 'median over frames; rows b and d pair the two students '
                     'frame by frame before taking the median',
        'split': 'ridge regression on held-out 60-degree azimuthal sectors, so '
                 'the real part is an out-of-sample fit',
        'normalisation': 'divided by the Dense teacher’s own band energy',
        'bands': {'count': NBANDS, 'edges': list(BAND_EDGES),
                  'high_frequency': 'λ ≥ 0.60 (bands 7-10)'},
        'row_axes': NUM['row_axes_fig2'],
    }
    doc['units'] = {
        'x': 'graph-frequency band index 0-10 (tick label = lower λ edge of the band)',
        'y_a_c': 'band energy divided by the Dense teacher’s band energy (1 = the teacher’s own level)',
        'y_b_d': 'LiEqualizer minus LiSeer-P, divided by the Dense teacher’s band energy',
    }
    doc['colors'] = dict(C)
    doc['figure_size_in'] = [7.2, 7.6]
    doc['figure_legends'] = [{
        'entries': [
            {'label': 'LiSeer-P real / pseudo', 'kind': 'patch2',
             'color': C['C_LISEER'], 'color2': C['C_LISEER_LIGHT']},
            {'label': 'LiEqualizer real / pseudo', 'kind': 'patch2',
             'color': C['C_LIEQ'], 'color2': C['C_LIEQ_LIGHT']},
            {'label': 'energy distribution', 'kind': 'line', 'color': C['C_LIEQ'],
             'linewidth_pt': 1.2, 'style': 'dashed'},
            {'label': 'Dense teacher = 1', 'kind': 'line', 'color': C['INK2'],
             'linewidth_pt': 1.0, 'style': 'dashed'},
            {'label': 'broken bar', 'kind': 'break', 'color': C['C_LISEER']},
            {'label': 'unsupported by sparse scan', 'kind': 'hatch', 'color': C['C_LIEQ_LIGHT']},
        ],
        'loc': 'lower center', 'ncol': 6,
    }]
    doc['figure_texts'] = [{
        'text': 'Graph-frequency band  (lower edge of λ)',
        'x_fig_fraction': 0.5, 'y_fig_fraction': 0.03,
        'ha': 'center', 'va': 'bottom', 'color': C['INK'],
        'fontsize_pt': 7.5, 'bold': False,
    }]
    doc['grid'] = {'cols': 3, 'rows': rows,
                   'row_titles': [m[2] for m in ROW_META]}
    doc['panels'] = panels
    return dump(doc, 'fig4_energy_real.json')


def main():
    out = [
        fig3('tpv', 'Fig. 3a', 'the TPV xy feature plane',
             'fig3a_band_cka_feature_planes.json'),
        fig3('logits', 'Fig. 3b', 'the 8-class voxel logits over the teacher support',
             'fig3b_band_cka_voxel_logits.json'),
        fig4(),
    ]
    for p in out:
        print('wrote %s (%.1f kB)' % (p, os.path.getsize(p) / 1024.0))


if __name__ == '__main__':
    main()
