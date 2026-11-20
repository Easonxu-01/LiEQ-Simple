"""Readers for the per-frame spectral summaries written by the dump stage.

A dump directory has the layout

    <dump>/TPV/FRAMES.json                 frame manifest ('eval', 'dev')
    <dump>/TPV/summary200/frame_XXXXXX.json     feature-plane (TPV) summaries
    <dump>/LOGITS/summary200/frame_XXXXXX.json  voxel-logit summaries (M1..M7)

Every array built here is a plain stack of the per-frame summary fields (frames on axis 0,
bands on axis 1). Missing entries and JSON nulls become NaN, with the same conventions as the
reference aggregation that produced the published numbers, so all downstream reductions see
bit-identical float64 inputs.

Role names: the models are whatever names were passed to the dump stage. The teacher arms
use the fixed protocol names of the summary schema:
    reference  dense teacher, the target every model is compared with
    ceiling    teacher re-run on the re-truncated input (the attainable ceiling)
    sparse     teacher fed the sparse scan directly (the teacher-on-sparse control)
"""
import glob
import json
import os

import numpy as np

NB = 11
BAND_EDGES = [-0.05, 0.01, 0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, 1.50, 2.02]
HF_BANDS = [k for k in range(NB) if BAND_EDGES[k] >= 0.60 - 1e-12]       # bands 7..10
PLANES = ('xy', 'yz', 'zx')
DEFAULT_ROLES = dict(reference='T_union', ceiling='T_union2', sparse='T_raw')


# ----------------------------------------------------------------------------- conversions
def band_vec_planes(x, n=NB):
    """Band vector from a feature-plane summary: None -> all NaN, null entries -> NaN."""
    return np.array([np.nan if v is None else v for v in (x if x is not None else [None] * n)], float)


def band_vec_logits(x, n=NB):
    """Band vector from a voxel-logit summary: None or wrong length -> all NaN, null entries -> NaN."""
    if x is None:
        return np.full(n, np.nan)
    a = np.array([np.nan if v is None else v for v in x], float)
    return a if a.size == n else np.full(n, np.nan)


def scalar(x):
    return np.nan if x is None else float(x)


def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


# ----------------------------------------------------------------------------- frames
def frame_ids(summary_dir):
    return sorted(int(os.path.basename(p)[6:12]) for p in glob.glob(os.path.join(summary_dir, 'frame_*.json')))


def eval_frames(dump, frames_json=None):
    """Evaluation frames = manifest 'eval' minus 'dev' (development frames never enter a statistic); without an 'eval'
    list, every summarised frame minus 'dev'; without a manifest, every frame that has both summaries."""
    tdir, ldir = os.path.join(dump, 'TPV', 'summary200'), os.path.join(dump, 'LOGITS', 'summary200')
    have_t, have_l = set(frame_ids(tdir)), set(frame_ids(ldir))
    path = frames_json or os.path.join(dump, 'TPV', 'FRAMES.json')
    if os.path.exists(path):
        fr = json.load(open(path))
        dev = set(map(int, fr.get('dev', [])))
        want = sorted(set(map(int, fr['eval'])) - dev) if 'eval' in fr else sorted((have_t & have_l) - dev)
        missing = [f for f in want if f not in have_t or f not in have_l]
        if missing:
            raise SystemExit(f'{len(missing)} evaluation frames have no summary (first: {missing[:5]}); '
                             'the dump is incomplete')
        return want
    return sorted(have_t & have_l)


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------- roles
def resolve_roles(dump, frames, reference=None, ceiling=None, sparse=None, models=None):
    """Teacher arms (schema names, overridable) and model roles (auto: every summary variant that is
    not a teacher arm and is present in both summaries)."""
    roles = dict(reference=reference or DEFAULT_ROLES['reference'], ceiling=ceiling or DEFAULT_ROLES['ceiling'],
                 sparse=sparse or DEFAULT_ROLES['sparse'])
    f0 = frames[0]
    L = read_json(os.path.join(dump, 'LOGITS', 'summary200', f'frame_{f0:06d}.json'))
    T = read_json(os.path.join(dump, 'TPV', 'summary200', f'frame_{f0:06d}.json'))
    lv = list(L.get('variants') or [])
    tv = list(get(T, 'xy', 'b_dct_standard', 'cka') or {})
    for k in ('ceiling', 'sparse'):
        if roles[k] not in lv or roles[k] not in tv:
            raise SystemExit(f'{k} role {roles[k]!r} not found in the summaries (voxel variants {lv}, plane variants {tv})')
    teachers = set(roles.values())
    if models is None:
        models = [v for v in lv if v not in teachers and v in tv]
    for m in models:
        if m not in lv or m not in tv:
            raise SystemExit(f'model role {m!r} not found in the summaries (voxel variants {lv}, plane variants {tv})')
    if not models:
        raise SystemExit('no model role found in the summaries')
    roles['models'] = list(models)
    roles['voxel_variants'] = lv
    return roles


# ----------------------------------------------------------------------------- per-frame arrays
class DumpArrays:
    """Per-frame arrays of one dump directory (one run; no arrays are ever combined across runs).

    planes['cka'][v]            [F, 11]  CKA to the reference per band on the teacher support (xy, neck s0, DCT)
    planes['excess_gc'][v]      [F]      mean HF excess over the reference on gate-closed cells (teacher-smooth domain)
    planes['sbr'|'mbr'][p][v]   [F]      spurious / missed boundary rate on the argmax planes p in xy, yz, zx
    rp['planes'|'voxel_logits'][v][q]    real/pseudo split (q in recovery, useful_share, pseudo_share [F, 11];
                                         pseudo_share_hf [F])
    voxel['cka'][v]             [F, 11]  CKA to the reference per band on the teacher support (voxel logits)
    voxel['excess_gc'][v]       [F]      mean HF excess over the reference where the HF gate is closed
    """

    RP_FIELDS_BAND = ('recovery', 'useful_share', 'pseudo_share')
    RP_FIELDS_SCALAR = ('pseudo_share_hf',)
    RP_KEY = {'planes': 'M7_phi', 'voxel_logits': 'M6_phi'}

    def __init__(self, dump, frames, roles):
        self.dump, self.frames, self.roles = dump, list(frames), roles
        tdir, ldir = os.path.join(dump, 'TPV', 'summary200'), os.path.join(dump, 'LOGITS', 'summary200')
        arms = [roles['ceiling'], roles['sparse']] + roles['models']
        acc = {}

        def put(key, val):
            acc.setdefault(key, []).append(val)

        for f in self.frames:
            d = read_json(os.path.join(tdir, f'frame_{f:06d}.json'))
            xb = d['xy']['b_dct_standard']
            for v in arms:
                put(('cka_planes', v), band_vec_planes(xb['cka'][v].get('OmegaT')))
                e = xb['excess'][v].get('gate_closed_oldint') or {}
                put(('exc_planes', v), e.get('excess_mean', np.nan))
            for pl in PLANES:
                for v in roles['models']:
                    sm = d[pl]['d_probC_sbr_mbr'][v]
                    put(('sbr', pl, v), sm['SBR'])
                    put(('mbr', pl, v), sm['MBR'])
            put(('crop_xy',), d['xy'].get('rho_crop_rows'))
            L = read_json(os.path.join(ldir, f'frame_{f:06d}.json'))
            for layer, key in self.RP_KEY.items():
                for v in arms:
                    for q in self.RP_FIELDS_BAND:
                        put(('rp', layer, v, q), band_vec_logits(get(L, key, v, q)))
                    for q in self.RP_FIELDS_SCALAR:
                        put(('rp', layer, v, q), scalar(get(L, key, v, q)))
            for v in arms:
                put(('cka_voxel', v), band_vec_logits(get(L, 'M1', v, 'OmegaT')))
                put(('exc_voxel', v), scalar(get(L, 'M3', v, 'gate_closed', 'excess_mean')))

        A = {k: np.array(v, float) if k != ('crop_xy',) else v for k, v in acc.items()}
        self.planes = dict(cka={v: A[('cka_planes', v)] for v in arms}, excess_gc={v: A[('exc_planes', v)] for v in arms},
                           sbr={pl: {v: A[('sbr', pl, v)] for v in roles['models']} for pl in PLANES},
                           mbr={pl: {v: A[('mbr', pl, v)] for v in roles['models']} for pl in PLANES})
        self.voxel = dict(cka={v: A[('cka_voxel', v)] for v in arms}, excess_gc={v: A[('exc_voxel', v)] for v in arms})
        self.rp = {layer: {v: {q: A[('rp', layer, v, q)] for q in self.RP_FIELDS_BAND + self.RP_FIELDS_SCALAR}
                           for v in arms} for layer in self.RP_KEY}
        crops = set(A[('crop_xy',)])
        self.rho_crop_rows = crops.pop() if len(crops) == 1 else None

    def cka(self, layer):
        return self.planes['cka'] if layer == 'planes' else self.voxel['cka']

    def excess_gc(self, layer):
        return self.planes['excess_gc'] if layer == 'planes' else self.voxel['excess_gc']
