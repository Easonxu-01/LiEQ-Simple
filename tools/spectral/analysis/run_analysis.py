#!/usr/bin/env python
"""Reported spectral quantities from one dump directory (CPU only).

    python tools/spectral/analysis/run_analysis.py --dump DUMP_DIR --compare A B --out OUT_DIR

A and B are model role names given at dump time (A = model under study, B = baseline). Every quantity is
computed from the summaries of DUMP_DIR alone; runs are never combined. To compare models that were dumped
in different runs, dump them together (one run, same frames, same teacher arms).

Writes to OUT_DIR (per-frame values, band medians and quartiles where reported):
    fig3_band_cka.json            Fig. 3  band-wise CKA / ceiling, both layers, and the gap-closed strip for (A, B)
    fig4_energy.json              Fig. 4  real / pseudo energy per band (ceiling units) and the paired change A - B
    planes_boundary.json          spurious / missed boundary rate per TPV plane, frames where A < B
    appendix_d_spurious_hf.json   HF excess where the teacher is smooth / the HF gate is closed (ceiling = 1)
    appendix_d_pseudo_hf.json     HF pseudo-energy share per role
    appendix_d_paired_sparse.json each model vs the teacher-on-sparse control (and the ceiling)
    analysis_info.json            roles, frames, band edges
The raw-dump quantities of Appendix D have their own scripts: gate_test.py and crossterm.py.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import reducers as RD  # noqa: E402
import spectral_io as IO  # noqa: E402

LAYER_TITLE = {'planes': 'TPV feature planes (xy, neck scale 0)', 'voxel_logits': 'voxel logits'}


def plain(x, keep_per_frame=True):
    if isinstance(x, dict):
        return {str(k): plain(v, keep_per_frame) for k, v in x.items()
                if keep_per_frame or not str(k).startswith('per_frame')}
    if isinstance(x, (list, tuple)):
        return [plain(v, keep_per_frame) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist(), keep_per_frame)
    if isinstance(x, np.generic):
        return plain(x.item(), keep_per_frame)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def write(out, name, obj, keep_per_frame):
    p = os.path.join(out, name)
    tmp = p + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(plain(obj, keep_per_frame), f, indent=1, allow_nan=False)
        f.write('\n')
    os.replace(tmp, p)
    return p


def analyse(dump, A, B, roles_kw, frames_json=None):
    frames = IO.eval_frames(dump, frames_json)
    roles = IO.resolve_roles(dump, frames, **roles_kw)
    for m in (A, B):
        if m not in roles['models']:
            raise SystemExit(f'--compare: {m!r} is not a model role of this dump (models: {roles["models"]})')
    if A == B:
        raise SystemExit('--compare needs two different model roles')
    X = IO.DumpArrays(dump, frames, roles)
    res = {}
    res['fig3'] = {layer: RD.band_cka(X, layer, A, B) for layer in RD.LAYERS}
    fig4 = {}
    for layer in RD.LAYERS:
        E = RD.energy(X, layer, A, B)
        E['bands_unsupported_by_sparse_scan'] = RD.unsupported_bands(E, roles['sparse'], A, B)
        fig4[layer] = E
    res['fig4'] = fig4
    res['boundary'] = RD.boundary_rates(X, A, B)
    res['spurious_hf'] = {layer: RD.spurious_hf(X, layer) for layer in RD.LAYERS}
    res['pseudo_hf'] = {layer: RD.pseudo_hf_shares(X, layer) for layer in RD.LAYERS}
    res['paired_sparse'] = {layer: RD.paired_vs_sparse(X, layer) for layer in RD.LAYERS}
    info = dict(frames=frames, n_frames=len(frames), roles={k: roles[k] for k in ('reference', 'ceiling', 'sparse', 'models')},
                compare=dict(A=A, B=B), band_edges=IO.BAND_EDGES, hf_bands=IO.HF_BANDS,
                rho_crop_rows_xy=X.rho_crop_rows, layers=LAYER_TITLE)
    return res, info, X


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', required=True, help='dump directory (contains TPV/ and LOGITS/)')
    ap.add_argument('--compare', nargs=2, required=True, metavar=('A', 'B'), help='model roles: A = model under study, B = baseline')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--models', nargs='+', default=None, help='model roles to report (default: every non-teacher variant)')
    ap.add_argument('--ceiling-role', default=None, help=f'teacher re-run on the re-truncated input (default {IO.DEFAULT_ROLES["ceiling"]})')
    ap.add_argument('--sparse-role', default=None, help=f'teacher fed the sparse scan (default {IO.DEFAULT_ROLES["sparse"]})')
    ap.add_argument('--reference-role', default=None, help=f'dense teacher (default {IO.DEFAULT_ROLES["reference"]})')
    ap.add_argument('--frames-json', default=None, help='frame manifest (default DUMP/TPV/FRAMES.json)')
    ap.add_argument('--no-per-frame', action='store_true', help='omit per-frame arrays from the outputs')
    a = ap.parse_args(argv)
    A, B = a.compare
    res, info, _ = analyse(a.dump, A, B, dict(reference=a.reference_role, ceiling=a.ceiling_role, sparse=a.sparse_role,
                                              models=a.models), a.frames_json)
    os.makedirs(a.out, exist_ok=True)
    keep = not a.no_per_frame
    head = dict(info=dict(compare=info['compare'], roles=info['roles'], n_frames=info['n_frames']))
    written = [
        write(a.out, 'analysis_info.json', info, keep),
        write(a.out, 'fig3_band_cka.json', dict(head, definition=(
            'per frame and band: CKA(role, reference) / CKA(ceiling, reference) on the teacher support; curve = median over '
            'frames, ribbon = q25..q75; gap_closed.median_pct = 100 * median_f((A - B) / (ceiling - B))'), layers=res['fig3']), keep),
        write(a.out, 'fig4_energy.json', dict(head, definition=(
            'per frame and band: real = clip(recovery, 0, 1); pseudo = pseudo_share / total share of the ceiling. Reported '
            'values: median_f(real) / median_f(real of the ceiling) and median_f(pseudo) / median_f(useful/total of the ceiling); '
            'total = real + pseudo; paired = median_f of per-frame A - B in the same units'), layers=res['fig4']), keep),
        write(a.out, 'planes_boundary.json', dict(head, unit='percent', definition=(
            'per plane: spurious-boundary rate (teacher-same 4-neighbour pairs the model splits) and missed-boundary rate '
            '(teacher-different pairs the model merges) on the class-argmax planes; quartiles over frames finite for both; '
            'frames_A_lower = frames with A < B'), rates=res['boundary']), keep),
        write(a.out, 'appendix_d_spurious_hf.json', dict(head, definition=(
            'per frame: mean HF energy above the reference on gate-closed cells (planes: inside teacher class interiors) or '
            'where the HF gate is closed (voxel logits), divided by the median of the ceiling'), layers=res['spurious_hf']), keep),
        write(a.out, 'appendix_d_pseudo_hf.json', dict(head, definition='median over frames of the pseudo-energy share above lambda = 0.60',
                                                      layers=res['pseudo_hf']), keep),
        write(a.out, 'appendix_d_paired_sparse.json', dict(head, definition=(
            'per model: frame-paired HF pseudo share vs the teacher-on-sparse control and the ceiling'), layers=res['paired_sparse']), keep),
    ]
    for p in written:
        print('wrote', p)


if __name__ == '__main__':
    main()
