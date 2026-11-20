#!/usr/bin/env python
"""Appendix D.1 cross-term of the recovery decomposition, from the raw frames of a dump (CPU).

    python tools/spectral/analysis/crossterm.py --dump DUMP_DIR --out OUT_DIR

For a model's band signal Y and its ridge prediction Yhat from the reference (dense teacher) band signal X on held-out
folds (nested ridge, outer folds = even / odd 60-degree phi sectors; the recovery of the summaries), with P = Yhat - mean
and E = Y - Yhat:  1 = a + b + c,  a = |P|^2 / SST,  b = |E|^2 / SST,  c = 2 <P, E> / SST,  forward R^2 = 1 - b = a + c.
Reported per layer, model and band, over the raw frames that are evaluation frames: median_f c, median_f c / R^2 and the
number of frames with c > 0.

Layers
  planes        xy neck features (scale 0) on the teacher support, rho-cropped like the summaries; exact DCT band split
                (float64)
  voxel_logits  class logits on the 3D teacher support; Chebyshev band split of the HF distiller's support Laplacian
                (float32, K = 1024; all variants in one channel batch, in the summary's variant order)
Inputs: DUMP/LOGITS/dump200/<variant>/frame_XXXXXX.npz (neck_xy_s0; cs_logit, and sup_idx / sup_logit for the
reference), DUMP/LOGITS/dump200/_masks/frame_XXXXXX.npz (OmegaT_xy, field_shape), DUMP/LOGITS/summary200 (variant order;
recovery and energy share used as a per-frame consistency check), DUMP/TPV/summary200 (xy.rho_crop_rows).

The voxel layer is slow (about 25 min per frame with 4 threads). Per-frame results are cached under
OUT_DIR/crossterm_frames/; run several processes with disjoint --frames and finish with one run without --frames (it
reuses the cache and writes OUT_DIR/appendix_d_crossterm.json).
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.dirname(HERE))       # tools/spectral -> the shared 'common' package
import numpy as np  # noqa: E402
import torch  # noqa: E402

import _ridge_decomp as RDC  # noqa: E402
import spectral_io as IO  # noqa: E402

NB = IO.NB
LAYER_KEY = {'planes': 'M7_phi', 'voxel_logits': 'M6_phi'}
TOL = {'planes': 1e-12, 'voxel_logits': 1e-5}    # recomputed recovery / share vs the summary (max abs, per frame)


def kernels(repo_root=None):
    from common import model_io
    if repo_root:
        model_io.set_repo_root(repo_root)
    from common import logits_core as LC
    from common import tpv_metrics as TM
    return LC, TM


def rho_crop_xy(a, rows):
    return np.ascontiguousarray(a[..., rows:, :]) if rows else a


def load_raw(dump, layer, frame, variants, reference):
    d = os.path.join(dump, 'LOGITS', 'dump200')
    V = {}
    for v in variants:
        with np.load(os.path.join(d, v, f'frame_{frame:06d}.npz'), allow_pickle=False) as z:
            if layer == 'voxel_logits':
                fields = ('sup_idx', 'sup_logit') if v == reference else ('cs_logit',)
            else:
                fields = ('neck_xy_s0',)
            V[v] = {k: z[k] for k in fields}
    with np.load(os.path.join(d, '_masks', f'frame_{frame:06d}.npz'), allow_pickle=False) as z:
        masks = {k: z[k] for k in (('field_shape',) if layer == 'voxel_logits' else ('OmegaT_xy',))}
    return V, masks


def decompose(LC, bands, sector, dec, reference):
    out = {}
    for v in dec:
        energy = (bands[v].double() ** 2).sum(1).sum(1).numpy()
        d = RDC.decomp_bands(LC, bands[v], bands[reference], sector, NB)
        d['E'] = energy.tolist()
        d['share'] = (energy / max(float(energy.sum()), 1e-300)).tolist()
        out[v] = d
    return out


def voxel_frame(LC, V, masks, variants, dec, reference):
    fshape = tuple(map(int, masks['field_shape']))
    idx_np = np.asarray(V[reference]['sup_idx']).astype(np.int64)
    idx = torch.from_numpy(idx_np)
    assert ((idx >= 0).all() and (idx < torch.tensor(fshape)).all())
    flat = (idx[:, 0] * fshape[1] + idx[:, 1]) * fshape[2] + idx[:, 2]
    assert torch.unique(flat).numel() == flat.numel()
    vals = {v: np.asarray(V[v]['sup_logit' if v == reference else 'cs_logit'], np.float32) for v in variants}
    assert all(x.shape == (len(idx_np), vals[reference].shape[1]) for x in vals.values())
    C = vals[reference].shape[1]
    lap = LC.support3d('cpu')
    mask = torch.zeros(int(np.prod(fshape)), dtype=torch.float32)
    mask[flat] = 1
    lap.set_support(mask.reshape((1, 1) + fshape))
    xs = [LC.preprocess_sparse(torch.from_numpy(vals[v]).T.contiguous(), 'standard') for v in variants]
    xc = torch.cat(xs, 0)
    dense = torch.zeros((1, xc.shape[0], mask.numel()), dtype=torch.float32)
    dense[0][:, flat] = xc
    y = LC.cheb_band_signals(lap, dense.reshape((1, xc.shape[0]) + fshape), flat, LC.K_DEFAULT)
    bands = {v: y[:, C * i:C * i + C] for i, v in enumerate(variants)}
    return decompose(LC, bands, LC.phi_sector(idx[:, 1]), dec, reference), dict(n_support=int(flat.numel()))


def planes_frame(LC, TM, V, masks, variants, dec, reference, crop):
    om = rho_crop_xy(masks['OmegaT_xy'].astype(bool), crop)
    H, W = om.shape
    pb = TM.PlaneBasis.get(H, W, 'cpu', (False, False))
    dom = torch.from_numpy(om)
    cells = dom.reshape(-1).nonzero().flatten()
    cc = cells % W
    bands = {}
    for v in variants:
        field = rho_crop_xy(np.asarray(V[v]['neck_xy_s0'], np.float32), crop)
        x = TM.to_t(field, 'cpu', torch.float64)
        yp, _ = TM.preprocess(x, dom, 'standard')
        xh = pb.fwd(yp)
        bands[v] = torch.stack([TM.band_signal(xh, pb, pb.w_lam[k]).reshape(x.shape[0], -1)[:, cells] for k in range(NB)])
    return decompose(LC, bands, LC.phi_sector(cc), dec, reference), dict(n_cells=int(cells.numel()), rho_crop_rows=crop)


def check_against_summary(values, summary, layer, dec):
    """Recomputed recovery and energy share vs the summary of the same frame (guards the raw inputs and the kernels)."""
    key = LAYER_KEY[layer]
    worst = 0.0
    for v in dec:
        for mine, ref in (('r2', 'r2_student_from_teacher'), ('share', 'energy_share_student')):
            x = np.asarray(values[v][mine], float)
            y = np.asarray(summary[key][v][ref], float)
            assert x.shape == y.shape == (NB,) and np.isfinite(x).all() and np.isfinite(y).all(), (layer, v, mine)
            worst = max(worst, float(np.max(np.abs(x - y))))
    return dict(max_abs=worst, tolerance=TOL[layer], passed=bool(worst <= TOL[layer]))


def frame_result(dump, layer, frame, roles, crop, LC, TM):
    summary = json.load(open(os.path.join(dump, 'LOGITS', 'summary200', f'frame_{frame:06d}.json')))
    ref = roles['reference']
    dec = [roles['ceiling']] + roles['models']
    if layer == 'voxel_logits':
        variants = list(summary['variants'])
    else:
        variants = [ref, roles['ceiling']] + roles['models']
    assert variants[0] == ref and all(v in variants for v in dec), (variants, dec)
    V, masks = load_raw(dump, layer, frame, variants, ref)
    t0 = time.time()
    with torch.no_grad():
        if layer == 'voxel_logits':
            values, geom = voxel_frame(LC, V, masks, variants, dec, ref)
        else:
            values, geom = planes_frame(LC, TM, V, masks, variants, dec, ref, crop)
    return dict(frame=frame, layer=layer, values=values, geometry=geom, check=check_against_summary(values, summary, layer, dec),
                seconds=time.time() - t0)


def reduce_layer(recs, models):
    out = {}
    for v in models:
        g = {q: np.array([r['values'][v][q] for r in recs], float) for q in ('pred2', 'res2', 'cross2', 'SST', 'r2')}
        assert np.isfinite(np.stack(list(g.values()))).all() and (g['SST'] > 0).all(), v
        a, b, c = g['pred2'] / g['SST'], g['res2'] / g['SST'], g['cross2'] / g['SST']
        identity = float(np.max(np.abs(a + b + c - 1)))
        r2_identity = float(np.max(np.abs(1 - b - g['r2'])))
        assert identity < 1e-9 and r2_identity < 1e-9, (v, identity, r2_identity)
        frac = np.divide(c, g['r2'], out=np.full_like(c, np.nan), where=np.abs(g['r2']) > 1e-12)
        out[v] = dict(c_median=np.nanmedian(c, 0), c_over_r2_median=np.nanmedian(frac, 0), n_c_positive=(c > 0).sum(0),
                      per_frame_c=c, per_frame_r2=g['r2'], identity_max_abs=identity)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--layers', nargs='+', choices=tuple(LAYER_KEY), default=list(LAYER_KEY))
    ap.add_argument('--frames', type=int, nargs='+', default=None, help='compute only these frames (worker mode, no reduction)')
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--ceiling-role', default=None)
    ap.add_argument('--sparse-role', default=None)
    ap.add_argument('--reference-role', default=None)
    ap.add_argument('--rho-crop-rows', type=int, default=None, help='xy rho rows removed before the plane transform (default: summaries)')
    ap.add_argument('--repo-root', default=None, help='code checkout that contains projects/ (default: inferred)')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    ev = IO.eval_frames(a.dump)
    roles = IO.resolve_roles(a.dump, ev, a.reference_role, a.ceiling_role, a.sparse_role, a.models)
    rawdir = os.path.join(a.dump, 'LOGITS', 'dump200', roles['reference'])
    raw = sorted(int(os.path.basename(p)[6:12]) for p in glob.glob(os.path.join(rawdir, 'frame_*.npz')))
    frames = [f for f in raw if f in set(ev)]
    if not frames:
        raise SystemExit(f'no raw evaluation frame under {rawdir}')
    todo = frames if a.frames is None else [f for f in a.frames if f in frames]
    if a.frames is not None and len(todo) != len(a.frames):
        raise SystemExit(f'--frames must be raw evaluation frames ({frames})')
    crop = a.rho_crop_rows
    if crop is None:
        s = json.load(open(os.path.join(a.dump, 'TPV', 'summary200', f'frame_{frames[0]:06d}.json')))
        crop = int(s['xy'].get('rho_crop_rows') or 0)
    LC, TM = kernels(a.repo_root)
    cache = os.path.join(a.out, 'crossterm_frames')
    for layer in a.layers:
        os.makedirs(os.path.join(cache, layer), exist_ok=True)
        for f in todo:
            p = os.path.join(cache, layer, f'frame_{f:06d}.json')
            if os.path.exists(p):
                continue
            rec = frame_result(a.dump, layer, f, roles, crop, LC, TM)
            tmp = p + f'.{os.getpid()}.tmp'
            json.dump(rec, open(tmp, 'w'))
            os.replace(tmp, p)
            print(layer, f, 'seconds', round(rec['seconds'], 1), 'check', rec['check'], flush=True)
    if a.frames is not None:
        return
    result = dict(frames=frames, n_frames=len(frames), roles={k: roles[k] for k in ('reference', 'ceiling', 'models')},
                  rho_crop_rows_xy=crop, definition=('c = 2<P,E>/SST (P = prediction - mean, E = residual) of the held-out ridge prediction of the model band signal '
                                                     'from the reference band signal; R^2 = a + c; medians over frames'), layers={})
    for layer in a.layers:
        recs = [json.load(open(os.path.join(cache, layer, f'frame_{f:06d}.json'))) for f in frames]
        bad = [r['frame'] for r in recs if not r['check']['passed']]
        if bad:
            raise SystemExit(f'{layer}: recomputed recovery/share differ from the summaries on frames {bad} '
                             f'(worst {max(r["check"]["max_abs"] for r in recs)}, tolerance {TOL[layer]}); no result written')
        result['layers'][layer] = dict(models=reduce_layer(recs, roles['models']),
                                       check_max_abs=max(r['check']['max_abs'] for r in recs), check_tolerance=TOL[layer])
    sys.path.insert(0, HERE)
    from run_analysis import write
    print('wrote', write(a.out, 'appendix_d_crossterm.json', result, True))


if __name__ == '__main__':
    main()
