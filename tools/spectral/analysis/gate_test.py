#!/usr/bin/env python
"""Appendix D gate test: does the teacher's LF probability gate separate class boundaries from class interiors? (CPU)

    python tools/spectral/analysis/gate_test.py --dump DUMP_DIR --out OUT_DIR/appendix_d_gate_test.json

Inputs (the raw frames of the dump, i.e. every frame with a raw dump):
    DUMP/TPV/dump200/_p2dom/frame_XXXXXX.npz      gate_v_mean (mean gate value over the suppression scales b3..b8) and R
                                                  (gate support) on the original xy grid ('<name>_full' if present)
    DUMP/TPV/dump200/<reference>/frame_XXXXXX.npz grid_ind, gt_new (point labels, 8 classes)
    DUMP/TPV/summary200/frame_XXXXXX.json         xy.rho_crop_rows (default crop) and xy.p2_domain_cells.R (check)

Procedure (fixed before any statistic was computed):
  regions  cls = per-cell majority class on the xy plane; occupied = cls >= 0;
           interior I = R & no class change (any class, class 0 included) within Chebyshev radius 1;
           boundary B = R & occupied & not interior.
           The gate, the class map and the neighbourhoods stay on the original grid; only the evaluated support R
           excludes the first rho_crop_rows rho rows (the cropped analysis domain).
  primary  per-frame AUROC of the gate value for B (positive) vs I (Mann-Whitney); one-sided Wilcoxon signed-rank test
           of AUROC - 0.5 > 0 over frames, alpha = 0.05; also reported: pooled AUROC with a frame-bootstrap 95% interval
           (2000 resamples, seed 0), gate means and opening rates (v > 0.5) on B and I.
  secondary (only if the primary test is significant)
           curb proxy: B cells with classes 1 and 7 meeting in the 4-neighbourhood vs interior cells of class 1 or 7;
           sensitivity: radius 2 instead of 1. Pooled AUROC and opening rates only.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

import _plane_regions as PR  # noqa: E402
from spectral_io import DEFAULT_ROLES  # noqa: E402

N_BOOT = 2000
N_HIST = 4096


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    u = mannwhitneyu(pos, neg, alternative='two-sided').statistic
    return float(u / (len(pos) * len(neg)))


def pair_cells(cls, a, b):
    """Cells with a class in {a, b} whose 4-neighbourhood contains the other class of {a, b}."""
    p = np.pad(cls, 1, constant_values=-1)
    out = np.zeros(cls.shape, bool)
    for di, dj in ((0, 1), (2, 1), (1, 0), (1, 2)):
        nb = p[di:di + cls.shape[0], dj:dj + cls.shape[1]]
        out |= ((cls == a) & (nb == b)) | ((cls == b) & (nb == a))
    return out


def load_gate(path, crop_rows):
    """Gate value and support on the original grid; the support loses the first crop_rows rho rows."""
    with np.load(path) as z:
        key = lambda k: k + '_full' if k + '_full' in z.files else k
        v = z[key('gate_v_mean')].astype(np.float64)
        R = z[key('R')].astype(bool).copy()
    if v.shape != R.shape:
        raise SystemExit(f'{path}: gate and support shapes differ {v.shape} {R.shape}')
    if crop_rows:
        R[:crop_rows] = False
    return v, R


def raw_frames(p2dir, refdir):
    ids = lambda d: {int(os.path.basename(p)[6:12]) for p in glob.glob(os.path.join(d, 'frame_*.npz'))}
    return sorted(ids(p2dir) & ids(refdir))


def run(dump, frames=None, crop_rows=None, reference=None, p2dir=None, refdir=None):
    reference = reference or DEFAULT_ROLES['reference']
    p2dir = p2dir or os.path.join(dump, 'TPV', 'dump200', '_p2dom')
    refdir = refdir or os.path.join(dump, 'TPV', 'dump200', reference)
    frames = frames or raw_frames(p2dir, refdir)
    if len(frames) < 2:
        raise SystemExit(f'gate test needs raw frames in {p2dir} and {refdir}; found {frames}')
    sumdir = os.path.join(dump, 'TPV', 'summary200')
    rng = np.random.default_rng(0)
    rows, store, crops = [], [], set()
    for f in frames:
        sp = os.path.join(sumdir, f'frame_{f:06d}.json')
        s = json.load(open(sp))['xy'] if os.path.exists(sp) else None
        c = crop_rows if crop_rows is not None else int((s or {}).get('rho_crop_rows') or 0)
        crops.add(c)
        v, R = load_gate(os.path.join(p2dir, f'frame_{f:06d}.npz'), c)
        if s is not None and 'p2_domain_cells' in s and int(R.sum()) != int(s['p2_domain_cells']['R']):
            raise SystemExit(f'frame {f}: evaluated support has {int(R.sum())} cells, the summary {s["p2_domain_cells"]["R"]} '
                             '(gate arrays not on the original grid, or a different crop)')
        with np.load(os.path.join(refdir, f'frame_{f:06d}.npz')) as ref:
            grid_ind, gt = ref['grid_ind'], ref['gt_new'].astype(np.int64)
        H, W = v.shape
        cls = PR.plane_majority_classes(grid_ind, gt, H, W, 'xy')
        occ = cls >= 0
        I1 = R & PR.interior_radius(cls, 1)
        B1 = R & occ & ~PR.interior_radius(cls, 1)
        I2 = R & PR.interior_radius(cls, 2)
        B2 = R & occ & ~PR.interior_radius(cls, 2)
        curb = B1 & pair_cells(cls, 1, 7)
        curb_ctrl = I1 & ((cls == 1) | (cls == 7))
        rows.append(dict(frame=f, n_R=int(R.sum()), n_B=int(B1.sum()), n_I=int(I1.sum()),
                         auroc=auroc(v[B1], v[I1]), mean_B=float(v[B1].mean()), mean_I=float(v[I1].mean()),
                         open_B=float((v[B1] > 0.5).mean()), open_I=float((v[I1] > 0.5).mean()),
                         open_R=float((v[R] > 0.5).mean()),
                         n_curb=int(curb.sum()), n_curb_ctrl=int(curb_ctrl.sum())))
        store.append(dict(vB=v[B1], vI=v[I1], vB2=v[B2], vI2=v[I2], vC=v[curb], vCc=v[curb_ctrl]))
    au = np.array([r['auroc'] for r in rows])
    w = wilcoxon(au - 0.5, alternative='greater')
    edges = np.linspace(0.0, 1.0, N_HIST + 1)

    def hist(x):
        return np.histogram(np.clip(x, 0.0, 1.0), bins=edges)[0].astype(np.float64)

    def auroc_h(hp, hn):
        cn = np.cumsum(hn) - hn                       # negatives strictly below each bin
        return float((hp * (cn + 0.5 * hn)).sum() / (hp.sum() * hn.sum()))

    def pooled(kp, kn):
        HP = np.array([hist(s[kp]) for s in store]); HN = np.array([hist(s[kn]) for s in store])
        P = np.concatenate([s[kp] for s in store]); N = np.concatenate([s[kn] for s in store])
        est_exact = auroc(P, N)
        est_h = auroc_h(HP.sum(0), HN.sum(0))
        bs = []
        for _ in range(N_BOOT):
            ii = rng.integers(0, len(store), len(store))
            bs.append(auroc_h(HP[ii].sum(0), HN[ii].sum(0)))
        return dict(auroc=est_exact, auroc_hist=est_h, hist_vs_exact_absdiff=abs(est_h - est_exact),
                    ci95=np.percentile(bs, [2.5, 97.5]).tolist(), n_pos=int(P.size), n_neg=int(N.size),
                    mean_pos=float(P.mean()), mean_neg=float(N.mean()),
                    open_pos=float((P > 0.5).mean()), open_neg=float((N > 0.5).mean()))
    primary = dict(per_frame_auroc_median=float(np.median(au)), per_frame_auroc_min=float(au.min()),
                   per_frame_auroc_max=float(au.max()), frames_auroc_gt_half=int((au > 0.5).sum()),
                   wilcoxon_one_sided_p=float(w.pvalue), significant=bool(w.pvalue < 0.05),
                   pooled=pooled('vB', 'vI'))
    res = dict(frames=frames, primary=primary, per_frame=rows)
    if primary['significant']:
        res['secondary'] = dict(curb_pair_1_7=pooled('vC', 'vCc'), sensitivity_r2=pooled('vB2', 'vI2'))
    else:
        res['secondary'] = 'not reported: primary test not significant (fixed procedure)'
    res['analysis_domain'] = dict(rho_crop_rows=sorted(crops)[0] if len(crops) == 1 else sorted(crops),
                                  original_shape=list(v.shape),
                                  geometry='gate and class-neighbourhood geometry on the original grid; the evaluated '
                                           'support excludes the first rho_crop_rows rho rows')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dump', required=True)
    ap.add_argument('--out', required=True, help='output json')
    ap.add_argument('--frames', type=int, nargs='+', default=None, help='raw frames (default: every frame with a raw dump)')
    ap.add_argument('--rho-crop-rows', type=int, default=None, help='rho rows excluded from the support (default: the summaries)')
    ap.add_argument('--reference-role', default=None, help=f'dense teacher raw dump (default {DEFAULT_ROLES["reference"]})')
    ap.add_argument('--p2dom-dir', default=None, help='gate arrays (default DUMP/TPV/dump200/_p2dom)')
    ap.add_argument('--reference-dir', default=None, help='reference raw dump (default DUMP/TPV/dump200/<reference>)')
    a = ap.parse_args(argv)
    res = run(a.dump, a.frames, a.rho_crop_rows, a.reference_role, a.p2dom_dir, a.reference_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps(dict(primary=res['primary'], secondary=res['secondary']), indent=1))


if __name__ == '__main__':
    main()
