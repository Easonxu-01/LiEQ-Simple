#!/usr/bin/env python
"""Gate-domain arrays of the raw frames (CPU), consumed by the gate test of the analysis stage.

  python tools/spectral/dump/export_gate_domains.py --tpv OUT/TPV [--frames 0,20] [--rho-crop-rows 16]

For each raw frame of the TPV stage (default: FRAMES.json 'raw'), reads dump200/_masks and dump200/T_union and writes
dump200/_p2dom/frame_XXXXXX.npz on the uncropped xy grid [240, 180]:
  gate_closed         R AND the final LF probability gate closed (mean gate value over the suppression scales b3-b8 <= 0.5)
  gate_closed_oldint  gate_closed AND the interior of the gt_old classes (no gt_old boundary within Chebyshev radius 2;
                      gt_old equals the current-map GT)
  R                   the gate support; OmegaT; gate_v_mean (mean gate value, float32)
The code path is the one of tpv_summary.summarize_frame (lf_gate_closed / plane_majority_classes / interior_radius,
same inputs). The cell counts, re-cropped like the summary, are checked against xy.p2_domain_cells of the summary and
written to dump200/_p2dom/export_check.json. run_dump.py --stage tpv calls export_frame for every raw frame.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


def export_frame(TSUM, dump_dir, f):
    """Write dump_dir/_p2dom/frame_XXXXXX.npz; return the cell counts (uncropped and cropped as in the summary)."""
    D = str(dump_dir)
    OUT = os.path.join(D, '_p2dom')
    os.makedirs(OUT, exist_ok=True)
    mk = np.load(os.path.join(D, '_masks', f'frame_{f:06d}.npz'))
    ref = np.load(os.path.join(D, 'T_union', f'frame_{f:06d}.npz'))
    om = mk['OmegaT_xy'].astype(bool)
    H, W = om.shape
    pp = mk['plane_prob_xy']
    gopen, gclosed, vmean, Rn = TSUM.lf_gate_closed(pp.astype(np.float64), om.astype(np.float64), 'cpu')
    cls_old = TSUM.plane_majority_classes(ref['grid_ind'], ref['gt_old'].astype(np.int64), H, W, 'xy')
    old_int = TSUM.interior_radius(np.where(om, cls_old, -1), 2)
    gco = gclosed & old_int
    tmp = os.path.join(OUT, f'.frame_{f:06d}.{os.getpid()}.tmp.npz')
    np.savez_compressed(tmp, gate_closed=gclosed, gate_closed_oldint=gco, R=Rn, OmegaT=om, gate_v_mean=vmean.astype(np.float32))
    os.replace(tmp, os.path.join(OUT, f'frame_{f:06d}.npz'))
    c = lambda a: TSUM.rho_crop(a, 'xy')
    return dict(uncropped=dict(gate_closed=int(gclosed.sum()), gate_closed_oldint=int(gco.sum()), R=int(Rn.sum()),
                               old_interior_in_OmegaT=int((old_int & om).sum())),
                cropped=dict(gate_closed=int(c(gclosed).sum()), gate_closed_oldint=int(c(gco).sum()), R=int(c(Rn).sum()),
                             old_interior_in_OmegaT=int((c(old_int) & c(om)).sum())),
                rho_crop_rows=int(TSUM.RHO_CROP))


def check_against_summary(stage_dir, f, got):
    p = os.path.join(str(stage_dir), 'summary200', f'frame_{f:06d}.json')
    if not os.path.isfile(p):
        return None
    s = json.load(open(p))['xy']['p2_domain_cells']
    return dict(summary=s, match=all(got['cropped'][k] == s[k] for k in s if k in got['cropped']))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tpv', required=True, help='the TPV stage directory (contains FRAMES.json, dump200/, summary200/)')
    ap.add_argument('--frames', default=None, help='comma-separated frames (default: FRAMES.json raw)')
    ap.add_argument('--rho-crop-rows', type=int, default=16)
    ap.add_argument('--repo-root', default=str(HERE.parents[2]))
    a = ap.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from common import model_io
    model_io.set_repo_root(a.repo_root)
    import tpv_summary as TSUM
    TSUM.set_rho_crop_rows(a.rho_crop_rows)
    frames = ([int(x) for x in a.frames.split(',')] if a.frames
              else json.load(open(os.path.join(a.tpv, 'FRAMES.json')))['raw'])
    D = os.path.join(a.tpv, 'dump200')
    log = {}
    for f in frames:
        got = export_frame(TSUM, D, f)
        chk = check_against_summary(a.tpv, f, got)
        log[f] = dict(got, check=chk)
        print(f'frame {f:6d}: gate_closed {got["uncropped"]["gate_closed"]} gate_closed_oldint '
              f'{got["uncropped"]["gate_closed_oldint"]} R {got["uncropped"]["R"]} '
              f'summary check: {"n/a" if chk is None else ("match" if chk["match"] else "MISMATCH")}', flush=True)
    json.dump(log, open(os.path.join(D, '_p2dom', 'export_check.json'), 'w'), indent=1)
    bad = [f for f, v in log.items() if v['check'] is not None and not v['check']['match']]
    print('ALL_MATCH' if not bad else f'MISMATCH {bad}')
    return 0 if not bad else 1


if __name__ == '__main__':
    sys.exit(main())
