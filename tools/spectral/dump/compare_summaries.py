#!/usr/bin/env python
"""Compare per-frame summary JSONs (or _info JSONs) of two runs.

  python tools/spectral/dump/compare_summaries.py --a RUN_A/TPV/summary200 --b RUN_B/TPV/summary200 [--frames 0,20,41]
      [--atol 5e-5] [--ignore timing,M9_bev_file]

--a is the run to check, --b the reference. For every frame present in both directories (or the listed frames) the
two JSON trees are walked together. Numbers
pass when |a - b| <= atol + rtol * |b|; NaN / inf are stored as null and must match; strings, booleans, list lengths
and key sets must match exactly. Prints the worst differences per frame and a summary line; exit code 0 if all pass.
--npz compares raw dumps instead (--a / --b = dump200 directories): for every variant directory and frame present in
both, the arrays of each common key must have the same shape and dtype; reports the max |a - b| per key (0 = identical).
--subset ignores keys that are present only in --b (use it when --b holds reference outputs of a larger schema).
--info compares the _info/ records instead and checks only the input identity fields (point / grid-index / tokenizer
input md5 and counts of every variant), which must be identical.
"""
import argparse
import json
import math
import os
import sys

INFO_KEYS = ('n_loaded', 'n_used', 'points_md5', 'grid_ind_md5', 'tokenizer_in_n', 'tokenizer_in_md5',
             'tokenizer_in_gi_md5', 'kept_idx_md5_int32')


def walk(a, b, path, ign, out, subset=False):
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a) - ign, set(b) - ign
        for k in sorted(ka ^ kb):
            if subset and k in kb:
                continue
            out['struct'].append(f'{path}/{k}: only in {"a" if k in ka else "b"}')
        for k in sorted(ka & kb):
            walk(a[k], b[k], f'{path}/{k}', ign, out, subset)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out['struct'].append(f'{path}: list length {len(a)} vs {len(b)}')
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f'{path}[{i}]', ign, out, subset)
        return
    num = (int, float)
    if isinstance(a, bool) or isinstance(b, bool) or not (isinstance(a, num) and isinstance(b, num)):
        if a != b:
            out['struct'].append(f'{path}: {str(a)[:60]} vs {str(b)[:60]}')
        return
    out['n'] += 1
    d = abs(float(a) - float(b))
    if d == 0:
        out['exact'] += 1
        return
    rel = d / max(abs(float(b)), 1e-300)
    out['diffs'].append((d, rel, path, abs(float(b))))


def compare_file(fa, fb, ign, atol, rtol, subset=False):
    out = dict(n=0, exact=0, diffs=[], struct=[])
    walk(json.load(open(fa)), json.load(open(fb)), '', ign, out, subset)
    bad = [x for x in out['diffs'] if x[0] > atol + rtol * x[3]]
    return out, bad


def compare_npz(da, db, frames, atol):
    import numpy as np
    ok_all = True
    subs = sorted(d for d in os.listdir(da) if os.path.isdir(os.path.join(da, d)) and os.path.isdir(os.path.join(db, d)))
    for sd in subs:
        fa_dir, fb_dir = os.path.join(da, sd), os.path.join(db, sd)
        names = ([f'frame_{int(x):06d}.npz' for x in frames.split(',')] if frames else
                 sorted(set(os.listdir(fa_dir)) & set(os.listdir(fb_dir))))
        for nm in names:
            pa, pb = os.path.join(fa_dir, nm), os.path.join(fb_dir, nm)
            if not (os.path.isfile(pa) and os.path.isfile(pb)):
                continue
            za, zb = np.load(pa), np.load(pb)
            ka, kb = set(za.files), set(zb.files)
            res, bad = [], []
            for k in sorted(ka & kb):
                x, y = za[k], zb[k]
                if x.shape != y.shape or x.dtype != y.dtype:
                    bad.append(f'{k}: {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}'); continue
                d = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
                if not d <= atol:
                    bad.append(f'{k}: max_abs={d:.3g}')
                res.append(d)
            ok = not bad and ka == kb
            ok_all &= ok
            print(f'{sd}/{nm}: {"PASS" if ok else "FAIL"} keys={len(ka & kb)} identical={sum(1 for d in res if d == 0)} '
                  f'max_abs={max(res) if res else 0:.3g} only_a={sorted(ka - kb)} only_b={sorted(kb - ka)} {bad[:4]}')
    print('ALL PASS' if ok_all else 'SOME FAIL')
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--frames', default=None)
    ap.add_argument('--atol', type=float, default=5e-5)
    ap.add_argument('--rtol', type=float, default=0.0)
    ap.add_argument('--ignore', default='timing,M9_bev_file')
    ap.add_argument('--info', action='store_true')
    ap.add_argument('--subset', action='store_true')
    ap.add_argument('--npz', action='store_true')
    ap.add_argument('--top', type=int, default=5)
    a = ap.parse_args(argv)
    ign = set(x for x in a.ignore.split(',') if x)
    if a.npz:
        return compare_npz(a.a, a.b, a.frames, a.atol)
    da, db = (os.path.join(d, '_info') if a.info else d for d in (a.a, a.b))
    if a.frames:
        names = [f'frame_{int(x):06d}.json' for x in a.frames.split(',')]
    else:
        names = sorted(set(f for f in os.listdir(da) if f.startswith('frame_')) & set(f for f in os.listdir(db) if f.startswith('frame_')))
    ok_all = True
    for nm in names:
        fa, fb = os.path.join(da, nm), os.path.join(db, nm)
        if not (os.path.isfile(fa) and os.path.isfile(fb)):
            print(f'{nm}: missing in {"a" if not os.path.isfile(fa) else "b"}'); ok_all = False; continue
        if a.info:
            ia, ib = json.load(open(fa)), json.load(open(fb))
            vs = sorted(k for k in ia if isinstance(ia[k], dict) and 'points_md5' in ia[k])
            mism = [(v, k) for v in vs for k in INFO_KEYS if k in ia[v] and ia[v].get(k) != (ib.get(v) or {}).get(k)]
            miss = [v for v in vs if v not in ib]
            ok = not mism and not miss
            ok_all &= ok
            print(f'{nm}: {"PASS" if ok else "FAIL"} variants={len(vs)} mismatches={mism[:6]} missing={miss}')
            continue
        out, bad = compare_file(fa, fb, ign, a.atol, a.rtol, a.subset)
        ok = not bad and not out['struct']
        ok_all &= ok
        worst = sorted(out['diffs'], reverse=True)[:a.top]
        maxabs = worst[0][0] if worst else 0.0
        maxrel = max((x[1] for x in out['diffs']), default=0.0)
        print(f'{nm}: {"PASS" if ok else "FAIL"} numbers={out["n"]} exact={out["exact"]} max_abs={maxabs:.3g} '
              f'max_rel={maxrel:.3g} over_tol={len(bad)} struct={len(out["struct"])}')
        for d, r, p, _ in worst:
            print(f'    abs={d:.3g} rel={r:.3g} {p}')
        for s in out['struct'][:a.top]:
            print(f'    struct: {s}')
    print('ALL PASS' if ok_all else 'SOME FAIL')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
