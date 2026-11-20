#!/usr/bin/env python
"""Frequency-band distance (FBD) between the voxel logit fields of a dense teacher and of student models.

Reports one number per student model: the FBD of its voxel logit field relative to the teacher's on the
teacher's dense support, with channel energies summed first, averaged over evenly spaced test frames.
Definitions: fbd_math.py (bands and FBD) and field_pass.py (fields, inputs, graph operator).

GPU pass + FBD (run from the repository root; one GPU, select it with CUDA_VISIBLE_DEVICES):

  python tools/spectral/fbd/run_fbd.py --dataset semantickitti \\
      --teacher-checkpoint TEACHER_CKPT \\
      --model baseline=projects/configs/pointocc/pointtpv_semantickitti_seg_random.py:BASELINE_CKPT \\
      --model student=projects/configs/pointocc/pointtpv_semantickitti_seg_random.py:STUDENT_CKPT \\
      --num-frames 200 --seed 0 --out OUT

  Writes OUT/energies/<name>/frame_XXXXXX.npz (per-frame fine-bin energies of teacher and student),
  OUT/student_inputs/XXXXXX.pt (simulated student inputs, shared by all models), OUT/fbd.json, OUT/fbd.csv.
  Interrupted runs resume: frames whose energies exist for every model are skipped.

FBD from saved energies (CPU only):

  python tools/spectral/fbd/run_fbd.py --from-dump OUT [--out DIR]

CPU self-test of the operator and the FBD numerics (no data, no checkpoints):

  python tools/spectral/fbd/run_fbd.py --self-test
"""
import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import fbd_math  # noqa: E402

DATASET_CONFIGS = dict(
    semantickitti='projects/configs/pointocc/pointtpv_semantickitti_seg_random.py',
    nuscenes='projects/configs/pointocc/pointtpv_nusc_seg_random.py',
    waymo='projects/configs/pointocc/pointtpv_waymo_seg_random.py',
)
DEFINITION = ('FBD of the voxel logit field on the teacher dense support (support-graph Laplacian, Chebyshev '
              'order 1024, float64); channel energies summed; bands cut at teacher cumulative-energy quantiles '
              '0.5/0.9; weights 0.2/0.3/0.5; eps 1e-8; per-frame value averaged over the evaluated frames.')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=sorted(DATASET_CONFIGS), help='dataset of the evaluated split')
    ap.add_argument('--data-root', default=None,
                    help="dataset root; overrides data.<split>.data_root of the config (default: the config's own)")
    ap.add_argument('--teacher-config', default=None,
                    help='config the teacher network and the data pipeline are built from. This is the student '
                         'training (distillation) config: as in distillation training, the teacher is the student '
                         'architecture loaded with the teacher weights. Default: the seg_random config of --dataset')
    ap.add_argument('--teacher-checkpoint', default=None, help='dense teacher checkpoint')
    ap.add_argument('--model', action='append', default=[], metavar='NAME=CONFIG:CHECKPOINT',
                    help='student model to evaluate (repeatable); NAME labels the output')
    ap.add_argument('--num-frames', type=int, default=200, help='number of evenly spaced test frames')
    ap.add_argument('--seed', type=int, default=0, help='base seed of the teacher cap and the student inputs')
    ap.add_argument('--split', default='test', choices=['test', 'val'], help='cfg.data.<split>')
    ap.add_argument('--frame-ids', default=None,
                    help='comma list: evaluate only these frames (must belong to the evenly spaced selection)')
    ap.add_argument('--sample-cache', default=None,
                    help='directory of the simulated student inputs (read if present, written otherwise); '
                         'default OUT/student_inputs')
    ap.add_argument('--workers', type=int, default=2, help='data loader workers')
    ap.add_argument('--allow-missing-keys', action='store_true',
                    help='do not stop when a checkpoint lacks non-distillation parameters of its config')
    ap.add_argument('--from-dump', default=None, metavar='DIR',
                    help='compute FBD from DIR/energies/<name>/frame_*.npz (no GPU)')
    ap.add_argument('--self-test', action='store_true', help='CPU self-test (operator and FBD numerics)')
    ap.add_argument('--out', default=None, help='output directory')
    return ap.parse_args(argv)


def parse_model(spec):
    name, sep, rest = spec.partition('=')
    i = rest.find('.py:')
    if not sep or not name or i < 0:
        raise SystemExit(f'--model expects NAME=CONFIG.py:CHECKPOINT, got {spec!r}')
    if any(c in name for c in '/\\') or name.startswith('.'):
        raise SystemExit(f'--model NAME must be a plain label: {name!r}')
    return name, rest[:i + 3], rest[i + 4:]


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- offline FBD
def load_dump(root):
    """{name: {frame: (teacher_energy, student_energy)}} from root/energies/<name>/frame_XXXXXX.npz."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, 'energies', '*'))):
        if not os.path.isdir(d):
            continue
        per = {}
        for p in sorted(glob.glob(os.path.join(d, 'frame_*.npz'))):
            z = np.load(p)
            if not np.array_equal(z['fine_edges'], fbd_math.FINE_EDGES):
                raise SystemExit(f'{p}: fine lambda-grid differs from the measurement grid')
            per[int(os.path.basename(p)[6:12])] = (z['teacher_energy'], z['student_energy'])
        if per:
            out[os.path.basename(d)] = per
    if not out:
        raise SystemExit(f'no energies found under {root}/energies/<name>/frame_*.npz')
    return out


def fbd_from_dump(root, out_dir, extra=None):
    dump = load_dump(root)
    res = dict(quantity='FBD', definition=DEFINITION, models={})
    if extra:
        res.update(extra)
    frame_sets = {n: sorted(p) for n, p in dump.items()}
    for name, per in dump.items():
        vals = {fi: fbd_math.fbd_frame(et, es) for fi, (et, es) in sorted(per.items())}
        agg = fbd_math.aggregate(list(vals.values()))
        res['models'][name] = dict(fbd=agg['fbd'], n_frames=agg['n_frames'],
                                   per_frame={str(k): v for k, v in vals.items()})
    first = next(iter(frame_sets.values()))
    res['same_frames_for_all_models'] = all(v == first for v in frame_sets.values())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'fbd.json').write_text(json.dumps(res, indent=1) + '\n')
    with open(out_dir / 'fbd.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'fbd', 'n_frames'])
        for name, r in res['models'].items():
            w.writerow([name, repr(r['fbd']), r['n_frames']])
    print('model                          FBD     frames')
    for name, r in res['models'].items():
        print(f'{name:<28s} {r["fbd"]:8.3f}  {r["n_frames"]:6d}')
    if not res['same_frames_for_all_models']:
        print('WARNING: the models were evaluated on different frame sets; their FBD values are not paired')
    print(f'written: {out_dir / "fbd.json"}, {out_dir / "fbd.csv"}')
    return res


# ----------------------------------------------------------------------------- GPU pass
def run_gpu(args):
    import torch
    from mmcv.parallel import collate
    from torch.utils.data import DataLoader, Subset

    import field_pass as FP

    if not args.dataset or not args.teacher_checkpoint or not args.model or not args.out:
        raise SystemExit('the GPU pass needs --dataset, --teacher-checkpoint, at least one --model and --out')
    models = [parse_model(s) for s in args.model]
    if len({m[0] for m in models}) != len(models):
        raise SystemExit('--model names must be unique')
    for p in [args.teacher_checkpoint] + [m[2] for m in models]:
        if not Path(p).is_file():
            raise SystemExit(f'checkpoint not found: {p}')
    if args.dataset == 'waymo' and args.data_root and 'waymo' not in args.data_root:
        # The Waymo label format is selected by the loader from the lowercase substring 'waymo' in the path.
        raise SystemExit('--data-root for Waymo must contain the lowercase substring "waymo" (the loader selects '
                         'the Waymo label format from it); use a path or a symbolic link named accordingly')
    if not torch.cuda.is_available():
        raise SystemExit('the GPU pass needs a CUDA device')
    # The build step changes into the repository root (config-relative data paths); paths given relative to
    # the current directory are made absolute first. Config paths that do not exist relative to the current
    # directory are taken relative to the repository root.
    # (os.path.abspath keeps symbolic-link names, which the Waymo loader inspects.)
    def here_or_root(p):
        if not p or not Path(p).exists():
            return p
        return os.path.abspath(p) + (os.sep if p.endswith(os.sep) else '')
    out = Path(os.path.abspath(args.out))
    cache_dir = Path(os.path.abspath(args.sample_cache)) if args.sample_cache else out / 'student_inputs'
    args.teacher_checkpoint = os.path.abspath(args.teacher_checkpoint)
    args.data_root = here_or_root(args.data_root)
    models = [(n, here_or_root(c), os.path.abspath(k)) for n, c, k in models]
    tcfg = here_or_root(args.teacher_config or DATASET_CONFIGS[args.dataset])
    device = torch.device('cuda:0')
    tf32 = FP.set_tf32(args.dataset)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'progress.log', 'a', buffering=1)

    def P(msg):
        line = f'[{_dt.datetime.now().isoformat(timespec="seconds")}] {msg}'
        print(line, flush=True)
        log.write(line + '\n')

    P(f'building teacher, input simulator and {len(models)} student model(s) from {tcfg}')
    cfg, dataset, teacher, sampler, students, cap, report = FP.build_all(
        tcfg, args.teacher_checkpoint, models, args.split, args.data_root, device, args.allow_missing_keys)
    hfd = FP.load_hf_distiller(device)
    n = len(dataset)
    idxs = FP.frame_indices(n, args.num_frames)
    if args.frame_ids:
        sel = [int(x) for x in args.frame_ids.split(',') if x.strip()]
        bad = [x for x in sel if x not in idxs]
        if bad:
            raise SystemExit(f'--frame-ids {bad} are not in the evenly spaced selection of {args.num_frames} frames')
        idxs = sel
    manifest = dict(
        started=_dt.datetime.now().isoformat(timespec='seconds'), definition=DEFINITION, dataset=args.dataset,
        split=args.split, dataset_len=n, num_frames=args.num_frames, seed=args.seed, frame_indices=idxs,
        teacher_config=tcfg, data_root=args.data_root or cfg.data[args.split].get('data_root'),
        teacher_cap=cap, simulator_settings=FP.SIMULATOR_SETTINGS, tf32=tf32, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), sample_cache=str(cache_dir),
        checkpoints_md5=dict(teacher=md5(args.teacher_checkpoint), **{nm: md5(k) for nm, _, k in models}),
        model_configs={nm: c for nm, c, _ in models}, load_report=report,
        tool_md5={p.name: md5(p) for p in sorted(HERE.glob('*.py'))})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1, default=str) + '\n')

    def done(fi):
        return all((out / 'energies' / nm / f'frame_{fi:06d}.npz').exists() for nm, _ in students)
    todo = [i for i in idxs if not done(i)]
    P(f'{n} frames in the split, {len(idxs)} selected, {len(todo)} to compute; teacher cap {cap}; '
      f'student inputs in {cache_dir}')
    loader = DataLoader(Subset(dataset, todo), batch_size=1, shuffle=False, num_workers=args.workers,
                        collate_fn=lambda b: collate(b, samples_per_gpu=1))
    frames_log = open(out / 'frames.jsonl', 'a')
    t_start = time.time()
    for j, (fi, data) in enumerate(zip(todo, loader)):
        t0 = time.time()
        d = FP.to_device(data, device)
        tg, gi, gv, pl, flag = FP.dense_input(d, device)
        tg_t, gi_t, gv_t = FP.cap_teacher_input(tg, gi, gv, fi, args.seed, cap, device)
        FP.seed_all(int(args.seed) * FP.FRAME_SEED_MULT + int(fi))     # forward-pass repeatability
        t_field = FP.teacher_field(teacher, tg_t, gi_t, gv_t, flag)
        ps, gs, gvs, generated = FP.student_input(sampler, tg, gi, gv, pl, fi, args.seed, cache_dir, device)
        lap = FP.SupportLaplacian(hfd, t_field)
        e_t = fbd_math.energies_from_moments(FP.cheb_moments(lap, t_field))
        rec = dict(frame=int(fi), n_dense=int(tg.shape[0]), n_teacher_in=int(tg_t.shape[0]),
                   n_student_in=int(ps.shape[0]), support_voxels=int(lap.m.sum().item()),
                   student_input_generated=bool(generated), fbd={})
        for nm, m in students:
            FP.seed_all(int(args.seed) * FP.FRAME_SEED_MULT + int(fi))
            s_field = FP.student_field(m, ps, gs, gvs, gv_t, flag)
            if s_field.shape != t_field.shape:
                raise RuntimeError(f'{nm}: student field {tuple(s_field.shape)} != teacher field {tuple(t_field.shape)}')
            e_s = fbd_math.energies_from_moments(FP.cheb_moments(lap, s_field))
            dst = out / 'energies' / nm
            dst.mkdir(parents=True, exist_ok=True)
            tmp = dst / f'frame_{fi:06d}.tmp.npz'
            np.savez(tmp, fine_edges=fbd_math.FINE_EDGES, teacher_energy=e_t, student_energy=e_s,
                     support_voxels=np.int64(rec['support_voxels']))
            os.replace(tmp, dst / f'frame_{fi:06d}.npz')
            rec['fbd'][nm] = fbd_math.fbd_frame(e_t, e_s)
            del s_field
        rec['seconds'] = round(time.time() - t0, 2)
        frames_log.write(json.dumps(rec) + '\n')
        frames_log.flush()
        eta = (time.time() - t_start) / (j + 1) * (len(todo) - j - 1)
        P(f'frame {fi} ({j + 1}/{len(todo)}) {rec["seconds"]}s support={rec["support_voxels"]} '
          f'n_student={rec["n_student_in"]} ' + ' '.join(f'{k}={v:.4f}' for k, v in rec['fbd'].items()) +
          f' eta={eta / 60:.1f}min')
        del t_field, lap
        torch.cuda.empty_cache()
    frames_log.close()
    P('computing FBD from the saved energies')
    fbd_from_dump(str(out), str(out), extra=dict(dataset=args.dataset, split=args.split,
                                                  frame_indices=idxs, seed=args.seed))
    log.close()


# ----------------------------------------------------------------------------- self-test
def self_test():
    import torch

    import field_pass as FP
    ok = True

    def check(name, cond, info=''):
        nonlocal ok
        ok &= bool(cond)
        print(f'[{"PASS" if cond else "FAIL"}] {name} {info}')

    co = fbd_math.fine_bin_coeffs(fbd_math.FINE_EDGES, fbd_math.CHEB_ORDER)
    unity = co.sum(0)
    check('fine-bin projectors sum to T_0', abs(unity[0] - 1) < 1e-12 and np.abs(unity[1:]).max() < 1e-10,
          f'(max |residual| {np.abs(unity[1:]).max():.1e})')
    # support Laplacian of the released distiller on a small random support (CPU)
    g = torch.Generator().manual_seed(0)
    shape = (1, 3, 7, 6, 5)
    mask = (torch.rand(shape[2:], generator=g) < 0.6).to(torch.float32)
    t = torch.randn(shape, generator=g) * mask
    s = (t + 0.3 * torch.randn(shape, generator=g)) * mask
    lap = FP.SupportLaplacian(FP.load_hf_distiller(torch.device('cpu')), t)
    nv = int(np.prod(shape[2:]))
    idx = mask.reshape(-1).nonzero().squeeze(1)
    cols = []
    for k in idx.tolist():
        e = torch.zeros((1, 1) + shape[2:], dtype=torch.float64)
        e.view(-1)[k] = 1.0
        cols.append(lap(e).reshape(-1)[idx])
    A = torch.stack(cols, 1).numpy()
    ev = np.linalg.eigvalsh(0.5 * (A + A.T))
    check('L symmetric', np.abs(A - A.T).max() < 1e-14)
    check('spectrum in [0, 2)', ev.min() > -1e-12 and ev.max() < 2.0, f'(min {ev.min():.2e}, max {ev.max():.4f})')
    mu_t = FP.cheb_moments(lap, t)
    mu_s = FP.cheb_moments(lap, s)
    e_t, e_s = fbd_math.energies_from_moments(mu_t), fbd_math.energies_from_moments(mu_s)
    rel = abs(e_t.sum() - float((t.double() ** 2).sum())) / float((t.double() ** 2).sum())
    check('fine-bin energies add up to ||x||^2', rel < 1e-3, f'(rel {rel:.1e}; clipped negative leakage only)')
    check('FBD(teacher, teacher) = 0', fbd_math.fbd_frame(e_t, e_t) == 0.0)
    v = fbd_math.fbd_frame(e_t, e_s)
    check('FBD(teacher, perturbed) > 0', v > 0, f'({v:.4f})')
    print('SELF-TEST', 'PASS' if ok else 'FAIL', f'({nv} grid nodes, {len(idx)} on the support)')
    return 0 if ok else 1


def main():
    args = parse_args()
    if args.self_test:
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
        return self_test()
    if args.from_dump:
        fbd_from_dump(args.from_dump, args.out or args.from_dump)
        return 0
    run_gpu(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
