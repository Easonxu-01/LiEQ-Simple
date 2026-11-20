#!/usr/bin/env python
"""Spectral-measurement suite, dump stage: per-frame TPV-plane and logits-level spectral summaries (read-only; GPU).

Two stages, run as two separate invocations with the same arguments except --stage:
  --stage tpv     TPV feature plane (neck s0) and class-probability planes on the teacher support; summary = tpv_summary
  --stage logits  common-support voxel logits and the xy feature plane; summary = logits_summary
The full tensors of the raw frames are written in the same format by both stages; resummarize.py recomputes the
summaries from them.

Usage (from the repository root):
  python tools/spectral/dump/run_dump.py --stage tpv --dataset semantickitti \\
      --teacher-config projects/configs/pointocc/pointtpv_semantickitti_teacher_dense.py --teacher-checkpoint TEACHER.pth \\
      --model baseline=BASELINE_CONFIG.py:BASELINE.pth \\
      --model student=projects/configs/pointocc/pointtpv_semantickitti_seg_random.py:STUDENT.pth \\
      --frames-rule paper --raw-frames paper --rho-crop-rows 16 --out OUT [--shard 0/4] [--gpu 0]

Roles (one checkpoint each; every model is built from its own config, the config it was trained with):
  teacher   --teacher-config / --teacher-checkpoint
  student   the distilled model to analyse      (--model student=CONFIG:CHECKPOINT); its config also supplies the HF
            gate of the logits stage, and it is the channel-aligned model of the TPV plane metrics
  baseline  the model it is compared with       (--model baseline=CONFIG:CHECKPOINT)
The per-frame summaries key their output by these role names. A checkpoint whose embedded label map differs from the
current one is evaluated as it is (no remapping); all GT and regions use the current map.

Data config (--data-config, default: the student config): its data.test defines the inputs, its label map is the current
map, and its model block builds the serving teacher (student structure + teacher weights, dense_train = d2skd = True)
with its teacher_serve_max_points cap. Every model config must have the same data.test.

Inputs per frame (teacher input variants):
  T_union   teacher on the dense union (raw scan U dense completion), randomly truncated to the serving cap (500k points)
            with Generator(seed*1000003 + frame); reference of all comparisons; its distillation field defines Omega_T
  T_union2  as T_union, re-truncated with Generator((seed+1)*1000003 + frame) (noise ceiling)
  T_raw     teacher on the sparse raw scan
  student / baseline   the model on the sparse raw scan
Forward passes are seeded with seed*1000003 + 29 (the tokenizer uses randperm). In the logits stage the teacher variants
T_union2 / T_raw are also queried at the Omega_T voxel centres by the serving teacher.

Frames: --frames-rule paper = round(linspace(0, N-1, 200)) plus the development frames {0, N//3, 2N//3}; the development
frames are dumped and summarised but excluded from the statistics (199 evaluated frames); see frames.py. Raw frames
(--raw-frames paper = development frames + 10 evenly spaced grid frames, 13 in total) are summarised in 'full' mode
and also write their full tensors; every other frame is summarised in 'eval' mode.

Data: dataset annotation files are read from the config paths (relative to the repository root, e.g. data/...).
--data-root overrides data.test.data_root only (where scans, dense completions and labels are resolved). Dense
completions are found next to the raw scans by the loader's path rule, as in training. Waymo runs with TF32 off.

Output (OUT/TPV or OUT/LOGITS):
  summary200/frame_XXXXXX.json      per-frame summary (schema of tpv_summary / logits_summary)
  summary200/_info/frame_XXXXXX.json per-frame record: input identity checks, point counts, 7-class mIoU, timings
  summary200/manifest_*.json        per-process record (arguments, code and checkpoint md5, load reports)
  dump200/<variant>/frame_XXXXXX.npz and dump200/_masks/frame_XXXXXX.npz   full tensors of the raw frames
  dump200/_p2dom/frame_XXXXXX.npz   (tpv stage, raw frames) gate-domain arrays, see export_gate_domains.py
  FRAMES.json                       frame lists of the run
Processes may run in parallel on disjoint shards (--shard I/N) with the same --out.
The per-frame quantities do not depend on the frame order or on sharding (every random step has its own seed).
"""
import argparse
import copy
import hashlib
import json
import os
import resource
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECTRAL = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(SPECTRAL))
import dataset_io as DIO   # noqa: E402
import frames as FR        # noqa: E402

PLANES = ('xy', 'yz', 'zx')
T_VARIANTS = ('T_union', 'T_union2', 'T_raw')
MODEL_ROLES = ('baseline', 'student')      # canonical processing order
STAGE_DIR = {'tpv': 'TPV', 'logits': 'LOGITS'}
MIN_FREE_GB = 30.0
MODEL_FILES = ['projects/unilidar_plugin/pointocc_model/pointtpv_seg.py',
               'projects/unilidar_plugin/pointocc_model/Spectraldistiller.py',
               'projects/unilidar_plugin/pointocc_model/cylinder_encoder.py',
               'projects/unilidar_plugin/pointocc_model/tpv_aggregator.py',
               'projects/unilidar_plugin/pointocc_model/fpn.py',
               'projects/unilidar_plugin/pointocc_model/swin.py',
               'projects/unilidar_plugin/datasets/pipelines/PointoccMapping.py',
               'projects/unilidar_plugin/datasets/pipelines/loading_voxels_sk.py']


def stage_variants(stage, names):
    return list(T_VARIANTS) + list(names)


def md5(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def resolve(p, repo_root):
    q = Path(p)
    if q.is_absolute():
        return q
    for b in (Path.cwd(), Path(repo_root)):
        if (b / q).exists():
            return (b / q).resolve()
    raise FileNotFoundError(f'{p} not found (relative to the current directory or the repository root)')


def parse_model_spec(spec):
    """NAME=CONFIG:CHECKPOINT (the config path ends with .py)."""
    if '=' not in spec:
        raise SystemExit(f'--model needs NAME=CONFIG:CHECKPOINT: {spec}')
    name, rest = spec.split('=', 1)
    i = rest.find('.py:')
    if i < 0:
        raise SystemExit(f'--model needs NAME=CONFIG.py:CHECKPOINT: {spec}')
    return name, rest[:i + 3], rest[i + 4:]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=sorted(STAGE_DIR))
    ap.add_argument('--dataset', required=True, choices=sorted(DIO.DATASETS))
    ap.add_argument('--repo-root', default=str(HERE.parents[2]),
                    help='repository root (contains projects/); default: three levels above this file')
    ap.add_argument('--data-root', default=None, help='override of data.test.data_root (default: the config value)')
    ap.add_argument('--teacher-config', required=True)
    ap.add_argument('--teacher-checkpoint', required=True)
    ap.add_argument('--model', action='append', required=True, metavar='NAME=CONFIG:CHECKPOINT',
                    help=f'repeatable; NAME in {list(MODEL_ROLES)}')
    ap.add_argument('--data-config', default=None,
                    help='config whose data.test, label map and serving-teacher structure are used (default: student config)')
    fg = ap.add_mutually_exclusive_group(required=True)
    fg.add_argument('--frames-rule', choices=['paper'], help='paper: 200-frame grid + development frames (202 frames)')
    fg.add_argument('--frames', help='frame list: file (JSON list or integers) or comma-separated indices')
    ap.add_argument('--raw-frames', default='paper',
                    help="frames that are summarised in full mode and write full tensors: 'paper' (13), an integer n "
                         "(development frames + n grid frames), 'none', or a frame list / file")
    ap.add_argument('--rho-crop-rows', type=int, default=16,
                    help='near-sensor rho rows of the (b) grid dropped before the plane spectra (paper: 16; 0 = none)')
    ap.add_argument('--shard', default='0/1', help='I/N: process the I-th of N interleaved shards of the frames')
    ap.add_argument('--gpu', default=None, help='physical GPU index (sets CUDA_VISIBLE_DEVICES)')
    ap.add_argument('--seed', type=int, default=0, help='base seed of the truncation and forward seeds (paper: 0)')
    ap.add_argument('--tf32', default='dataset', choices=['dataset', 'default', 'off'],
                    help='dataset: off for waymo, torch default otherwise')
    ap.add_argument('--out', required=True, help='output root; the stage writes OUT/TPV or OUT/LOGITS')
    a = ap.parse_args(argv)
    a.tf32 = DIO.resolve_tf32(a.dataset, a.tf32)
    a.data_root = DIO.normalize_data_root(a.dataset, a.data_root)
    specs = []
    for s in a.model:
        name, cfg, ck = parse_model_spec(s)
        if name not in MODEL_ROLES:
            raise SystemExit(f'model name {name} not in {list(MODEL_ROLES)}')
        if any(x[0] == name for x in specs):
            raise SystemExit(f'model {name} given twice')
        specs.append((name, cfg, ck))
    a.model_specs = sorted(specs, key=lambda x: MODEL_ROLES.index(x[0]))
    return a


def make_ds_cfg(cfg, src):
    """Input variants of the dataset: 'union' = raw scan U dense completion (the serving-teacher input and the ray-cast
    source in training), 'refine' = dense completion only; 'asis' / 'raw' leave data.test unchanged."""
    ds_cfg = copy.deepcopy(cfg.data['test'])
    ds_cfg.test_mode = True
    if src in ('union', 'refine'):
        ds_cfg.random = True
        ds_cfg.dense7sparse = (src == 'union')
        ds_cfg.reset_random_prob = 0.0
        n_hit = 0
        for step in ds_cfg.pipeline:
            if step.get('type') == 'LoadPointsFromFile_RPR':
                step['Random'] = True; step['dense7sparse'] = (src == 'union'); step['load_dim'] = 3; n_hit += 1
        if n_hit != 1:
            raise RuntimeError(f'{src}: {n_hit} LoadPointsFromFile_RPR steps')
    ds_cfg.pop('samples_per_gpu', None)
    return ds_cfg


def atomic_write_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp')
    tmp.write_text(text); os.replace(tmp, path)


def rss_mb():
    try:
        for ln in open('/proc/self/status'):
            if ln.startswith('VmRSS:'):
                return int(ln.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def canonical(obj):
    return json.dumps(obj, sort_keys=True, default=str)


def main():
    a = parse_args()
    import numpy as np, torch
    tf32_info = DIO.set_tf32(a.tf32)   # before any CUDA work
    from mmcv import Config
    from mmcv.parallel import collate, scatter
    t_start = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    pid = os.getpid()
    ROOT = Path(a.repo_root).resolve()
    LOGITS = (a.stage == 'logits')
    map_name = DIO.get(a.dataset)['map_name']

    # ---- paths (resolved before chdir) ----
    tea_cfg_path = str(resolve(a.teacher_config, ROOT)); tea_ck = str(resolve(a.teacher_checkpoint, ROOT))
    roles = []
    for name, cfgp, ck in a.model_specs:
        roles.append(dict(name=name, config=str(resolve(cfgp, ROOT)), ckpt=str(resolve(ck, ROOT))))
    snames = [s['name'] for s in roles]
    if a.data_config:
        data_cfg_path = str(resolve(a.data_config, ROOT))
    else:
        data_cfg_path = next((s['config'] for s in roles if s['name'] == 'student'), roles[0]['config'])
    variants = stage_variants(a.stage, snames)
    stage_out = Path(a.out).resolve() / STAGE_DIR[a.stage]
    out = stage_out / 'dump200'; out.mkdir(parents=True, exist_ok=True)
    sum_out = stage_out / 'summary200'; sum_out.mkdir(parents=True, exist_ok=True)

    # ---- suite modules (repository root set before the spectral modules load the distillers) ----
    from common import model_io
    model_io.set_repo_root(ROOT)
    from common import plane_cells as lfx
    import tpv_summary as TS
    TS.set_rho_crop_rows(a.rho_crop_rows)
    if LOGITS:
        import logits_summary as LS
        LS.set_gate_config(next((s['config'] for s in roles if s['name'] == 'student'), data_cfg_path))
        frame_mod = LS
    else:
        import export_gate_domains as EGD
        frame_mod = TS
    code_md5 = {str(p.relative_to(SPECTRAL)): md5(p) for d in (HERE, SPECTRAL / 'common') for p in sorted(d.glob('*.py'))}

    os.chdir(ROOT); sys.path.insert(0, str(ROOT))
    model_code_md5 = {f: md5(ROOT / f) for f in MODEL_FILES if (ROOT / f).is_file()}
    from mmdet3d.models import build_model
    from mmdet3d.datasets import build_dataset
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False

    # ---- configs: data config (datasets, serving teacher) + each model's own config ----
    base_cfg = Config.fromfile(data_cfg_path)
    data_ovr = dict(data=DIO.apply_data_overrides(base_cfg, a.dataset, 'data', a.data_root))
    model_io.import_plugin(base_cfg, data_cfg_path)
    tcfg = Config.fromfile(tea_cfg_path)
    for c in (base_cfg, tcfg):
        c.model.pretrained = None; c.model.train_cfg = None
    base_test = canonical(base_cfg.data.test.to_dict())
    for st in roles:
        c = Config.fromfile(st['config'])
        DIO.apply_data_overrides(c, a.dataset, 'data', a.data_root)
        c.model.pretrained = None; c.model.train_cfg = None
        st['cfg'] = c
        st['data_test_same_as_base'] = (canonical(c.data.test.to_dict()) == base_test)
        if not st['data_test_same_as_base']:
            raise RuntimeError(f"{st['name']}: data.test of {st['config']} differs from the data config; "
                               'the inputs are built from the data config only')
    from projects.unilidar_plugin.pointocc_model.Spectraldistiller import TPVLowFreqSpectralDistiller as LFD

    # ---- label map (one map for all models) ----
    new_map = DIO.parse_map_file(data_cfg_path, map_name)
    assert new_map == DIO.parse_map_file(tea_cfg_path, map_name), 'teacher / data config label maps differ'

    def ck_space(path):
        ck = torch.load(path, map_location='cpu')
        txt = (ck.get('meta') or {}).get('config')
        meta_ep = (ck.get('meta') or {}).get('epoch')
        del ck
        if not isinstance(txt, str):
            return 'unknown (no meta config)', None, meta_ep
        try:
            mp = DIO.parse_map_text(txt, map_name)
        except RuntimeError:
            return 'unknown (no label map in meta config)', None, meta_ep
        return ('new' if mp == new_map else 'old'), mp, meta_ep
    t_space, _, _ = ck_space(tea_ck)
    for st in roles:
        st['label_space'], mp, st['meta_epoch'] = ck_space(st['ckpt'])
        st['own_map_diff_vs_current'] = ({int(k_): [mp.get(k_), new_map[k_]] for k_ in new_map if mp.get(k_) != new_map[k_]}
                                         if isinstance(mp, dict) else None)
    # All GT, regions and coarse statistics use the current map ("old" map := current map); a model whose checkpoint
    # carries another map is evaluated as it is, without remapping.
    old_map = dict(new_map)
    OLD_IS_NEW = True
    diff_keys = sorted(k for k in new_map if new_map[k] != old_map.get(k))

    # ---- datasets ----
    def build_ds(cfg, src):
        return build_dataset(make_ds_cfg(cfg, src))
    ds = {}
    for src in ('raw', 'union'):
        ds[(src, 'new')] = build_ds(base_cfg, src)
        ds[(src, 'old')] = ds[(src, 'new')]
    N = len(ds[('raw', 'new')])
    for key in [k for k in ds if k[1] == 'new' and k[0] != 'raw']:
        if len(ds[key]) != N:
            raise RuntimeError(f'{key[0]} / raw datasets differ in length: {len(ds[key])} vs {N}')
    pf = FR.paper_frames(N)
    all_frames = pf['all'] if a.frames_rule == 'paper' else sorted(set(FR.parse_frame_list(a.frames)))
    if any(f < 0 or f >= N for f in all_frames):
        raise SystemExit(f'frame index outside [0, {N})')
    frames = FR.shard(all_frames, a.shard)
    full_frames = set(FR.raw_frames(a.raw_frames, N))
    atomic_write_text(stage_out / 'FRAMES.json', json.dumps(dict(
        dataset=a.dataset, N=N, lin=pf['lin'], dev=pf['dev'], eval=pf['eval'], n_eval=pf['n_eval'], review=pf['review'],
        raw=sorted(full_frames), frames=all_frames, frames_rule=(a.frames_rule or 'list'),
        rule='lin = round(linspace(0, N-1, 200)); dev = {0, N//3, 2N//3} (excluded from statistics); eval = lin - dev; '
             'raw frames are summarised in full mode'), indent=1))

    def mode_for(fi):
        return 'full' if fi in full_frames else 'eval'

    # ---- models ----
    reports = {}

    def load_and_check(name, m, ck, role):
        rep = model_io.load_weights_report(m, ck, False, role=role)
        sd = torch.load(ck, map_location='cpu'); sd = sd.get('state_dict', sd)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        msd = m.state_dict()
        rep.update(missing_all=[k for k in msd if k not in sd], unexpected_all=[k for k in sd if k not in msd],
                   shape_mismatch=[k for k in msd if k in sd and tuple(sd[k].shape) != tuple(msd[k].shape)])
        if rep['shape_mismatch']:
            raise RuntimeError(f"{name}: shape mismatch {rep['shape_mismatch'][:5]}")
        reports[name] = rep
    teacher_own = build_model(tcfg.model, test_cfg=tcfg.get('test_cfg'))
    load_and_check('teacher_own_cfg', teacher_own, tea_ck, 'teacher')
    teacher_serve = build_model(base_cfg.model, test_cfg=base_cfg.get('test_cfg'))   # student structure (data config)
    load_and_check('teacher_serve', teacher_serve, tea_ck, 'teacher')
    teacher_serve.tpv_aggregator.dense_train = True; teacher_serve.tpv_aggregator.d2skd = True; teacher_serve.dense_train = True
    models = [teacher_own, teacher_serve]
    for st in roles:
        m = build_model(st['cfg'].model, test_cfg=st['cfg'].get('test_cfg'))
        load_and_check(st['name'], m, st['ckpt'], 'student')
        m.tpv_aggregator.dense_train = False; m.tpv_aggregator.d2skd = False; m.dense_train = False
        st['model'] = m; models.append(m)
    for m in models:
        for p in m.parameters():
            p.requires_grad_(False)
        m.to(device).eval()
    smodel = {st['name']: st['model'] for st in roles}
    teacher_md5 = md5(tea_ck)
    for st in roles:
        st['ckpt_md5'] = md5(st['ckpt']); st['config_md5'] = md5(st['config'])
    sspace = {st['name']: st['label_space'] for st in roles}

    # ---- hooks ----
    cap_buf = {}

    def reg(model):
        return [model.lidar_tokenizer.register_forward_pre_hook(lambda mod, inp: cap_buf.__setitem__('tok_in', inp)),
                model.lidar_tokenizer.register_forward_hook(lambda mod, inp, o: cap_buf.__setitem__('tok', o)),
                model.lidar_neck.register_forward_hook(lambda mod, inp, o: cap_buf.setdefault('neck', []).append(o)),
                model.tpv_aggregator.register_forward_hook(lambda mod, inp, o: cap_buf.setdefault('agg', []).append(o))]

    def md5_arr(t):
        return hashlib.md5(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()

    TEST_SEED = int(a.seed) * 1000003 + 29

    def test_path(model, it):
        cap_buf.clear(); hs = reg(model)
        model_io.seed_all(TEST_SEED)   # the tokenizer's randperm causes <= 3e-3 jitter -> fixed seed
        L = dict(train_grid=[it['tg']], grid_ind=[it['gi']], grid_ind_vox=[it['gv']],
                 train_voxel_label=[it['vl']] if it['vl'] is not None else None, train_pts_label=[it['pl']])
        try:
            with torch.no_grad():
                model.forward_test(dataset_flag=it['flag'], return_loss=False, **L)
        finally:
            for h in hs:
                h.remove()
        tok = cap_buf['tok']; neck = cap_buf['neck']; agg = cap_buf['agg']
        assert len(neck) == 3 and len(agg) == 1, (len(neck), len(agg))
        used = dict(tg=L['train_grid'][0], gi=L['grid_ind'][0], gv=L['grid_ind_vox'][0], pl=L['train_pts_label'][0])
        assert torch.equal(cap_buf['tok_in'][0][0], used['tg']), 'tokenizer input != truncated list'
        used['tok_in_n'] = int(cap_buf['tok_in'][0][0].shape[0])
        used['tok_in_md5'] = md5_arr(cap_buf['tok_in'][0][0]); used['tok_in_gi_md5'] = md5_arr(cap_buf['tok_in'][1][0])
        logit = agg[0][1]
        logit = logit.reshape(logit.shape[0], logit.shape[1], -1)[0].T.contiguous()
        return tok, neck, logit, agg[0][2], used

    # common coarsening of the label maps (identity with a single map)
    par = list(range(8))

    def _find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for k in diff_keys:
        ra_, rb_ = _find(old_map[k]), _find(new_map[k])
        if ra_ != rb_:
            par[max(ra_, rb_)] = min(ra_, rb_)
    roots = sorted({_find(c) for c in range(8)})
    coarse_of = [roots.index(_find(c)) for c in range(8)]
    coarse_components = [[c for c in range(8) if coarse_of[c] == j] for j in range(len(roots))]
    KC = len(roots)
    coarse_lut = torch.tensor(coarse_of, dtype=torch.long, device=device)
    for k in set(new_map) & set(old_map):
        assert coarse_of[new_map[k]] == coarse_of[old_map[k]], k
    if [int(x) for x in TS.CG] != coarse_of:
        raise RuntimeError(f'summary coarse groups {list(TS.CG)} != dumper coarsening {coarse_of}')

    def confmat(pred, gt, K):
        gt = gt.long().reshape(-1); pred = pred.long().reshape(-1)
        assert int(gt.min()) >= 0 and int(gt.max()) < K and int(pred.max()) < K
        return torch.bincount(gt * K + pred, minlength=K * K).reshape(K, K).cpu().numpy().astype(np.int64)

    def miou_from_conf(cm, skip=()):
        ious = {}
        for c in range(cm.shape[0]):
            if c in skip or cm[c].sum() == 0:
                continue
            ious[c] = float(cm[c, c] / max(cm[c].sum() + cm[:, c].sum() - cm[c, c], 1))
        return dict(miou=float(np.mean(list(ious.values()))) if ious else float('nan'), iou={int(k): round(v, 4) for k, v in ious.items()})

    def miou7(pred, gt):
        ious = {}
        for c in range(1, 8):
            i = int(((pred == c) & (gt == c)).sum()); u_ = int(((pred == c) | (gt == c)).sum())
            if int((gt == c).sum()) > 0:
                ious[c] = i / max(u_, 1)
        valid = gt != 0
        acc = float((pred[valid] == gt[valid]).float().mean()) if valid.any() else float('nan')
        return dict(miou7_present=float(np.mean(list(ious.values()))) if ious else float('nan'),
                    n_cls_present=len(ious), acc_excl0=acc, iou_per_cls={int(k): round(v, 4) for k, v in ious.items()})

    def save_npz(path, arr):
        path.parent.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(path.parent).free / 1e9
        if free_gb < MIN_FREE_GB:
            raise RuntimeError(f'{free_gb:.1f} GB free < {MIN_FREE_GB} GB, refusing to write full tensors')
        tmp = path.with_name(f'.{path.stem}.{pid}.{uuid.uuid4().hex[:8]}.tmp.npz')
        np.savez(tmp, **arr); os.replace(tmp, path)

    tpl = lambda t: t[0].half().cpu().numpy()   # [1,C,H,W] -> [C,H,W] fp16
    vox_cast = (lambda t: t.float()) if LOGITS else (lambda t: t.half())   # logits stage stores fp32 support logits
    cap = int(base_cfg.model.get('teacher_serve_max_points') or 0)
    frame_meta = {}

    def write_manifest(tag):
        meta = dict(
            status=tag, started_utc=t_start, written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            stage=a.stage, dataset=a.dataset, argv=sys.argv, python=sys.executable, torch=torch.__version__,
            cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
            code_md5=code_md5, model_code_md5=model_code_md5,
            teacher=dict(ckpt=tea_ck, md5=teacher_md5, config=tea_cfg_path, config_md5=md5(tea_cfg_path), label_space=t_space),
            models=[{k: v for k, v in st.items() if k not in ('cfg', 'model')} for st in roles],
            data_config=data_cfg_path, data_config_md5=md5(data_cfg_path), variants=variants, N=N,
            frames=frames, shard=a.shard, raw_frames=sorted(full_frames), rho_crop_rows=a.rho_crop_rows,
            seeds=dict(seed=a.seed, T_union='Generator(seed*1000003+frame) randperm[:cap]',
                       T_union2='Generator((seed+1)*1000003+frame) randperm[:cap]', test_forward=TEST_SEED),
            teacher_serve=dict(built_from='data config model + teacher weights, dense_train = d2skd = True',
                               max_points=cap, cap_scope=base_cfg.model.get('teacher_serve_cap_scope')),
            label_spaces=dict(teacher=t_space, models=sspace, old_map='current map'),
            common_coarsening=dict(components=coarse_components, class_to_coarse=coarse_of),
            load_reports=reports, tf32=tf32_info, data_overrides=data_ovr)
        txt = json.dumps(meta, indent=1, ensure_ascii=False, default=str)
        i_, n_ = a.shard.split('/')
        name = f'manifest_{t_start.replace(":", "").replace("-", "")}_shard{i_}of{n_}_pid{pid}.json'
        atomic_write_text(out / name, txt)
        atomic_write_text(sum_out / name, txt)
        return sum_out / name

    mf = write_manifest('started')
    print(f'manifest -> {mf}', flush=True)

    for fi in frames:
        tf0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        fm = frame_meta[fi] = dict(frame=fi, mode=mode_for(fi), full=(fi in full_frames))
        is_full = fi in full_frames
        items = {}

        def get(src, mp):
            if mp == 'old' and OLD_IS_NEW:   # same dataset object: no second load
                return get(src, 'new')
            if (src, mp) not in items:
                d = scatter(collate([ds[(src, mp)][fi]], samples_per_gpu=1), [0])[0]
                g = lambda k: (model_io._first(d[k]).to(device) if d.get(k) is not None else None)
                flag = d.get('dataset_flag'); flag = model_io._first(flag) if isinstance(flag, (list, tuple)) else flag
                items[(src, mp)] = dict(tg=g('train_grid'), gi=g('grid_ind'), gv=g('grid_ind_vox'), vl=g('train_voxel_label'),
                                        pl=g('train_pts_label'), flag=flag)
            return items[(src, mp)]

        def gt_old_for(src, sel=None):
            it_new, it_old = get(src, 'new'), get(src, 'old')
            for k in ('tg', 'gi', 'gv'):
                assert torch.equal(it_old[k], it_new[k]), f'{src}: {k} differs between label maps'
            pl = it_old['pl'].reshape(-1)
            return pl[sel] if sel is not None else pl

        # ---------- T_union (and T_union2) truncation + serving-teacher distillation field ----------
        u = get('union', 'new')
        n_union = int(u['tg'].shape[0])

        def capped(seed_val):
            sel_ = None
            if cap and n_union > cap:
                gen = torch.Generator().manual_seed(int(seed_val))
                sel_ = torch.randperm(n_union, generator=gen)[:cap].to(device)
            it_ = {k: (v[sel_] if (sel_ is not None and torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n_union) else v)
                   for k, v in u.items()}
            it_['vl'] = u['vl']; it_['flag'] = u['flag']
            return it_, sel_
        tu, sel_u = capped(int(a.seed) * 1000003 + int(fi))
        t0 = time.time()
        model_io.seed_all(TEST_SEED)
        with torch.no_grad():
            x_t = teacher_serve.extract_feat(train_grid=[tu['tg']], voxel_label=[tu['gi']])
            t_field = teacher_serve.tpv_aggregator(x_t, points=tu['gv'], voxel_label=None, point_labels=None,
                                                   dataset_flag=tu['flag'], return_loss=False, sensor_vec=None,
                                                   skip_consistency=True, distill_points=tu['gv'],
                                                   return_distill_field=True)[3]
        shapes = {k: tuple(x_t[i].shape[-2:]) for i, k in enumerate(PLANES)}
        sup_vox = t_field[0].ne(0).any(0); sup_idx = sup_vox.nonzero(as_tuple=False)
        sup, prob = LFD.plane_inputs_from_field(t_field)
        fm['T_union_field_sec'] = round(time.time() - t0, 2)
        fm['n_union_before_cap'] = n_union
        # ---------- masks ----------
        r = get('raw', 'new')
        cnt_raw = lfx.counts(r['gi'], shapes)
        masks, san = {}, {}
        for k, p in enumerate(PLANES):
            m = sup[k].to(torch.float32)
            deg = LFD._nbr_sum_2d(m) * m
            R = m * (deg > 0).to(m.dtype)
            OT = sup[k][0, 0].cpu().numpy(); OS = cnt_raw[p] > 0
            masks.update({f'OmegaT_{p}': OT, f'OmegaS_{p}': OS, f'ST_{p}': OS & OT, f'T_minus_S_{p}': OT & ~OS,
                          f'S_minus_T_{p}': OS & ~OT, f'R_{p}': R[0, 0].cpu().numpy().astype(bool),
                          f'deg_{p}': deg[0, 0].cpu().numpy().astype(np.int8),
                          f'plane_prob_{p}': prob[k][0].half().cpu().numpy()})
            ra, ca = lfx.PL[p]
            proj = torch.zeros(shapes[p], dtype=torch.bool, device=device)
            proj[sup_idx[:, ra], sup_idx[:, ca]] = True
            san[p] = dict(OmegaT_eq_proj_supvox=bool(torch.equal(proj, sup[k][0, 0])),
                          size_OmegaT=int(OT.sum()), size_OmegaS=int(OS.sum()), size_ST=int((OS & OT).sum()),
                          size_T_minus_S=int((OT & ~OS).sum()), size_S_minus_T=int((OS & ~OT).sum()),
                          size_R=int(R.sum()), plane_cells=int(OT.size))
            if not san[p]['OmegaT_eq_proj_supvox']:
                raise RuntimeError(f'frame {fi} {p}: Omega_T differs from the projection of the support voxels')
        fm['mask_sanity'] = san
        fm['n_support_voxels'] = int(sup_idx.shape[0])
        if LOGITS:
            # ---------- voxel level: Omega_S (voxels hit by the student's raw scan) on Omega_T, field shape, frame ----------
            # voxel = floor(gv / 2) (Omega_T = unique(floor(gv_t / 2))); the rule is checked on T_union's own points.
            fshape = tuple(int(s) for s in t_field.shape[-3:])

            def vox_hit(gv):
                vi = torch.floor(gv.reshape(-1, 3).double() / 2.0).long()
                for d in range(3):
                    vi[:, d] = vi[:, d].clamp(0, fshape[d] - 1)
                h = torch.zeros(fshape, dtype=torch.bool, device=gv.device)
                h[vi[:, 0], vi[:, 1], vi[:, 2]] = True
                return h
            hit_t, hit_s = vox_hit(tu['gv']), vox_hit(r['gv'])
            fm['vox_sanity'] = dict(OmegaT_eq_floor_gv_half=bool(torch.equal(hit_t, sup_vox)), n_OmegaT=int(sup_vox.sum()),
                                    n_floor_gv_half=int(hit_t.sum()), n_OmegaS=int(hit_s.sum()),
                                    n_OmegaS_in_OmegaT=int((hit_s & sup_vox).sum()))
            masks['OmegaS_supvox'] = hit_s[sup_vox].cpu().numpy()
            masks['field_shape'] = np.array(fshape, dtype=np.int64)
            masks['frame'] = np.array(fi, dtype=np.int64)
        if is_full:
            save_npz(out / '_masks' / f'frame_{fi:06d}.npz', masks)

        # ---------- per variant ----------
        V = {}
        for v in variants:
            tv = time.time(); info = {}
            if v == 'T_union':
                model, it, src, sel, space = teacher_own, dict(tu), 'union', sel_u, t_space
            elif v == 'T_union2':
                it, sel = capped((int(a.seed) + 1) * 1000003 + int(fi))
                model, src, space = teacher_own, 'union', t_space
            elif v == 'T_raw':
                model, it, src, sel, space = teacher_own, r, 'raw', None, t_space
            else:
                model, it, src, sel, space = smodel[v], r, 'raw', None, sspace[v]
            n_loaded = int(it['tg'].shape[0])
            tok, neck, logit, sens, used = test_path(model, it)
            tmp_cap = getattr(model, 'test_max_points', None)
            gt_new = used['pl'].reshape(-1)
            gt_old = gt_old_for(src, sel)
            assert gt_old.shape == gt_new.shape
            pred = logit.argmax(1)
            info.update(label_space=space, n_loaded=n_loaded, n_used=int(used['tg'].shape[0]),
                        points_md5=md5_arr(used['tg']), grid_ind_md5=md5_arr(used['gi']),
                        tokenizer_in_n=used['tok_in_n'], tokenizer_in_md5=used['tok_in_md5'], tokenizer_in_gi_md5=used['tok_in_gi_md5'],
                        acc_own_space=miou7(pred, gt_new if space == 'new' else gt_old),
                        acc_other_space=miou7(pred, gt_old if space == 'new' else gt_new),
                        frac_gt_old_ne_new=float((gt_old != gt_new).float().mean()),
                        sensor_pred=(sens.flatten().tolist() if torch.is_tensor(sens) else None), test_max_points=tmp_cap)
            if sel is not None:
                info['kept_idx_md5_int32'] = hashlib.md5(sel.cpu().numpy().astype(np.int32).tobytes()).hexdigest()
            gt_own = gt_new if space == 'new' else gt_old
            # integer counts (a float32 mean of all ones can give 1 - 2^-24)
            n_coarse_dis = int((coarse_lut[gt_new.long()] != coarse_lut[gt_old.long()]).sum())
            coarse_agree = 1.0 - n_coarse_dis / max(int(gt_new.numel()), 1)
            assert n_coarse_dis == 0, f'coarsened GT differs between label maps at {n_coarse_dis} points'
            info['coarse_gt_new_old_agree'] = coarse_agree
            conf_own = confmat(pred, gt_own, 8)
            conf_coarse = confmat(coarse_lut[pred.long()], coarse_lut[gt_new.long()], KC)
            info.update(conf_own=conf_own.tolist(), conf_own_gt_space=space, conf_coarse=conf_coarse.tolist(),
                        miou7_from_conf=miou_from_conf(conf_own, skip=(0,)), coarse_miou_all=miou_from_conf(conf_coarse),
                        coarse_miou_excl_comp_with_cls0=miou_from_conf(conf_coarse, skip=(coarse_of[0],)))
            tokl = list(tok) if isinstance(tok, (list, tuple)) else [tok]
            neckl = [n if isinstance(n, (list, tuple)) else [n] for n in neck]
            info['tok_shapes'] = [list(t.shape) for t in tokl]
            info['neck_shapes'] = {p: [list(t.shape) for t in nl] for p, nl in zip(PLANES, neckl)}
            cnt = lfx.counts(used['gi'], shapes)
            if v == 'T_union':
                info['serve_vs_own_neck_s0_maxabs'] = {p: float((neckl[i][0][0] - x_t[i][0]).abs().max()) for i, p in enumerate(PLANES)}
            arr = {}
            if v in smodel:
                agg = model.tpv_aggregator; saved = getattr(agg, 'distill_field_only', True)
                agg.distill_field_only = False
                try:
                    with torch.no_grad():
                        s_field = agg([neckl[i][0] for i in range(3)], points=[used['gv']], voxel_label=None, point_labels=None,
                                      dataset_flag=it['flag'], return_loss=False, sensor_vec=None,
                                      skip_consistency=True, distill_points=tu['gv'], return_distill_field=True)[3]
                finally:
                    agg.distill_field_only = saved
                assert s_field.shape == t_field.shape
                arr['cs_logit'] = vox_cast(s_field[0][:, sup_vox].T.contiguous()).cpu().numpy()
                info['cs_support_nonzero_frac'] = float(s_field[0][:, sup_vox].ne(0).any(0).float().mean())
                del s_field
            elif LOGITS and v in ('T_union2', 'T_raw'):
                # teacher variants queried at the Omega_T voxel centres: the serving teacher (built as for T_union) takes
                # this variant's input, distill_points = T_union's points; seeded before the forward as for T_union.
                model_io.seed_all(TEST_SEED)
                with torch.no_grad():
                    x_v = teacher_serve.extract_feat(train_grid=[it['tg']], voxel_label=[it['gi']])
                    f_v = teacher_serve.tpv_aggregator(x_v, points=it['gv'], voxel_label=None, point_labels=None,
                                                       dataset_flag=it['flag'], return_loss=False, sensor_vec=None,
                                                       skip_consistency=True, distill_points=tu['gv'],
                                                       return_distill_field=True)[3]
                assert f_v.shape == t_field.shape
                arr['cs_logit'] = f_v[0][:, sup_vox].T.contiguous().float().cpu().numpy()
                info['cs_support_nonzero_frac'] = float(f_v[0][:, sup_vox].ne(0).any(0).float().mean())
                del x_v, f_v
            for i, p in enumerate(PLANES):
                arr[f'tok_{p}'] = tpl(tokl[i])
                for j, t in enumerate(neckl[i]):
                    arr[f'neck_{p}_s{j}'] = tpl(t)
                arr[f'cnt_{p}'] = cnt[p]
            arr.update(conf_own=conf_own, conf_coarse=conf_coarse)
            arr.update(logits=logit.half().cpu().numpy(), gt_new=gt_new.cpu().numpy().astype(np.int8),
                       gt_old=gt_old.cpu().numpy().astype(np.int8),
                       train_grid=used['tg'].float().cpu().numpy(), grid_ind=used['gi'].float().cpu().numpy(),
                       aggregator_sampled_scale=np.array(0))
            if sel is not None:
                arr['kept_idx'] = sel.cpu().numpy().astype(np.int32)
            if v == 'T_union':
                arr['sup_idx'] = sup_idx.cpu().numpy().astype(np.int16)
                arr['sup_logit'] = vox_cast(t_field[0][:, sup_vox].T.contiguous()).cpu().numpy()
            if torch.is_tensor(sens):
                arr['sensor_pred'] = sens.float().cpu().numpy()
            if is_full:
                fp = out / v / f'frame_{fi:06d}.npz'
                save_npz(fp, arr)
                info['file_bytes'] = fp.stat().st_size
            V[v] = arr
            info['wall_sec'] = round(time.time() - tv, 2)
            fm[v] = info
            print(f'[frame {fi}] {v}: n={info["n_used"]} md5={info["points_md5"][:12]} '
                  f'miou7={info["acc_own_space"]["miou7_present"]:.3f} ({space}) {info["wall_sec"]}s', flush=True)
            del tok, neck, neckl, tokl, logit; torch.cuda.empty_cache()
        del x_t

        # ---------- per-frame input identity (taken at the tokenizer input) ----------
        ident = {}
        raw_group = [v for v in variants if v in smodel or v == 'T_raw']
        for gname, grp in (('raw', raw_group),):
            sig = {v: [fm[v]['tokenizer_in_n'], fm[v]['tokenizer_in_md5'], fm[v]['tokenizer_in_gi_md5']] for v in grp}
            ident[gname] = dict(members=sig, all_equal=(len({tuple(x) for x in sig.values()}) <= 1))
        fm['student_input_identity'] = ident
        fm['dump_sec'] = round(time.time() - tf0, 2)
        fm['memory'] = dict(dump_gpu_peak_alloc_MB=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                            dump_gpu_peak_reserved_MB=round(torch.cuda.max_memory_reserved() / 2 ** 20, 1))
        bad = [g for g, d in ident.items() if not d['all_equal']]
        if bad:
            atomic_write_text(sum_out / '_info' / f'frame_{fi:06d}.json', json.dumps(fm, ensure_ascii=False, default=str))
            raise RuntimeError(f'frame {fi}: inputs differ between the members of {bad}: {ident}')

        # ---------- per-frame summary (in memory) ----------
        torch.cuda.reset_peak_memory_stats()
        ts = time.time()
        md = mode_for(fi)
        res = frame_mod.summarize_frame(V, masks, device='cuda', mode=md)
        fm['summary_sec'] = round(time.time() - ts, 2)
        fm['summary_mode'] = md
        tw = time.time()
        atomic_write_text(sum_out / f'frame_{fi:06d}.json', json.dumps(frame_mod._jsonable(res)))
        fm['summary_json_bytes'] = (sum_out / f'frame_{fi:06d}.json').stat().st_size
        fm['summary_write_sec'] = round(time.time() - tw, 2)
        if not LOGITS and is_full:
            got = EGD.export_frame(TS, out, fi)
            chk = EGD.check_against_summary(stage_out, fi, got)
            fm['p2dom_cells'] = dict(got, check=chk)
            if chk is not None and not chk['match']:   # the export runs on the CPU, the summary on the GPU
                print(f'[frame {fi}] warning: gate-domain counts differ from the summary: {chk} vs {got}', flush=True)
        fm['memory'].update(summary_gpu_peak_alloc_MB=round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
                            summary_gpu_peak_reserved_MB=round(torch.cuda.max_memory_reserved() / 2 ** 20, 1))
        del res
        fm['memory'].update(host_rss_MB=round(rss_mb() or -1, 1),
                            host_maxrss_MB=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
                            V_host_MB=round(sum(x.nbytes for d in V.values() for x in d.values()) / 2 ** 20, 1))
        fm['frame_total_sec'] = round(time.time() - tf0, 2)
        atomic_write_text(sum_out / '_info' / f'frame_{fi:06d}.json', json.dumps(fm, ensure_ascii=False, default=str))
        print(f'[frame {fi}] done mode={fm.get("summary_mode")} dump={fm["dump_sec"]}s summary={fm.get("summary_sec")}s '
              f'total={fm["frame_total_sec"]}s mem={fm["memory"]}', flush=True)
        del V, masks, items, t_field, sup, prob; torch.cuda.empty_cache()

    mf = write_manifest('finished')
    print(f'manifest -> {mf}', flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    _g = argparse.ArgumentParser(add_help=False); _g.add_argument('--gpu'); _ga, _ = _g.parse_known_args()
    if _ga.gpu is not None:
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_ga.gpu)
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc(); sys.exit(1)
