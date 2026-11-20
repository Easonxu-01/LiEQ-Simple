"""GPU pass of the FBD measurement: voxel logit fields of the teacher and of each student on the teacher's
dense support, and their Chebyshev moments under the support-graph Laplacian.

Per evaluated frame (all models see exactly the same inputs):

1. Dense input. The test split is read with the dense union input: original scan + dense completion
   (the teacher's serving input during distillation and the source cloud of the student's sensor
   simulation). The training-time augmentations are not applied.
2. Teacher field. The dense input is capped to ``model.teacher_serve_max_points`` points by a seeded
   uniform permutation (CPU generator, seed ``seed * 1000003 + frame``). The teacher runs its dense path
   and returns the voxel logit field on the voxels of the capped input (support Omega_T).
3. Student input. Each student receives a simulated sparse scan generated from the full dense input by the
   model's own sensor simulator (``_sampling``, ray casting with randomly drawn sensor parameters), with
   all global RNGs seeded to ``seed * 1000003 + frame + 17`` immediately before the call; the first of the
   simulated views is used. Inputs are written once to a cache directory and re-read for every model, so
   all students are compared on identical inputs. See ``student_input`` for the exact rule.
4. Student field. The student runs its full deployment path (sensor head -> FiLM conditioning on its own
   prediction) and its field is queried at the same teacher voxels, so teacher and student fields share
   the support Omega_T.
5. Spectrum. The graph is the subgraph of the 6-neighbour voxel grid induced by Omega_T, with the
   Laplacian used by the high-frequency distiller (``hf_operator="support"``):
   L x = m * (deg * x - sum_neighbours x) / 6, spectrum in [0, 2). For each field x and channel c the
   Chebyshev moments mu_n = <x_c, T_n(L~) x_c>, L~ = (2 / 2.02) L - I, n = 0..1024 are computed in
   float64; fbd_math turns them into fine-bin energies.
"""
import copy
import importlib
import importlib.util
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

import fbd_math

CODE_ROOT = Path(__file__).resolve().parents[3]          # repository root (contains projects/ and tools/)

# Sensor-simulator settings used to generate the student inputs of the measurement. These equal the
# simulator defaults (the values the paper's measurement inputs were generated with). A training config
# may set other values for its own training-time augmentation (e.g. the SemanticKITTI config pastes
# class 5 only); that choice is not part of the measurement, so the simulator is pinned here.
SIMULATOR_SETTINGS = dict(minority_mixup_prob=0.15, minority_mixup_classes=(5, 6), minority_mixup_num_add=2)

STUDENT_SEED_OFFSET = 17     # student-input simulation seed = seed * 1000003 + frame + 17
FRAME_SEED_MULT = 1000003

# TF32 policy of the measurement: SemanticKITTI / nuScenes were run with TF32 enabled (the PyTorch 1.10
# default), Waymo with TF32 disabled.
TF32 = dict(semantickitti=True, nuscenes=True, waymo=False)


def set_tf32(dataset):
    on = TF32[dataset]
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on
    return dict(matmul_allow_tf32=on, cudnn_allow_tf32=on)


def seed_all(s):
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def frame_indices(n, num_frames):
    """Evaluated frames: num_frames evenly spaced indices over the split, first and last frame included."""
    return sorted(set(np.linspace(0, n - 1, min(num_frames, n)).round().astype(int).tolist()))


def _first(x):
    if isinstance(x, (list, tuple)):
        return x[0]
    if torch.is_tensor(x) and x.dim() >= 3:
        return x[0]
    return x


# ----------------------------------------------------------------------------- build
def _import_plugin(cfg, config_path):
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    if getattr(cfg, 'plugin', False):
        pdir = cfg.plugin_dir if hasattr(cfg, 'plugin_dir') else os.path.dirname(config_path)
        mod = '.'.join(p for p in os.path.dirname(pdir).split('/') if p)
        importlib.import_module(mod)


def _load_cfg(path):
    from mmcv import Config
    cfg = Config.fromfile(path)
    _import_plugin(cfg, path)
    if cfg.get('test_dual', False):
        raise RuntimeError(f'{path}: test_dual configs are not supported')
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    return cfg


def check_measurement_config(cfg, path):
    """The teacher is instantiated from the student training config (as in distillation training, where
    the teacher network is a deep copy of the student architecture loaded with the teacher weights).
    A dense-teacher training config builds a different module set and is refused."""
    if cfg.model.get('dense_train', False) or not cfg.model.get('teacher_serve_max_points'):
        raise SystemExit(f'--teacher-config {path}: expected the student training (distillation) config, which '
                         'defines model.teacher_serve_max_points; the teacher network is built from it as in '
                         'distillation training. Dense-teacher training configs are not used by this measurement.')


def build_dataset(cfg, split, data_root=None):
    """Test split with the dense union input (scan + dense completion), no training augmentation."""
    from mmdet3d.datasets import build_dataset as _bd
    ds_cfg = cfg.data[split]
    ds_cfg.test_mode = True
    if data_root:
        ds_cfg.data_root = data_root
    ds_cfg.random = True
    ds_cfg.dense7sparse = True
    ds_cfg.reset_random_prob = 0.0
    n_hit = 0
    for step in ds_cfg.pipeline:
        if step.get('type') == 'LoadPointsFromFile_RPR':
            step['Random'] = True
            step['dense7sparse'] = True
            step['load_dim'] = 3
            n_hit += 1
    if n_hit != 1:
        raise RuntimeError(f'the union input needs exactly one LoadPointsFromFile_RPR step, found {n_hit}')
    ds_cfg.pop('samples_per_gpu', None)
    return _bd(ds_cfg)


def load_weights_report(model, path, allow_missing, role):
    ck = torch.load(path, map_location='cpu')
    sd = ck.get('state_dict', ck)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    try:
        res = model.load_state_dict(sd, strict=False)
    except RuntimeError as e:
        raise RuntimeError(f'{path} does not match the architecture of its config: {str(e)[:300]}')
    benign = ('distill', 'ema_', '_balance', 'vanilla_kd', 'cwd', 'pvkd', 'num_batches_tracked')
    if role == 'teacher':
        # The dense teacher path does not use the sensor head / FiLM (no sensor conditioning), so these
        # keys are structurally absent from a dense-teacher checkpoint.
        benign = benign + ('sensor_encoder', 'sensor_head', 'film_layer', '_film_alpha', 'sensor_mean', 'sensor_std')
    crit = [k for k in res.missing_keys if not any(b in k for b in benign)]
    rep = dict(role=role, n_missing=len(res.missing_keys), n_unexpected=len(res.unexpected_keys),
               critical_missing=crit[:50])
    if crit and not allow_missing:
        raise RuntimeError(f'{path}: {len(crit)} non-distillation parameters missing (e.g. {crit[:5]}); '
                           'wrong checkpoint for this config? (--allow-missing-keys to override)')
    return rep


def _freeze(m, device):
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device).eval()


def build_all(teacher_config, teacher_ckpt, models, split, data_root, device, allow_missing=False):
    """models: list of (name, config, checkpoint).
    Returns cfg, dataset, teacher, sampler, students [(name, model)], teacher input cap, load report."""
    from mmdet3d.models import build_model
    os.chdir(CODE_ROOT)
    sys.path.insert(0, str(CODE_ROOT))
    cfg = _load_cfg(teacher_config)
    check_measurement_config(cfg, teacher_config)
    dataset = build_dataset(cfg, split, data_root)
    base = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    teacher = copy.deepcopy(base)
    report = dict(teacher=load_weights_report(teacher, teacher_ckpt, allow_missing, 'teacher'))
    teacher.tpv_aggregator.dense_train = True
    teacher.tpv_aggregator.d2skd = True
    teacher.dense_train = True
    teacher = _freeze(teacher, device)
    # Input simulator: an unweighted student-mode network of the measurement config (the simulator uses no
    # learned parameters), with the measurement's simulator settings.
    sampler = base
    for k, v in SIMULATOR_SETTINGS.items():
        setattr(sampler, k, v)
    sampler.dense_train = False
    sampler.random = True
    sampler = _freeze(sampler, device)
    students = []
    for name, mcfg_path, ckpt in models:
        mcfg = _load_cfg(mcfg_path)
        m = build_model(mcfg.model, test_cfg=mcfg.get('test_cfg'))
        report[name] = load_weights_report(m, ckpt, allow_missing, 'student')
        m.tpv_aggregator.dense_train = False
        m.tpv_aggregator.d2skd = False
        m.dense_train = False
        students.append((name, _freeze(m, device)))
    cap = int(cfg.model.get('teacher_serve_max_points') or 0)
    return cfg, dataset, teacher, sampler, students, cap, report


# ----------------------------------------------------------------------------- operator and moments
def load_hf_distiller(device):
    p = CODE_ROOT / 'projects/unilidar_plugin/pointocc_model/Spectraldistiller.py'
    spec = importlib.util.spec_from_file_location('hf_distiller_module', p)
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    return sm.TPVHighFreqSpectralDistiller(gate_domain='support', hf_operator='support',
                                           use_checkpoint=False).to(device)


class SupportLaplacian:
    """L_Omega of the high-frequency distiller on the support of a teacher field (any channel non-zero)."""

    def __init__(self, distiller, teacher_field):
        self.d = distiller
        self.m, self.deg = distiller._support_geometry(teacher_field)

    def __call__(self, x):
        return self.d._lap_op(x, self.m.to(x.dtype), self.deg.to(x.dtype))


@torch.no_grad()
def cheb_moments(lap, x, K=fbd_math.CHEB_ORDER, dtype=torch.float64):
    """mu[c, n] = <x_c, T_n(L~) x_c>, L~ = (2 / LMAX_MAP) L - I, n = 0..K. x: [1, C, ...] -> [C, K+1] float64."""
    x = x.to(dtype)
    dims = tuple(range(2, x.dim()))
    sc = 2.0 / fbd_math.LMAX_MAP
    mu = torch.zeros(x.shape[1], K + 1, dtype=torch.float64, device=x.device)

    def rec(n, t):
        mu[:, n] = (x * t).sum(dim=dims)[0].to(torch.float64)

    def Lt(v):
        return sc * lap(v) - v

    t_prev = x
    rec(0, t_prev)
    if K == 0:
        return mu.cpu().numpy()
    t_cur = Lt(x)
    rec(1, t_cur)
    for n in range(2, K + 1):
        t_next = 2.0 * Lt(t_cur) - t_prev
        t_prev, t_cur = t_cur, t_next
        rec(n, t_cur)
    return mu.cpu().numpy()


# ----------------------------------------------------------------------------- per-frame fields
def to_device(data, device):
    from mmcv.parallel import scatter
    return scatter(data, [0])[0] if device.type == 'cuda' else data


def dense_input(data, device):
    tg = _first(data['train_grid']).to(device)
    gi = _first(data['grid_ind']).to(device)
    gv = data.get('grid_ind_vox')
    if gv is None:
        raise ValueError('the sample has no grid_ind_vox; the distillation support is defined on it')
    gv = _first(gv).to(device)
    pl = _first(data['train_pts_label']).to(device)
    flag = data.get('dataset_flag')
    flag = _first(flag) if isinstance(flag, (list, tuple)) else flag
    return tg, gi, gv, pl, flag


def cap_teacher_input(tg, gi, gv, frame, seed, cap, device):
    n_dense = int(tg.shape[0])
    if cap and n_dense > cap:
        g = torch.Generator().manual_seed(int(seed) * FRAME_SEED_MULT + int(frame))
        idx = torch.randperm(n_dense, generator=g)[:cap].to(device)
        return tg[idx], gi[idx], gv[idx]
    return tg, gi, gv


@torch.no_grad()
def teacher_field(teacher, tg_t, gi_t, gv_t, flag):
    x_t = teacher.extract_feat(train_grid=[tg_t], voxel_label=[gi_t])
    return teacher.tpv_aggregator(
        x_t, points=gv_t, voxel_label=None, point_labels=None, dataset_flag=flag,
        return_loss=False, sensor_vec=None, skip_consistency=True,
        distill_points=gv_t, return_distill_field=True)[3]


@torch.no_grad()
def student_input(sampler, tg, gi, gv, pl, frame, seed, cache_dir, device):
    """Simulated sparse student input of one frame, read from / written to cache_dir/<frame:06d>.pt.
    Generation rule: seed_all(seed * 1000003 + frame + 17), then the simulator on the full dense input
    (uncapped), keep the first simulated view. Returns (points, grid, grid_vox, generated)."""
    cf = Path(cache_dir) / f'{frame:06d}.pt'
    if cf.exists():
        c = torch.load(cf, map_location='cpu')
        return c['points'].to(device), c['grid'].to(device), c['grid_vox'].to(device), False
    seed_all(int(seed) * FRAME_SEED_MULT + int(frame) + STUDENT_SEED_OFFSET)
    pts, grd, grdv, _, sp = sampler._sampling([tg], [gi], [gv], [pl], raw_mask=[False])
    ps, gs, gvs = pts[0], grd[0], grdv[0]
    sensor = sp[0:1] if sp is not None else None
    cf.parent.mkdir(parents=True, exist_ok=True)
    tmp = cf.with_suffix('.tmp')
    torch.save(dict(points=ps.cpu(), grid=gs.cpu(), grid_vox=gvs.cpu(),
                    sensor=sensor.cpu() if sensor is not None else torch.empty(0)), tmp)
    os.replace(tmp, cf)
    return ps, gs, gvs, True


@torch.no_grad()
def student_field(model, ps, gs, gvs, gv_t, flag):
    x_s = model.extract_feat([ps], [gs])
    agg = model.tpv_aggregator
    saved = getattr(agg, 'distill_field_only', True)
    agg.distill_field_only = False          # full deployment path (sensor head -> FiLM)
    try:
        out = agg(x_s, points=[gvs], voxel_label=None, point_labels=None, dataset_flag=flag,
                  return_loss=False, sensor_vec=None, skip_consistency=True,
                  distill_points=gv_t, return_distill_field=True)
    finally:
        agg.distill_field_only = saved
    return out[3]
