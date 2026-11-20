#!/usr/bin/env python
"""Logits-level spectral analysis: per-frame summary (called once per frame by run_dump.py --stage logits):
    summarize_frame(V, masks, device='cuda', mode='eval'|'full')
V[variant] has the keys of the logits-stage npz; T_union carries sup_idx / sup_logit, the other variants carry cs_logit
(8-class logits queried at the Omega_T voxel centres, fp32).

Object: the common-support voxel logit field [8, 240, 180, 16] (zero outside Omega_T), i.e. the field the HF
distillation acts on. Variants: T_union (reference), T_union2 (re-truncation seed, noise ceiling), T_raw (teacher on the
student's raw scan, control), baseline, student (the two model roles, see run_dump.py).
Operator: 3D sub-graph L_Omega (spectral_core.SupportLaplacian = the HF distiller's _support_geometry / _lap_op).
Bands: 11 Chebyshev-Jackson bands (K = 1024), see common/logits_core. Pre-processing: per-channel mean removed inside
Omega_T, divided by the RMS inside Omega_T.
Quantities:
  M1      per-band CKA (each variant vs T_union; regions O_ST / O_T / class boundary / class interior / gate open /
          gate closed / distance);
  M3      spurious high frequency where the gate is closed: excess / excess fraction / deficit (gate = the HF
          distiller's _teacher_gate_b1, w <= 0.5 is closed);
  M6_phi  per-band useful / spurious decomposition of the voxel logits: cross-fitted ridge regression across channels
          with outer folds over phi sectors and inner leave-one-sector-out selection of the ridge parameter; recovery
          R^2(T_k <- S_k) and R^2(S_k <- T_k), useful / spurious energy shares, pooled recovery of the band groups;
  M7_phi  as M6_phi, for the TPV feature plane xy neck s0 (128 channels, DCT tight-frame bands, local attribution on
          Omega_T, near-sensor rho rows cropped as in the TPV analysis).
The HF gate is the distiller built from model.tpv_aggregator.highfreq_distiller_cfg of the student model config
(set_gate_config), i.e. the gate used in its training.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from common import logits_core as LC        # noqa: E402
from common import tpv_metrics as TM        # noqa: E402
import tpv_summary as TS                    # noqa: E402

REF = 'T_union'
VARS = ('T_union', 'T_union2', 'T_raw', 'baseline', 'student')
M6_VARS = ('T_union2', 'T_raw', 'baseline', 'student')
GATE_CFG = None   # model.tpv_aggregator.highfreq_distiller_cfg of the student config (set_gate_config)
RHO_BINS = {'rho_near': (0, 70), 'rho_mid': (70, 141), 'rho_far': (141, 10 ** 6)}


def set_gate_config(config_path):
    """Read the HF distiller config of the student model config (the gate of M3 / regions gate_open, gate_closed)."""
    global GATE_CFG
    from mmcv import Config
    cfg = Config.fromfile(str(config_path))
    GATE_CFG = cfg.model.tpv_aggregator.highfreq_distiller_cfg
    _CACHE.clear()


K = LC.K_DEFAULT
HF, LOW, NB = LC.HF_BANDS, LC.LOW_BANDS, LC.NB

_CACHE = {}


def hf_gate_distiller(device):
    """HF distiller of the student training config: the same gate as in training (gate_kappa 0.5, prob, voxel, support)."""
    key = ('hfd', str(device))
    if key not in _CACHE:
        if GATE_CFG is None:
            raise RuntimeError('logits_summary.set_gate_config(<student model config>) must be called first')
        h = dict(GATE_CFG)
        h['use_checkpoint'] = False
        d = LC.SD.TPVHighFreqSpectralDistiller(**h).to(device)
        _CACHE[key] = (d, {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in h.items()})
    return _CACHE[key]


# ------------------------------------------------------------------ voxel regions
def vox_regions(cls_sup, sup_idx, fshape, in_s, device):
    """cls_sup: [N] teacher 8-class argmax; sup_idx: [N,3]. Boundary = a 6-neighbour inside Omega_T has another class;
    interior = has a neighbour inside Omega_T and none of another class."""
    X, Y, Z = fshape
    g = torch.full((X + 2, Y + 2, Z + 2), -1, dtype=torch.long, device=device)
    a, b, c = sup_idx[:, 0] + 1, sup_idx[:, 1] + 1, sup_idx[:, 2] + 1
    g[a, b, c] = cls_sup
    diff = torch.zeros_like(cls_sup, dtype=torch.bool); has = torch.zeros_like(diff)
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        nb = g[a + dx, b + dy, c + dz]
        ok = nb >= 0
        has |= ok
        diff |= ok & (nb != cls_sup)
    reg = dict(OmegaT=torch.ones_like(diff), O_ST=in_s, O_T=~in_s, boundary=diff, interior=has & ~diff)
    for k, (lo, hi) in RHO_BINS.items():
        reg[k] = (sup_idx[:, 0] >= lo) & (sup_idx[:, 0] < hi)
    return reg


# ------------------------------------------------------------------ shared computations
def band_cka(yv, yt, regs, names):
    out = {}
    for r in names:
        m = regs[r]
        if int(m.sum()) < 50:
            out[r] = [float('nan')] * NB
            continue
        out[r] = [LC.cka(yv[k][:, m].T.double(), yt[k][:, m].T.double()) for k in range(NB)]
    return out


def energies(y):
    """y: [nb, C, N] -> local energy e [nb, N]."""
    return (y.double() ** 2).sum(1)


def m6_decomp_nested(yv, yt, ev, sector):
    """Same fields as m6_decomp, with outer two folds over phi sectors and inner leave-one-sector-out selection of the
    ridge parameter (logits_core.ridge_r2_nested; the R^2 reference model = training-half mean). sector: [N] phi
    60-degree sector id."""
    rec, fwd, sst_r, lam_r, lam_f = [], [], [], [], []
    for k in range(NB):
        r1, s1, l1 = LC.ridge_r2_nested(yv[k].T, yt[k].T, sector)
        r2, _s2, l2 = LC.ridge_r2_nested(yt[k].T, yv[k].T, sector)
        rec.append(r1); fwd.append(r2); sst_r.append(s1); lam_r.append(l1); lam_f.append(l2)
    Es = ev.sum(1).cpu().numpy()
    share = Es / max(Es.sum(), 1e-300)
    f_cl = np.clip(np.nan_to_num(np.array(fwd), nan=0.0), 0.0, 1.0)

    def grp(idx):
        sse = sum((1.0 - rec[k]) * sst_r[k] for k in idx if np.isfinite(rec[k]))
        sst = sum(sst_r[k] for k in idx if np.isfinite(rec[k]))
        return (1.0 - sse / sst) if sst > 0 else float('nan')
    return dict(recovery=rec, r2_student_from_teacher=fwd, energy_share_student=share.tolist(),
                useful_share=(f_cl * share).tolist(), pseudo_share=((1.0 - f_cl) * share).tolist(),
                recovery_low=grp(LOW), recovery_mid=grp(LC.MID_BANDS), recovery_hf=grp(HF),
                pseudo_share_hf=float(((1.0 - f_cl) * share)[HF].sum()), useful_share_hf=float((f_cl * share)[HF].sum()),
                ridge_lambda_rel_T_from_S=lam_r, ridge_lambda_rel_S_from_T=lam_f)


# ------------------------------------------------------------------ M7: TPV features
def m7_tpv(V, masks, device):
    """M7_phi: per-band recovery decomposition of the TPV feature plane xy neck s0 (phi-sector cross-fitting)."""
    om = TS.rho_crop(masks['OmegaT_xy'].astype(bool), 'xy')   # no-op when TS.RHO_CROP == 0
    H, W = om.shape
    pb = TM.PlaneBasis.get(H, W, device, (False, False))
    dom = torch.from_numpy(om).to(device)
    cells = dom.reshape(-1).nonzero().flatten()
    rr, cc = cells // W, cells % W
    bands = {}
    for v in VARS:
        if v not in V or 'neck_xy_s0' not in V[v]:
            continue
        x = TM.to_t(np.ascontiguousarray(TS.rho_crop(np.asarray(V[v]['neck_xy_s0'], np.float32), 'xy')), device, torch.float64)
        y, _ = TM.preprocess(x, dom, 'standard')
        Xh = pb.fwd(y)
        bands[v] = torch.stack([TM.band_signal(Xh, pb, pb.w_lam[k]).reshape(x.shape[0], -1)[:, cells] for k in range(NB)])   # [nb, C, n]
    out_phi = {}
    yt = bands[REF]
    sector = LC.phi_sector(cc)                     # the columns of the xy plane are phi
    for v in M6_VARS:
        if v not in bands:
            continue
        ev = energies(bands[v])
        out_phi[v] = m6_decomp_nested(bands[v], yt, ev, sector)
    out_phi['_n_cells'] = int(cells.numel())
    return out_phi


# ------------------------------------------------------------------ M8: prediction planes (2D sub-graph operator)


# ------------------------------------------------------------------ main function
def summarize_frame(V, masks, device='cuda', mode='eval'):
    """Per-frame quantities of the paper (voxel logits on the common support, and the xy feature plane):
      M1       per-band CKA of each variant against T_union, per region
      M3       spurious high frequency (excess / excess fraction / deficit) per region, gate regions included
      M6_phi   per-band recovery decomposition of the voxel logits (phi-sector cross-fitting)
      M7_phi   the same for the TPV feature plane xy neck s0
      n_support, region_n, gate_cfg, support_eq_gate_support   bookkeeping
    mode ('full' for raw frames, 'eval' otherwise) is recorded; both modes compute the same quantities."""
    t0 = time.time()
    tm = {}
    out = dict(mode=mode, variants=[v for v in VARS if v in V], timing=tm)
    sup_idx_np = np.asarray(V[REF]['sup_idx']).astype(np.int64)
    fshape = tuple(int(s) for s in masks['field_shape']) if 'field_shape' in masks else (240, 180, 16)
    sup_idx = torch.from_numpy(sup_idx_np).to(device)
    N = sup_idx.shape[0]
    flat = (sup_idx[:, 0] * fshape[1] + sup_idx[:, 1]) * fshape[2] + sup_idx[:, 2]
    vals = {REF: np.asarray(V[REF]['sup_logit'], np.float32)}
    for v in VARS[1:]:
        if v in V and 'cs_logit' in V[v]:
            vals[v] = np.asarray(V[v]['cs_logit'], np.float32)
    vs = list(vals)
    out['variants'] = vs
    in_s = torch.from_numpy(np.asarray(masks['OmegaS_supvox'], bool)).to(device)
    t_log = torch.from_numpy(vals[REF]).to(device).T.contiguous()             # [8, N]
    cls = t_log.argmax(0)
    regs = vox_regions(cls, sup_idx, fshape, in_s, device)
    # gate (the HF distiller itself)
    hfd, hcfg = hf_gate_distiller(device)
    tdense = torch.zeros((1, 8, fshape[0] * fshape[1] * fshape[2]), dtype=torch.float32, device=device)
    tdense[0][:, flat] = t_log
    tdense = tdense.reshape((1, 8) + fshape)
    with torch.no_grad():
        mm, deg = hfd._support_geometry(tdense)
        w = hfd._teacher_gate_b1(tdense, mm, deg)
        Rg = hfd._gate_region(mm, deg)
    wv = w.reshape(-1)[flat]; Rv = Rg.reshape(-1)[flat] > 0
    regs['gate_closed'] = Rv & (wv <= 0.5)
    regs['gate_open'] = Rv & (wv > 0.5)
    out['n_support'] = int(N)
    out['region_n'] = {k: int(v.sum()) for k, v in regs.items()}
    out['gate_cfg'] = {k: hcfg[k] for k in ('gate_kappa', 'gate_source', 'gate_share', 'gate_domain', 'hf_operator', 'alpha') if k in hcfg}
    out['support_eq_gate_support'] = bool(int(mm.reshape(-1)[flat].sum()) == N and int(mm.sum()) == N)
    del tdense, w, Rg, mm, deg
    tm['setup'] = time.time() - t0
    # band-limited signals (one recursion for all variants), standard scale
    lap = LC.support3d(device)
    msk = torch.zeros(fshape[0] * fshape[1] * fshape[2], dtype=torch.float32, device=device)
    msk[flat] = 1.0
    lap.set_support(msk.reshape((1, 1) + fshape))
    ts = time.time()
    xs = [LC.preprocess_sparse(torch.from_numpy(vals[v]).to(device).T.contiguous(), 'standard') for v in vs]
    xc = torch.cat(xs, 0)
    dense = torch.zeros((1, xc.shape[0], msk.numel()), dtype=torch.float32, device=device)
    dense[0][:, flat] = xc
    y = LC.cheb_band_signals(lap, dense.reshape((1, xc.shape[0]) + fshape), flat, K)
    ys = {v: y[:, 8 * i:8 * i + 8] for i, v in enumerate(vs)}
    del dense
    tm['bands_standard'] = time.time() - ts
    E = {v: energies(ys[v]) for v in vs}
    # M1
    cka_regs = ('OmegaT', 'O_ST', 'O_T', 'boundary', 'interior', 'gate_open', 'gate_closed', 'rho_near', 'rho_mid', 'rho_far')
    out['M1'] = {v: band_cka(ys[v], ys[REF], regs, cka_regs) for v in vs if v != REF}
    # M3
    eh = {v: E[v][HF].sum(0) / E[v].sum(0).mean().clamp_min(1e-300) for v in vs}
    m3 = {}
    for v in vs:
        if v != REF:
            d = eh[v] - eh[REF]
            m3[v] = {k: dict(excess_mean=float(torch.relu(d[regs[k]]).mean()), excess_frac=float((d[regs[k]] > 0).double().mean()),
                             deficit_mean=float(torch.relu(-d[regs[k]]).mean()))
                     for k in ('gate_closed', 'gate_open', 'OmegaT', 'boundary', 'interior') if int(regs[k].sum()) >= 50}
    out['M3'] = m3
    # M6_phi: outer phi sectors + inner ridge selection; voxel coordinate 1 is phi
    ts = time.time()
    sector3 = LC.phi_sector(sup_idx[:, 1])
    out['M6_phi'] = {v: m6_decomp_nested(ys[v], ys[REF], E[v], sector3) for v in M6_VARS if v in ys}
    tm['M6'] = time.time() - ts
    del y, ys
    # M7_phi (TPV features)
    ts = time.time()
    out['M7_phi'] = m7_tpv(V, masks, device)
    tm['M7'] = time.time() - ts
    tm['total'] = time.time() - t0
    return out


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def load_frame(dump_dir, frame):
    V = {}
    for v in os.listdir(dump_dir):
        f = os.path.join(dump_dir, v, 'frame_%06d.npz' % frame)
        if v.startswith('_') or not os.path.isfile(f):
            continue
        V[v] = dict(np.load(f))
    masks = dict(np.load(os.path.join(dump_dir, '_masks', 'frame_%06d.npz' % frame)))
    return V, masks

