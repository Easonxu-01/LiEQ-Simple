#!/usr/bin/env python
"""TPV-plane spectral analysis: per-frame summary (called once per frame by run_dump.py --stage tpv).
Input: one frame of the TPV dump, V[variant] = the arrays of <dump>/<variant>/frame_XXXXXX.npz and
masks = <dump>/_masks/frame_XXXXXX.npz. Output: one json-friendly dict per frame with the quantities the paper uses:
  - xy.b_dct_standard: plane metrics of neck s0 (level b: LF distiller input, aggregator sampling level) against the
    reference T_union (tpv_metrics.plane_metrics: 11 lambda bands of the tight DCT frame on both axes, statistics
    domain Omega_T, standard scale = per-channel mean removed + RMS; per-band CKA per region, excess per region,
    spectral shares, attribution, occupancy residue).
  - <plane>.d_probC_sbr_mbr: split / merge boundary rates (plane 4-neighbourhood, inside Omega_T) of the argmax of the
    class-probability planes (level d: voxel class probabilities on the Omega_T support voxels, averaged along the
    collapsed axis, common coarse groups).
  - Regions: four observation classes (O_ST, O_T, O_S, E); semantics (teacher probability and GT, common coarsening);
    on xy the LF region R and the final LF probability gate open / closed; three rho distance bins.
  - rho crop: before any spectral transform the near-sensor ring (the first RHO_CROP rows of the (b) grid on the planes
    with a rho axis, xy rows and zx columns; 16 cells = 3.41 m) is dropped; masks and regions are built at full size
    first, then cropped. RHO_CROP = 0 disables the crop.
Variant names (keys of the output): T_union (reference), T_union2, T_raw, student, baseline; see run_dump.py.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))   # tools/spectral, for the shared helpers in common/
from common import tpv_metrics as M          # noqa: E402
from common import graph_spectral as G       # noqa: E402
from common import plane_cells as L          # noqa: E402

PLANES = ('xy', 'yz', 'zx')
# Near-sensor ring dropped before any spectral transform on the planes that have a rho axis (xy: rows, zx: columns).
# Counted on the (b) grid (240 rho cells over 51.2 m, so 16 cells = 3.41 m); the tokenizer grid (480 cells) drops twice
# as many. The paper uses 16; 0 disables the crop.
RHO_CROP = 16


def set_rho_crop_rows(k):
    """Set the number of near-sensor rho rows dropped on the (b) grid (paper: 16; 0 = no crop)."""
    global RHO_CROP
    RHO_CROP = int(k)


def rho_crop(a, which, scale=1):
    k = RHO_CROP * scale
    if k <= 0 or which == 'yz' or a is None:
        return a
    return a[..., k:, :] if which == 'xy' else a[..., k:]


RHO_AXIS = {'xy': 0, 'zx': 1}                                                 # rho axis (distance bins)
RHO_BINS = {'near': (0, 70), 'mid': (70, 141), 'far': (141, 10 ** 6)}        # row index (0.213 m/cell): < 15 m, 15-30 m, > 30 m
REF = 'T_union'
CKA_REGIONS = ('OmegaT', 'O_ST', 'O_T', 'O_S', 'semT_boundary', 'semT_interior', 'gate_open', 'gate_closed',
               'rho_near', 'rho_mid', 'rho_far')
FAST = torch.float32
FEATURE_PAIRS = ('student', 'baseline', 'T_raw', 'T_union2')   # variants compared with the reference
MODELS = ('student', 'baseline')                                  # model roles
LF_CFG = dict(moment_norm='joint', suppress_ref='teacher_floor', block_size=(30, 30), lf_gate=True, gate_kappa=0.5,
              lambda_suppress=0.25, profile_metric='l1', lambda_profile=8.0, block_pad='fit', gate_domain='support',
              gate_source='prob', profile_domain='open', plane_gate=(True, False, False), suppress_target='zero',
              suppress_ref_mode='log_tail', ref_floor_rel=1e-4, ungated_domain='support')

# Common coarsening of the 8 classes. Edges (a, b) join classes that one raw class maps to under two different label
# maps. All models are evaluated on the current label map, so there are no edges and the coarsening is the identity.
MAP_DIFF_EDGES = ()


def coarse_groups(n=8, edges=MAP_DIFF_EDGES):
    """Merge the 8 classes joined by the edges into connected components (both maps share the class ids 0..7)."""
    parent = list(range(n))

    def f(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in edges:
        parent[f(a)] = f(b)
    roots = sorted({f(i) for i in range(n)})
    gid = {r: j for j, r in enumerate(roots)}
    return np.array([gid[f(i)] for i in range(n)], np.int64)


CG = coarse_groups()           # class -> coarse group
NG = int(CG.max() + 1)


# ------------------------------------------------------------------ projections and construction


def voxel_to_planes(idx, val, shape=(240, 180, 16), reduce='mean'):
    """Values on support voxels (a,b,c) [N,K] -> three planes [K,H,W] (mean over the support voxels along the collapsed
    axis), plus the number of support voxels per cell. xy=(a,b), yz=(b,c), zx=(c,a) (plane_inputs_from_field convention)."""
    a, b, c = (idx[:, 0].astype(np.int64), idx[:, 1].astype(np.int64), idx[:, 2].astype(np.int64))
    out, cnt = {}, {}
    K = val.shape[1]
    for name, (i, j, H, W) in {'xy': (a, b, shape[0], shape[1]), 'yz': (b, c, shape[1], shape[2]),
                               'zx': (c, a, shape[2], shape[0])}.items():
        lin = i * W + j
        n = np.bincount(lin, minlength=H * W).astype(np.float64)
        acc = np.stack([np.bincount(lin, weights=val[:, k].astype(np.float64), minlength=H * W) for k in range(K)])
        with np.errstate(invalid='ignore', divide='ignore'):
            out[name] = np.where(n > 0, acc / np.maximum(n, 1), 0.0).reshape(K, H, W)
        cnt[name] = n.reshape(H, W)
    return out, cnt


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def coarse_prob(p):
    """[N,8] probabilities -> [N,NG] coarse-group probabilities."""
    out = np.zeros((p.shape[0], NG))
    for k in range(p.shape[1]):
        out[:, CG[k]] += p[:, k]
    return out


def gt_plane_classes(grid_ind, labels, H, W, which):
    """Point GT (current map, 0..7) -> majority coarse group per plane cell; -1 for cells without points.
    grid_ind: fine-grid float indices [N,3]."""
    g = np.floor(grid_ind).astype(np.int64) // 2
    i, j = {'xy': (g[:, 0], g[:, 1]), 'yz': (g[:, 1], g[:, 2]), 'zx': (g[:, 2], g[:, 0])}[which]
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W) & (labels >= 0)
    lab = CG[np.clip(labels[ok], 0, 7)]
    lin = i[ok] * W + j[ok]
    tab = np.zeros((H * W, NG), np.int64)
    np.add.at(tab, (lin, lab), 1)
    cls = tab.argmax(1)
    cls[tab.sum(1) == 0] = -1
    return cls.reshape(H, W)


def plane_majority_classes(grid_ind, labels, H, W, which, ncls=8):
    """Point labels (0..ncls-1, not coarsened) -> majority class per plane cell; -1 for cells without points."""
    g = np.floor(grid_ind).astype(np.int64) // 2
    i, j = {'xy': (g[:, 0], g[:, 1]), 'yz': (g[:, 1], g[:, 2]), 'zx': (g[:, 2], g[:, 0])}[which]
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W) & (labels >= 0) & (labels < ncls)
    lin = i[ok] * W + j[ok]
    tab = np.zeros((H * W, ncls), np.int64)
    np.add.at(tab, (lin, labels[ok].astype(np.int64)), 1)
    cls = tab.argmax(1)
    cls[tab.sum(1) == 0] = -1
    return cls.reshape(H, W)


def interior_radius(cls, r=2):
    """Labelled cells with no class boundary within Chebyshev radius r (any class change is a boundary, class 0
    included; conservative)."""
    from scipy.ndimage import binary_dilation
    occ = cls >= 0
    p = np.pad(cls, 1, constant_values=-1)
    bd = np.zeros_like(occ)
    for di, dj in ((0, 1), (2, 1), (1, 0), (1, 2)):
        nb = p[di:di + cls.shape[0], dj:dj + cls.shape[1]]
        bd |= occ & (nb >= 0) & (nb != cls)
    near = binary_dilation(bd, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    return occ & ~near


def sem_regions(cls_plane, dom):
    """Boundary / interior / deep interior of the common coarse classes (plane_cells.cell_categories; no ignored class
    after coarsening)."""
    c = np.where(dom, cls_plane, -1)
    cat = L.cell_categories(c, ignore_classes=())
    return dict(boundary=cat['sem_boundary'] & dom, interior=cat['interior'] & dom, deep_interior=cat['deep_interior'] & dom)


def lf_gate_closed(prob_plane, m, device):
    """Final LF probability gate on xy: v_b = e_b / (e_b + kappa * mean_R e_b + eps) * R, e_b = the distiller's
    _lf_prob_saliency; gate closed = mean of v over the suppression scales b3-b8 <= 0.5 inside R.
    Returns (gate_open, gate_closed, v_mean, R)."""
    d = G.SD.TPVLowFreqSpectralDistiller(**LF_CFG).to(device).double()
    p = torch.from_numpy(prob_plane[None]).to(device=device, dtype=torch.float64)
    mt = torch.from_numpy(m[None, None].astype(np.float64)).to(device)
    deg = d._nbr_sum_2d(mt) * mt
    R = mt * (deg > 0).to(mt.dtype)
    sal = d._lf_prob_saliency(p, mt, deg)
    vs = []
    for b in sorted(d._suppress_scales):
        e = sal[b]
        tau = d.gate_kappa * (e * R).sum() / R.sum().clamp_min(1.0) + d.eps
        vs.append((e / (e + tau) * R)[0, 0])
    vm = torch.stack(vs).mean(0).cpu().numpy()
    Rn = R[0, 0].cpu().numpy().astype(bool)
    return Rn & (vm > 0.5), Rn & (vm <= 0.5), vm, Rn


def rho_bins(which, shape):
    if which not in RHO_AXIS:
        return {}
    H, W = shape
    idx = np.arange(H)[:, None].repeat(W, 1) if RHO_AXIS[which] == 0 else np.arange(W)[None, :].repeat(H, 0)
    return {'rho_' + k: (idx >= lo) & (idx < hi) for k, (lo, hi) in RHO_BINS.items()}


# ------------------------------------------------------------------ boundary rates
def sbr_mbr(ls, lt, dom):
    """Adjacent cell pairs (plane 4-neighbourhood, both ends in dom): SBR = fraction of teacher same-class pairs that
    the student splits; MBR = fraction of teacher different-class pairs that the student merges."""
    pairs = []
    for sl_a, sl_b in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                       ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        ok = dom[sl_a] & dom[sl_b]
        pairs.append((lt[sl_a][ok] == lt[sl_b][ok], ls[sl_a][ok] == ls[sl_b][ok]))
    tsame = np.concatenate([p[0] for p in pairs]); ssame = np.concatenate([p[1] for p in pairs])
    sbr = float((~ssame[tsame]).mean()) if tsame.any() else float('nan')
    mbr = float(ssame[~tsame].mean()) if (~tsame).any() else float('nan')
    return dict(SBR=sbr, MBR=mbr, n_same=int(tsame.sum()), n_diff=int((~tsame).sum()))


# ------------------------------------------------------------------ main function
def load_frame(dump_dir, frame):
    V = {}
    for v in os.listdir(dump_dir):
        f = os.path.join(dump_dir, v, 'frame_%06d.npz' % frame)
        if v.startswith('_') or not os.path.isfile(f):
            continue
        V[v] = dict(np.load(f))
    masks = dict(np.load(os.path.join(dump_dir, '_masks', 'frame_%06d.npz' % frame)))
    return V, masks


def summarize_frame(V, masks, device='cuda', mode='eval'):
    """Per-frame quantities of the paper (TPV planes):
      xy.b_dct_standard        plane metrics of neck s0 (level b) on Omega_T: per-band CKA per region, excess per region,
                               shares, attribution (tpv_metrics.plane_metrics, standard scale, DCT on both axes)
      <plane>.d_probC_sbr_mbr  split / merge boundary rates of the class-probability planes (level d), every plane
      <plane>.n_cells, <plane>.rho_crop_rows, xy.p2_domain_cells, xy.gate_closed_frac_of_R   domain bookkeeping
    mode ('full' for raw frames, 'eval' otherwise) is recorded; both modes compute the same quantities."""
    t0 = time.time()
    out = dict(variants=sorted(V), coarse_groups=CG.tolist(), timing={}, mode=mode)
    ref = V[REF]
    pairs = [v for v in FEATURE_PAIRS if v in V]
    # ---- (d) class fields on the support voxels -> planes (coarse-group probabilities)
    sup_idx = ref['sup_idx']
    dfieldsC = {}
    fields_vox = {REF: ref['sup_logit'].astype(np.float64)}
    for v in MODELS:
        if v in V and 'cs_logit' in V[v]:
            fields_vox[v] = V[v]['cs_logit'].astype(np.float64)
    for v, lg in fields_vox.items():
        p = softmax(lg)
        pl, _ = voxel_to_planes(sup_idx, coarse_prob(p))
        dfieldsC[v] = pl
    for which in PLANES:
        rec = {}
        om = masks['OmegaT_' + which].astype(bool)
        osm = masks['OmegaS_' + which].astype(bool)
        H, W = om.shape
        regions = dict(OmegaT=om, full=np.ones_like(om), O_ST=masks['ST_' + which].astype(bool),
                       O_T=masks['T_minus_S_' + which].astype(bool), O_S=masks['S_minus_T_' + which].astype(bool),
                       E=~(om | osm))
        # semantics: teacher probability (coarsened) and GT (points of T_union, coarsened)
        pp = masks['plane_prob_' + which]
        pc = np.zeros((NG, H, W))
        for k in range(pp.shape[0]):
            pc[CG[k]] += pp[k]
        cls_t = np.where(om, pc.argmax(0), -1)
        sem_t = sem_regions(cls_t, om)
        cls_g = gt_plane_classes(ref['grid_ind'], ref['gt_new'].astype(np.int64), H, W, which)
        sem_g = sem_regions(np.where(om, cls_g, -1), om)
        for k, mm in sem_t.items():
            regions['semT_' + k] = mm
        for k, mm in sem_g.items():
            regions['semG_' + k] = mm
        if which == 'xy':
            gopen, gclosed, vmean, Rn = lf_gate_closed(pp.astype(np.float64), om.astype(np.float64), device)
            # main domain of the gate statistics: gate closed AND interior of the gt_old classes (no gt_old boundary within
            # radius 2); gt_old equals the current-map GT, see run_dump.py
            cls_old = plane_majority_classes(ref['grid_ind'], ref['gt_old'].astype(np.int64), H, W, which)
            old_int = interior_radius(np.where(om, cls_old, -1), 2)
            regions.update(R=Rn, gate_open=gopen, gate_closed=gclosed, gate_closed_oldint=gclosed & old_int)
            rec['gate_closed_frac_of_R'] = float(gclosed.sum() / max(Rn.sum(), 1))
            rec['p2_domain_cells'] = dict(gate_closed=int(gclosed.sum()), gate_closed_oldint=int((gclosed & old_int).sum()),
                                          old_interior_in_OmegaT=int((old_int & om).sum()), R=int(Rn.sum()))
        for k, mm in rho_bins(which, (H, W)).items():
            regions[k] = mm & om
        if RHO_CROP > 0 and which != 'yz':
            regions = {k: rho_crop(v, which) for k, v in regions.items()}
            sem_t = {k: rho_crop(v, which) for k, v in sem_t.items()}
            om = rho_crop(om, which)
            H, W = om.shape
            rec['rho_crop_rows'] = RHO_CROP
            if which == 'xy':
                rec['gate_closed_frac_of_R'] = float(regions['gate_closed'].sum() / max(regions['R'].sum(), 1))
                rec['p2_domain_cells'] = dict(gate_closed=int(regions['gate_closed'].sum()),
                                              gate_closed_oldint=int(regions['gate_closed_oldint'].sum()),
                                              old_interior_in_OmegaT=int((rho_crop(old_int, which) & om).sum()), R=int(regions['R'].sum()))
        rec['n_cells'] = {k: int(v.sum()) for k, v in regions.items()}
        counts = {v: rho_crop(V[v]['cnt_' + which], which) for v in V if ('cnt_' + which) in V[v]}
        occ = {v: (c > 0).astype(np.float64) for v, c in counts.items()}
        # ---- level b (neck s0), xy plane: standard scale, DCT on both axes
        if which == 'xy':
            flds = {v: rho_crop(V[v]['neck_%s_s0' % which].astype(np.float64), which) for v in V if ('neck_%s_s0' % which) in V[v]}
            rec['b_dct_standard'] = M.plane_metrics(flds, REF, [v for v in pairs if v in flds], regions,
                                                    sem_t, counts, occ,
                                                    device, dom_key='OmegaT', periodic=(False, False), scale_mode='standard',
                                                    attribution=True, dtype=FAST, cka_regions=CKA_REGIONS)
        # ---- level d (coarse-group probabilities on the support): split / merge boundary rates
        dC = {v: rho_crop(x, which)[:, :H, :W] for v, x in ((v, dfieldsC[v][which]) for v in dfieldsC)}
        prs = [v for v in MODELS if v in dC]
        if prs:
            lt = np.where(om, dC[REF].argmax(0), -1)
            rec['d_probC_sbr_mbr'] = {v: sbr_mbr(np.where(om, dC[v].argmax(0), -1), lt, om) for v in prs}
        out[which] = rec
        out['timing'][which] = time.time() - t0
    out['timing']['total'] = time.time() - t0
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
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o

