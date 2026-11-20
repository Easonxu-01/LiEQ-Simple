#!/usr/bin/env python
"""Computational core of the logits-level spectral analysis.

Operators are the distillers' own implementations; no formula is re-written:
  - 3D: spectral_core.SupportLaplacian = TPVHighFreqSpectralDistiller._support_geometry / _lap_op,
        L_Omega x = m * (deg * x - sum_6nbr x) / 6, support = any channel of the teacher field non-zero (the HF
        distiller's rule);
  - 2D: graph_spectral.SupportOp, built from TPVLowFreqSpectralDistiller._nbr_sum_2d:
        L_Omega x = m * (deg * x - sum_4nbr x) / 4, the operator inside the LF distiller's _lf_prob_saliency
        (checked bit for bit in the self-test).
Bands: the 11 bands (V2_BAND_EDGES), Chebyshev-Jackson band-pass filters h_k (spectral_core.band_coeffs, K = 1024,
lmax 2.02, Lt = (2/2.02) L - I). One recursion with 11 accumulators gives the band-limited signals h_k(L_Omega) x;
sum_k h_k == 1, so the band-limited signals sum to x.
T_n x is zero outside the support, so accumulation runs on the support nodes only ([nb, C, N]); no 11 dense grids.
"""
import math
import os
import sys

import numpy as np
import torch

from . import graph_spectral as G   # G.B0 = spectral_core; G.SD = Spectraldistiller of the checkout

B0 = G.B0
SD = G.SD
BAND_EDGES = list(B0.V2_BAND_EDGES)
NB = len(BAND_EDGES) - 1
HF_BANDS = [k for k, lo in enumerate(BAND_EDGES[:-1]) if lo >= 0.6 - 1e-12]      # 7..10
LOW_BANDS = [0, 1, 2]
MID_BANDS = [3, 4, 5, 6]
K_DEFAULT = 1024
RIDGE_REL = 1e-3          # M6 ridge parameter: lambda = RIDGE_REL * tr(X^T X) / C (relative, fixed)
FOLD_BLOCK3 = (8, 8, 4)   # M6 two-fold cross-fitting: voxels split in an 8x8x4 block checkerboard
FOLD_BLOCK2 = (16, 16)    # M7: plane cells split in a 16x16 block checkerboard

_COEF = {}


def band_coeffs_t(K, device, dtype):
    key = (K, str(device), str(dtype))
    if key not in _COEF:
        _COEF[key] = torch.from_numpy(B0.band_coeffs(BAND_EDGES, K)).to(device=device, dtype=dtype)   # [nb, K+1]
    return _COEF[key]


@torch.no_grad()
def cheb_band_signals(lap, x, sup_flat_idx, K=K_DEFAULT):
    """x: [1, C, *spatial] (zero outside the support); lap: linear operator (same shape in and out); sup_flat_idx: flat
    indices of the support nodes [N]. Returns y [nb, C, N]: h_k(L) x on the support nodes."""
    c = band_coeffs_t(K, x.device, x.dtype)
    sc = 2.0 / B0.LMAX_MAP
    C = x.shape[1]
    y = torch.zeros((NB, C, sup_flat_idx.numel()), dtype=x.dtype, device=x.device)

    def acc(n, t):
        ts = t.reshape(C, -1).index_select(1, sup_flat_idx)       # [C, N]
        y.add_(c[:, n].view(NB, 1, 1) * ts.unsqueeze(0))

    def Lt(v):
        return sc * lap(v) - v
    t_prev = x
    acc(0, t_prev)
    t_cur = Lt(x)
    acc(1, t_cur)
    for n in range(2, K + 1):
        t_next = 2.0 * Lt(t_cur) - t_prev
        t_prev, t_cur = t_cur, t_next
        acc(n, t_cur)
    return y


def support3d(device):
    return B0.SupportLaplacian(device)


def support2d(m):
    """m: [1,1,H,W] 0/1 -> the 2D L_Omega of the LF distiller."""
    return G.SupportOp(m)


# ------------------------------------------------------------------ statistics helpers
def preprocess_sparse(x, mode='standard'):
    """x: [C, N] (values on the support nodes). standard: per-channel mean removed, divided by the total RMS; robust:
    per-channel median removed, clipped at the 99.99% quantile of |x - med|, divided by MAD x 1.4826 of all non-zero
    values (RMS if too few) -- the formula of tpv_metrics.preprocess, applied on the support nodes only (zero outside)."""
    if mode == 'robust':
        med = x.median(dim=1, keepdim=True).values
        dev = (x - med).abs()
        kq = max(int(math.ceil(0.9999 * dev.shape[1])) - 1, 0)
        cap = dev.kthvalue(kq + 1, dim=1, keepdim=True).values
        y = torch.maximum(torch.minimum(x - med, cap), -cap)
        nz = y[y != 0]
        mad = (nz - nz.median()).abs().median() * 1.4826 if nz.numel() > 100 else y.new_tensor(0.0)
        if not float(mad) > 0:
            mad = torch.sqrt((y ** 2).mean())
        return y / mad.clamp_min(1e-30)
    y = x - x.mean(dim=1, keepdim=True)
    return y / torch.sqrt((y ** 2).mean()).clamp_min(1e-30)


def cka(Ys, Yt):
    """Linear CKA; Y: [n, C], centred per channel inside the function."""
    Ys = Ys - Ys.mean(0, keepdim=True)
    Yt = Yt - Yt.mean(0, keepdim=True)
    a = torch.linalg.norm(Ys.T @ Yt) ** 2
    b = torch.linalg.norm(Ys.T @ Ys) * torch.linalg.norm(Yt.T @ Yt)
    return float(a / b) if float(b) > 0 else float('nan')


def ridge_r2(X, Y, fold, rel=RIDGE_REL):
    """Two-fold cross-fitted multi-output ridge regression Y <- X. X: [n, Cx], Y: [n, Cy], fold: [n] bool (True = fold 1).
    Each fold is predicted by a fit on the other fold (centred on the training fold, lambda = rel * tr(X^T X) / Cx);
    R^2 = 1 - sum SSE / sum SST (SST around the test-fold mean). Returns (R^2, total SST)."""
    X = X.double(); Y = Y.double()
    sse = sst = 0.0
    for test in (fold, ~fold):
        tr = ~test
        if int(tr.sum()) < X.shape[1] + 2 or int(test.sum()) < 2:
            return float('nan'), 0.0
        mx, my = X[tr].mean(0, keepdim=True), Y[tr].mean(0, keepdim=True)
        Xa, Ya = X[tr] - mx, Y[tr] - my
        G_ = Xa.T @ Xa
        lam = rel * float(torch.trace(G_)) / X.shape[1]
        W = torch.linalg.solve(G_ + lam * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), Xa.T @ Ya)
        pred = (X[test] - mx) @ W + my
        sse += float(((Y[test] - pred) ** 2).sum())
        sst += float(((Y[test] - Y[test].mean(0, keepdim=True)) ** 2).sum())
    return (1.0 - sse / sst) if sst > 0 else float('nan'), sst


RIDGE_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)   # relative ridge grid (x tr(X^T X) / C)
PHI_SECTOR = 30                                              # phi axis has 180 cells (2 deg/cell) -> 30 cells = 60 deg sector


def phi_sector(phi_idx):
    """phi index -> 60-degree sector id 0..5 (spanning all rho and z). Outer two folds = even / odd sectors."""
    return torch.div(phi_idx, PHI_SECTOR, rounding_mode='floor')


def _ridge_stats(X, Y):
    mx, my = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xa, Ya = X - mx, Y - my
    G = Xa.T @ Xa
    return mx, my, G, Xa.T @ Ya, float(torch.trace(G)) / X.shape[1]


def ridge_r2_nested(X, Y, sector, grid=RIDGE_GRID):
    """Multi-output ridge regression Y <- X with two-fold outer cross-fitting and inner selection of the ridge parameter.
    Outer: even / odd phi sectors (contiguous 60-degree strips spanning all rho and z), which avoids the leak of low
    bands where a test cell is an interpolation of neighbouring training cells.
    Inner: leave-one-sector-out (3 folds) inside the training half selects lambda (grid x tr(X^T X) / C), separately per
    band, variant and direction.
    The reference model of R^2 predicts the training-half mean: R^2 = 1 - SSE / sum ||Y_test - ybar_train||^2. A
    regression without predictive power gives exactly 0 and is not pushed negative by different half means.
    Returns (R^2, reference sum of squares, relative lambda chosen in each fold)."""
    X = X.double(); Y = Y.double()
    outer = (sector % 2) == 0
    sse = sst = 0.0
    chosen = []
    for test in (outer, ~outer):
        tr = ~test
        if int(tr.sum()) < X.shape[1] + 2 or int(test.sum()) < 2:
            return float('nan'), 0.0, chosen
        secs = torch.unique(sector[tr]).tolist()
        inner = []
        for s in secs:
            ti = tr & (sector == s); fi = tr & (sector != s)
            if int(ti.sum()) < 2 or int(fi.sum()) < X.shape[1] + 2:
                continue
            inner.append((ti, _ridge_stats(X[fi], Y[fi])))
        best, best_lam = float('inf'), grid[-1]
        for lam in grid:
            s_in = 0.0
            for ti, (mx, my, G, B, tr_c) in inner:
                W = torch.linalg.solve(G + lam * tr_c * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), B)
                s_in += float(((Y[ti] - ((X[ti] - mx) @ W + my)) ** 2).sum())
            if s_in < best:
                best, best_lam = s_in, lam
        mx, my, G, B, tr_c = _ridge_stats(X[tr], Y[tr])
        W = torch.linalg.solve(G + best_lam * tr_c * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), B)
        sse += float(((Y[test] - ((X[test] - mx) @ W + my)) ** 2).sum())
        sst += float(((Y[test] - my) ** 2).sum())
        chosen.append(best_lam)
    return (1.0 - sse / sst) if sst > 0 else float('nan'), sst, chosen


def block_fold(coords, block):
    """coords: [N, D] integer grid coordinates -> checkerboard fold [N] bool."""
    b = torch.stack([torch.div(coords[:, d], block[d], rounding_mode='floor') for d in range(coords.shape[1])], 1)
    return (b.sum(1) % 2) == 0


def auroc(pos, neg):
    from scipy.stats import rankdata
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if pos.size < 10 or neg.size < 10:
        return float('nan')
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


# ------------------------------------------------------------------ self-test
def _dense_operator(lap, mask):
    """Expand the operator into a dense matrix on the support nodes of a small grid. mask: [1,1,*sp] 0/1 (float64)."""
    idx = mask.reshape(-1).nonzero().flatten()
    n = idx.numel()
    E = torch.zeros((n, 1) + tuple(mask.shape[2:]), dtype=torch.float64)
    E.view(n, -1)[torch.arange(n), idx] = 1.0
    out = torch.stack([lap(E[i:i + 1]).reshape(-1)[idx] for i in range(n)], 1)
    return out.numpy(), idx


def self_test(K=K_DEFAULT, verbose=True):
    res = {}
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    # (1) 3D: SupportLaplacian is the HF distiller's _lap_op; the recursion equals the polynomial on eigh exactly;
    #     difference between the band-pass filters and ideal projectors
    shape3 = (10, 8, 4)
    mask = rng.random(shape3) < 0.55
    m3 = torch.tensor(mask[None, None].astype(np.float64))
    lap3 = support3d('cpu')
    lap3.set_support(m3)
    hfd = SD.TPVHighFreqSpectralDistiller(gate_domain='support', hf_operator='support', use_checkpoint=False)
    mm, deg = hfd._support_geometry(m3)
    xr = torch.tensor(rng.standard_normal((1, 3) + shape3)) * m3
    res['3d_supportlaplacian_eq_distiller_lap_op_maxabs'] = float((lap3(xr) - hfd._lap_op(xr, mm, deg)).abs().max())
    Ls, idx = _dense_operator(lap3, m3)
    res['3d_operator_symmetric_maxabs'] = float(np.abs(Ls - Ls.T).max())
    lam, U = np.linalg.eigh(0.5 * (Ls + Ls.T))
    C = 3
    x = torch.zeros((1, C) + shape3, dtype=torch.float64)
    coef = rng.standard_normal((C, idx.numel())) / (1e-2 + np.clip(lam, 0, None))[None] ** 0.5
    x.view(1, C, -1)[0][:, idx] = torch.tensor(coef @ U.T)
    y = cheb_band_signals(lap3, x, idx, K).numpy()                 # [nb, C, N]
    xs = x.view(C, -1)[:, idx].numpy()
    cfs = B0.band_coeffs(BAND_EDGES, K)
    tl = 2.0 * np.clip(lam, 0, None) / B0.LMAX_MAP - 1.0
    Tm = np.polynomial.chebyshev.chebvander(tl, K)                # [n, K+1]
    hk = Tm @ cfs.T                                                # [n, nb]: h_k(lambda_i)
    proj = xs @ U                                                  # [C, n]
    y_poly = np.stack([(proj * hk[:, k][None]) @ U.T for k in range(NB)])
    res['3d_recursion_vs_eigh_poly_relerr'] = float(np.abs(y - y_poly).max() / np.abs(xs).max())
    res['3d_sum_k_reconstruction_relerr'] = float(np.abs(y.sum(0) - xs).max() / np.abs(xs).max())
    ideal = np.stack([((lam >= a) & (lam < b)).astype(float) for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])], 1)
    e_cheb = (y ** 2).sum(axis=(1, 2)); e_ideal = np.array([((proj * ideal[:, k][None]) ** 2).sum() for k in range(NB)])
    res['3d_band_share_vs_ideal_projector_maxabs'] = float(np.abs(e_cheb / e_cheb.sum() - e_ideal / e_ideal.sum()).max())
    res['3d_energy_sum_bands_over_total'] = float(e_cheb.sum() / (xs ** 2).sum())
    # (2) 2D: SupportOp is the operator of the LF distiller's _lf_prob_saliency (its diffusion chain recomputed bit for bit)
    H, W = 24, 18
    m2 = torch.tensor((rng.random((H, W)) < 0.6)[None, None].astype(np.float64))
    op2 = support2d(m2)
    lfd = SD.TPVLowFreqSpectralDistiller()
    deg2 = SD.TPVLowFreqSpectralDistiller._nbr_sum_2d(m2) * m2
    p = torch.softmax(torch.tensor(rng.standard_normal((1, 4, H, W))), 1) * m2
    sal = lfd._lf_prob_saliency(p, m2, deg2)
    g = p * m2; mine = []
    for _ in range(lfd.num_scales):
        g = g - lfd.alpha * op2(g)
        Lg = op2(g)
        mine.append((Lg * Lg).sum(dim=1, keepdim=True))
    res['2d_supportop_eq_lf_prob_saliency_maxabs'] = float(max((a - b).abs().max() for a, b in zip(sal, mine)))
    L2, idx2 = _dense_operator(op2, m2)
    lam2, U2 = np.linalg.eigh(0.5 * (L2 + L2.T))
    x2 = torch.zeros((1, 2, H, W), dtype=torch.float64)
    x2.view(1, 2, -1)[0][:, idx2] = torch.tensor(rng.standard_normal((2, idx2.numel())))
    y2 = cheb_band_signals(op2, x2, idx2, K).numpy()
    xs2 = x2.view(2, -1)[:, idx2].numpy()
    hk2 = np.polynomial.chebyshev.chebvander(2.0 * np.clip(lam2, 0, None) / B0.LMAX_MAP - 1.0, K) @ cfs.T
    pr2 = xs2 @ U2
    y2p = np.stack([(pr2 * hk2[:, k][None]) @ U2.T for k in range(NB)])
    res['2d_recursion_vs_eigh_poly_relerr'] = float(np.abs(y2 - y2p).max() / np.abs(xs2).max())
    res['2d_sum_k_reconstruction_relerr'] = float(np.abs(y2.sum(0) - xs2).max() / np.abs(xs2).max())
    res['2d_spectrum_range'] = [float(lam2.min()), float(lam2.max())]
    # (3) M6 sanity: synthetic "teacher" field (8 channels, low-frequency biased), student = A * teacher + sigma * white
    #     noise; the recovery must decrease monotonically with sigma, ~1 at sigma = 0, ~0 for an independent field
    shape = (40, 30, 8)
    msk = torch.tensor((rng.random(shape) < 0.6)[None, None].astype(np.float32))
    lapm = support3d('cpu'); lapm.set_support(msk)
    sidx = msk.reshape(-1).nonzero().flatten()
    N = sidx.numel()
    t = torch.randn((1, 8) + shape) * msk
    for _ in range(6):                                            # a few diffusion steps: low-frequency biased, energy in every band
        t = t - 0.25 * lapm(t)
    t = t + 0.05 * torch.randn_like(t) * msk
    A = torch.tensor(rng.standard_normal((8, 8)), dtype=torch.float32)
    coords = torch.stack(torch.meshgrid(*[torch.arange(s) for s in shape], indexing='ij'), -1).reshape(-1, 3)[sidx]
    fold = block_fold(coords, FOLD_BLOCK3)
    yt = cheb_band_signals(lapm, t, sidx, 256)
    tts = t.view(8, -1)[:, sidx]
    rec = {}
    for sig in (0.0, 0.1, 0.3, 1.0, 3.0):
        s = torch.einsum('ij,bjxyz->bixyz', A, t) + sig * float(tts.std()) * torch.randn_like(t) * msk
        ys = cheb_band_signals(lapm, s, sidx, 256)
        rec[sig] = [ridge_r2(ys[k].T, yt[k].T, fold)[0] for k in range(NB)]
    ind = torch.randn_like(t) * msk
    for _ in range(6):
        ind = ind - 0.25 * lapm(ind)
    yi = cheb_band_signals(lapm, ind, sidx, 256)
    rec['independent'] = [ridge_r2(yi[k].T, yt[k].T, fold)[0] for k in range(NB)]
    res['m6_recovery_by_noise'] = {str(k): [round(v, 4) for v in vals] for k, vals in rec.items()}
    sigs = (0.0, 0.1, 0.3, 1.0, 3.0)
    mono = all(all(rec[a][k] >= rec[b][k] - 1e-3 for a, b in zip(sigs, sigs[1:])) for k in range(NB))
    res['m6_monotone_in_noise'] = bool(mono)
    res['m6_sigma0_min_recovery'] = float(min(rec[0.0]))
    # Independent field: a negative cross-fitted R^2 means "worse than predicting the mean", i.e. no predictive power;
    # bands with few effective modes (lowest, highest) are noisy and biased negative. What must be excluded is a spurious
    # positive recovery, so the criterion is one-sided: max R^2 < 0.05.
    res['m6_independent_max_recovery'] = float(max(rec['independent']))
    res['m6_independent_min_recovery'] = float(min(rec['independent']))
    ok = (res['3d_supportlaplacian_eq_distiller_lap_op_maxabs'] == 0.0 and res['3d_operator_symmetric_maxabs'] < 1e-12
          and res['3d_recursion_vs_eigh_poly_relerr'] < 1e-9 and res['3d_sum_k_reconstruction_relerr'] < 1e-9
          and res['2d_supportop_eq_lf_prob_saliency_maxabs'] == 0.0 and res['2d_recursion_vs_eigh_poly_relerr'] < 1e-9
          and res['2d_sum_k_reconstruction_relerr'] < 1e-9 and mono and res['m6_sigma0_min_recovery'] > 0.99
          and res['m6_independent_max_recovery'] < 0.05)
    res['ALL_PASS'] = bool(ok)
    return res


if __name__ == '__main__':
    import json
    if len(sys.argv) > 1 and sys.argv[1] == '--self-test':
        r = self_test()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r['ALL_PASS'] else 1)
