"""Graph-spectral decomposition on TPV plane grids, with the same bands and formulas as the voxel analysis
(11 bands V2_BAND_EDGES, HFER at lambda >= 0.6, per-band signal-to-noise ratio). Formulas are those of
spectral_core; nothing is re-derived here.

Two graphs:
  - full: the full-plane grid graph. Operator = the distiller's _GraphSpectralBase._apply_laplacian (4-neighbour mean,
    replicate padding = Neumann boundary), exactly diagonalised by the 2D DCT-II (lambda = 1/2 sum (1 - cos pi k / n)).
    Main path: spectral_core.exact_bins (exact).
  - support: the teacher-support induced sub-graph Omega. Operator = L_Omega x = m (deg x - sum_nbr x) / 4 with
    deg = _nbr_sum_2d(m) m (the formula inside the LF distiller's _lf_prob_saliency; _nbr_sum_2d is the distiller's
    own static method). Chebyshev path only.
Plane tensors: [1, C, H, W] (the LF distiller's input layout).
"""
import json
import sys

import numpy as np
import torch

from . import model_io
from . import spectral_core as B0

SD = model_io.load_spectral_module()
BAND_EDGES = list(B0.V2_BAND_EDGES)                       # 11 bands, as for the voxel analysis
HF_EDGE = float(B0.PAPER_HF_EDGE)                         # 0.60
HF_BANDS = [k for k, lo in enumerate(BAND_EDGES[:-1]) if lo >= HF_EDGE - 1e-12]   # bands with lower edge >= 0.6


class GridOp:
    """Full-plane grid operator: the distiller's own _apply_laplacian (replicate padding)."""

    def __init__(self, device, dtype=torch.float64):
        self.base = SD._GraphSpectralBase(alpha=0.25, pad_mode='replicate', use_checkpoint=False).to(device)
        self.dtype = dtype

    def __call__(self, x):
        return self.base._apply_laplacian(x)


class SupportOp:
    """Teacher-support induced sub-graph operator L_Omega (LF distiller formula). m: [1, 1, H, W] 0/1."""

    def __init__(self, m):
        self.m = m
        self.deg = SD.TPVLowFreqSpectralDistiller._nbr_sum_2d(m) * m

    def __call__(self, x):
        return self.m * (self.deg * x - SD.TPVLowFreqSpectralDistiller._nbr_sum_2d(x)) / 4.0


# ------------------------------------------------------------------ full-plane, exact (DCT)
def full_exact(s, t, fine_edges=None):
    """-> dict(tt, ss, st [C, nb_fine], dims [nb_fine], lsum [nb_fine], fine_edges), via spectral_core.exact_bins."""
    fe = B0.fine_edges_for() if fine_edges is None else fine_edges
    tt, ss, st, dims, lsum = B0.exact_bins(s, t, fe)
    return dict(tt=tt, ss=ss, st=st, dims=dims, lsum=lsum, fine_edges=fe)


def fine_to_bands(ex, band_edges=BAND_EDGES):
    """Fine bins -> 11 bands (spectral_core.fine_to_band). Returns (tt, ss, st [C, nb]), dims [nb], lbar [nb]."""
    Bm, _ = B0.fine_to_band(ex['fine_edges'], band_edges)
    tt, ss, st = (ex[k] @ Bm.T for k in ('tt', 'ss', 'st'))
    dims = Bm @ ex['dims']
    lbar = (Bm @ ex['lsum']) / np.maximum(dims, 1e-300)
    return tt, ss, st, dims, lbar


# ------------------------------------------------------------------ Chebyshev (any operator)
def cheb_bands(op, s, t, K=1024, band_edges=BAND_EDGES, dtype=torch.float64):
    """spectral_core.cheb_moments + band_quadratics: per-channel band energies (tt, ss, st) [C, nb]."""
    coeffs = B0.band_coeffs(band_edges, K)
    mu_tt, mu_st, mu_ss = B0.field_moments(op, s, t, K, dtype)
    return (B0.band_quadratics(mu_tt, coeffs), B0.band_quadratics(mu_ss, coeffs),
            B0.band_quadratics(mu_st, coeffs))


@torch.no_grad()
def hutchinson_dims(op, m, K=1024, n_probe=32, band_edges=BAND_EDGES, seed=0, dtype=torch.float64):
    """Per-band dimension of the support sub-graph, d_k ~ E[z^T h_k(L_Omega) z] with Rademacher z on Omega.
    Returns (d_mean [nb], d_se [nb], n_support). sum_k d_k ~ |Omega|."""
    coeffs = B0.band_coeffs(band_edges, K)
    g = torch.Generator(device='cpu').manual_seed(seed)
    mm = m.to(dtype)
    vals = []
    for _ in range(n_probe):
        z = (torch.randint(0, 2, mm.shape, generator=g).to(dtype) * 2 - 1).to(mm.device) * mm
        (mu,) = B0.cheb_moments(op, z, [z], K, dtype)
        vals.append(B0.band_quadratics(mu.numpy(), coeffs)[0])
    v = np.stack(vals)
    return v.mean(0), v.std(0, ddof=1) / np.sqrt(n_probe), int(mm.sum().item())


# ------------------------------------------------------------------ metrics
def snr_bands(tt, ss, st):
    """Channel-summed per-band snr_k (spectral_core.snr_from), gain, noise and correlation."""
    T, S, X = tt.sum(0), ss.sum(0), st.sum(0)
    snr, gain, noise, corr = B0.snr_from(T, S, X)
    return dict(T=T, S=S, X=X, snr=snr, gain=gain, noise=noise, corr=corr,
                spearman_band_vs_snr=B0.spearman(np.arange(len(snr), dtype=float)[np.isfinite(snr)],
                                                 snr[np.isfinite(snr)]))


def shares(E):
    tot = E.sum()
    return E / tot if tot > 0 else E * np.nan


# ------------------------------------------------------------------ self-test
def self_test(device='cpu'):
    res = {}
    torch.manual_seed(0)
    H, W, C = 24, 20, 3
    x = torch.randn(1, C, H, W, dtype=torch.float64)
    y = x + 0.3 * torch.randn(1, C, H, W, dtype=torch.float64)
    # 1 Parseval: fine-bin energies sum to ||x||^2
    ex = full_exact(y, x)
    res['parseval_rel_err'] = float(np.max(np.abs(ex['tt'].sum(1) - (x[0] ** 2).sum((1, 2)).numpy()) / (x[0] ** 2).sum((1, 2)).numpy()))
    # 2 exact vs Chebyshev (full plane)
    tt, ss, st, dims, _ = fine_to_bands(ex)
    ctt, css, cst = cheb_bands(GridOp(device), y, x, K=512)
    res['cheb_vs_exact_max_share_diff'] = float(np.nanmax(np.abs(shares(ctt.sum(0)) - shares(tt.sum(0)))))
    # 3 full support: L_Omega == grid operator
    m1 = torch.ones(1, 1, H, W, dtype=torch.float64)
    res['support_op_full_eq_grid_maxabs'] = float((SupportOp(m1)(x) - GridOp(device)(x)).abs().max())
    # 4 Hutchinson: sum d_k ~ |Omega|
    m = (torch.rand(1, 1, H, W, generator=torch.Generator().manual_seed(1)) > 0.4).to(torch.float64)
    d, dse, n = hutchinson_dims(SupportOp(m), m, K=512, n_probe=64)
    res['hutch_dims_sum_over_n'] = float(d.sum() / n)
    # 5 constant field -> all energy in band 0; the highest DCT-II basis vector -> all energy in the top band
    const = torch.ones(1, 1, H, W, dtype=torch.float64)
    e0 = fine_to_bands(full_exact(const, const))[0][0]
    res['const_share_band0'] = float(e0[0] / e0.sum())
    import scipy.fft as sfft
    delta = np.zeros((H, W)); delta[H - 1, W - 1] = 1.0
    top = torch.from_numpy(sfft.idctn(delta, type=2, norm='ortho')[None, None])
    ec = fine_to_bands(full_exact(top, top))[0][0]
    res['dct_top_mode_share_top_band'] = float(ec[-1] / ec.sum())
    ok = (res['parseval_rel_err'] < 1e-10 and res['cheb_vs_exact_max_share_diff'] < 0.05
          and res['support_op_full_eq_grid_maxabs'] < 1e-12 and abs(res['hutch_dims_sum_over_n'] - 1) < 0.05
          and res['const_share_band0'] > 0.999 and res['dct_top_mode_share_top_band'] > 0.999)
    res['ALL_PASS'] = bool(ok)
    return res


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--self-test':
        r = self_test()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r['ALL_PASS'] else 1)
