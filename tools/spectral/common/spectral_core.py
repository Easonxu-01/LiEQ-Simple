"""Graph-spectral primitives shared by the spectral-measurement suite.

Operators are the distillers' own operators (imported from the checkout, see model_io):
  * grid operator: the Neumann grid Laplacian of _GraphSpectralBase (4/6-neighbour mean, replicate padding),
    exactly diagonalised by the orthonormal DCT-II, eigenvalues lambda_j = (1/D) sum_i (1 - cos(pi j_i / n_i)),
    spec(L) in [0, 2];
  * support operator (3D): TPVHighFreqSpectralDistiller._support_geometry / _lap_op,
    L_Omega x = m * (deg * x - sum_6nbr x) / 6, m = support indicator, deg = true neighbour count inside the support.

Bands: 11 bands on the lambda axis (V2_BAND_EDGES); "high frequency" means lambda >= 0.6 (PAPER_HF_EDGE).
Band filters: Jackson-damped Chebyshev expansions of band indicators, variable t = 2 lambda / LMAX_MAP - 1.
Exact path: per-channel energies of DCT-II coefficients binned on a fine lambda grid, aggregated to bands afterwards.
"""
import numpy as np
import torch

from . import model_io


# =============================================================================
# Support operator (3D voxel field)
# =============================================================================
class SupportLaplacian:
    """The operator of the HF distiller with hf_operator="support" (calls _support_geometry / _lap_op directly):
        L_Omega x = m * (deg * x - sum_6nbr x) / 6,
    spec in [0, 2) (the same LMAX_MAP = 2.02 Chebyshev mapping as the grid operator applies); equal to the grid
    operator when the support is the full grid. The support is set per frame by set_support(field): a voxel is in
    the support when any channel of the field is non-zero (the distiller's rule)."""

    def __init__(self, device):
        sm = model_io.load_spectral_module()
        self.d = sm.TPVHighFreqSpectralDistiller(gate_domain='support', hf_operator='support',
                                                 use_checkpoint=False).to(device)
        self.m = self.deg = None

    def set_support(self, field):
        self.m, self.deg = self.d._support_geometry(field)

    def __call__(self, x):
        return self.d._lap_op(x, self.m.to(x.dtype), self.deg.to(x.dtype))


# =============================================================================
# Chebyshev-Jackson band filters
# =============================================================================
LMAX_MAP = 2.02            # 1% safety margin above 2, so that |eig(Lt)| < 1
V2_BAND_EDGES = [-0.05, 0.01, 0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, 1.50, 2.02]


def jackson(m):
    """Jackson damping coefficients g_k, k = 0..m."""
    k = np.arange(m + 1, dtype=np.float64)
    a = np.pi / (m + 2)
    return ((m + 2 - k) * np.sin(a) * np.cos(k * a)
            + np.cos(a) * np.sin(k * a)) / ((m + 2) * np.sin(a))


def projector_cheb_coeffs(lo, hi, m, lmax_map):
    """Chebyshev coefficients of the band indicator 1_[lo,hi)(lambda)."""
    def to_t(lam):
        return float(np.clip(2.0 * lam / lmax_map - 1.0, -1.0, 1.0))
    a = np.arccos(to_t(hi))
    b = np.arccos(to_t(lo))
    c = np.empty(m + 1, dtype=np.float64)
    c[0] = (b - a) / np.pi
    k = np.arange(1, m + 1, dtype=np.float64)
    c[1:] = 2.0 / np.pi * (np.sin(k * b) - np.sin(k * a)) / k
    return c


def lam_to_t(lam):
    return 2.0 * np.asarray(lam, dtype=np.float64) / LMAX_MAP - 1.0


def band_coeffs(edges, K):
    """Jackson-damped Chebyshev coefficients per band [nb, K+1] (variable t = 2 lambda / lmax_map - 1)."""
    g = jackson(K)
    return np.stack([g * projector_cheb_coeffs(a, b, K, LMAX_MAP) for a, b in zip(edges[:-1], edges[1:])])


@torch.no_grad()
def cheb_moments(lap, x, ys, K, dtype):
    """mu_n[y] = <y, T_n(Lt) x> per channel, Lt = (2 / lmax_map) L - I, n = 0..K. x, y: [1, C, ...].
    Returns a list of [C, K+1] float64 tensors."""
    x = x.to(dtype)
    ys = [y.to(dtype) for y in ys]
    dims = tuple(range(2, x.dim()))
    sc = 2.0 / LMAX_MAP
    mus = [torch.zeros(x.shape[1], K + 1, dtype=torch.float64, device=x.device) for _ in ys]

    def rec(n, t):
        for m, y in zip(mus, ys):
            m[:, n] = (y * t).sum(dim=dims)[0].to(torch.float64)

    def Lt(v):
        return sc * lap(v) - v

    t_prev = x
    rec(0, t_prev)
    if K == 0:
        return [m.cpu() for m in mus]
    t_cur = Lt(x)
    rec(1, t_cur)
    for n in range(2, K + 1):
        t_next = 2.0 * Lt(t_cur) - t_prev
        t_prev, t_cur = t_cur, t_next
        rec(n, t_cur)
    return [m.cpu() for m in mus]


def field_moments(lap, s, t, K, dtype):
    """(tt, st, ss) per-channel moments, each [C, K+1]."""
    mu_tt, mu_st = cheb_moments(lap, t, [t, s], K, dtype)
    (mu_ss,) = cheb_moments(lap, s, [s], K, dtype)
    return mu_tt.numpy(), mu_st.numpy(), mu_ss.numpy()


def band_quadratics(mu, coeffs):
    """mu [..., K+1] -> [..., nb]."""
    return mu @ coeffs.T


# =============================================================================
# Exact spectrum of the Neumann grid Laplacian (separable DCT-II eigenbasis)
# =============================================================================
def grid_eigenvalues(shape):
    D = len(shape)
    comps = [(1.0 - np.cos(np.pi * np.arange(n) / n)) for n in shape]
    lam = np.zeros([1] * D)
    for i, c in enumerate(comps):
        sh = [1] * D
        sh[i] = len(c)
        lam = lam + c.reshape(sh)
    return (lam / D).reshape(-1)


def fine_edges_for(extra_edges=()):
    e = [-1.0, 1e-12] + list(np.geomspace(1e-6, LMAX_MAP, 481)[1:])
    e += [v for v in list(extra_edges) + V2_BAND_EDGES if 1e-12 < v < LMAX_MAP]
    return np.unique(np.array(e, dtype=np.float64))


_EIG_CACHE = {}


def _mode_bins(shape, fine_edges):
    key = (tuple(shape), len(fine_edges), float(fine_edges.sum()))
    if key not in _EIG_CACHE:
        lam = grid_eigenvalues(shape)
        idx = np.searchsorted(fine_edges, lam, side='right') - 1
        nb = len(fine_edges) - 1
        dims = np.bincount(idx, minlength=nb)[:nb].astype(np.float64)
        lsum = np.bincount(idx, weights=lam, minlength=nb)[:nb]
        _EIG_CACHE.clear()
        _EIG_CACHE[key] = (idx, dims, lsum)
    return _EIG_CACHE[key]


def exact_bins(s, t, fine_edges, workers=4):
    """s, t: [1, C, *spatial] tensors -> (tt, ss, st [C, nb], dims [nb], lamsum [nb]), all float64."""
    import scipy.fft as sfft
    sa = s.detach()[0].double().cpu().numpy()
    ta = t.detach()[0].double().cpu().numpy()
    axes = tuple(range(1, sa.ndim))
    sh = sfft.dctn(sa, type=2, norm='ortho', axes=axes, workers=workers)
    th = sfft.dctn(ta, type=2, norm='ortho', axes=axes, workers=workers)
    del sa, ta
    idx, dims, lsum = _mode_bins(sh.shape[1:], fine_edges)
    nb = len(fine_edges) - 1
    C = sh.shape[0]
    tt, ss, st = (np.zeros((C, nb)) for _ in range(3))
    for c in range(C):
        a, b = th[c].ravel(), sh[c].ravel()
        tt[c] = np.bincount(idx, weights=a * a, minlength=nb)[:nb]
        ss[c] = np.bincount(idx, weights=b * b, minlength=nb)[:nb]
        st[c] = np.bincount(idx, weights=a * b, minlength=nb)[:nb]
    return tt, ss, st, dims, lsum


def fine_to_band(fine_edges, band_edges):
    """Fine bin j (by its lower edge lo_j) -> band k, as a 0/1 aggregation matrix [nb_band, nb_fine].
    The DC bin (lo = -1) goes to the first band."""
    lo = fine_edges[:-1]
    B = np.zeros((len(band_edges) - 1, len(lo)))
    for k, (a, b) in enumerate(zip(band_edges[:-1], band_edges[1:])):
        sel = (lo >= a) & (lo < b)
        if k == 0:
            sel |= lo < a
        if k == len(band_edges) - 2:
            sel |= lo >= b
        B[k, sel] = 1.0
    hi = fine_edges[1:]
    snap = 0.0
    for e in band_edges[1:-1]:
        j = np.searchsorted(fine_edges, e)
        cand = fine_edges[max(j - 1, 0):j + 1]
        snap = max(snap, float(np.min(np.abs(cand - e))) if len(cand) else 0.0)
    return B, snap


# =============================================================================
# Summary statistics
# =============================================================================
def snr_from(T, S, X):
    denom = T * S - X * X
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(denom > 0, X * X / denom, np.inf)
        gain = X / T
        noise = S - X * X / T
        corr = X / np.sqrt(T * S)
    return snr, gain, noise, corr


def spearman(x, y):
    if len(x) < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


PAPER_HF_EDGE = 0.60          # fixed high-frequency threshold of HFER on the lambda axis (spec(L) in [0, 2])
FBD_QUANTILES = (0.5, 0.9)    # FBD bands = teacher cumulative energy [0, 0.5) / [0.5, 0.9) / [0.9, 1]
FBD_WEIGHTS = (0.2, 0.3, 0.5)
EPS_NUM = 1e-8
PROFILE_ALPHA = 0.25          # same as the HF distiller default
PROFILE_SCALES = tuple(range(1, 9))   # b = 1..8 (HF distiller num_scales = 8)


def _frac_below(Et, q):
    """Per fine bin, the fraction below the teacher cumulative-energy quantile q (linear split inside a bin).
    Et: [C, nb] -> [C, nb]."""
    tot = Et.sum(1, keepdims=True)
    tot = np.where(tot > 0, tot, 1.0)
    share = Et / tot
    c_prev = np.cumsum(share, 1) - share
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(share > 0, (q - c_prev) / share, np.where(c_prev < q, 1.0, 0.0))
    return np.clip(f, 0.0, 1.0)


def _profiles(E, lam, kind):
    """E: [C, nb] fine-bin energies; lam: [nb] mean eigenvalue per bin. kind='smooth' -> e_b = sum (1 - alpha lam)^(2b) E;
    'residual' -> r_b = sum (1 - (1 - alpha lam)^b)^2 E (the residual profile used by the HF distiller).
    Returns [C, B] (not normalised)."""
    r = 1.0 - PROFILE_ALPHA * lam
    cols = []
    for b in PROFILE_SCALES:
        w = r ** (2 * b) if kind == 'smooth' else (1.0 - r ** b) ** 2
        cols.append(E @ w)
    return np.stack(cols, 1)


def _scc_flat(Ps, Pt):
    """Ps, Pt: [C, B] profiles. Per channel q = P / sum P, SCC = Pearson(log q_s, log q_t), Flat = GM / AM.
    Returns (scc[C], Fs[C], Ft[C])."""
    def norm(P):
        s_ = P.sum(1, keepdims=True)
        return P / np.where(s_ > 0, s_, 1.0)
    qs, qt = norm(Ps), norm(Pt)
    ok = (qs > 0).all(1) & (qt > 0).all(1)
    scc = np.full(len(qs), np.nan); Fs = np.full(len(qs), np.nan); Ft = np.full(len(qs), np.nan)
    for c in np.where(ok)[0]:
        a, b = np.log(qs[c]), np.log(qt[c])
        if a.std() > 0 and b.std() > 0:
            scc[c] = float(np.corrcoef(a, b)[0, 1])
        Fs[c] = float(np.exp(a.mean()) / qs[c].mean())
        Ft[c] = float(np.exp(b.mean()) / qt[c].mean())
    return scc, Fs, Ft


def paper_metrics(ex, fine_edges, fdims, flsum, n_boot=1000, seed=0):
    """Voxel-field metrics of the paper, computed per frame and per channel, averaged over channels, then over frames
    (frame bootstrap gives the CI). The suite uses the HFER entry:
      HFER  = [E_hi / E_tot]_s / [E_hi / E_tot]_t with E_hi over lambda >= 0.6 (1 is ideal).
    Also returned: FBD (teacher-quantile bands 0.5 / 0.9, weights 0.2 / 0.3 / 0.5), SCC and flatness gap of the
    smooth (e_b = ||D^b x||^2) and residual (r_b = ||x - D^b x||^2) diffusion profiles."""
    lo = fine_edges[:-1]
    hi_sel = lo >= PAPER_HF_EDGE - 1e-12
    lam = flsum / np.maximum(fdims, 1e-300)
    per_frame = {k: [] for k in ('hfer', 'fbd_low', 'fbd_mid', 'fbd_high', 'fbd',
                                 'scc_smooth', 'flatgap_smooth', 'scc_residual', 'flatgap_residual',
                                 'scc_residual_chsum', 'flatgap_residual_chsum')}
    for Et, Es, _ in ex:
        tt_tot, ss_tot = Et.sum(1), Es.sum(1)
        tt_hi, ss_hi = Et[:, hi_sel].sum(1), Es[:, hi_sel].sum(1)
        ok = (tt_tot > 0) & (ss_tot > 0) & (tt_hi > 0)
        per_frame['hfer'].append(np.nanmean(np.where(ok, (ss_hi / np.where(ss_tot > 0, ss_tot, 1)) /
                                                     np.where(tt_hi > 0, tt_hi / np.where(tt_tot > 0, tt_tot, 1), 1), np.nan)))
        f5, f9 = _frac_below(Et, FBD_QUANTILES[0]), _frac_below(Et, FBD_QUANTILES[1])
        wts = (f5, f9 - f5, 1.0 - f9)
        fb = []
        for name, w in zip(('fbd_low', 'fbd_mid', 'fbd_high'), wts):
            Tb = (Et * w).sum(1) / np.where(tt_tot > 0, tt_tot, 1)
            Sb = (Es * w).sum(1) / np.where(ss_tot > 0, ss_tot, 1)
            v = np.where(tt_tot > 0, np.abs(Sb - Tb) / (Tb + EPS_NUM), np.nan)
            per_frame[name].append(np.nanmean(v)); fb.append(v)
        per_frame['fbd'].append(np.nanmean(sum(wi * v for wi, v in zip(FBD_WEIGHTS, fb))))
        for kind in ('smooth', 'residual'):
            scc, Fs, Ft = _scc_flat(_profiles(Es, lam, kind), _profiles(Et, lam, kind))
            per_frame['scc_' + kind].append(np.nanmean(scc))
            per_frame['flatgap_' + kind].append(np.nanmean(Fs - Ft))
        scc, Fs, Ft = _scc_flat(_profiles(Es.sum(0, keepdims=True), lam, 'residual'),
                                _profiles(Et.sum(0, keepdims=True), lam, 'residual'))
        per_frame['scc_residual_chsum'].append(float(scc[0])); per_frame['flatgap_residual_chsum'].append(float(Fs[0] - Ft[0]))
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in per_frame.items():
        a = np.asarray(v, dtype=float); a = a[np.isfinite(a)]
        if not a.size:
            out[k] = dict(mean=float('nan'), ci95=[float('nan')] * 2, n_frames=0); continue
        bs = [rng.choice(a, a.size).mean() for _ in range(n_boot)]
        out[k] = dict(mean=float(a.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                      n_frames=int(a.size))
    # SCC is read on the smooth per-channel profile; the flatness gap on the channel-summed residual profile
    # (the profile the HF hinge constrains). The two profiles differ on purpose.
    out['paper_SCC'] = out['scc_smooth']
    out['paper_flatness_gap'] = out['flatgap_residual_chsum']
    out['_definition'] = ('HFER lambda>=0.6 ratio; FBD teacher-quantile bands (0.5,0.9), shares, weights (0.2,0.3,0.5), '
                          'eps 1e-8; SCC/flat on profile q, alpha=0.25, b=1..8; paper_SCC=smooth (per-channel); '
                          'paper_flatness_gap=residual_chsum')
    return out
