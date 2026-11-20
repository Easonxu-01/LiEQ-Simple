"""Frequency-band distance (FBD) of the voxel logit field: numerics (NumPy only, no GPU).

This module holds every number-defining step of the measurement that runs after the GPU pass:

1. The fixed fine lambda-grid on which spectral energy is accumulated (``FINE_EDGES``).
2. The Jackson-damped Chebyshev projector coefficients of every fine bin (``fine_bin_coeffs``), which turn
   the Chebyshev moments mu_n = <x, T_n(L~) x> computed on the GPU into fine-bin energies
   (``energies_from_moments``).
3. The FBD itself (``fbd_frame``) and its aggregation over frames (``aggregate``).

FBD definition (as reported in the paper)
-----------------------------------------
For one frame, let E_t[j] and E_s[j] be the teacher and student field energies in fine bin j, summed over
the output channels (class logits) first. Bands are cut on the teacher's cumulative energy share at the
quantiles 0.5 and 0.9 (low / mid / high; a fine bin that straddles a cut is split linearly by its teacher
energy), and the student uses the same cuts. With E_beta the share of band beta in each field's own total,

    FBD_beta = |E_beta(s) - E_beta(t)| / (E_beta(t) + 1e-8),
    FBD      = 0.2 * FBD_low + 0.3 * FBD_mid + 0.5 * FBD_high.

The reported value is the mean of the per-frame FBD over the evaluated frames (frames with zero teacher
energy, if any, are skipped). FBD = 0 when the student reproduces the teacher's band distribution.
"""
import warnings

import numpy as np

# ----------------------------------------------------------------------------- spectral grid
LMAX_MAP = 2.02        # lambda -> t = 2*lambda/LMAX_MAP - 1; 1% margin above the spectral bound 2
CHEB_ORDER = 1024      # Chebyshev order K of the fine-bin projectors

# The fine grid contains, besides 480 log-spaced edges, the boundaries of two coarser band sets used
# elsewhere in the spectral analysis. They do not enter the FBD definition, but they are edges of the
# grid on which the energies are accumulated, so they are kept to make every number identical.
BAND_EDGES_11 = [-0.05, 0.01, 0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, 1.50, 2.02]
BAND_EDGES_LOG16 = [-0.05] + [float('%.4g' % v) for v in np.geomspace(1e-4, 2.02, 17)[:-1]] + [2.02]

# ----------------------------------------------------------------------------- FBD constants
FBD_QUANTILES = (0.5, 0.9)      # teacher cumulative-energy cuts: [0, 0.5) / [0.5, 0.9) / [0.9, 1]
FBD_WEIGHTS = (0.2, 0.3, 0.5)   # low / mid / high
EPS_NUM = 1e-8


def fine_edges_for(extra_edges=()):
    """Fine lambda-bin edges: [-1, 1e-12] (the DC bin), 480 log-spaced edges up to LMAX_MAP, plus the
    extra edges that fall inside (0, LMAX_MAP)."""
    e = [-1.0, 1e-12] + list(np.geomspace(1e-6, LMAX_MAP, 481)[1:])
    e += [v for v in list(extra_edges) + BAND_EDGES_11 if 1e-12 < v < LMAX_MAP]
    return np.unique(np.array(e, dtype=np.float64))


FINE_EDGES = fine_edges_for(BAND_EDGES_11 + BAND_EDGES_LOG16)   # 508 edges -> 507 fine bins


# ----------------------------------------------------------------------------- Chebyshev-Jackson projectors
def jackson(m):
    """Jackson damping coefficients g_k, k = 0..m."""
    k = np.arange(m + 1, dtype=np.float64)
    a = np.pi / (m + 2)
    return ((m + 2 - k) * np.sin(a) * np.cos(k * a)
            + np.cos(a) * np.sin(k * a)) / ((m + 2) * np.sin(a))


def projector_cheb_coeffs(lo, hi, m, lmax_map):
    """Chebyshev coefficients of the indicator 1_[lo, hi)(lambda) in t = 2*lambda/lmax_map - 1."""
    def to_t(lam):
        return float(np.clip(2.0 * lam / lmax_map - 1.0, -1.0, 1.0))
    a = np.arccos(to_t(hi))
    b = np.arccos(to_t(lo))
    c = np.empty(m + 1, dtype=np.float64)
    c[0] = (b - a) / np.pi
    k = np.arange(1, m + 1, dtype=np.float64)
    c[1:] = 2.0 / np.pi * (np.sin(k * b) - np.sin(k * a)) / k
    return c


def band_coeffs(edges, K):
    """Jackson-damped Chebyshev coefficients of every band [edges[i], edges[i+1]): array [nb, K+1]."""
    g = jackson(K)
    return np.stack([g * projector_cheb_coeffs(a, b, K, LMAX_MAP) for a, b in zip(edges[:-1], edges[1:])])


_FINE_COEF_CACHE = {}


def fine_bin_coeffs(fine_edges, K):
    """Projector coefficients of the fine bins, [nb_fine, K+1]. The bins cover [-LMAX_MAP, LMAX_MAP), so
    the coefficients of all bins sum to [1, 0, ..., 0] and the bin energies add up to ||x||^2."""
    key = (len(fine_edges), float(np.sum(fine_edges)), int(K))
    if key not in _FINE_COEF_CACHE:
        _FINE_COEF_CACHE.clear()
        lo = np.maximum(fine_edges[:-1], -LMAX_MAP)
        hi = np.minimum(fine_edges[1:], LMAX_MAP)
        _FINE_COEF_CACHE[key] = band_coeffs(np.concatenate([lo[:1], hi]), K)
    return _FINE_COEF_CACHE[key]


def energies_from_moments(mu, fine_edges=FINE_EDGES, K=CHEB_ORDER):
    """Per-channel moments mu [C, K+1] (mu_n = <x_c, T_n(L~) x_c>) -> fine-bin energies [C, nb_fine].
    Small negative values produced by the finite expansion are clipped to 0."""
    return np.clip(mu @ fine_bin_coeffs(fine_edges, K).T, 0.0, None)


# ----------------------------------------------------------------------------- FBD
def _frac_below(Et, q):
    """Per fine bin, the fraction that lies below the teacher cumulative-energy quantile q (bins are cut
    linearly by their teacher energy). Et: [C, nb] -> [C, nb]."""
    tot = Et.sum(1, keepdims=True)
    tot = np.where(tot > 0, tot, 1.0)
    share = Et / tot
    c_prev = np.cumsum(share, 1) - share
    with np.errstate(divide='ignore', invalid='ignore'):
        f = np.where(share > 0, (q - c_prev) / share, np.where(c_prev < q, 1.0, 0.0))
    return np.clip(f, 0.0, 1.0)


def fbd_frame(teacher_energy, student_energy):
    """FBD of one frame from the per-channel fine-bin energies of the teacher and the student field,
    both [C, nb_fine] on the same fine grid. Channels are summed first. Returns a float (NaN if the
    teacher field has zero energy)."""
    Et = np.asarray(teacher_energy, dtype=np.float64).sum(0, keepdims=True)
    Es = np.asarray(student_energy, dtype=np.float64).sum(0, keepdims=True)
    tt_tot, ss_tot = Et.sum(1), Es.sum(1)
    f5, f9 = _frac_below(Et, FBD_QUANTILES[0]), _frac_below(Et, FBD_QUANTILES[1])
    wts = (f5, f9 - f5, 1.0 - f9)
    fb = []
    for w in wts:
        Tb = (Et * w).sum(1) / np.where(tt_tot > 0, tt_tot, 1)
        Sb = (Es * w).sum(1) / np.where(ss_tot > 0, ss_tot, 1)
        v = np.where(tt_tot > 0, np.abs(Sb - Tb) / (Tb + EPS_NUM), np.nan)
        fb.append(v)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmean(sum(wi * v for wi, v in zip(FBD_WEIGHTS, fb))))


def aggregate(values):
    """Reported FBD = mean of the finite per-frame values."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return dict(fbd=float(x.mean()) if x.size else float('nan'), n_frames=int(x.size))
