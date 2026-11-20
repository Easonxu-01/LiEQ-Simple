"""Reductions over evaluation frames for the reported spectral quantities.

Each function takes the per-frame arrays of ONE dump directory (spectral_io.DumpArrays) and returns
plain dicts of numpy arrays. Frames are on axis 0, bands on axis 1. A, B denote the user-chosen
model pair (A = the model under study, B = the baseline it is compared with).

Fig. 3   band-wise CKA relative to the ceiling (planes and voxel logits)
Fig. 4   real / pseudo energy per band relative to the ceiling, and the paired A - B change
Planes   spurious / missed boundary rates per TPV plane
App. D   HF excess where the teacher is smooth / the HF gate is closed; HF pseudo-energy shares;
         paired comparison of each model with the teacher-on-sparse control
"""
import numpy as np

from spectral_io import HF_BANDS, NB, PLANES

LAYERS = ('planes', 'voxel_logits')


def _q(x, axis=0):
    lo, md, hi = np.nanpercentile(x, [25, 50, 75], axis=axis)
    return lo, md, hi


# ============================================================================= Fig. 3: band-wise CKA
def band_cka(X, layer, A, B):
    """Per frame and band: CKA(model, reference) / CKA(ceiling, reference). Curves are the median over frames,
    ribbons the interquartile range. Gap closed (strip) = 100 * median_f((A - B) / (ceiling - B))."""
    r = X.roles
    g = X.cka(layer)
    ceil = g[r['ceiling']]
    curves = {}
    for v in [r['sparse']] + r['models']:
        ratio = g[v] / ceil
        q25, _, q75 = _q(ratio)
        curves[v] = dict(per_frame=ratio, median=np.nanmedian(ratio, 0), q25=q25, q75=q75)
    gap_pf = (g[A] - g[B]) / (ceil - g[B])
    gap = dict(pair=[A, B], per_frame=gap_pf, median_pct=100 * np.nanmedian(gap_pf, 0))
    return dict(curves=curves, gap_closed=gap, n_frames=int(ceil.shape[0]))


# ============================================================================= Fig. 4: real / pseudo energy
def energy_frames(X, layer):
    """Per frame and band:
         real   = clip(recovery, 0, 1)   (R^2 of the reference band signal predicted from the model; NaN -> 0)
         pseudo = pseudo_share / (useful_share + pseudo_share) of the ceiling
       and the band units
         unit_real   = median_f real of the ceiling
         unit_pseudo = median_f useful_share / (useful_share + pseudo_share) of the ceiling."""
    r = X.roles
    R = X.rp[layer]
    c = r['ceiling']
    total = R[c]['useful_share'] + R[c]['pseudo_share']
    F = {}
    for v in [c, r['sparse']] + r['models']:
        F[v] = dict(real=np.clip(np.nan_to_num(R[v]['recovery'], nan=0.0), 0.0, 1.0), pseudo=R[v]['pseudo_share'] / total)
    unit_real = np.median(F[c]['real'], 0)
    unit_pseudo = np.median(R[c]['useful_share'] / total, 0)
    return F, unit_real, unit_pseudo


def energy(X, layer, A, B):
    """Band medians of real and pseudo energy in units of the ceiling (ceiling real = 1), the stacked total for
    the models, and the paired change median_f(A - B) in the same units."""
    r = X.roles
    F, ur, up = energy_frames(X, layer)
    rows = {}
    for v in F:
        real, pseudo = np.median(F[v]['real'], 0) / ur, np.median(F[v]['pseudo'], 0) / up
        rows[v] = dict(real=real, pseudo=pseudo, per_frame_real=F[v]['real'], per_frame_pseudo=F[v]['pseudo'])
        if v in r['models']:
            rows[v]['total'] = real + pseudo
    d_real, d_pseudo = F[A]['real'] - F[B]['real'], F[A]['pseudo'] - F[B]['pseudo']
    paired = dict(pair=[A, B], real=np.median(d_real, 0) / ur, pseudo=np.median(d_pseudo, 0) / up,
                  per_frame_real=d_real, per_frame_pseudo=d_pseudo)
    return dict(roles=rows, paired=paired, unit_real=ur, unit_pseudo=up, n_frames=len(X.frames))


def unsupported_bands(E, sparse, A, B, thr=0.05):
    """Bands where neither the teacher fed the sparse scan nor models A and B keep real energy (< thr, in
    ceiling units); drawn hatched in Fig. 4."""
    rr = E['roles']
    return [k for k in range(NB) if all(rr[v]['real'][k] < thr for v in (sparse, A, B))]


# ============================================================================= planes figure: boundary rates
def boundary_rates(X, A, B):
    """Spurious-boundary rate (SBR) and missed-boundary rate (MBR) per plane, in percent, over frames where
    both models are finite: quartiles of each model and the number of frames where A < B."""
    out = {}
    for m in ('sbr', 'mbr'):
        for pl in PLANES:
            a, b = X.planes[m][pl][A] * 100, X.planes[m][pl][B] * 100
            ok = np.isfinite(a) & np.isfinite(b)
            out[f'{m}_{pl}'] = {A: dict(per_frame=a, q25_median_q75=np.percentile(a[ok], [25, 50, 75])),
                                B: dict(per_frame=b, q25_median_q75=np.percentile(b[ok], [25, 50, 75])),
                                'frames_A_lower': int((a[ok] < b[ok]).sum()), 'n': int(ok.sum())}
    return out


# ============================================================================= Appendix D
def spurious_hf(X, layer):
    """HF energy above the reference where the teacher is smooth (planes: gate-closed cells inside the teacher's
    class interiors) / where the HF gate is closed (voxel logits), per frame, divided by the median of the ceiling
    (the teacher's own re-truncation fluctuation = 1). Quartiles over finite frames."""
    r = X.roles
    g = X.excess_gc(layer)
    unit = float(np.nanmedian(g[r['ceiling']]))
    if not unit > 0:
        raise SystemExit(f'{layer}: ceiling gate-closed excess has a non-positive median ({unit})')
    out = {}
    for v in [r['ceiling']] + r['models']:
        x = g[v] / unit
        m = np.isfinite(x)
        lo, md, hi = np.percentile(x[m], [25, 50, 75])
        out[v] = dict(per_frame=x, q25=float(lo), median=float(md), q75=float(hi), n_finite=int(m.sum()),
                      n_nonfinite=int((~m).sum()))
    return dict(roles=out, unit=unit)


def pseudo_hf_shares(X, layer):
    """Share of pseudo energy in the HF bands (lambda >= 0.60), median over frames, per role."""
    r = X.roles
    R = X.rp[layer]
    out = {}
    for v in r['models'] + [r['sparse'], r['ceiling']]:
        x = np.asarray(R[v]['pseudo_share_hf'], float)
        out[v] = dict(per_frame=x, median=float(np.median(x)))
    return dict(roles=out, n=len(X.frames))


def paired_vs_sparse(X, layer):
    """Per model M, frame-paired comparison of the HF pseudo share with the teacher-on-sparse control T_s and
    the ceiling (re-truncation floor) T_c: medians, paired median and mean of M - T_s, ratio of medians, median
    per-frame ratio, and the frame counts M > T_s, M < T_s, M > T_c."""
    r = X.roles
    R = X.rp[layer]
    t = np.asarray(R[r['sparse']]['pseudo_share_hf'], float)
    u = np.asarray(R[r['ceiling']]['pseudo_share_hf'], float)
    out = {}
    for v in r['models']:
        s = np.asarray(R[v]['pseudo_share_hf'], float)
        diff = s - t
        out[v] = dict(n=int(s.size), median_model=float(np.median(s)), median_sparse=float(np.median(t)),
                      median_ceiling=float(np.median(u)),
                      paired_median_model_minus_sparse=float(np.median(diff)), paired_mean_model_minus_sparse=float(diff.mean()),
                      ratio_of_medians=float(np.median(s) / np.median(t)), median_per_frame_ratio=float(np.median(s / t)),
                      frames_above_sparse=int((diff > 0).sum()), frames_below_sparse=int((s < t).sum()),
                      frames_above_ceiling=int((s > u).sum()))
    return out


__all__ = ['band_cka', 'energy_frames', 'energy', 'unsupported_bands', 'boundary_rates', 'spurious_hf',
           'pseudo_hf_shares', 'paired_vs_sparse', 'LAYERS', 'HF_BANDS']
