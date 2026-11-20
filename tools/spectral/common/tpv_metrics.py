#!/usr/bin/env python
"""TPV-plane spectral metrics.
- Transform: cylindrical grid (xy = rho x phi, yz = phi x z, zx = z x rho). Periodic phi -> orthonormal real Fourier basis
  (constant, cos/sin pairs, Nyquist row for even length); non-periodic rho, z -> orthonormal DCT-II. Both are matrix
  products (float64); the DCT equals scipy.fft.dctn(type=2, norm='ortho') element-wise (self-test).
  Note: the distiller's grid operator uses replicate padding (non-periodic) on phi; the decomposition with the same
  operator as the voxel analysis is in graph_spectral.py (DCT on all axes).
- Bands: main definition = 11 lambda bands (V2_BAND_EDGES, lambda = 1/2 sum (1 - cos pi k / N), high frequency =
  lambda >= 0.6); control definition = 6 octaves of |f| (cycles per cell).
  Tight frame: square-root raised-cosine transitions, sum_k w_k^2 == 1, hence sum_k ||B_k x||^2 = ||x||^2 (Parseval,
  checked in the self-test).
- Local band energy e_k(u) = sum_c (B_k x)_c(u)^2; regional quantities are the mean of e_k over the region (density).
- Pre-processing: per channel, subtract the mean inside the statistics domain D, then divide by the total RMS inside
  D; spectra describe shape only, the scale is reported separately.
Plane arrays: [C, H, W] (numpy or torch); masks [H, W] bool.
"""
import math

import numpy as np
import torch
import scipy.fft as sfft

LAMBDA_EDGES = [-0.05, 0.01, 0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, 1.50, 2.02]   # = b0 V2_BAND_EDGES
HF_EDGE = 0.60
OCT_EDGES = [0.0, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1.0]   # ascending: B6 ... B1 (|f| in cycles/cell; B1 includes the corners above 1/2)
OCT_NAMES = ['B6', 'B5', 'B4', 'B3', 'B2', 'B1']
TRANS_FRAC = 0.25        # transition half-width = 0.25 x the width of the narrower neighbouring band
MIN_CELLS = 50           # minimum number of cells of a region; below it the regional quantity is NaN


def dct_matrix(n):
    """Orthonormal DCT-II matrix D (y = D x equals scipy.fft.dct(x, type=2, norm='ortho'))."""
    return sfft.dct(np.eye(n), type=2, norm='ortho', axis=0)


def dct_axis(n):
    """DCT-II axis: matrix, per-row frequency f = k/(2n) (cycles/cell), axial Laplacian eigenvalue 1 - cos(pi k / n)
    (Neumann path graph)."""
    k = np.arange(n)
    return dct_matrix(n), k / (2.0 * n), 1.0 - np.cos(np.pi * k / n)


def fourier_axis(n):
    """Orthonormal real Fourier basis of a periodic axis: rows are the constant, (cos_k, sin_k) for k = 1..floor((n-1)/2),
    and the Nyquist row for even n. Returns the matrix, per-row frequency f = k/n and axial eigenvalue
    1 - cos(2 pi k / n) (cycle graph, same normalisation as the Neumann path graph)."""
    t = np.arange(n)
    rows, ks = [np.full(n, 1.0 / np.sqrt(n))], [0]
    for k in range(1, (n - 1) // 2 + 1):
        rows.append(np.sqrt(2.0 / n) * np.cos(2 * np.pi * k * t / n)); ks.append(k)
        rows.append(np.sqrt(2.0 / n) * np.sin(2 * np.pi * k * t / n)); ks.append(k)
    if n % 2 == 0:
        rows.append(np.cos(np.pi * t) / np.sqrt(n)); ks.append(n // 2)
    ks = np.asarray(ks, float)
    return np.stack(rows), ks / n, 1.0 - np.cos(2 * np.pi * ks / n)


PLANE_PERIODIC = {'xy': (False, True), 'yz': (True, False), 'zx': (False, False)}   # whether the (H axis, W axis) is phi


def _tight_windows(val, edges):
    """Tight-frame windows w_k on the scalar frequency coordinate val (sum w_k^2 = 1). edges ascending, outer edges open."""
    nb = len(edges) - 1
    ws = []
    inner = edges[1:-1]
    half = []
    for j, e in enumerate(inner):
        left = edges[j + 1] - edges[j]
        right = edges[j + 2] - edges[j + 1]
        half.append(TRANS_FRAC * min(left, right))

    def rise(x, e, d):          # square-root raised cosine from 0 to 1 (x < e-d: 0; x > e+d: 1)
        t = np.clip((x - (e - d)) / (2 * d), 0.0, 1.0)
        return np.sin(0.5 * np.pi * t)
    for k in range(nb):
        w = np.ones_like(val)
        if k > 0:
            w = w * rise(val, inner[k - 1], half[k - 1])
        if k < nb - 1:
            fall = np.cos(0.5 * np.pi * np.clip((val - (inner[k] - half[k])) / (2 * half[k]), 0.0, 1.0))
            w = w * fall
        ws.append(w)
    return np.stack(ws)


def _cast_basis(pb, dtype):
    key = ('cast', pb.H, pb.W, str(pb.device), pb.periodic, str(dtype))
    if key not in PlaneBasis._cache:
        import copy
        q = copy.copy(pb)
        q.DH, q.DW = pb.DH.to(dtype), pb.DW.to(dtype)
        q.w_lam, q.w_oct = pb.w_lam.to(dtype), pb.w_oct.to(dtype)
        PlaneBasis._cache[key] = q
    return PlaneBasis._cache[key]


class PlaneBasis:
    """DCT matrices, frequency grids and band windows of one plane shape (H, W) (cached)."""
    _cache = {}

    @classmethod
    def get(cls, H, W, device, periodic=(False, False)):
        key = (H, W, str(device), tuple(periodic))
        if key not in cls._cache:
            cls._cache[key] = cls(H, W, device, periodic)
        return cls._cache[key]

    def __init__(self, H, W, device, periodic=(False, False)):
        self.H, self.W, self.device, self.periodic = H, W, device, tuple(periodic)
        MH, fh, lh = (fourier_axis if periodic[0] else dct_axis)(H)
        MW, fw, lw = (fourier_axis if periodic[1] else dct_axis)(W)
        self.DH = torch.from_numpy(MH).to(device)
        self.DW = torch.from_numpy(MW).to(device)
        lam = 0.5 * (lh[:, None] + lw[None, :])
        frad = np.sqrt(fh[:, None] ** 2 + fw[None, :] ** 2)
        self.lam = lam
        self.frad = frad
        self.w_lam = torch.from_numpy(_tight_windows(lam, LAMBDA_EDGES)).to(device)      # [11,H,W]
        self.w_oct = torch.from_numpy(_tight_windows(frad, OCT_EDGES)).to(device)        # [6,H,W]
        self.hf_lam = [k for k, lo in enumerate(LAMBDA_EDGES[:-1]) if lo >= HF_EDGE - 1e-12]
        # frequency indices for the axial marginal spectra
        self.fh, self.fw = fh, fw

    def fwd(self, x):          # [C,H,W] -> DCT coefficients
        return self.DH @ x @ self.DW.T

    def inv(self, X):
        return self.DH.T @ X @ self.DW


def to_t(x, device, dtype=torch.float64):   # noqa: E302
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x.to(device=device, dtype=dtype)


ROBUST_Q = 0.9999


def preprocess(x, dom, mode='standard'):
    """Per channel, subtract the mean inside dom, then divide by the total RMS inside dom. Returns (x', scale).
    dom=None means the full plane.
    mode='robust': per channel, clip at the 99.99% quantile of |x - median| inside dom and subtract the median, then
    divide by the median absolute deviation of all channels (MAD x 1.4826); this keeps single outlier activations from
    dominating the spectral shape."""
    if mode == 'robust':
        d = torch.ones(x.shape[1:], dtype=torch.bool, device=x.device) if dom is None else dom.bool()
        v = x[:, d]                                            # [C, n]
        med = v.median(dim=1, keepdim=True).values
        dev = (v - med).abs()
        k = max(int(math.ceil(ROBUST_Q * dev.shape[1])) - 1, 0)
        cap = dev.kthvalue(k + 1, dim=1, keepdim=True).values if dev.shape[1] > 0 else dev.new_ones(x.shape[0], 1)
        y = x - med[:, :, None]
        y = torch.maximum(torch.minimum(y, cap[:, :, None]), -cap[:, :, None])
        yv = y[:, d]
        nz = yv[yv != 0]
        mad = (nz - nz.median()).abs().median() * 1.4826 if nz.numel() > 100 else yv.new_tensor(0.0)
        if not float(mad) > 0:
            mad = torch.sqrt((yv ** 2).mean())
        return y / mad.clamp_min(1e-30), float(mad)
    if dom is None:
        mu = x.mean(dim=(1, 2), keepdim=True)
        y = x - mu
        rms = torch.sqrt((y ** 2).mean())
    else:
        d = dom.to(x.dtype)
        n = d.sum().clamp_min(1.0)
        mu = (x * d).sum(dim=(1, 2), keepdim=True) / n
        y = (x - mu)
        rms = torch.sqrt(((y * d) ** 2).sum() / (n * x.shape[0]))
    return y / rms.clamp_min(1e-30), float(rms)


def global_bands(Xhat, pb):
    """Full-plane band energies (coefficient domain, weighted by squared windows; sum_k = ||x||^2).
    -> dict(lam [11], oct [6], axis_h [H], axis_w [W], tot)"""
    P = (Xhat.double() ** 2).sum(0)                          # [H,W] channel-summed power (float64 accumulation)
    lam = (pb.w_lam.double() ** 2 * P).sum(dim=(1, 2))
    octb = (pb.w_oct.double() ** 2 * P).sum(dim=(1, 2))
    return dict(lam=lam.cpu().numpy(), oct=octb.cpu().numpy(), axis_h=P.sum(1).cpu().numpy(),
                axis_w=P.sum(0).cpu().numpy(), tot=float(P.sum()))


def band_signal(Xhat, pb, w):
    """B_k x = D_H^T (Xhat * w_k) D_W, [C,H,W]."""
    return pb.inv(Xhat * w)


def spectral_slope(P2d, pb, lo=1 / 32, hi=1 / 4):
    """Log-log slope -beta of the radial power spectrum on |f| in [lo, hi] (log-binned means, linear fit)."""
    f = pb.frad.ravel()
    p = P2d.ravel()
    sel = (f >= lo) & (f <= hi) & (p > 0)
    if sel.sum() < 10:
        return float('nan')
    bins = np.geomspace(lo, hi, 13)
    idx = np.digitize(f[sel], bins)
    xs, ys = [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() >= 3:
            xs.append(np.log(np.sqrt(bins[b - 1] * bins[b])))
            ys.append(np.log(p[sel][m].mean()))
    if len(xs) < 4:
        return float('nan')
    return float(-np.polyfit(xs, ys, 1)[0])


def flatness(E):
    E = np.asarray(E, dtype=float)
    E = E[E > 0]
    return float(np.exp(np.mean(np.log(E))) / np.mean(E)) if E.size else float('nan')


def cka(Ys, Yt):
    """Linear CKA, Y: [n, C] (already centred per channel inside the region)."""
    a = torch.linalg.norm(Ys.T @ Yt) ** 2
    b = torch.linalg.norm(Ys.T @ Ys) * torch.linalg.norm(Yt.T @ Yt)
    return float(a / b) if b > 0 else float('nan')


def _center_region(B, reg):
    Y = B[:, reg].T                                           # [n, C]
    return Y - Y.mean(0, keepdim=True)


def auroc(pos, neg):
    """Mann-Whitney AUROC: P(score_pos > score_neg), ties count 0.5."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if pos.size < 10 or neg.size < 10:
        return float('nan')
    allv = np.concatenate([pos, neg])
    ranks = _rankdata(allv)
    rp = ranks[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def _rankdata(a):
    from scipy.stats import rankdata
    return rankdata(a, method='average')


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float('nan')
    rx, ry = _rankdata(x[ok]), _rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def n_empty_neighbors(occ):
    """Number of empty cells among the 4 neighbours (outside the grid not counted). occ: [H,W] bool."""
    o = occ.astype(np.int32)
    p = np.pad(1 - o, 1, constant_values=0)
    inside = np.pad(np.ones_like(o), 1, constant_values=0)
    empty = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]
    return empty.astype(np.int32)


def plane_metrics(fields, ref, pairs, regions, sem, counts, occ_fields, device, dom_key='OmegaT',
                  aligned=('student',), attribution=True, periodic=(False, False), scale_mode='standard',
                  dtype=torch.float64, cka_regions=None):
    """All metrics of one plane at one level.
    fields: {variant: [C,H,W]} (same level, same plane); ref: reference variant name ('T_union'); pairs: variants
    compared with ref; regions: {name: [H,W] bool} (at least dom_key = Omega_T); sem: {'boundary', 'interior',
    'deep_interior': bool} (common coarsening); counts: {variant: [H,W] int} (the variant's own input point counts);
    occ_fields: {variant: [H,W] float} (the variant's input occupancy, for the occupancy residue).
    Returns a json-friendly dict."""
    H, W = next(iter(fields.values())).shape[-2:]
    pb = PlaneBasis.get(H, W, device, periodic)
    if dtype != torch.float64:
        pb = _cast_basis(pb, dtype)
    regt = {k: torch.from_numpy(np.asarray(v, bool)).to(device) for k, v in regions.items()}
    dom = regt.get(dom_key)
    out = dict(shape=[int(H), int(W)], n_cells={k: int(v.sum()) for k, v in regions.items()})
    X, Xh, scale = {}, {}, {}
    for v, x in fields.items():
        xt = to_t(x, device)
        y, rms = preprocess(xt, dom, scale_mode)
        y = y.to(dtype)
        X[v] = y
        Xh[v] = pb.fwd(y)
        scale[v] = rms
    out['scale_rms_in_dom'] = scale
    # ---- full-plane spectral shape; the hard-masked Omega_T spectrum is recorded only (*_hard).
    # The hard mask X * Omega_T creates jumps at the jagged boundary of Omega_T; this edge high frequency is the same for
    # all models and swamps the differences between models. The main masked_* definition is therefore the local energy
    # attribution below.
    shape = {}
    for v in fields:
        g = global_bands(Xh[v], pb)
        P2 = (Xh[v] ** 2).sum(0).cpu().numpy()
        gm = global_bands(pb.fwd(X[v] * dom.to(X[v].dtype)), pb) if dom is not None else None
        shape[v] = dict(lam_share=(g['lam'] / g['tot']).tolist(), oct_share=(g['oct'] / g['tot']).tolist(),
                        H=float(g['lam'][pb.hf_lam].sum() / g['tot']), beta=spectral_slope(P2, pb),
                        flatness=flatness(g['lam']),
                        aniso_hf_h_over_w=float(P2[pb.fh[:, None].repeat(W, 1) >= 1 / 8].sum() /
                                                max(P2[pb.fw[None, :].repeat(H, 0) >= 1 / 8].sum(), 1e-30)),
                        axis_h=(g['axis_h'] / g['tot']).tolist(), axis_w=(g['axis_w'] / g['tot']).tolist())
        if gm is not None:
            shape[v]['masked_lam_share_hard'] = (gm['lam'] / max(gm['tot'], 1e-30)).tolist()
            shape[v]['masked_oct_share_hard'] = (gm['oct'] / max(gm['tot'], 1e-30)).tolist()
            shape[v]['masked_H_hard'] = float(gm['lam'][pb.hf_lam].sum() / max(gm['tot'], 1e-30))
    # Secondary definition: per model, average the per-channel high-frequency shares with equal weights (invariant to
    # channel permutation), then take the ratio.
    whf = (pb.w_lam[pb.hf_lam].double() ** 2).sum(0)
    for v in fields:
        for tag, Xc in (('H_chmean', Xh[v]), ('masked_H_chmean_hard', pb.fwd(X[v] * dom.to(X[v].dtype)) if dom is not None else None)):
            if Xc is None:
                continue
            Pc = Xc.double() ** 2                                           # [C,H,W]
            tot_c = Pc.sum(dim=(1, 2))
            hf_c = (Pc * whf).sum(dim=(1, 2))
            okc = tot_c > 0
            shape[v][tag] = float((hf_c[okc] / tot_c[okc]).mean()) if bool(okc.any()) else float('nan')
    # ---- per band: local energy -> regional density; CKA; eps_k (region masks stacked into one matrix, one product
    # gives all regions; moved to the CPU at the end). The same pass accumulates the local energy on the statistics
    # domain dom (channel-summed and per channel) for the local energy attribution.
    nb = pb.w_lam.shape[0]
    loc_lam = {v: torch.zeros(nb, dtype=torch.float64, device=device) for v in fields}
    loc_ch_tot = {v: torch.zeros(X[v].shape[0], dtype=torch.float64, device=device) for v in fields}
    loc_ch_hf = {v: torch.zeros(X[v].shape[0], dtype=torch.float64, device=device) for v in fields}
    rnames = list(regt)
    RM = torch.stack([regt[r].reshape(-1).double() for r in rnames])        # [nR, HW]
    rcnt = RM.sum(1)
    rvalid = (rcnt >= MIN_CELLS)
    dens_t = {v: torch.full((len(rnames), nb), float('nan'), dtype=torch.float64, device=device) for v in fields}
    cka_t = {v: torch.full((len(rnames), nb), float('nan'), dtype=torch.float64, device=device) for v in pairs}
    eps_t = {v: torch.full((len(rnames), nb), float('nan'), dtype=torch.float64, device=device) for v in pairs}
    e_hf = {v: torch.zeros(H, W, dtype=torch.float64, device=device) for v in fields}
    e_tot = {v: torch.zeros(H, W, dtype=torch.float64, device=device) for v in fields}
    cka_reg = set(regt) if cka_regions is None else set(cka_regions) & set(regt)
    cka_idx = [(i, r) for i, r in enumerate(rnames) if r in cka_reg and int(rcnt[i]) >= MIN_CELLS]
    for k in range(nb):
        Bk = {v: band_signal(Xh[v], pb, pb.w_lam[k]) for v in fields}
        Yt_cache = {}
        for v in fields:
            ek = (Bk[v] ** 2).sum(0).double()
            e_tot[v] += ek
            if k in pb.hf_lam:
                e_hf[v] += ek
            dens_t[v][:, k] = torch.where(rvalid, RM @ ek.reshape(-1) / rcnt.clamp_min(1), dens_t[v][:, k])
            if dom is not None:
                loc_lam[v][k] = ek[dom].sum()
                pc = (Bk[v][:, dom].double() ** 2).sum(1)                   # [C]
                loc_ch_tot[v] += pc
                if k in pb.hf_lam:
                    loc_ch_hf[v] += pc
        for v in pairs:
            for i, r in cka_idx:
                m = regt[r]
                if r not in Yt_cache:                        # teacher side once per band and region (float32 Gram, precision ~1e-5, far below the tolerance 0.02)
                    yt = _center_region(Bk[ref], m).float()
                    Yt_cache[r] = (yt, torch.linalg.norm(yt.T @ yt))
                yt, gtt = Yt_cache[r]
                ys = _center_region(Bk[v], m).float()
                num = torch.linalg.norm(ys.T @ yt).double() ** 2
                den = torch.linalg.norm(ys.T @ ys).double() * gtt.double()
                cka_t[v][i, k] = num / den
                if v in aligned and Bk[v].shape[0] == Bk[ref].shape[0]:
                    d_ = ((Bk[v] - Bk[ref])[:, m].double() ** 2).sum()
                    tn = (Bk[ref][:, m].double() ** 2).sum()
                    eps_t[v][i, k] = d_ / tn
        del Bk
    # ---- local energy attribution on the statistics domain: full-plane band signals B_k x, sum
    # e_k(u) = sum_c (B_k x)_c(u)^2 over the cells of dom, then normalise over bands. With dom = the full plane this equals
    # the full-plane shares bit for bit (self-test).
    if dom is not None:
        for v in fields:
            lam_d = loc_lam[v].cpu().numpy()
            oct_d = np.array([float((band_signal(Xh[v], pb, pb.w_oct[k]) ** 2).sum(0).double()[dom].sum())
                              for k in range(pb.w_oct.shape[0])])
            tl, to = float(lam_d.sum()), float(oct_d.sum())
            shape[v]['masked_lam_share'] = (lam_d / max(tl, 1e-30)).tolist()
            shape[v]['masked_oct_share'] = (oct_d / max(to, 1e-30)).tolist()
            shape[v]['masked_H'] = float(lam_d[pb.hf_lam].sum() / max(tl, 1e-30))
            tc, hc = loc_ch_tot[v], loc_ch_hf[v]
            okc = tc > 0
            shape[v]['masked_H_chmean'] = float((hc[okc] / tc[okc]).mean()) if bool(okc.any()) else float('nan')
    out['shape'] = shape
    # ---- distribution level: HFER, r_k (main = local energy attribution on dom; _fullplane = full plane;
    # _hard = hard-masked spectrum, recorded only)
    def _ratio(a, b):
        return a / b if b > 0 else float('nan')

    def _logr(a, b):
        return np.log(np.maximum(a, 1e-300) / np.maximum(b, 1e-300)).tolist()
    dist = {}
    for v in pairs:
        s, t = shape[v], shape[ref]
        e = dict(HFER_fullplane=_ratio(s['H'], t['H']), HFER_chmean_fullplane=_ratio(s['H_chmean'], t['H_chmean']),
                 r_lam_fullplane=_logr(s['lam_share'], t['lam_share']), r_oct_fullplane=_logr(s['oct_share'], t['oct_share']))
        for suf in ('', '_hard'):
            if 'masked_H' + suf in s:
                e['HFER' + suf] = _ratio(s['masked_H' + suf], t['masked_H' + suf])
                e['HFER_chmean' + suf] = _ratio(s['masked_H_chmean' + suf], t['masked_H_chmean' + suf])
                e['r_lam' + suf] = _logr(s['masked_lam_share' + suf], t['masked_lam_share' + suf])
                e['r_oct' + suf] = _logr(s['masked_oct_share' + suf], t['masked_oct_share' + suf])
        dist[v] = e
    out['dist'] = dist
    dens = {v: {r: dv[i] for i, r in enumerate(rnames)} for v, dv in ((v, dens_t[v].cpu().numpy()) for v in fields)}
    ckas = {v: {r: cv[i] for i, r in enumerate(rnames)} for v, cv in ((v, cka_t[v].cpu().numpy()) for v in pairs)}
    eps_k = {v: {r: ev[i] for i, r in enumerate(rnames)} for v, ev in ((v, eps_t[v].cpu().numpy()) for v in pairs)}
    out['density'] = {v: {r: d.tolist() for r, d in dv.items()} for v, dv in dens.items()}
    out['region_H'] = {v: {r: (float(np.nansum(d[pb.hf_lam]) / np.nansum(d)) if np.isfinite(d).all() else float('nan'))
                           for r, d in dv.items()} for v, dv in dens.items()}
    out['cka'] = {v: {r: c.tolist() for r, c in cv.items()} for v, cv in ckas.items()}
    out['eps'] = {v: {r: c.tolist() for r, c in cv.items()} for v, cv in eps_k.items() if v in aligned}
    # ---- excess (both sides), normalised by each model's mean total energy on Omega_T
    exc = {}
    if dom is not None and int(dom.sum()) >= MIN_CELLS:
        norm = {v: e_tot[v][dom].mean() for v in fields}
        eh = {v: (e_hf[v] / norm[v].clamp_min(1e-30)).reshape(-1) for v in fields}
        for v in pairs:
            dlt = eh[v] - eh[ref]
            ex_m = (RM @ torch.relu(dlt)) / rcnt.clamp_min(1)
            ex_f = (RM @ (dlt > 0).double()) / rcnt.clamp_min(1)
            de_m = (RM @ torch.relu(-dlt)) / rcnt.clamp_min(1)
            a_, b_, c_ = ex_m.cpu().numpy(), ex_f.cpu().numpy(), de_m.cpu().numpy()
            exc[v] = {r: dict(excess_mean=float(a_[i]), excess_frac=float(b_[i]), deficit_mean=float(c_[i]))
                      for i, r in enumerate(rnames) if int(rcnt[i]) >= MIN_CELLS}
    out['excess'] = exc
    # ---- attribution (inside Omega_T) and occupancy residue
    if attribution and dom is not None:
        att = {}
        domn = regions[dom_key]
        for v in fields:
            ehv = e_hf[v].cpu().numpy()
            a = {}
            if sem is not None:
                bd, it = np.asarray(sem['boundary'], bool) & domn, np.asarray(sem['interior'], bool) & domn
                a['auroc_sem'] = auroc(ehv[bd], ehv[it])
            if v in counts:
                cnt = np.asarray(counts[v])
                occ = cnt > 0
                a['rho_geo'] = spearman(ehv[domn], n_empty_neighbors(occ)[domn])
                a['rho_pts'] = spearman(ehv[domn], cnt[domn])
            att[v] = a
        out['attribution'] = att
    if occ_fields:
        res = {}
        for v, o in occ_fields.items():
            if v not in fields:
                continue
            ot = to_t(np.asarray(o, float)[None], device)
            oy, _ = preprocess(ot, dom)
            oy = oy.to(pb.DH.dtype)
            go = global_bands(pb.fwd(oy), pb)
            po = np.log(np.maximum(go['lam'] / go['tot'], 1e-300))
            pv = np.log(np.maximum(np.asarray(shape[v]['lam_share']), 1e-300))
            ok = np.isfinite(po) & np.isfinite(pv)
            # scan-pattern leak: Spearman, inside Omega_T, between the variant's local high-frequency energy and that of
            # its input occupancy
            ehf_o = torch.zeros(H, W, dtype=torch.float64, device=device)
            Oh = pb.fwd(oy)
            for k in pb.hf_lam:
                ehf_o += (band_signal(Oh, pb, pb.w_lam[k]) ** 2).sum(0).double()
            leak = spearman(e_hf[v][dom].cpu().numpy(), ehf_o[dom].cpu().numpy()) if dom is not None else float('nan')
            res[v] = dict(profile_corr_with_occupancy=float(np.corrcoef(po[ok], pv[ok])[0, 1]) if ok.sum() > 2 else float('nan'),
                          occupancy_H=float(go['lam'][pb.hf_lam].sum() / go['tot']), scan_pattern_leak_rho=leak)
        out['occupancy_residue'] = res
    return out


# ------------------------------------------------------------------ self-test
def self_test(device='cpu'):
    r = {}
    torch.manual_seed(0)
    H, W, C = 40, 30, 4
    pb = PlaneBasis.get(H, W, device)
    x = torch.randn(C, H, W, dtype=torch.float64)
    # DCT matrix = scipy dctn
    ref = sfft.dctn(x.numpy(), type=2, norm='ortho', axes=(1, 2))
    r['dct_vs_scipy_maxabs'] = float(np.abs(pb.fwd(x).numpy() - ref).max())
    # real Fourier basis of a periodic axis: orthonormal; all energy of cos(2 pi 5 t / 30) falls on the two k=5 rows
    Mf, ff, lf = fourier_axis(30)
    r['fourier_orth_maxdev'] = float(np.abs(Mf @ Mf.T - np.eye(30)).max())
    sig = np.cos(2 * np.pi * 5 * np.arange(30) / 30)
    co = Mf @ sig
    r['fourier_pure_tone_share_k5'] = float((co[np.isclose(ff, 5 / 30)] ** 2).sum() / (co ** 2).sum())
    pbp = PlaneBasis.get(H, W, device, (False, True))
    xp = pbp.inv(pbp.fwd(x))
    r['periodic_roundtrip_maxabs'] = float((xp - x).abs().max())
    r['periodic_tight_lam_maxdev'] = float((pbp.w_lam ** 2).sum(0).sub(1).abs().max())
    # tight frame: sum w^2 = 1, sum_k ||B_k x||^2 = ||x||^2
    r['tight_lam_maxdev'] = float((pb.w_lam ** 2).sum(0).sub(1).abs().max())
    r['tight_oct_maxdev'] = float((pb.w_oct ** 2).sum(0).sub(1).abs().max())
    Xh = pb.fwd(x)
    tot = sum(float((band_signal(Xh, pb, pb.w_lam[k]) ** 2).sum()) for k in range(pb.w_lam.shape[0]))
    r['parseval_bands_rel_err'] = abs(tot / float((x ** 2).sum()) - 1)
    # CKA with itself = 1; invariant to orthogonal channel rotations
    Q, _ = torch.linalg.qr(torch.randn(C, C, dtype=torch.float64))
    y = torch.einsum('ij,jhw->ihw', Q, x)
    reg = torch.ones(H, W, dtype=torch.bool)
    Bx = band_signal(Xh, pb, pb.w_lam[3]); By = band_signal(pb.fwd(y), pb, pb.w_lam[3])
    r['cka_self'] = cka(_center_region(Bx, reg), _center_region(Bx, reg))
    r['cka_rot_invariant'] = cka(_center_region(Bx, reg), _center_region(By, reg))
    # power-law field recovers beta (synthetic P ~ |f|^-2)
    beta_true = 2.0
    amp = np.where(pb.frad > 0, pb.frad ** (-beta_true / 2), 0.0)
    rng = np.random.default_rng(0)
    coef = rng.standard_normal((8, H, W)) * amp
    field = torch.from_numpy(sfft.idctn(coef, type=2, norm='ortho', axes=(1, 2)))
    P2 = (pb.fwd(field) ** 2).sum(0).numpy()
    r['beta_recovered'] = spectral_slope(P2, pb)
    # plane_metrics identity: a field against itself gives r_k = 0, CKA = 1
    f = {'T_union': x.numpy(), 'S_copy': x.numpy().copy()}
    regs = {'OmegaT': np.ones((H, W), bool)}
    pm = plane_metrics(f, 'T_union', ['S_copy'], regs, None, {}, {}, device)
    r['identity_r_maxabs'] = float(np.abs(pm['dist']['S_copy']['r_lam']).max())
    r['identity_cka_min'] = float(np.nanmin(pm['cka']['S_copy']['OmegaT']))
    r['identity_excess'] = pm['excess']['S_copy']['OmegaT']['excess_mean']
    # local energy attribution: with dom = full plane it equals the full-plane shares (orthogonal transform + tight frame)
    sv = pm['shape']['T_union']
    r['local_full_vs_global_maxabs'] = float(max(np.abs(np.subtract(sv['masked_lam_share'], sv['lam_share'])).max(),
                                                 np.abs(np.subtract(sv['masked_oct_share'], sv['oct_share'])).max(),
                                                 abs(sv['masked_H'] - sv['H']), abs(sv['masked_H_chmean'] - sv['H_chmean'])))
    # against itself on a jagged random domain (about 60% of the cells): HFER = 1, r_k = 0
    jag = np.random.default_rng(1).random((H, W)) < 0.6
    pmj = plane_metrics(f, 'T_union', ['S_copy'], {'OmegaT': jag}, None, {}, {}, device)
    r['jagged_identity_HFER'] = pmj['dist']['S_copy']['HFER']
    r['jagged_identity_r_maxabs'] = float(np.abs(pmj['dist']['S_copy']['r_lam']).max())
    # demonstration (not part of the pass condition): 1/f^2 power-law field vs the same field plus high frequency, HFER
    # with local attribution, hard mask and full plane on a jagged domain
    sm = field[:C]
    hfn = torch.from_numpy(sfft.idctn(rng.standard_normal((C, H, W)) * np.where(pb.lam >= 0.6, 1.0, 0.0),
                                      type=2, norm='ortho', axes=(1, 2)))
    ro = sm + 0.5 * hfn * float(sm.std() / hfn.std())
    pmd = plane_metrics({'T_union': sm.numpy(), 'S_rough': ro.numpy()}, 'T_union', ['S_rough'], {'OmegaT': jag}, None, {}, {}, device)
    dd = pmd['dist']['S_rough']
    r['demo_smooth_vs_rough_jagged'] = dict(HFER_local=dd['HFER'], HFER_hard=dd['HFER_hard'], HFER_fullplane=dd['HFER_fullplane'])
    ok = (r['fourier_orth_maxdev'] < 1e-12 and r['fourier_pure_tone_share_k5'] > 1 - 1e-12
          and r['local_full_vs_global_maxabs'] < 1e-12 and abs(r['jagged_identity_HFER'] - 1) < 1e-12
          and r['jagged_identity_r_maxabs'] < 1e-12
          and r['periodic_roundtrip_maxabs'] < 1e-12 and r['periodic_tight_lam_maxdev'] < 1e-12
          and r['dct_vs_scipy_maxabs'] < 1e-12 and r['tight_lam_maxdev'] < 1e-12 and r['tight_oct_maxdev'] < 1e-12
          and r['parseval_bands_rel_err'] < 1e-10 and abs(r['cka_self'] - 1) < 1e-10
          and abs(r['cka_rot_invariant'] - 1) < 1e-10 and abs(r['beta_recovered'] - beta_true) < 0.3
          and r['identity_r_maxabs'] < 1e-12 and r['identity_cka_min'] > 1 - 1e-10 and r['identity_excess'] == 0.0)
    r['ALL_PASS'] = bool(ok)
    return r


if __name__ == '__main__':
    import json
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--self-test':
        res = self_test()
        print(json.dumps(res, indent=1))
        sys.exit(0 if res['ALL_PASS'] else 1)
