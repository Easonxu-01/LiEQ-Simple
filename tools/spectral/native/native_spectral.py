"""Graph-spectral measurements of one sparse/dense frame pair.

Per side (sparse raw scan, dense completed cloud subsampled to MAX_DENSE
points), a symmetric Gaussian kNN graph (k = 16) with normalized Laplacian
L = I - D^{-1/2} W D^{-1/2} is built. Signals on the graph:
  * three global signals: density (inverse mean kNN distance), normal_z
    (|z| of the local PCA normal) and surface_variation (RMS plane distance
    over RMS neighbour radius);
  * per coarse class c present on both sides with >= MIN_POINTS_PER_CLASS
    points: the class-masked, within-class-centered height z.
Every signal is centered and l2-normalized before analysis.

Two estimators are computed:
  * Diffusion. With D_a = I - a L (a = 0.25), the normalized response energy
    profile q_t = ||D_a^t x||^2 / sum_t' ||D_a^t' x||^2 for t in DIFF_STEPS and
    the high-frequency residual ||x - D_a^b0 x||^2 for b0 in B0_LIST. The
    per-class retention is the sparse/dense ratio of that residual.
  * Full-spectrum Chebyshev / kernel polynomial method (KPM). Moments
    mu_k = x^T T_k(Lt) x, Lt = (2 / LMAX_MAP) L - I, k = 0..CHEB_M, are stored
    per frame. Band energies over BAND_EDGES follow from the closed-form
    Chebyshev expansion of each band indicator with Jackson damping, so any
    band layout can be recomputed offline from the stored moments.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

import native_data as nd

# ---- protocol constants -----------------------------------------------------
K_GRAPH = 16
ALPHA = 0.25
MIN_POINTS_PER_CLASS = 300
MAX_DENSE = 500000            # dense side: uniform random subsample without replacement
SUBSAMPLE_SEED = 0
CHEB_M = 1024                 # spectral resolution about pi * 2.02 / 1024 = 0.006
LMAX_MAP = 2.02               # normalized Laplacian has lambda_max <= 2; 1% margin
# 11 bands covering the whole spectrum; the first lower edge is negative and the
# last upper edge exceeds 2, so lambda = 0 and lambda = 2 lie inside a band.
BAND_EDGES = [-0.05, 0.01, 0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, 1.50, 2.02]
TRUE_HF_EDGE = 0.60           # bands with lower edge >= 0.60 form the true high-frequency share
ROLLOFF_THETA = 0.85
DIFF_STEPS = (1, 4, 16, 32)
B0_LIST = (2, 8, 16, 32)
GLOBAL_SIGNALS = ("density", "normal_z", "surface_variation")


# =============================================================================
# Graph and signals
# =============================================================================

class GraphData:
    def __init__(self, L, knn_dists, knn_indices, sigma):
        self.L = L
        self.knn_dists = knn_dists
        self.knn_indices = knn_indices
        self.sigma = sigma


def build_knn_graph(points, k=K_GRAPH):
    """Gaussian kNN graph (sigma = mean kNN distance) and its normalized Laplacian."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = points.shape[0]
    if n < 3:
        raise ValueError("At least 3 points are required.")
    k = int(min(max(1, k), n - 1))
    tree = cKDTree(points)
    dists, idx = tree.query(points, k=k + 1)
    sigma = float(np.mean(dists[:, 1:]) + 1e-8)
    src = np.repeat(np.arange(n), k)
    dst = idx[:, 1:].reshape(-1)
    wts = np.exp(-(dists[:, 1:].reshape(-1) ** 2) / (2.0 * sigma ** 2))
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    vals = np.concatenate([wts, wts])
    W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    W = (W + W.T) * 0.5
    deg = np.asarray(W.sum(axis=1)).reshape(-1) + 1e-8
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    L = sp.eye(n, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt
    return GraphData(L=L.tocsr(), knn_dists=dists, knn_indices=idx, sigma=sigma)


def l2_normalize(f):
    f = np.asarray(f, dtype=np.float64).reshape(-1)
    f = f - np.nanmean(f)
    return f / (np.linalg.norm(f) + 1e-12)


def signal_density(graph):
    mean_dist = np.mean(graph.knn_dists[:, 1:], axis=1)
    return 1.0 / (mean_dist + 1e-8)


def local_pca_geometry(points, graph, chunk_size=50000):
    """normal_z and surface_variation from the PCA of each point's kNN neighbourhood."""
    pts = np.asarray(points, dtype=np.float64)
    idx = graph.knn_indices[:, 1:]
    n = pts.shape[0]
    normal_z = np.zeros(n, dtype=np.float64)
    surface_variation = np.zeros(n, dtype=np.float64)
    eps = 1e-12
    for st in range(0, n, chunk_size):
        ed = min(st + chunk_size, n)
        centers = pts[st:ed, None, :]
        neigh = pts[idx[st:ed]]
        rel = neigh - centers
        cov = np.einsum("bki,bkj->bij", rel, rel) / max(idx.shape[1], 1)
        vals, vecs = np.linalg.eigh(cov)
        normals = vecs[:, :, 0]
        normal_z[st:ed] = np.abs(normals[:, 2])
        signed_dist = np.einsum("bki,bi->bk", rel, normals)
        rms_plane = np.sqrt(np.mean(signed_dist ** 2, axis=1))
        rms_radius = np.sqrt(np.mean(np.sum(rel ** 2, axis=2), axis=1)) + eps
        surface_variation[st:ed] = rms_plane / rms_radius
    return normal_z, surface_variation


def class_masked_z(points, labels, cls):
    """Class-masked, within-class-centered z: zero outside the class, zero-mean overall."""
    z = points[:, 2].astype(np.float64)
    mask = labels == cls
    f = np.zeros(len(z), dtype=np.float64)
    f[mask] = z[mask] - z[mask].mean()
    return f


def build_frame_signals(points, labels, graph, class_ids):
    """Ordered signals: 3 global signals, then class_<c> for the given classes."""
    normal_z, surface_variation = local_pca_geometry(points, graph)
    sigs = {}
    sigs["density"] = signal_density(graph)
    sigs["normal_z"] = normal_z
    sigs["surface_variation"] = surface_variation
    for c in class_ids:
        sigs["class_{}".format(c)] = class_masked_z(points, labels, c)
    return sigs


# =============================================================================
# Diffusion estimator
# =============================================================================

def batched_diffusion(L, signals, alpha, steps, b0_list):
    """One diffusion chain G <- (I - alpha L) G over all l2-normalized signal columns.

    Returns per signal the normalized energy profile over `steps`, the
    high-frequency residual energy ||x - D^b0 x||^2 for each b0 (the energy of
    the normalized column is 1, so this is also the residual ratio), and a
    validity flag (False for a constant signal).
    """
    names = list(signals.keys())
    raw_norms, cols = {}, []
    for nm in names:
        f = np.asarray(signals[nm], dtype=np.float64).reshape(-1)
        raw_norms[nm] = float(np.linalg.norm(f - np.nanmean(f)))
        cols.append(l2_normalize(f))
    Fn = np.column_stack(cols)

    steps = [int(t) for t in steps]
    b0s = [int(b) for b in b0_list]
    ckpts = sorted(set(steps) | set(b0s))
    D = sp.eye(L.shape[0], format="csr") - alpha * L
    G = Fn.copy()
    t_done = 0
    energy_at, hf_at = {}, {}
    for t in ckpts:
        for _ in range(t - t_done):
            G = D @ G
        t_done = t
        energy_at[t] = np.sum(G * G, axis=0)
        if t in b0s:
            R = Fn - G
            hf_at[t] = np.sum(R * R, axis=0)

    out = {}
    for j, nm in enumerate(names):
        e = np.array([energy_at[t][j] for t in steps], dtype=np.float64)
        out[nm] = {
            "profile": (e / (e.sum() + 1e-12)).tolist(),
            "hf_ratio": {str(b): float(hf_at[b][j]) for b in b0s},
            "valid": bool(raw_norms[nm] > 1e-9),
        }
    return out, Fn, names


def js_divergence(p, q):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12); q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12))) +
                        np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12)))))


def safe_hf_retention(hf_sparse, hf_dense, floor=1e-9):
    """Sparse/dense residual ratio; NaN when the dense residual is ~0."""
    if hf_dense < floor:
        return float("nan")
    return float(hf_sparse / hf_dense)


# =============================================================================
# Chebyshev / KPM estimator
# =============================================================================

def jackson_coeffs(m):
    """Jackson damping coefficients g_k, k = 0..m."""
    k = np.arange(m + 1, dtype=np.float64)
    a = np.pi / (m + 2)
    return ((m + 2 - k) * np.sin(a) * np.cos(k * a)
            + np.cos(a) * np.sin(k * a)) / ((m + 2) * np.sin(a))


def projector_cheb_coeffs(lo, hi, m, lmax_map):
    """Closed-form Chebyshev coefficients of the indicator 1_[lo,hi)(lambda).

    With t(lambda) = 2 lambda / lmax_map - 1:
      c_0 = (arccos(t_lo) - arccos(t_hi)) / pi
      c_k = (2 / pi) (sin(k arccos(t_lo)) - sin(k arccos(t_hi))) / k
    lo may be negative and hi may exceed 2 (edges outside the spectrum).
    """
    def to_t(lam):
        return float(np.clip(2.0 * lam / lmax_map - 1.0, -1.0, 1.0))
    a = np.arccos(to_t(hi))
    b = np.arccos(to_t(lo))
    c = np.empty(m + 1, dtype=np.float64)
    c[0] = (b - a) / np.pi
    k = np.arange(1, m + 1, dtype=np.float64)
    c[1:] = 2.0 / np.pi * (np.sin(k * b) - np.sin(k * a)) / k
    return c


def chebyshev_signal_moments(L, X, m=CHEB_M, lmax_map=LMAX_MAP):
    """Chebyshev moments of all signal columns, shape (m + 1, s).

    X: (n, s), columns centered and l2-normalized (mu_0 = 1). lmax_map = 2.02
    keeps |eig(Lt)| < 1, so the three-term recurrence is stable.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    n, s = X.shape
    Lt = (sp.csr_matrix(L) * (2.0 / lmax_map)) - sp.identity(n, format="csr")
    mu = np.empty((m + 1, s), dtype=np.float64)
    T_prev = X
    mu[0] = np.einsum("ns,ns->s", X, T_prev)
    T_curr = Lt @ X
    mu[1] = np.einsum("ns,ns->s", X, T_curr)
    for k in range(2, m + 1):
        T_next = 2.0 * (Lt @ T_curr) - T_prev
        mu[k] = np.einsum("ns,ns->s", X, T_next)
        T_prev, T_curr = T_curr, T_next
    return mu


def band_weight_matrix(band_edges, m, lmax_map):
    """(nb, m + 1) Jackson-damped band projector coefficients; E = clip(W @ mu, 0)."""
    g = jackson_coeffs(m)
    C = np.stack([projector_cheb_coeffs(band_edges[b], band_edges[b + 1], m, lmax_map)
                  for b in range(len(band_edges) - 1)])
    return C * g[None, :]


def band_energies_from_moments(mu, band_edges, lmax_map):
    """Band energies (nb, s) from the moments; small negative leakage clipped to 0."""
    m = mu.shape[0] - 1
    return np.clip(band_weight_matrix(band_edges, m, lmax_map) @ np.asarray(mu, dtype=np.float64),
                   0.0, None)


def spectral_summary_from_moments(mu, band_edges, lmax_map, rolloff_theta=ROLLOFF_THETA,
                                  hf_edge=TRUE_HF_EDGE):
    """Per signal column: normalized band profile (nb, s), roll-off lambda at
    rolloff_theta (linear interpolation inside the band) and the true
    high-frequency share (sum of bands with lower edge >= hf_edge)."""
    E = band_energies_from_moments(mu, band_edges, lmax_map)
    tot = E.sum(axis=0)
    P = E / (tot + 1e-12)

    edges = np.asarray(band_edges, dtype=np.float64)
    los = np.clip(edges[:-1], 0.0, None)
    his = np.clip(edges[1:], None, 2.0)
    cum = np.cumsum(P, axis=0)
    s = P.shape[1]
    rolloff = np.empty(s)
    for j in range(s):
        b = int(np.searchsorted(cum[:, j], rolloff_theta))
        b = min(b, P.shape[0] - 1)
        prev = cum[b - 1, j] if b > 0 else 0.0
        frac = (rolloff_theta - prev) / (P[b, j] + 1e-12)
        rolloff[j] = float(los[b] + np.clip(frac, 0.0, 1.0) * (his[b] - los[b]))

    hf_mask = los >= float(hf_edge)
    hf_share = P[hf_mask, :].sum(axis=0)
    return dict(band_profile=P, rolloff=rolloff, hf_share=hf_share)


# =============================================================================
# Per-frame record
# =============================================================================

def process_frame(task):
    """All per-frame quantities for one sparse/dense pair (see module docstring)."""
    t_start = time.time()
    dense_pts, dense_lab = nd.load_side(
        task["dense_points"], task["dense_labels"], task["dense_bin_dim"],
        task["dense_label_format"])
    sparse_pts, sparse_lab = nd.load_side(
        task["sparse_points"], task["sparse_labels"], task["sparse_bin_dim"],
        task["sparse_label_format"])
    n_dense_in = int(len(dense_pts))
    dense_pts, dense_lab = nd.deterministic_subsample(
        dense_pts, dense_lab, MAX_DENSE, SUBSAMPLE_SEED)
    if len(dense_pts) < K_GRAPH + 2 or len(sparse_pts) < K_GRAPH + 2:
        raise ValueError("too few points: dense=%d sparse=%d"
                         % (len(dense_pts), len(sparse_pts)))

    dense_graph = build_knn_graph(dense_pts, k=K_GRAPH)
    sparse_graph = build_knn_graph(sparse_pts, k=K_GRAPH)

    counts_d = {c: int(np.sum(dense_lab == c)) for c in nd.CLASS_IDS}
    counts_s = {c: int(np.sum(sparse_lab == c)) for c in nd.CLASS_IDS}
    valid_classes = [c for c in nd.CLASS_IDS
                     if counts_d[c] >= MIN_POINTS_PER_CLASS
                     and counts_s[c] >= MIN_POINTS_PER_CLASS]

    sigs_d = build_frame_signals(dense_pts, dense_lab, dense_graph, valid_classes)
    sigs_s = build_frame_signals(sparse_pts, sparse_lab, sparse_graph, valid_classes)

    diff_d, Fn_d, names_d = batched_diffusion(dense_graph.L, sigs_d, ALPHA, DIFF_STEPS, B0_LIST)
    diff_s, Fn_s, names_s = batched_diffusion(sparse_graph.L, sigs_s, ALPHA, DIFF_STEPS, B0_LIST)

    mu_d = chebyshev_signal_moments(dense_graph.L, Fn_d, CHEB_M, LMAX_MAP)
    mu_s = chebyshev_signal_moments(sparse_graph.L, Fn_s, CHEB_M, LMAX_MAP)

    signals_out = {}
    for s in GLOBAL_SIGNALS:
        signals_out[s] = {
            "valid": bool(diff_d[s]["valid"] and diff_s[s]["valid"]),
            "q_d": diff_d[s]["profile"], "q_s": diff_s[s]["profile"],
            "drd_js": js_divergence(np.asarray(diff_d[s]["profile"]),
                                    np.asarray(diff_s[s]["profile"])),
        }
    per_class_out = {}
    for c in valid_classes:
        key = "class_%d" % c
        per_class_out[str(c)] = {
            "valid": bool(diff_d[key]["valid"] and diff_s[key]["valid"]),
            "hf_retention": {b: safe_hf_retention(diff_s[key]["hf_ratio"][b],
                                                  diff_d[key]["hf_ratio"][b])
                             for b in diff_d[key]["hf_ratio"]},
        }

    rec = {
        "ok": True,
        "dataset": task["dataset"], "seq": task["seq"], "frame": task["frame"],
        "steps": list(DIFF_STEPS), "b0_list": list(B0_LIST),
        "signals": signals_out, "per_class": per_class_out,
        "cheb": {
            "m": CHEB_M, "lmax_map": LMAX_MAP, "band_edges": BAND_EDGES,
            "signal_names_d": names_d, "signal_names_s": names_s,
            "mu_d": np.asarray(mu_d, dtype=np.float32).tolist(),
            "mu_s": np.asarray(mu_s, dtype=np.float32).tolist(),
        },
    }
    info = dict(n_dense_in=n_dense_in, n_dense=int(len(dense_pts)),
                n_sparse=int(len(sparse_pts)), seconds=time.time() - t_start)
    return rec, info


# =============================================================================
# Self-check of the KPM band energies against an exact eigendecomposition
# =============================================================================

def regression_check(seed=0, n=3000, verbose=True):
    """KPM band energies vs exact eigh on three small synthetic graphs.

    (a) Identity test: moments vs the same Jackson-smoothed filters applied to
        the exact eigenpairs; must agree to 1e-6.
    (b) Hard-edge test: vs hard band indicators. Spectral mass within +-2
        kernel widths of a band edge is legitimately split between the two
        neighbouring bands, so per signal
            max_b |KPM - hard| <= 0.6 * (exact energy in the edge strips) + 5e-2.
    """
    rng = np.random.default_rng(seed)
    cases = {
        "uniform3d": rng.uniform(-1, 1, size=(n, 3)),
        "surface": np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n),
                                    0.05 * rng.standard_normal(n)]),
        "two_clusters": np.vstack([rng.normal(-2, 0.3, size=(n // 2, 3)),
                                   rng.normal(+2, 0.3, size=(n - n // 2, 3))]),
    }
    nb = len(BAND_EDGES) - 1
    Cg = band_weight_matrix(BAND_EDGES, CHEB_M, LMAX_MAP)
    ok_all = True
    for name, pts in cases.items():
        g = build_knn_graph(pts, k=K_GRAPH)
        sigs = np.column_stack([
            l2_normalize(pts[:, 2]),
            l2_normalize(rng.standard_normal(n)),               # white noise (flat spectrum)
            l2_normalize(np.sin(3 * pts[:, 0]) + pts[:, 1]),
        ])
        lam, V = np.linalg.eigh(g.L.toarray())
        A2 = (V.T @ sigs) ** 2
        exact_hard = np.stack([
            np.sum(A2[(lam >= max(0.0, BAND_EDGES[b])) & (lam < BAND_EDGES[b + 1])], axis=0)
            if b > 0 else np.sum(A2[lam < BAND_EDGES[1]], axis=0)
            for b in range(nb)])
        theta = np.arccos(np.clip(2.0 * lam / LMAX_MAP - 1.0, -1.0, 1.0))
        T_at_lam = np.cos(np.outer(np.arange(CHEB_M + 1), theta))
        exact_smooth = (Cg @ T_at_lam) @ A2
        mu = chebyshev_signal_moments(g.L, sigs, CHEB_M, LMAX_MAP)
        approx = band_energies_from_moments(mu, BAND_EDGES, LMAX_MAP)
        err_ident = np.abs(approx - np.clip(exact_smooth, 0.0, None)).max()
        delta = 2.0 * np.pi * LMAX_MAP / CHEB_M
        edge_mass = np.zeros(A2.shape[1])
        for e in BAND_EDGES[1:-1]:
            edge_mass += A2[np.abs(lam - e) < delta].sum(axis=0)
        dev = np.abs(approx - exact_hard).max(axis=0)
        bound = 0.6 * edge_mass + 5e-2
        ok = err_ident < 1e-6 and bool(np.all(dev <= bound))
        ok_all = ok_all and ok
        if verbose:
            j = int(np.argmax(dev - bound))
            print("[selfcheck] %-12s identity err = %.2e (<1e-6); worst signal: "
                  "hard-edge dev = %.2e vs bound %.2e (edge mass %.3f)  %s"
                  % (name, err_ident, dev[j], bound[j], edge_mass[j],
                     "PASS" if ok else "FAIL"), flush=True)
    print("[selfcheck] overall:", "PASS" if ok_all else "FAIL", flush=True)
    return ok_all
