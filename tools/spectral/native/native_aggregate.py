"""Aggregation of per-frame records into the reported dataset-level numbers.

Outputs (written by `run_native.py aggregate`):
  summary_<dataset>.json  quantities drawn in the native figures
  tables.json / tables.txt the table numbers quoted in the paper

Conventions: per-frame statistics are reported as the median over frames
(with IQR and the fraction of frames above parity, i.e. ratio > 1), except the
diffusion profiles q_t, the dense/sparse true-HF shares, the band profiles and
the band-profile gaps, which are means over frames. Class statistics use only
frames where the class has >= 300 points on both sides.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

import native_data as nd
import native_spectral as ns

B0_KEYS = [str(b) for b in ns.B0_LIST]


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def _frac_gt1(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean([x > 1.0 for x in xs])) if xs else float("nan")


def _pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def frame_files(in_dir, dataset):
    return sorted(glob.glob(os.path.join(in_dir, dataset, "*.json")))


def load_frames(in_dir, dataset):
    """Per-frame records (ok only, sorted by uid) with their KPM spectral summaries."""
    out = []
    for p in frame_files(in_dir, dataset):
        try:
            with open(p, "r", encoding="utf-8") as f:
                r = json.load(f)
        except Exception:
            continue
        if not r.get("ok"):
            continue
        ch = r["cheb"]
        sd = ns.spectral_summary_from_moments(
            np.asarray(ch["mu_d"], dtype=np.float64), ch["band_edges"],
            ch["lmax_map"], ns.ROLLOFF_THETA, ns.TRUE_HF_EDGE)
        ss = ns.spectral_summary_from_moments(
            np.asarray(ch["mu_s"], dtype=np.float64), ch["band_edges"],
            ch["lmax_map"], ns.ROLLOFF_THETA, ns.TRUE_HF_EDGE)
        idx_d = {n: i for i, n in enumerate(ch["signal_names_d"])}
        idx_s = {n: i for i, n in enumerate(ch["signal_names_s"])}
        out.append((r, idx_d, idx_s, sd, ss))
    return out


# =============================================================================
# Dataset summary (figure quantities)
# =============================================================================

def summarize(dataset, frames):
    if not frames:
        return {"dataset": dataset, "n_frames_ok": 0}
    summary = {"dataset": dataset, "n_frames_ok": len(frames),
               "band_edges": ns.BAND_EDGES, "true_hf_edge": ns.TRUE_HF_EDGE,
               "rolloff_theta": ns.ROLLOFF_THETA, "steps": list(ns.DIFF_STEPS),
               "b0_list": list(ns.B0_LIST), "signals": {}, "per_class": {}}

    for s in ns.GLOBAL_SIGNALS:
        rol_d, rol_s, hf_d, hf_s, band_d, band_s = [], [], [], [], [], []
        js, qd, qs = [], [], []
        for r, idx_d, idx_s, sd, ss in frames:
            sig = r["signals"].get(s)
            if not sig or not sig.get("valid", True):
                continue
            js.append(sig["drd_js"])
            qd.append(sig["q_d"]); qs.append(sig["q_s"])
            if s in idx_d and s in idx_s:
                jd, jsg = idx_d[s], idx_s[s]
                rol_d.append(sd["rolloff"][jd]); rol_s.append(ss["rolloff"][jsg])
                hf_d.append(sd["hf_share"][jd]); hf_s.append(ss["hf_share"][jsg])
                band_d.append(sd["band_profile"][:, jd]); band_s.append(ss["band_profile"][:, jsg])
        ratio = [b / (a + 1e-12) for a, b in zip(hf_d, hf_s)]
        summary["signals"][s] = {
            "n_frames": len(js),
            "q_d_mean": np.mean(qd, axis=0).tolist() if qd else None,
            "q_s_mean": np.mean(qs, axis=0).tolist() if qs else None,
            "drd_js_mean": _mean(js),
            "rolloff_d_median": _median(rol_d), "rolloff_s_median": _median(rol_s),
            "hf_share_d_mean": _mean(hf_d), "hf_share_s_mean": _mean(hf_s),
            "hf_share_ratio_median": _median(ratio),
            "hf_share_ratio_frac_gt1": _frac_gt1(ratio),
            "band_d_mean": np.mean(band_d, axis=0).tolist() if band_d else None,
            "band_s_mean": np.mean(band_s, axis=0).tolist() if band_s else None,
        }

    for c in nd.CLASS_IDS:
        cs = str(c); key = "class_%d" % c
        ret = {b: [] for b in B0_KEYS}
        hfr_ratio, n_frames = [], 0
        for r, idx_d, idx_s, sd, ss in frames:
            e = r["per_class"].get(cs)
            if not e or not e.get("valid", True):
                continue
            n_frames += 1
            for b in B0_KEYS:
                ret[b].append(e["hf_retention"].get(b))
            if key in idx_d and key in idx_s:
                hfr_ratio.append(ss["hf_share"][idx_s[key]] / (sd["hf_share"][idx_d[key]] + 1e-12))
        if n_frames == 0:
            continue
        summary["per_class"][cs] = {
            "n_frames": n_frames,
            "hf_retention": {b: {"median": _median(v), "frac_gt1": _frac_gt1(v)}
                             for b, v in ret.items()},
            "spectral_hf_share_ratio_median": _median(hfr_ratio),
            "spectral_hf_share_ratio_frac_gt1": _frac_gt1(hfr_ratio),
        }
    return summary


# =============================================================================
# Table numbers
# =============================================================================

def band_gap_rows(frames):
    """Mean band-profile gap G = P_dense - P_sparse per row (3 global signals, then
    classes 1..7). Class rows use frames where the class signal exists on both
    sides. Returns [(row_key, n_frames, merged_10_or_None, unmerged_11_or_None)];
    'merged' sums the two lowest bands (both lie inside the Jackson soft edge)."""
    gaps = {k: [] for k in list(ns.GLOBAL_SIGNALS) + list(nd.CLASS_IDS)}
    for r, idx_d, idx_s, sd, ss in frames:
        P_d, P_s = sd["band_profile"], ss["band_profile"]
        for sig in ns.GLOBAL_SIGNALS:
            gaps[sig].append(P_d[:, idx_d[sig]] - P_s[:, idx_s[sig]])
        for c in nd.CLASS_IDS:
            nm = "class_{}".format(c)
            if nm in idx_d and nm in idx_s:
                gaps[c].append(P_d[:, idx_d[nm]] - P_s[:, idx_s[nm]])
    rows = []
    for key in list(ns.GLOBAL_SIGNALS) + list(nd.CLASS_IDS):
        lst = gaps[key]
        if not lst:
            rows.append((str(key), 0, None, None))
            continue
        m = np.stack(lst).mean(axis=0)
        rows.append((str(key), len(lst), np.concatenate([[m[0] + m[1]], m[2:]]).tolist(), m.tolist()))
    return rows


def band_labels(edges, merged=False):
    """Band-range labels; the first lower edge (negative) is shown as 0."""
    los = [max(0.0, e) for e in edges[:-1]]
    his = list(edges[1:])
    labels = ["{:g}–{:g}".format(lo, hi) for lo, hi in zip(los, his)]
    if merged:
        labels = ["<{:g}".format(his[1])] + labels[2:]
    return labels


def dataset_tables(summary, frames):
    """Table numbers of one dataset (global signals, per-class KPM, per-class diffusion,
    estimator agreement, band-profile gaps)."""
    rol = {g: ([], []) for g in ns.GLOBAL_SIGNALS}
    hfr = {g: [] for g in ns.GLOBAL_SIGNALS}
    qdc = {g: [] for g in ns.GLOBAL_SIGNALS}
    kpm = {c: [] for c in nd.CLASS_IDS}
    dif = {c: {b: [] for b in B0_KEYS} for c in nd.CLASS_IDS}
    # (KPM ratio, diffusion retention) pairs of the same (frame, class):
    # class-major order for the per-dataset value, frame-major for pooling.
    pairs_cls = {c: {b: [] for b in B0_KEYS} for c in nd.CLASS_IDS}
    pairs_frame = {b: ([], []) for b in B0_KEYS}
    for r, idx_d, idx_s, sd, ss in frames:
        for g in ns.GLOBAL_SIGNALS:
            sig = r["signals"].get(g)
            if not sig or not sig.get("valid", True) or g not in idx_d or g not in idx_s:
                continue
            jd, jsg = idx_d[g], idx_s[g]
            rol[g][0].append(sd["rolloff"][jd]); rol[g][1].append(ss["rolloff"][jsg])
            hfr[g].append(ss["hf_share"][jsg] / (sd["hf_share"][jd] + 1e-12))
            qdc[g].append(ss["band_profile"][0, jsg])
        for c in nd.CLASS_IDS:
            e = r["per_class"].get(str(c))
            if not e or not e.get("valid", True):
                continue
            for b in B0_KEYS:
                dif[c][b].append(e["hf_retention"].get(b))
            k = "class_%d" % c
            if k in idx_d and k in idx_s:
                cr = ss["hf_share"][idx_s[k]] / (sd["hf_share"][idx_d[k]] + 1e-12)
                kpm[c].append(cr)
                for b in B0_KEYS:
                    dv = e["hf_retention"].get(b)
                    if dv is not None and np.isfinite(dv) and np.isfinite(cr):
                        pairs_cls[c][b].append((cr, dv))
                        pairs_frame[b][0].append(cr); pairs_frame[b][1].append(dv)

    G = {}
    for g in ns.GLOBAL_SIGNALS:
        x = summary["signals"][g]
        rd, rs = np.asarray(rol[g][0]), np.asarray(rol[g][1])
        h = np.asarray(hfr[g])
        G[g] = dict(
            n_frames=x["n_frames"],
            q_dense_mean=x["q_d_mean"], q_sparse_mean=x["q_s_mean"],
            drd_js_mean=x["drd_js_mean"],
            rolloff85_dense_median=x["rolloff_d_median"],
            rolloff85_sparse_median=x["rolloff_s_median"],
            rolloff85_separation_ratio_of_medians=x["rolloff_d_median"] / x["rolloff_s_median"],
            rolloff85_separation_median_of_ratios=float(np.median(rd / rs)),
            hf_share_dense_mean=x["hf_share_d_mean"], hf_share_sparse_mean=x["hf_share_s_mean"],
            hfr_median=x["hf_share_ratio_median"],
            hfr_p25=_pct(hfr[g], 25), hfr_p75=_pct(hfr[g], 75),
            hfr_frac_over_parity=x["hf_share_ratio_frac_gt1"],
            hfr_frames_over_parity=int(np.sum(h > 1.0)),
            sparse_quasi_dc_share_mean=float(np.mean(qdc[g])),
        )
    K, D, agree = {}, {}, {}
    for c in nd.CLASS_IDS:
        pc = summary["per_class"].get(str(c))
        if pc is None:
            continue
        K[str(c)] = dict(n_frames=len(kpm[c]),
                         hfr_median=pc["spectral_hf_share_ratio_median"],
                         hfr_p25=_pct(kpm[c], 25), hfr_p75=_pct(kpm[c], 75),
                         hfr_frac_over_parity=pc["spectral_hf_share_ratio_frac_gt1"])
        D[str(c)] = dict(n_frames=pc["n_frames"],
                         retention_median={b: pc["hf_retention"][b]["median"] for b in B0_KEYS},
                         retention_frac_over_parity={b: pc["hf_retention"][b]["frac_gt1"]
                                                     for b in B0_KEYS},
                         retention16_p25=_pct(dif[c]["16"], 25),
                         retention16_p75=_pct(dif[c]["16"], 75))
    for b in B0_KEYS:
        xs = [p[0] for c in nd.CLASS_IDS for p in pairs_cls[c][b]]
        ys = [p[1] for c in nd.CLASS_IDS for p in pairs_cls[c][b]]
        agree[b] = dict(spearman_rho=float(spearmanr(xs, ys)[0]), n_pairs=len(xs))
    rows = band_gap_rows(frames)
    heat = dict(columns=band_labels(summary["band_edges"], merged=True),
                columns_unmerged=band_labels(summary["band_edges"]),
                rows=[dict(row=k, n_frames=n, gap_merged=m10, gap_unmerged=m11)
                      for k, n, m10, m11 in rows])
    return dict(n_frames=summary["n_frames_ok"], global_signals=G, per_class_kpm=K,
                per_class_diffusion=D, estimator_agreement=agree,
                band_gap=heat), pairs_frame


def build_tables(per_dataset):
    """per_dataset: {dataset: (summary, frames)} -> tables dict."""
    out, pooled = {}, {b: ([], []) for b in B0_KEYS}
    for ds in [d for d in nd.DATASETS if d in per_dataset]:
        summary, frames = per_dataset[ds]
        out[ds], pairs = dataset_tables(summary, frames)
        for b in B0_KEYS:
            pooled[b][0].extend(pairs[b][0]); pooled[b][1].extend(pairs[b][1])
    if len(per_dataset) > 1:
        out["pooled_estimator_agreement"] = {
            b: dict(spearman_rho=float(spearmanr(*pooled[b])[0]), n_pairs=len(pooled[b][0]))
            for b in B0_KEYS}
    return out


def _f3(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else "%.3f" % x


def tables_text(tables):
    """Plain-text rendering of tables.json."""
    L = []
    A = L.append
    dss = [d for d in nd.DATASETS if d in tables]
    A("Native spectral statistics: sparse raw scans vs dense completed clouds")
    A("Per-frame medians unless noted; IQR in brackets; 'over parity' = fraction of frames with ratio > 1.")
    A("")
    A("[A] Global signals. q_t: mean normalized diffusion profile at t = 1, 4, 16, 32. "
      "lambda85: KPM roll-off (median). HF share: mean share of lambda >= 0.6. "
      "HFR: per-frame sparse/dense true-HF share ratio.")
    for ds in dss:
        for g, x in tables[ds]["global_signals"].items():
            A("  {:<13} {:<17} n={:<3} q_d {} | q_s {} | DRD-JS {:.4f} | lambda85 d {:.4g} s {:.4g} "
              "sep {:.1f}x (median of ratios {:.1f}x) | HF share d {} s {} | HFR {} [{}, {}] "
              "over parity {} ({}/{}) | sparse share lambda<0.01 {}".format(
                  ds, g, x["n_frames"], "/".join("%.3f" % v for v in x["q_dense_mean"]),
                  "/".join("%.3f" % v for v in x["q_sparse_mean"]), x["drd_js_mean"],
                  x["rolloff85_dense_median"], x["rolloff85_sparse_median"],
                  x["rolloff85_separation_ratio_of_medians"],
                  x["rolloff85_separation_median_of_ratios"],
                  _f3(x["hf_share_dense_mean"]), _f3(x["hf_share_sparse_mean"]),
                  _f3(x["hfr_median"]), _f3(x["hfr_p25"]), _f3(x["hfr_p75"]),
                  _f3(x["hfr_frac_over_parity"]), x["hfr_frames_over_parity"], x["n_frames"],
                  _f3(x["sparse_quasi_dc_share_mean"])))
    A("")
    A("[B] Per-class KPM true-HF share ratio HFR^(c) (median [IQR], n, over parity).")
    for c in nd.CLASS_IDS:
        cells = []
        for ds in dss:
            k = tables[ds]["per_class_kpm"].get(str(c))
            cells.append("{}: -".format(ds) if k is None else "{}: {} [{}, {}] n={} {}".format(
                ds, _f3(k["hfr_median"]), _f3(k["hfr_p25"]), _f3(k["hfr_p75"]),
                k["n_frames"], _f3(k["hfr_frac_over_parity"])))
        A("  {} {:<17} {}".format(c, nd.CLASS_NAMES[c], " | ".join(cells)))
    A("")
    A("[C] Per-class diffusion retention R^(c)_b0 (median; b0 = 16 with IQR and over parity).")
    for c in nd.CLASS_IDS:
        for ds in dss:
            d = tables[ds]["per_class_diffusion"].get(str(c))
            if d is None:
                A("  {} {:<17} {:<13} n=0".format(c, nd.CLASS_NAMES[c], ds))
                continue
            m = d["retention_median"]
            A("  {} {:<17} {:<13} n={:<3} R2 {} R8 {} R16 {} [{}, {}] R32 {} | over parity "
              "@2/8/16/32 {}".format(
                  c, nd.CLASS_NAMES[c], ds, d["n_frames"], _f3(m["2"]), _f3(m["8"]), _f3(m["16"]),
                  _f3(d["retention16_p25"]), _f3(d["retention16_p75"]), _f3(m["32"]),
                  "/".join(_f3(d["retention_frac_over_parity"][b]) for b in B0_KEYS)))
    A("")
    A("[D] KPM vs diffusion estimator agreement: Spearman rho over (frame, class) pairs.")
    for ds in dss:
        a = tables[ds]["estimator_agreement"]
        A("  {:<13} ".format(ds) + "  ".join("b0={}: {:.3f} (n={})".format(
            b, a[b]["spearman_rho"], a[b]["n_pairs"]) for b in B0_KEYS))
    if "pooled_estimator_agreement" in tables:
        a = tables["pooled_estimator_agreement"]
        A("  {:<13} ".format("pooled") + "  ".join("b0={}: {:.3f} (n={})".format(
            b, a[b]["spearman_rho"], a[b]["n_pairs"]) for b in B0_KEYS))
    A("")
    A("[E] Mean band-profile gap G = P_dense - P_sparse (x1000; bands 1+2 merged).")
    for ds in dss:
        h = tables[ds]["band_gap"]
        A("  {}: {}".format(ds, " | ".join(h["columns"])))
        for row in h["rows"]:
            key = row["row"]
            name = key if key in ns.GLOBAL_SIGNALS else "{} {}".format(key, nd.CLASS_NAMES[int(key)])
            vals = row["gap_merged"]
            A("    {:<20} n={:<3} {}".format(name, row["n_frames"], " ".join(
                "-" if vals is None else "%7.1f" % (1000 * v) for v in (vals or [None] * len(h["columns"])))))
    return "\n".join(L) + "\n"
