"""Figures of the native spectral statistics (vector PDF + PNG).

  fig_native_b0_trend         median per-class diffusion retention vs b0
  fig_native_band_profiles    dense vs sparse mean KPM band profiles (3 x 3)
  fig_native_qprofiles        dataset-mean multi-scale diffusion profiles q_t
  fig_native_rolloff_hfshare  KPM roll-off medians and true-HF share means
  fig_native_class_retention  per-class retention at b0 = 16 (median + IQR) with
                              the KPM per-class true-HF share ratio
  fig_native_gap_heatmap      mean band-profile gap G = P_dense - P_sparse

Inputs: summary_<dataset>.json and tables.json written by the aggregate step.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import native_data as nd

PLOT_DATASETS = ("semantickitti", "waymo", "nuscenes")
DATASET_COLORS = {
    "semantickitti": "#4477AA",
    "waymo": "#EE6677",
    "nuscenes": "#228833",
}
GLOBAL_SIGNALS = ("density", "normal_z", "surface_variation")
SIGNAL_DISPLAY = {"density": "density", "normal_z": "normal_z",
                  "surface_variation": "surface_variation"}
CLASS_IDS = nd.CLASS_IDS
CLASS_COLORS = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(CLASS_IDS)}
SOFT_EDGE_NOTE = "kernel-width soft edge < 0.03"

FIGURES = ("band_profiles", "gap_heatmap", "class_retention", "rolloff_hfshare",
           "b0_trend", "qprofiles")


def _ds_color(ds):
    return DATASET_COLORS.get(ds, "#888888")


def _n_label(summaries):
    """Frame count for titles: one number if equal across datasets."""
    ns_ = [summaries[ds]["n_frames_ok"] for ds in PLOT_DATASETS]
    return str(ns_[0]) if len(set(ns_)) == 1 else "/".join(str(n) for n in ns_)


def _band_labels(edges, merged=False):
    los = [max(0.0, e) for e in edges[:-1]]
    his = list(edges[1:])
    labels = ["{:g}–{:g}".format(lo, hi) for lo, hi in zip(los, his)]
    if merged:
        labels = ["<{:g}".format(his[1])] + labels[2:]
    return labels


def _annotate_bar(ax, x, v, pad, fontsize, fmt, rotation=0):
    """Value annotation above a bar (pad is additive in data units)."""
    if v is None or not np.isfinite(v):
        return
    ax.text(x, v + pad, fmt.format(v), ha="center", va="bottom", fontsize=fontsize,
            color="#222222", rotation=rotation)


def fig_band_profiles(summaries):
    edges = summaries[PLOT_DATASETS[0]]["band_edges"]
    labels = _band_labels(edges)
    nb = len(labels)
    x = np.arange(nb)
    width = 0.4
    fig, axes = plt.subplots(3, 3, figsize=(15.0, 9.6), sharex=True, sharey="row")
    for r, ds in enumerate(PLOT_DATASETS):
        col = _ds_color(ds)
        for cidx, sig in enumerate(GLOBAL_SIGNALS):
            ax = axes[r, cidx]
            v = summaries[ds]["signals"][sig]
            bd = np.maximum(np.asarray(v["band_d_mean"], dtype=np.float64), 1e-8)
            bs = np.maximum(np.asarray(v["band_s_mean"], dtype=np.float64), 1e-8)
            ax.bar(x - width / 2, bd, width, color=col, alpha=0.92,
                   edgecolor="#555555", linewidth=0.5, label="Dense")
            ax.bar(x + width / 2, bs, width, color=col, alpha=0.38,
                   hatch="///", edgecolor=col, linewidth=0.5, label="Sparse")
            ax.set_yscale("log")
            ax.axvline(1.5, color="#999999", linewidth=0.9, linestyle=":")
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(SIGNAL_DISPLAY[sig], fontsize=11)
            if cidx == 0:
                ax.set_ylabel("{}\nmean band energy fraction".format(nd.DISPLAY[ds]),
                              fontsize=9)
                ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
            if r == 0 and cidx == 0:
                ax.text(1.44, 0.965, SOFT_EDGE_NOTE, transform=ax.get_xaxis_transform(),
                        rotation=90, fontsize=6.5, color="#777777",
                        ha="right", va="top", rotation_mode="anchor")
            if r == len(PLOT_DATASETS) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.8)
                ax.set_xlabel(r"graph-frequency band $\lambda$", fontsize=9)
    fig.suptitle("KPM band profiles: dense vs sparse mean normalized band energy "
                 "({} bands, N={} frames/dataset)\n"
                 "dotted line: first two bands lie inside the Jackson "
                 "{}".format(nb, _n_label(summaries), SOFT_EDGE_NOTE), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_gap_heatmap(summaries, tables):
    edges = summaries[PLOT_DATASETS[0]]["band_edges"]
    col_labels = _band_labels(edges, merged=True)
    ncols = len(col_labels)

    mats, row_labels_per_ds = {}, {}
    for ds in PLOT_DATASETS:
        rows_by_key = {r["row"]: r for r in tables[ds]["band_gap"]["rows"]}
        rows, labels = [], []
        for sig in GLOBAL_SIGNALS:
            rows.append(np.asarray(rows_by_key[sig]["gap_merged"], dtype=np.float64))
            labels.append(SIGNAL_DISPLAY[sig])
        for c in CLASS_IDS:
            r = rows_by_key[str(c)]
            if r["n_frames"]:
                rows.append(np.asarray(r["gap_merged"], dtype=np.float64))
            else:
                rows.append(np.full(ncols, np.nan))
            labels.append("{} (n={})".format(nd.CLASS_NAMES[c], r["n_frames"]))
        mats[ds] = np.stack(rows)
        row_labels_per_ds[ds] = labels

    vmax = max(float(np.nanmax(np.abs(m))) for m in mats.values())
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 6.2))
    im = None
    for ax, ds in zip(axes, PLOT_DATASETS):
        M = mats[ds]
        # pcolormesh keeps every cell a vector patch in the PDF; cell centres
        # sit on integer coordinates with row 0 on top, as with imshow.
        im = ax.pcolormesh(np.arange(ncols + 1) - 0.5, np.arange(M.shape[0] + 1) - 0.5,
                           np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="flat", edgecolors="none", rasterized=False)
        ax.set_xlim(-0.5, ncols - 0.5)
        ax.set_ylim(M.shape[0] - 0.5, -0.5)
        ax.set_xticks(np.arange(ncols))
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=6.8)
        ax.set_yticks(np.arange(M.shape[0]))
        ax.set_yticklabels(row_labels_per_ds[ds], fontsize=7.2)
        ax.set_title("{} (N={} frames)".format(nd.DISPLAY[ds],
                                               summaries[ds]["n_frames_ok"]),
                     fontsize=10.5, color=_ds_color(ds))
        ax.set_xlabel(r"graph-frequency band $\lambda$ (bands 1+2 merged)",
                      fontsize=8)
        for i in range(M.shape[0]):
            for j in range(ncols):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=5.4, color="#666666")
                    continue
                tc = "white" if abs(v) > 0.55 * vmax else "#222222"
                ax.text(j, i, "{:.1f}".format(v * 1000.0), ha="center",
                        va="center", fontsize=5.4, color=tc)
    fig.suptitle("mean band-profile gap  G = P_dense − P_sparse  "
                 "(red = dense excess, blue = sparse excess; cell text = G × 1000)\n"
                 "global signals over all frames; class rows over frames where the "
                 "class exists on both sides", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.925, 0.90])
    cax = fig.add_axes([0.938, 0.14, 0.012, 0.68])
    cb_ = fig.colorbar(im, cax=cax)
    # matplotlib rasterizes colorbar gradients with >= 50 levels; keep the PDF vector-only.
    if getattr(cb_, "solids", None) is not None:
        cb_.solids.set_rasterized(False)
    cb_.set_label(r"mean gap $G = P_d - P_s$", fontsize=8)
    cb_.ax.tick_params(labelsize=7)
    return fig


def fig_class_retention(summaries, tables):
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    x = np.arange(len(CLASS_IDS))
    width = 0.26
    tops = [1.0]
    for di, ds in enumerate(PLOT_DATASETS):
        off = (di - 1) * width
        med = np.full(len(CLASS_IDS), np.nan)
        p25 = np.full(len(CLASS_IDS), np.nan)
        p75 = np.full(len(CLASS_IDS), np.nan)
        kpm = np.full(len(CLASS_IDS), np.nan)
        nfr = np.zeros(len(CLASS_IDS), dtype=int)
        for i, c in enumerate(CLASS_IDS):
            pc = summaries[ds]["per_class"].get(str(c))
            if pc is None:
                continue
            med[i] = pc["hf_retention"]["16"]["median"]
            kpm[i] = pc["spectral_hf_share_ratio_median"]
            nfr[i] = pc["n_frames"]
            d = tables[ds]["per_class_diffusion"].get(str(c))
            if d is not None and d["n_frames"]:
                p25[i], p75[i] = d["retention16_p25"], d["retention16_p75"]
        yerr = np.stack([med - p25, p75 - med])
        ax.bar(x + off, med, width * 0.92, color=_ds_color(ds), alpha=0.9,
               edgecolor="#555555", linewidth=0.6,
               yerr=np.where(np.isfinite(yerr), yerr, 0.0),
               error_kw=dict(ecolor="#333333", elinewidth=1.0, capsize=2.5),
               label=nd.DISPLAY[ds])
        ax.plot(x + off, kpm, marker="D", markersize=5, linestyle="none",
                color="black", zorder=5)
        tops.extend(v for v in np.concatenate([p75, kpm, med]) if np.isfinite(v))
        for i in range(len(CLASS_IDS)):
            top_i = np.nanmax([p75[i], med[i], kpm[i]])
            if np.isfinite(med[i]):
                _annotate_bar(ax, x[i] + off, top_i, 0.05, fontsize=6.2,
                              fmt="{:.3f}", rotation=90)
                ax.text(x[i] + off, 0.03, "n={}".format(nfr[i]), rotation=90,
                        ha="center", va="bottom", fontsize=5.8, color="#333333")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    top = float(np.nanmax(tops))
    ax.set_ylim(0, top * 1.22)
    ax.set_xticks(x)
    ax.set_xticklabels([nd.CLASS_NAMES[c] for c in CLASS_IDS], fontsize=9)
    ax.set_ylabel("HF retention  $(E_{hi}/E_{tot})_{sparse} / "
                  "(E_{hi}/E_{tot})_{dense}$,  $b_0=16$")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="D", markersize=5, linestyle="none",
                          color="black"))
    labels.append("KPM spectral HF-share ratio (median)")
    ax.legend(handles, labels, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Per-class high-frequency retention (diffusion, $b_0=16$): "
                 "median over frames, whiskers = IQR (p25–p75)\n"
                 "black diamond = KPM spectral true-HF share ratio "
                 "(sparse/dense, median); dashed line = parity")
    fig.tight_layout()
    return fig


def _paired_bars_panel(ax, summaries, key_d, key_s, ylabel, title):
    group_w = 4.0
    width = 0.38
    for si, sig in enumerate(GLOBAL_SIGNALS):
        for di, ds in enumerate(PLOT_DATASETS):
            v = summaries[ds]["signals"][sig]
            xc = si * group_w + di
            col = _ds_color(ds)
            d, s = float(v[key_d]), float(v[key_s])
            ax.bar(xc - width / 2, d, width, color=col, alpha=0.92,
                   edgecolor="#555555", linewidth=0.5)
            ax.bar(xc + width / 2, s, width, color=col, alpha=0.38,
                   hatch="///", edgecolor=col, linewidth=0.5)
            for xx, vv in ((xc - width / 2, d), (xc + width / 2, s)):
                ax.text(xx, vv * 1.07, "{:.3g}".format(vv), ha="center",
                        va="bottom", fontsize=5.6, rotation=90, color="#222222")
    ax.set_yscale("log")
    ax.set_xticks([si * group_w + 1 for si in range(len(GLOBAL_SIGNALS))])
    ax.set_xticklabels([SIGNAL_DISPLAY[s] for s in GLOBAL_SIGNALS], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 2.2)  # headroom for rotated annotations


def fig_rolloff_hfshare(summaries):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    theta = summaries[PLOT_DATASETS[0]]["rolloff_theta"]
    hf_edge = summaries[PLOT_DATASETS[0]]["true_hf_edge"]
    _paired_bars_panel(axes[0], summaries, "rolloff_d_median", "rolloff_s_median",
                       r"spectral rolloff $\lambda$ (median over frames)",
                       r"Spectral rolloff ($\theta={}$): dense vs sparse".format(theta))
    _paired_bars_panel(axes[1], summaries, "hf_share_d_mean", "hf_share_s_mean",
                       r"true-HF share $P(\lambda \geq {})$ (mean over frames)".format(hf_edge),
                       r"True high-frequency share ($\lambda \geq {}$): "
                       "dense vs sparse".format(hf_edge))
    handles = [Patch(facecolor=_ds_color(ds), edgecolor="#555555",
                     label=nd.DISPLAY[ds]) for ds in PLOT_DATASETS]
    handles += [Patch(facecolor="#aaaaaa", edgecolor="#555555", label="Dense (solid)"),
                Patch(facecolor="#dddddd", edgecolor="#888888", hatch="///",
                      label="Sparse (hatched)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False)
    fig.suptitle("KPM global-signal spectral concentration (N={} frames/dataset)".format(
                 _n_label(summaries)), fontsize=11.5)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


def fig_b0_trend(summaries):
    b0s = tuple(summaries[PLOT_DATASETS[0]]["b0_list"])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), sharey=True)
    for ax, ds in zip(axes, PLOT_DATASETS):
        for c in CLASS_IDS:
            pc = summaries[ds]["per_class"].get(str(c))
            if pc is None:
                continue
            y = [pc["hf_retention"][str(b)]["median"] for b in b0s]
            ax.plot(b0s, y, marker="o", markersize=4, linewidth=1.8,
                    color=CLASS_COLORS[c], label=nd.CLASS_NAMES[c])
        ax.set_xscale("log", base=2)
        ax.set_xticks(b0s)
        ax.set_xticklabels([str(b) for b in b0s])
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0)
        ax.set_xlabel(r"diffusion cutoff $b_0$ (steps)")
        ax.set_title(nd.DISPLAY[ds], color=_ds_color(ds), fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("median HF retention (sparse/dense)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7.5,
               frameon=False)
    fig.suptitle("per-class HF retention vs diffusion cutoff $b_0$ "
                 "(median over frames; dashed line = parity)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    return fig


def fig_qprofiles(summaries):
    steps = tuple(summaries[PLOT_DATASETS[0]]["steps"])
    xs = np.arange(len(steps))
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), sharey=True)
    for ax, sig in zip(axes, GLOBAL_SIGNALS):
        for ds in PLOT_DATASETS:
            v = summaries[ds]["signals"][sig]
            col = _ds_color(ds)
            js = v.get("drd_js_mean", float("nan"))
            ax.plot(xs, v["q_d_mean"], marker="o", linewidth=2, color=col,
                    label="{} dense".format(nd.DISPLAY[ds]))
            ax.plot(xs, v["q_s_mean"], marker="s", linewidth=2, linestyle="--",
                    color=col,
                    label="{} sparse (DRD-JS={:.4f})".format(nd.DISPLAY[ds], js))
        ax.set_xticks(xs)
        ax.set_xticklabels([str(t) for t in steps])
        ax.set_xlabel("diffusion steps $t$")
        ax.set_title(SIGNAL_DISPLAY[sig], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.4, loc="best")
    axes[0].set_ylabel(r"normalized response energy $\bar{q}_t$")
    fig.suptitle("dataset-mean multi-scale diffusion profiles "
                 "(per-frame normalized, averaged over N={} frames)".format(
                     _n_label(summaries)), fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def make_figures(summaries, tables, out_dir, dpi=220, formats=("pdf", "png"), which=FIGURES):
    builders = {
        "band_profiles": lambda: fig_band_profiles(summaries),
        "gap_heatmap": lambda: fig_gap_heatmap(summaries, tables),
        "class_retention": lambda: fig_class_retention(summaries, tables),
        "rolloff_hfshare": lambda: fig_rolloff_hfshare(summaries),
        "b0_trend": lambda: fig_b0_trend(summaries),
        "qprofiles": lambda: fig_qprofiles(summaries),
    }
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for name in which:
        fig = builders[name]()
        for ext in formats:
            path = os.path.join(out_dir, "fig_native_{}.{}".format(name, ext))
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            written.append(path)
        plt.close(fig)
    return written
