#!/usr/bin/env python3
"""Native (point-level) spectral statistics: sparse raw scans vs dense completed clouds.

Data only, CPU only; no model or checkpoint is involved. For every frame pair
the sparse raw scan and the dense completed cloud (uniformly subsampled to
500k points) are cropped to a 50 m sphere, a kNN graph is built per side, and
geometric signals are analysed with a diffusion estimator and a full-spectrum
Chebyshev (KPM) estimator (see native_spectral.py).

Steps
  stats      per-frame records   -> OUT/<dataset>/<uid>.json  (resumable, shardable)
  aggregate  dataset summaries   -> summary_<dataset>.json, tables.json, tables.txt
  plot       figures             -> FIGDIR/fig_native_<name>.{pdf,png}
  selfcheck  KPM band energies vs exact eigendecomposition on small synthetic graphs

Examples
  python run_native.py stats --dataset semantickitti --raw-root <SEMANTICKITTI_ROOT> \\
      --dense-root <COMPLETED_ROOT> --frames paper --out <WORK_DIR>
  python run_native.py aggregate --in <WORK_DIR>
  python run_native.py plot --in <WORK_DIR> --out <FIG_DIR>

--frames paper selects the frames used in the paper (frames/<dataset>.txt);
--frames sample:N applies the stratified sampling rule (seed 0) to the
discovered frames; any other value is read as a file with one uid per line.
A frame takes about 2.5-4 minutes on 2-3 CPU threads (dominated by the
1024-step Chebyshev recurrence on the 500k-point graph); use --shard and
--num-shards to spread the frames over several processes. The BLAS thread
count can change the last digits of per-frame values (relative ~1e-7).
"""
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import native_data as nd          # noqa: E402
import native_spectral as ns      # noqa: E402


def _dataset(name):
    key = name.strip().lower()
    if key not in nd.DATASET_ALIASES:
        raise argparse.ArgumentTypeError("unknown dataset '{}' (choose from {})".format(
            name, ", ".join(nd.DATASETS)))
    return nd.DATASET_ALIASES[key]


def cmd_stats(args):
    ds = args.dataset
    dense_root = args.dense_root or args.raw_root
    sk_seqs = args.sk_seqs or (nd.sk_seqs_for_spec(args.frames) if ds == "semantickitti"
                               else nd.DEFAULT_SK_SEQS)
    tasks, skipped = nd.discover_frames(ds, args.raw_root, dense_root, sk_seqs=sk_seqs,
                                        nusc_info=args.nusc_info)
    print("[{}] discovered {} paired frames ({} skipped)".format(ds, len(tasks), len(skipped)),
          flush=True)
    if not tasks:
        for s in skipped[:5]:
            print("  skipped:", s)
        sys.exit(1)
    sel = nd.select_frames(tasks, ds, args.frames)
    if args.num_shards > 1:
        sel = [t for i, t in enumerate(sel) if i % args.num_shards == args.shard]
    out_dir = os.path.join(args.out, ds)
    os.makedirs(out_dir, exist_ok=True)
    done = new = failed = 0
    for t in sel:
        out_path = os.path.join(out_dir, "{}.json".format(t["uid"]))
        if os.path.exists(out_path) and not args.overwrite:
            try:
                with open(out_path) as f:
                    if json.load(f).get("ok"):
                        done += 1
                        continue
            except Exception:
                pass
        if args.max_new >= 0 and new >= args.max_new:
            print("PROGRESS {} {}/{} done (budget reached)".format(ds, done, len(sel)), flush=True)
            break
        t0 = time.time()
        try:
            rec, info = ns.process_frame(t)
        except Exception as exc:
            rec, info = {"ok": False, "dataset": ds, "seq": t["seq"], "frame": t["frame"],
                         "err": "{}: {}".format(type(exc).__name__, exc)}, None
            failed += 1
        nd.write_json_atomic(out_path, rec)
        new += 1
        if rec.get("ok"):
            done += 1
            print("[{}] {} ok in {:.1f}s (dense {} -> {}, sparse {}, classes {})".format(
                ds, t["uid"], time.time() - t0, info["n_dense_in"], info["n_dense"],
                info["n_sparse"], len(rec["per_class"])), flush=True)
        else:
            print("[{}] {} FAILED: {}".format(ds, t["uid"], rec.get("err")), flush=True)
    print("[{}] {}/{} frames done, {} new, {} failed".format(ds, done, len(sel), new, failed),
          flush=True)
    sys.exit(1 if failed else 0)


def cmd_aggregate(args):
    import native_aggregate as na
    out_dir = args.out or args.inp
    dss = [d for d in args.datasets] if args.datasets else \
        [d for d in nd.DATASETS if os.path.isdir(os.path.join(args.inp, d))]
    if not dss:
        sys.exit("no <dataset>/ directories with per-frame records under {}".format(args.inp))
    per_dataset = {}
    os.makedirs(out_dir, exist_ok=True)
    for ds in dss:
        frames = na.load_frames(args.inp, ds)
        summary = na.summarize(ds, frames)
        if not frames:
            print("[{}] no ok frames, skipped".format(ds))
            continue
        path = os.path.join(out_dir, "summary_{}.json".format(ds))
        nd.write_json_atomic(path, summary)
        print("[{}] aggregated {} frames -> {}".format(ds, summary["n_frames_ok"], path), flush=True)
        per_dataset[ds] = (summary, frames)
    tables = na.build_tables(per_dataset)
    path = os.path.join(out_dir, "tables.json")
    nd.write_json_atomic(path, tables)
    with open(os.path.join(out_dir, "tables.txt"), "w", encoding="utf-8") as f:
        f.write(na.tables_text(tables))
    print("[tables] -> {} (+ tables.txt)".format(path), flush=True)


def cmd_plot(args):
    import native_plots as npl
    summaries = {}
    for ds in npl.PLOT_DATASETS:
        path = os.path.join(args.inp, "summary_{}.json".format(ds))
        if not os.path.exists(path):
            sys.exit("missing {} (the figures show all three datasets; run aggregate first)".format(path))
        with open(path, "r", encoding="utf-8") as f:
            summaries[ds] = json.load(f)
    with open(os.path.join(args.inp, "tables.json"), "r", encoding="utf-8") as f:
        tables = json.load(f)
    which = [w.strip() for w in args.figures.split(",")] if args.figures else npl.FIGURES
    formats = [e.strip() for e in args.formats.split(",") if e.strip()]
    for p in npl.make_figures(summaries, tables, args.out, dpi=args.dpi, formats=formats,
                              which=which):
        print("[save] {}".format(p))


def cmd_selfcheck(args):
    sys.exit(0 if ns.regression_check() else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="compute per-frame records")
    p.add_argument("--dataset", type=_dataset, required=True,
                   help="semantickitti | nuscenes | waymo")
    p.add_argument("--raw-root", required=True,
                   help="dataset root holding the sparse raw scans and labels "
                        "(layout in native_data.py)")
    p.add_argument("--dense-root", default=None,
                   help="root holding the dense completed clouds and their labels, same "
                        "relative layout (default: --raw-root)")
    p.add_argument("--frames", default="paper",
                   help="'paper', 'sample:N' or a file with one frame uid per line")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--shard", type=int, default=0,
                   help="process the frames with index %% num_shards == shard")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-new", type=int, default=-1,
                   help="stop after this many newly computed frames (-1: no limit)")
    p.add_argument("--sk-seqs", default=None,
                   help="SemanticKITTI sequences to discover (default: those of the frame "
                        "list; '08' for sample:N)")
    p.add_argument("--nusc-info", default=None,
                   help="nuScenes mmdet3d info file (default: <raw-root>/{})".format(nd.NUSCENES_INFO))
    p.add_argument("--overwrite", action="store_true", help="recompute existing records")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("aggregate", help="dataset summaries and table numbers")
    p.add_argument("--in", dest="inp", required=True,
                   help="directory with <dataset>/<uid>.json records")
    p.add_argument("--out", default=None, help="output directory (default: --in)")
    p.add_argument("--datasets", type=lambda s: [_dataset(x) for x in s.split(",") if x.strip()],
                   default=None, help="comma list (default: all present under --in)")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("plot", help="render the figures")
    p.add_argument("--in", dest="inp", required=True,
                   help="directory with summary_<dataset>.json and tables.json")
    p.add_argument("--out", required=True, help="figure directory")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--formats", default="pdf,png")
    p.add_argument("--figures", default=None,
                   help="comma subset of: " + ", ".join(
                       ["band_profiles", "gap_heatmap", "class_retention", "rolloff_hfshare",
                        "b0_trend", "qprofiles"]))
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("selfcheck", help="KPM vs exact eigendecomposition on synthetic graphs")
    p.set_defaults(func=cmd_selfcheck)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
