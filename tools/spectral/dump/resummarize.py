#!/usr/bin/env python
"""Recompute per-frame summaries from saved full tensors (the raw frames written by run_dump.py), on CPU or GPU.

  python tools/spectral/dump/resummarize.py --stage tpv --dump OUT/TPV/dump200 --frames 0,20 --out DIR [--device cpu]
  python tools/spectral/dump/resummarize.py --stage logits --dump OUT/LOGITS/dump200 --frames 0,20 --out DIR \\
      --gate-config <student model config> [--device cpu]

Writes DIR/frame_XXXXXX.json with the schema of run_dump.py. The mode defaults to 'full' (raw frames); it is recorded
only. Results equal those of the dump stage up to floating-point differences between devices.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['tpv', 'logits'])
    ap.add_argument('--dump', required=True, help='the dump200 directory (<variant>/frame_XXXXXX.npz and _masks/)')
    ap.add_argument('--frames', required=True, help='comma-separated frame indices')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--mode', default='full', choices=['full', 'eval'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--rho-crop-rows', type=int, default=16)
    ap.add_argument('--gate-config', default=None, help='logits stage: the student model config (HF gate)')
    ap.add_argument('--repo-root', default=str(HERE.parents[2]))
    a = ap.parse_args(argv)
    from common import model_io
    model_io.set_repo_root(a.repo_root)
    import tpv_summary as TS
    TS.set_rho_crop_rows(a.rho_crop_rows)
    mod = TS
    if a.stage == 'logits':
        if not a.gate_config:
            raise SystemExit('--gate-config is required for the logits stage')
        import logits_summary as LS
        LS.set_gate_config(a.gate_config)
        mod = LS
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for fi in [int(x) for x in a.frames.split(',') if x.strip()]:
        t0 = time.time()
        V, masks = mod.load_frame(a.dump, fi)
        res = mod.summarize_frame(V, masks, device=a.device, mode=a.mode)
        (out / f'frame_{fi:06d}.json').write_text(json.dumps(mod._jsonable(res)))
        print(f'frame {fi}: {time.time() - t0:.1f}s -> {out / f"frame_{fi:06d}.json"}', flush=True)


if __name__ == '__main__':
    main()
