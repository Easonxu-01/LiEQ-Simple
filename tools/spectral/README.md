# Spectral measurement suite

These scripts compute the spectral quantities reported in the paper. They take models only as checkpoints you pass on the command line, one checkpoint per role. No checkpoint is bundled or referenced.

| directory | what it computes | needs |
|---|---|---|
| `dump/` | Per-frame spectral summaries of the TPV feature planes and the voxel logits, for the teacher input variants and the two models. | GPU, the model code of this repository |
| `analysis/` | The reported quantities, computed from one dump directory. | CPU |
| `fbd/` | The frequency-band distance (FBD) of the voxel logits. | GPU for the forward pass, CPU from saved energies |
| `native/` | Data-only spectral statistics of sparse scans vs dense completions. No model is involved. | CPU |
| `common/` | Shared operators (DCT bands, support Laplacian, Chebyshev–Jackson band-pass, CKA, plane cells). | – |

Run everything from the repository root, in the training environment.

## Roles

- **teacher** (`--teacher-config`, `--teacher-checkpoint`): the dense teacher. Three input variants are generated internally:
  - `T_union`: the dense union, randomly truncated to the 500k-point serving cap. This is the reference of every comparison, and its distillation field defines the support Ω_T.
  - `T_union2`: the same union truncated with another seed. This is the noise ceiling.
  - `T_raw`: the teacher on the sparse raw scan.
- **student** (`--model student=CONFIG:CHECKPOINT`): the distilled model under study.
- **baseline** (`--model baseline=CONFIG:CHECKPOINT`): the model it is compared with.

Each model is built from the config it was trained with. Compare models dumped in the same run, so that they share frames and teacher inputs; the analysis never combines runs.

## Frames

- **Rule:** 200 frames evenly spaced over the validation split, `round(linspace(0, N-1, 200))`. The development frames `{0, N//3, 2N//3}` are dumped but held out of the statistics, so only frame 0 is removed from the grid and 199 frames are evaluated. `dump/frames.py` implements this rule.
- **Raw frames:** 13 frames (the development frames plus 10 evenly spaced grid frames) also keep their full tensors. The gate test and the cross-term use them.
- **FBD:** uses all 200 grid frames.
- **`native/`:** uses its own frame lists, in `native/frames/`.

## Workflow

```
# 1. dump (GPU; run --stage tpv and --stage logits with otherwise identical arguments; --shard i/n to split)
python tools/spectral/dump/run_dump.py --stage tpv --dataset semantickitti \
    --teacher-config projects/configs/pointocc/pointtpv_semantickitti_teacher_dense.py --teacher-checkpoint TEACHER.pth \
    --model baseline=BASELINE_CONFIG.py:BASELINE.pth \
    --model student=projects/configs/pointocc/pointtpv_semantickitti_seg_random.py:STUDENT.pth \
    --frames-rule paper --raw-frames paper --rho-crop-rows 16 --out DUMP
python tools/spectral/dump/run_dump.py --stage logits ...same arguments... --out DUMP
# (the tpv stage also writes the gate domains of the raw frames, dump200/_p2dom/, used by gate_test.py)

# 2. reported quantities (CPU)
python tools/spectral/analysis/run_analysis.py --dump DUMP --compare student baseline --out RESULTS
python tools/spectral/analysis/gate_test.py   --dump DUMP --out RESULTS/appendix_d_gate_test.json
python tools/spectral/analysis/crossterm.py   --dump DUMP --out RESULTS

# 3. FBD (GPU forward, then CPU)
python tools/spectral/fbd/run_fbd.py --dataset semantickitti --teacher-checkpoint TEACHER.pth \
    --model baseline=CONFIG.py:BASELINE.pth --model student=CONFIG.py:STUDENT.pth \
    --num-frames 200 --seed 0 --out FBD
python tools/spectral/fbd/run_fbd.py --from-dump FBD

# 4. data-only statistics (CPU, no model)
python tools/spectral/native/run_native.py stats --dataset semantickitti --raw-root RAW --frames paper --out NATIVE
python tools/spectral/native/run_native.py aggregate --in NATIVE
python tools/spectral/native/run_native.py plot --in NATIVE --out NATIVE/figs
```

`--help` on each script lists every option. `fbd/README.md` documents the FBD tool, and the docstrings of `dump/run_dump.py`, `dump/frames.py` and `native/native_data.py` give the expected data layout.

## Outputs and where they appear in the paper

| file | paper |
|---|---|
| `fig3_band_cka.json` | Fig. 3: band-wise CKA divided by the ceiling, feature planes and voxel logits; gap-closed strip |
| `fig4_energy.json` | Fig. 4: real and pseudo energy per band in ceiling units, and the paired change |
| `planes_boundary.json` | spurious and missed boundary rates per TPV plane |
| `appendix_d_spurious_hf.json`, `appendix_d_pseudo_hf.json`, `appendix_d_paired_sparse.json` | Appendix D: high-frequency excess, pseudo-energy shares, comparison with the teacher-on-sparse control |
| `appendix_d_gate_test.json`, `appendix_d_crossterm.json` | Appendix D: gate test and cross-term |
| FBD output | frequency-band distance |
| `native/` figures and `tables.json` | data-only spectral figures and table numbers |

## Reproducibility notes

- Bands: 11 bands with edges −0.05, 0.01, 0.03, 0.08, 0.15, 0.25, 0.40, 0.60, 0.85, 1.15, 1.50, 2.02.
  - Feature planes use tight-frame raised-cosine windows on the DCT-II.
  - Voxel logits use a Chebyshev–Jackson expansion with K = 1024 on the 6-neighbour support Laplacian.
- The feature planes drop the first 16 range rows (ρ < 3.4 m), set with `--rho-crop-rows`. The voxel logits are not cropped.
- Seeds and determinism:
  - The truncations of the teacher input are drawn from seeded CPU generators and are the same on every machine.
  - Forward passes are seeded. Floating-point differences between GPU models can still change the last digits.
  - The FBD student inputs are drawn on the GPU; `fbd/README.md` says when they are reproduced exactly.
  - TF32 is set per dataset as documented in `run_dump.py` and `fbd/README.md`.
- `native/` needs open3d to read `.ply` dense completions. A single frame takes several minutes on CPU.
- CPU self-tests of the operators:
  - `python tools/spectral/fbd/run_fbd.py --self-test`
  - `python tools/spectral/native/run_native.py selfcheck`
  - from `tools/spectral/`: `python -m common.tpv_metrics --self-test`, and the same for `common.graph_spectral` and `common.logits_core`.
