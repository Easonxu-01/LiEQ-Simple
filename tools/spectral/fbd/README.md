# Frequency-band distance (FBD)

This tool computes the frequency-band distance (FBD) between the voxel logit field of a student model and that of the dense teacher. It is the quantity the paper reports when comparing the student before and after spectral distillation.

It reports one number per student model, and nothing else. The default setting is 200 frames and seed 0. Lower is better, and 0 means the student reproduces the teacher's band distribution.

## Files

| file | content |
|---|---|
| `run_fbd.py` | command line: the GPU pass, the offline computation from saved energies, and a CPU self-test |
| `field_pass.py` | GPU pass: data, teacher input cap, student input simulation, teacher and student fields, the graph operator, the Chebyshev moments |
| `fbd_math.py` | NumPy only: the fine spectral grid, the Chebyshev–Jackson projectors, the FBD and its aggregation over frames |

Every number-defining constant is a module constant in `fbd_math.py` or `field_pass.py`. None of them is a command-line option.

## Usage

Run from the repository root. The GPU pass uses one GPU; select it with `CUDA_VISIBLE_DEVICES`.

```bash
python tools/spectral/fbd/run_fbd.py --dataset semantickitti \
    --teacher-checkpoint TEACHER_CKPT \
    --model baseline=projects/configs/pointocc/pointtpv_semantickitti_seg_random.py:BASELINE_CKPT \
    --model student=projects/configs/pointocc/pointtpv_semantickitti_seg_random.py:STUDENT_CKPT \
    --num-frames 200 --seed 0 --out OUT
```

- **`--teacher-checkpoint`:** the dense teacher.
- **`--model NAME=CONFIG:CHECKPOINT`:** one student per flag. `NAME` is only a label for the output.
- **`--teacher-config`:** the config the teacher network and the data pipeline are built from. It defaults to the `seg_random` (student training) config of `--dataset`.
  - The teacher is built the way distillation training builds it: the student architecture, loaded with the teacher weights.
  - Dense-teacher training configs (`*_teacher_dense.py`) build a different module set. They are refused.
- **`--data-root`:** overrides `data.test.data_root` of the config. The default is the config's own path.
  - For Waymo, the path must contain the lowercase substring `waymo`. The loader uses it to select the Waymo label format; a symbolic link with such a name works.
- **`--frame-ids 0,20,41`:** restricts the run to a subset of the evenly spaced frames, for quick checks.
- **Resuming:** an interrupted run resumes. Frames whose energies exist for every model are skipped.

To recompute the FBD from saved energies (CPU only, no data or checkpoints needed):

```bash
python tools/spectral/fbd/run_fbd.py --from-dump OUT          # writes OUT/fbd.json, OUT/fbd.csv
python tools/spectral/fbd/run_fbd.py --self-test               # CPU check of the operator and the FBD numerics
```

## What is measured

The procedure for each evaluated frame is below.

1. **Frames.** The evaluated frames are `num_frames` evenly spaced frames of the test split, including the first and last frame: `sorted(set(round(linspace(0, N - 1, num_frames))))`. For SemanticKITTI (sequence 08, N = 4071) this gives frames 0, 20, 41, …, 4070.
2. **Dense input.** The dense input is the original scan plus the dense completion. This is the teacher's serving input during distillation, and the source cloud of the student's sensor simulation. Training augmentations are not applied.
3. **Teacher field.**
   - The dense input is capped to `model.teacher_serve_max_points` of the config (500 000) by a uniform permutation. It is drawn from a CPU generator seeded with `seed * 1000003 + frame`.
   - The teacher's voxel logit field is taken on the voxels of the capped input; this set is the support Ω_T.
4. **Student input.**
   - Each student receives a simulated sparse scan, produced by the model's own sensor simulator (`_sampling`: ray casting with randomly drawn sensor parameters) from the full dense input.
   - Python, NumPy and torch (CPU and CUDA) are seeded with `seed * 1000003 + frame + 17` immediately before the call, and the first simulated view is kept.
   - The simulator's minority copy-paste settings are fixed to probability 0.15, classes (5, 6) and 2 copies, the simulator defaults (`field_pass.SIMULATOR_SETTINGS`). Training configs may set other values for training.
   - The inputs are written once to `OUT/student_inputs/<frame>.pt` (or `--sample-cache DIR`) and re-read for every model, so all students are evaluated on identical inputs.
5. **Student field.** The student runs its deployment path: the sensor head, then FiLM conditioning on its own sensor prediction. Its field is queried at the same teacher voxels, so both fields live on Ω_T.
6. **Spectrum.**
   - The graph is the 6-neighbour voxel grid restricted to Ω_T, with the Laplacian of the high-frequency distiller (`hf_operator="support"` in `Spectraldistiller.py`): `L x = m ⊙ (deg ⊙ x − Σ_neighbours x) / 6`. Its spectrum lies in [0, 2).
   - For every field channel, the Chebyshev moments `μ_n = ⟨x, T_n(L̃) x⟩` are computed in float64, with `L̃ = (2/2.02) L − I` and `n = 0…1024`.
   - The moments are projected onto 507 fine λ-bins with Jackson-damped Chebyshev projectors.
7. **FBD.**
   - The fine-bin energies are summed over the channels.
   - Low/mid/high bands are cut at the 0.5 and 0.9 quantiles of the teacher's cumulative energy; the student uses the same cuts.
   - With `E_β` the band share of each field's total energy:

     `FBD_β = |E_β(student) − E_β(teacher)| / (E_β(teacher) + 1e-8)`

     `FBD = 0.2·FBD_low + 0.3·FBD_mid + 0.5·FBD_high`
8. **Reported value.** The reported value is the mean of the per-frame FBD. Frames with zero teacher energy would be skipped.

## Outputs

| path | content |
|---|---|
| `OUT/fbd.json`, `OUT/fbd.csv` | per model: FBD (mean over frames) and the number of frames; `fbd.json` also has the per-frame values |
| `OUT/energies/<name>/frame_XXXXXX.npz` | `fine_edges` [508]; `teacher_energy` and `student_energy` [C, 507] (per-channel fine-bin energies); `support_voxels` |
| `OUT/student_inputs/XXXXXX.pt` | simulated student input (`points`, `grid`, `grid_vox`, `sensor`) |
| `OUT/manifest.json`, `OUT/frames.jsonl`, `OUT/progress.log` | settings, checkpoint md5, environment, per-frame counts and FBD |

## Reproducibility notes

- **Stage 1, energies to FBD (`fbd_math.py`).** This stage is exact.
  - Recomputing from the per-frame energies of the original measurement reproduces the reported means to the last bit (SemanticKITTI 1.167 → 0.100, student before → after distillation).
  - The operator and moment code reproduces the original implementation bit for bit on CPU.
- **Stage 2, student inputs.**
  - The simulator draws random numbers on the GPU. The paper's SemanticKITTI inputs were generated on an RTX 4090 with PyTorch 1.10.1 (CUDA 11.3). With the same PyTorch version, regenerating them on an A800 gave bit-identical inputs (checked on three frames).
  - A different PyTorch version may draw a different random stream. The regenerated inputs are then different (but equally distributed) simulated scans, and the per-frame FBD changes within sampling variability, not systematically.
  - To reuse a given set of inputs, pass its directory with `--sample-cache`.
- **Stage 3, forward passes.**
  - Each forward pass is seeded with `seed * 1000003 + frame`; the encoder shuffles points before pooling.
  - Floating-point non-determinism on the GPU can still leave differences of order 1e-5 (relative) in the per-frame values.
- **TF32.** TF32 is enabled for SemanticKITTI and nuScenes, the PyTorch 1.10 default under which the paper's values were produced, and disabled for Waymo.
