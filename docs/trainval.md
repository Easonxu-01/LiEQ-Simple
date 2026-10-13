# Training & Evaluation

All commands assume the repo root as the working directory and `export PYTHONPATH="."` (the `run.sh` / `run_eval.sh`
wrappers set this for you). Edit the `conda activate <env>` line inside those wrappers to match the env name from
[`install.md`](install.md).

LiEqualizer is a **training-time** objective. The deployed network is the baseline network: the distillation branches
add no trainable parameter, are never invoked at inference, and leave the checkpoint's architecture and inference cost
unchanged. Evaluation and deployment therefore work exactly as they do for the baseline.

---

## Two stages

Reproducing a LiEqualizer model takes two training runs:

1. **Teacher** — the same architecture trained on the *densified* scenes (`prepare_data.md`, Step 2–3). It is frozen
   afterwards and is never updated by the distillation loss.
2. **Student** — the baseline randomized-sampling model trained on the sensor's own sparse scans, with the spectral
   distillation objective attached and the frozen teacher loaded alongside.

The teacher must be trained under the **same coarse taxonomy** as the student. Mixing a teacher trained under a
different label mapping with a student trained under the current one will silently change the meaning of the
per-channel quantities the loss compares.

---

## Configs

Per-dataset configs:

- `projects/configs/pointocc/pointtpv_semantickitti_seg_random.py`
- `projects/configs/pointocc/pointtpv_nusc_seg_random.py`
- `projects/configs/pointocc/pointtpv_waymo_seg_random.py`

Each one ships with the **exact hyper-parameters used for the paper's numbers written in as defaults** — including the
distillation switches, the gate settings and the scale schedule. Nothing has to be passed on the command line to
reproduce the reported results. A comment block at the top of each config records which table and which rows that
configuration corresponds to.

> **Changing the GPU layout changes the result.** The per-GPU batch size and gradient accumulation in these configs are
> the ones the paper used. Changing how many GPUs you train on without adjusting them changes the batch-normalization
> statistics and the iteration-counted schedules, which is why the paper's ablation marks the runs trained with one
> sample per GPU separately and does not compare them across that boundary.

### Pointing the student at its teacher

The student config loads a frozen teacher checkpoint. Set its path before training:

```python
# projects/configs/pointocc/pointtpv_<dataset>_seg_random.py
d2skd_load_from = 'path/to/teacher_<dataset>.pth'   # frozen dense-trained teacher
```

Teacher checkpoints are released together with the paper; until then, train your own with Stage 1 above.

---

## Training

Train with N GPUs:

```shell
bash run.sh ./projects/configs/pointocc/pointtpv_semantickitti_seg_random.py 8
```

`run.sh` forwards extra args to `tools/dist_train.sh`, so you can append e.g. `--cfg-options ...`. Note that the
released configs already contain the paper's values, so `--cfg-options` is only needed if you are deliberately
deviating from them.

### Turning the branches off

The two branches can be disabled independently, which is how the ablation rows were produced:

- low-frequency branch only — the encoder-side term that restores what sampling weakened;
- high-frequency branch only — the decoder-side term that removes what sampling faked, together with the soft-label
  term that shares its switch;
- both, with the teacher gate disabled — matching everywhere instead of only where the teacher marks something.

The switch names are documented in the comment block at the top of each config.

---

## Evaluation

```shell
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM
```

Because the distillation modules are not part of the deployed model, a LiEqualizer checkpoint is evaluated with exactly
the same command, and the same config, as a baseline checkpoint.

**On test-time augmentation.** The main table's LiEqualizer rows use test-time augmentation (8 views: rotations of 0,
90, 180 and 270° about the vertical axis, each with and without a mirror flip, per-point softmax averaged; on nuScenes
additionally accumulating the 4 preceding sweeps, ego-motion compensated, scoring only key-frame points). That is an
**evaluation setting, not part of the method** — the ablation and the comparison tables are reported single-pass
without it. Make sure you know which setting you are comparing against.

---

## Inference on real / custom LiDAR

To run inference on raw `.bin` scans (no labels) listed by a `list.txt`, use the real-data config and point `load_from`
at your checkpoint:

```shell
bash run_eval.sh ./projects/configs/pointocc/pointtpv_realdata_infer.py $PATH_TO_CKPT $GPU_NUM
```

Set `REAL_DATA_NAME` / `REAL_DATA_ROOTS` and the prediction output dir inside `pointtpv_realdata_infer.py`. Predictions
(and optional `.ply`) are written to `test_cfg.inference_pred_dir`.

---

## The spectral measurement protocol

The paper's diagnosis — what sparse sampling does to a signal, and whether distillation repaired it — is released as a
protocol rather than as a one-off script, so it can be run on any pair of sensors.

It reports two families of quantities:

- **Native suite**, on raw point clouds: high-frequency retention per signal and per class, the spectral roll-off, and
  the band-energy gap between a dense and a sparse observation of the same scene.
- **Voxel suite**, on the grid the network actually reads: the high-frequency energy ratio against the teacher, the
  per-band discrepancy, and the flatness of the residual profile.

Both are computed from the same graph-spectral machinery the losses use, so a number that moves in the diagnosis and a
number that moves in the loss are the same quantity. The entry points and their arguments are documented alongside the
protocol code.

---

## Visualization

Generate predictions to a directory and render them:

```shell
bash run_eval.sh $PATH_TO_CFG $PATH_TO_CKPT $GPU_NUM --show --show-dir $PATH
```
