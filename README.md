# LiEqualizer

**Seeing the Lost Spectrum of Sparse LiDARs via Spectral-Invariant Distillation**

Project page: **https://easonxu-01.github.io/LiEQ-Simple/**

> *Different LiDARs do not see different worlds; they see different spectra of the same world.*

Read a LiDAR scan as a spectrum and the sensor gap becomes a band problem. Sparse sampling keeps a scan's coarse
shape but **weakens real detail** and **invents false detail** — the second of which a filter cannot do, but a
sampler can. A teacher trained on densified scenes still holds the bands a sparse scan lost, so a student can be
taught to match them.

LiEqualizer attaches two training-time distillation branches to the baseline pipeline:

- a **low-frequency branch** on the encoder's tri-perspective-view planes, which restores what sampling weakened;
- a **high-frequency branch** on the decoder's common-support voxel logits, which removes what sampling faked.

Each acts only where the frozen teacher's own response marks something to restore or remove.

Across SemanticKITTI, Waymo and nuScenes this raises zero-shot cross-sensor mIoU by up to **11.5 points**, and it
does so **without changing the network, its parameter count or its inference cost** — the losses act during
training only, so the deployed model is the baseline network.

---

## What is in this repository

A trimmed release: the projection-branch code needed to train and evaluate LiEqualizer, plus the spectral
measurement protocol behind the paper's diagnosis. It is a sibling of
[LiSeer-Simple](https://github.com/Easonxu-01/LiSeer-Simple), on which it builds.

```
projects/
  configs/pointocc/
    pointtpv_{semantickitti,nusc,waymo}_seg_random.py     student configs (paper settings written in)
    pointtpv_{semantickitti,nusc,waymo}_teacher_dense.py  dense teacher configs, as trained
    pointtpv_realdata_infer.py                            inference on raw scans
  unilidar_plugin/
    pointocc_model/Spectraldistiller.py                   the LF/HF spectral distillers
    pointocc_model/pointtpv_seg.py                        model: LF branch, teacher forward, gradient-ratio control
    pointocc_model/tpv_aggregator.py                      HF branch and dark-knowledge wiring
    pointocc_model/vanillakd.py                           temperature-KL term
    datasets/pipelines/livb_instance_bank.py              cross-frame Living Being instance paste (SemanticKITTI)
    occupancy/apis/teacher_protocol.py                    refuses to distil against a mismatched teacher
    datasets/coarse_mapping.py                            the shared coarse taxonomy, single source of truth
docs/                                                     project page + install / data / train guides
tools/                                                    data preparation and training entry points
```

## Getting started

1. [`docs/install.md`](docs/install.md) — environment. Same as the baseline; LiEqualizer adds no CUDA extension
   and no new runtime dependency.
2. [`docs/prepare_data.md`](docs/prepare_data.md) — raw datasets, dense point clouds, dense labels. The dense data
   is **required**, because the teacher is trained on it.
3. [`docs/trainval.md`](docs/trainval.md) — the two training stages (teacher, then student), evaluation, and the
   spectral measurement protocol.

The released configs carry **the exact hyper-parameters the paper's numbers were produced with**, including the
distillation switches and the GPU layout. Nothing has to be passed on the command line to reproduce them.

## Two things worth knowing before you start

**Checkpoints are not interchangeable with LiSeer's.** The model code here differs from the public LiSeer release
(the distillation branches change some module shapes), so a LiSeer checkpoint will not load cleanly into this
tree, and vice versa.

**Changing the GPU layout changes the result.** The per-GPU batch size and gradient accumulation in these configs
are the ones the paper used. Changing how many GPUs you train on without adjusting them also changes the
batch-normalization statistics and the iteration-counted schedules, and the result is then no longer comparable
with the paper's; every run in the paper's tables shares one effective batch of 32.

## Training layout

The student configs assume 8 GPUs, 1 sample per GPU and 4 accumulation steps (effective batch 32). On
SemanticKITTI, 1 sample per GPU needs about 21 GB of GPU memory. On a different layout, keep the effective batch
at 32 and rescale the iteration-based keys by the change in iterations per epoch:

- `lr_config.warmup_iters`
- the EMA hook `momentum`
- `dynamic_loss_balance.interval` and `l0_burnin_iters`
- the distillation `warmup_iters`
- the curriculum hook `alpha_step` and `alpha_momentum`

The header of each student config gives the equivalent values for 4 GPUs x 4 samples x 2 accumulation steps.
The learning rate is not scaled with the number of GPUs.

## Teacher checkpoints

`teacher_dense_sk.pth`, `teacher_dense_nu.pth` and `teacher_dense_waymo.pth` are the teachers of the paper's
students. Place them at `ckpts/teacher_dense_{sk,nu,waymo}.pth`, the paths the student configs expect.

| Dataset | Checkpoint |
|---|---|
| SemanticKITTI | epoch 15, the highest point-level mIoU on the validation split under the input the teacher receives during distillation |
| nuScenes | epoch 24, the best validation epoch |
| Waymo | epoch 24, the last one; no validation split is used while training the Waymo teacher |

The files hold weights and label-mapping metadata only, with no optimizer state: they serve distillation and
evaluation, not resuming training. Training refuses to start if a teacher's label protocol does not match the
student's (`occupancy/apis/teacher_protocol.py`).

## Living Being instance bank (SemanticKITTI only)

The SemanticKITTI student and teacher configs paste Living Being instances across frames
(`LivBInstanceBankPaste`, probability 0.3, one instance per frame) from a bank read at
`data/livb_bank_sk_train_v1`. The bank is not distributed; it is built from your own copy of SemanticKITTI.
Without it, set `prob=0` in `LivBInstanceBankPaste`.

**What goes in.** Only the training sequences (00-07, 09, 10); the loader refuses sequence 08. From every
labelled training scan, Living Being instances (coarse class 5) are cut out by instance id, together with the
dense completion points within 0.4 m.

**Filters.** At least 20 real points; range 2.5-48 m; a dominant connected component (at least 85 % of the
points at 0.5 m linkage); height 0.9-2.3 m; horizontal extent at most 2.6 m; flat local ground underneath
(classes 1/7, at least 10 support points, p90 - p10 spread at most 0.35 m); observations of the same instance at
least 5 frames apart.

**File format.** The transform reads exactly these files:

- `meta.json`: at least `split='train'`, `sequences` and `obs_fields`;
- `raw_xyz.npy`, `raw_sem.npy`, `raw_rem.npy`: the concatenated real points of all observations;
- `dense_xyz.npy`, `dense_sem.npy`: the concatenated dense points;
- one `obs_<field>.npy` per observation field: `seq`, `frame`, `inst_id`, `inst_index`, `sem`, `n_raw`,
  `n_dense`, `raw_off`, `dense_off`, `ground_z`, `cx`, `cy`, `rho`, `phi`, `z_min`, `z_max`, `height`,
  `extent_major`, `extent_minor`, `radius_xy`, `n_ground`, `ground_spread`, `cc_frac`, `n_raw_dropped`.

## Citation

```bibtex
@article{xu2026liequalizer,
  title   = {LiEqualizer: Seeing the Lost Spectrum of Sparse LiDARs via Spectral-Invariant Distillation},
  author  = {Xu, Zikun and Yang, Shuocheng and Wang, Jianqiang and Li, Keqiang
             and Lee, Gim Hee and Xu, Shaobing},
  year    = {2026},
  note    = {Preprint}
}
```

## License

See [LICENSE](LICENSE).
