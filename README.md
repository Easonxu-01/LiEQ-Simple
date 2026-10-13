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
batch-normalization statistics and the iteration-counted schedules, which is why the paper's ablation keeps runs
trained under different layouts in separate groups and does not compare across that boundary.

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
