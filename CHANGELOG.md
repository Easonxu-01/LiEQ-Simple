# Changelog

## 2026-11 — final training code, layouts and teacher checkpoints

### Changed

- **Model code updated to the final training code.**
  - *Boundary bookkeeping fix.* `shrink_num_sparse_points` (`datasets/pipelines/PointoccMapping.py`) keeps
    `results['num_sparse_points']` in sync whenever rows are removed: by the ego-vehicle point filter, and by the
    Waymo label-0 removal in `loading_voxels_sk.py`. This changes the purified input on nuScenes and Waymo;
    SemanticKITTI is unchanged.
  - *Loss balancer.* `dynamic_loss_balance` gains an optional closed-loop GradNorm controller
    (`controller='gradnorm'`, `normalize='grad_budget'`) and optional diagnostics (`trace_file`, `log_grad_stats`).
    The default `controller='legacy'` behaves exactly as before.
  - *Minority-class recipe keys.* New keys `minority_mixup_prob`, `minority_mixup_classes`,
    `minority_mixup_num_add` and `consistency_minority_weight`. Their defaults equal the former fixed constants.
- **Configs.**
  - All three student configs use 8 GPUs x 1 sample per GPU x 4 accumulation steps, an effective batch of 32.
    Iteration-based values in these files are set for this layout.
  - The SemanticKITTI dense-teacher config describes the released SemanticKITTI teacher. It trains on the
    dense + sparse union input, with the Living Being paste and densification; its layout is 8 x 1 x 2, an
    effective batch of 16.

### Added

- **Data transform `LivBInstanceBankPaste`** (`datasets/pipelines/livb_instance_bank.py`). It pastes Living Being
  instances across frames from an offline bank built from the SemanticKITTI training sequences only. It must sit
  after `VoxelClassMapping` and before `PointsegMapping`; with `prob=0` it is a no-op. The SemanticKITTI student
  and teacher configs enable it; the nuScenes and Waymo configs do not use it. The bank itself is not
  distributed; see the README for how it is built and its file format.
- **Teacher checkpoints** for SemanticKITTI, nuScenes and Waymo; see the README.
- **Spectral measurement suite** (`tools/spectral/`, see its README). It contains the scripts that compute the
  paper's spectral quantities, and writes only the quantities the paper reports.
  - `dump/`: per-frame spectral summaries of the TPV feature planes and the voxel logits (GPU).
  - `analysis/`: the reported quantities, computed from one dump (CPU).
  - `fbd/`: the frequency-band discrepancy (FBD) of the voxel logits.
  - `native/`: data-only statistics of sparse scans against dense completions. No model is involved.
  - Models are passed only as checkpoints on the command line, one per role: `--teacher-checkpoint`,
    `--model student=CONFIG:CHECKPOINT`, `--model baseline=CONFIG:CHECKPOINT`. No checkpoint is bundled or
    referenced.

## 2026-10 — initial LiEqualizer release

First public release of the LiEqualizer code, built on the public
[LiSeer-Simple](https://github.com/Easonxu-01/LiSeer-Simple) tree.

### Added

- **Spectral distillers** (`projects/unilidar_plugin/pointocc_model/Spectraldistiller.py`): the low- and
  high-frequency branches, with teacher-derived gates, the support-subgraph operator, diffusion responses and the
  match / suppress terms.
- **Dark-knowledge term** (`pointocc_model/vanillakd.py`): the temperature-KL loss used when `hybrid_dark_kd` is
  on; usable standalone as plain KD.
- **Teacher protocol guard** (`occupancy/apis/teacher_protocol.py`): refuses to distil when the teacher checkpoint
  was trained under a different 8-class mapping. Training now fails loudly rather than distilling against a
  randomly initialised teacher.
- **Dense teacher configs** for the three datasets, written with the layouts the teachers were actually trained
  under.
- **Coarse taxonomy** (`datasets/coarse_mapping.py`): one source of truth for the shared label space.
- EMA evaluation hook, minority-class purification transform, and the hooks the released configs depend on.

### Changed

- The three student configs now carry the **final hyper-parameters written in as defaults** — the distillation
  switches, the gate settings, the scale schedule and the GPU layout. Reproducing the paper's numbers needs no
  command-line overrides.
- Model and aggregator build and call the distillers, run the frozen teacher forward, and apply the
  gradient-ratio controller.
- Ego-vehicle returns within 1 m are filtered in both train and test pipelines.
- Nine nuScenes raw classes are routed into the coarse taxonomy instead of being dropped.

### Removed

- The LiSeer-only sampler refactor (`sampling_common.py`, `solidstate_sampler.py`, the Risley example configs and
  the sampler test suite). They are incompatible with the model code here, and a silent fallback to mechanical
  spinning-LiDAR sampling would be worse than their absence.

### Notes

- Checkpoints are **not** interchangeable with the public LiSeer release.
- The deployed model is unchanged by this release: the distillation branches add no trainable parameter and are
  never invoked at inference.
