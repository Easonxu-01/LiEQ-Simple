# Changelog

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
