import torch
import torch.distributed as dist
from mmdet3d.models import builder
from mmdet.models import DETECTORS
from mmdet3d.models.detectors import CenterPoint
import yaml
import numpy as np
import copy
import random
import math
import time
import open3d as o3d
import os
import os.path as osp
from mmcv.parallel import DataContainer
from torch_scatter import scatter_min
from .Spectraldistiller import TPVLowFreqSpectralDistiller
from .tpv_aggregator import _cb_adc_consistency_loss, cb_adc_consistency_loss_ckpt
import torch.nn.functional as F

@DETECTORS.register_module()
class PointTPV_Seg(CenterPoint):
    
    def __init__(self,
                 lidar_tokenizer=None,
                 lidar_backbone=None,
                 lidar_neck=None,
                 tpv_aggregator=None,
                 empty_idx = None,
                 random = False,
                 reset_random_prob = 0.0,
                 sensor=None,
                 # Physical perturbation suite. None/default = all off,
                 # bit-identical to the unpatched behaviour. The three terms are **stacked in parallel** on top of the
                 # sampler's existing az_phase / sigma(r)=1cm+0.1%r / U[0,8%) dropout;
                 # they neither replace nor modify any existing constant.
                 sampler_perturb=None,
                 consistency = False,
                 dense_train = False,
                 d2skd = False,
                 # Choice of the distillation loss (derived from the config's KDbaseline (bool); mutually exclusive):
                 #  'spectral'   low-frequency spectral distillation (existing path, default)
                 #  'vanilla_kd' plain-KD baseline of the paper: pointwise MSE on TPV features (see the vanillakd.py module docstring)
                 distill_mode='spectral',
                 kd_cfg=None,           # dict(T=, lambda_kd=, lambda_feat=)
                 cwd_cfg=None,          # dict(T=, lambda_cwd=), effective when distill_mode='cwd'
                 pvkd_cfg=None,         # dict(...), effective when distill_mode='pvkd'
                 # Low-frequency spectral distiller hyperparameters (None = defaults). Since v3, block_size=(bh,bw) is supported for
                 # block-local response alignment (targeting the high floor of raw_inter caused by global normalization).
                 lowfreq_distiller_cfg=None,
                 # [Ablation switch] Master switch of the LF branch (low-frequency TPV spectral distillation).
                 # Note: setting lowfreq_distiller_cfg to None does **not** turn this branch off --
                 # the construction site is `**(lowfreq_distiller_cfg or {})`, so None degenerates to constructing an
                 # LF distiller with all default parameters. Ablations must use this switch.
                 # Used together with enable_hf_distill of tpv_aggregator:
                 #   LF-only arm: enable_lf_distill=True,  enable_hf_distill=False
                 #               (HF and dark knowledge are both off, see the note in tpv_aggregator)
                 #   HF-only arm: enable_lf_distill=False, enable_hf_distill=True
                 enable_lf_distill=True,
                 dynamic_loss_balance=None,
                 random_yaw=True,
                 # Teacher evaluation protocol: uniformly truncate each sample to N points before forward/evaluation. During teacher
                 # **training** the input is uniformly truncated to 500k (see the dense branch of _sampling); when the teacher **serves**
                 # for distillation it is not truncated by default (see teacher_serve_max_points), and test has no truncation by
                 # default (SK dense val 1.03M = 2.06x the training density); the density mismatch amplifies the apparent magnitude
                 # of the "post-peak decline" (measured when retraining both teachers: SK with a 2.06x mismatch declined
                 # -4.2pp vs nusc with a 1.18x mismatch declined -0.6pp). None = old behaviour, no truncation.
                 test_max_points=None,
                 # Teacher training: per-sample random range (lo, hi) of the dense uniform truncation cap. None = fixed 500k.
                 # [Current state] All three teacher configs use None, i.e. a fixed 500000,
                 # the same protocol as test_max_points / teacher_serve_max_points.
                 # ⚠ max_points is a **per-sample** cap, not a per-GPU cap:
                 #   total points per GPU = samples_per_gpu × 500k.
                 # ⚠ [Correction] This place used to say "about 3.82 GB per 100k points, linear", and on that basis
                 #   the OOM at spg=4 was attributed to the point count -- **that attribution is wrong**;
                 #   the "linear" relation was only an artifact back-derived from two measurements at spg=1/2 with the same point count.
                 #   Broken down, only a small part of the per-sample memory scales with the point count:
                 #     .dense() + flatten copies of the three tokenizer TPV projections
                 #       (grid_size=[480,360,32], split=[16,16,16], C=256, fp32)
                 #       = 2.637 + 0.176 + 0.234 GB, plus one copy each = about 6.09 GB/sample
                 #       **determined by the grid and split, completely independent of the point count, purely linear in batch**
                 #     per-point activations of point_mlp @500k points = about 3.00 GB/sample  <- only this part scales with the point count
                 #   i.e. limiting the per-sample point count only controls about 1/3 of the tokenizer memory; relative to the
                 #   total single-sample forward+backward peak of 17.64 GB, the point cap only controls about 1/6.
                 #   This is the real cause why spg=1 runs and spg=4 is bound to blow up; it is not a point-count problem.
                 #   [Reference measurement: per-stage forward hooks (single GPU, spg=1)]
                 #     lidar_tokenizer +5.703 GB   lidar_backbone  +7.428 GB
                 #     lidar_neck      +0.019 GB   tpv_aggregator  +2.803 GB
                 #     total forward+backward peak 17.64 GB -> extrapolated to 75.67 GB at spg=4
                 #     > GPU capacity of 79.15 GB -> OOM.
                 #   (The earlier set "spg=1 -> 19.1 GB, spg=2 -> 40.4 GB, marginal 21.3 GB,
                 #    extrapolated 82.9 GB" was a rough two-point estimate with a different method; the conclusion is OOM as well;
                 #    when quoting numbers, use the per-stage measurement above.)
                 #   The fix is gradient checkpointing, numerically strictly unchanged, **not** lowering spg nor lowering the cap:
                 #     - lidar_backbone (Swin) with_cp=True: measured total peak at spg=4 is 51.42 GB,
                 #       which fits (about 24.2 GB less than 75.67). This is how the 51.42 figure was measured.
                 #     - the with_cp of cylinder_encoder.py is **a separate, independent switch** that only acts on
                 #       the three TPV projections. ⚠ It **does not save memory**: measured, enabling it actually adds +344.8 MB/sample;
                 #       the theoretical upper bound of per-branch checkpointing saves only 0.820 GB (the pool_xy branch alone accounts
                 #       for 86.5% and still has to be fully materialized during recomputation). See the _tpv_branch comment in that file. Default False, do not enable.
                 #   Reentrant checkpointing under DDP requires DDPStaticGraphHook,
                 #   see projects/unilidar_plugin/core/evaluation/static_graph_hook.py.
                 dense_cap_range=None,
                 # Input cap for the teacher when **serving** (distillation).
                 # Teacher training goes through the dense branch of _sampling, where the input is uniformly truncated to
                 # the dense cap (a fixed 500k for all three teachers, dense_cap_range=None); but the distillation forward calls
                 # model_t.extract_feat directly (unsampled raw cloud), measured at 0.77M-1.20M points per SK frame,
                 # i.e. a density the teacher never saw in training. None = keep the current behaviour (no truncation, bit-identical);
                 # an integer truncates with the same indexing before the teacher forward, reproducing the teacher's training input protocol.
                 # ⚠ Changes the teacher output => changes the distillation targets => not comparable with existing distillation experiments; must be decided deliberately first.
                 teacher_serve_max_points=None,
                 # 'all'       : truncate both the tokenizer input and the distillation support set (fully reproduces the teacher's
                 #               training protocol; the HF/KL support set shrinks accordingly, measured -11%~-13%)
                 # 'tokenizer' : truncate only the tokenizer input; the distillation support set stays as is (HF/KL targets unchanged)
                 teacher_serve_cap_scope='all',
                 # None = independent random sampling every time (same distribution as the density jitter of teacher training);
                 # integer = reproducible sampling by call index, convenient for comparing experiments.
                 teacher_serve_cap_seed=None,
                 **kwargs,
                 ):

        super().__init__()

        if lidar_tokenizer:
            self.lidar_tokenizer = builder.build_middle_encoder(lidar_tokenizer)
        if lidar_backbone:
            self.lidar_backbone = builder.build_backbone(lidar_backbone)
        if lidar_neck:
            self.lidar_neck = builder.build_neck(lidar_neck)
        if tpv_aggregator:
            self.tpv_aggregator = builder.build_voxel_encoder(tpv_aggregator)

        self.fp16_enabled = False
        self.empty_idx = empty_idx
        self.random = random
        self.reset_random_prob = float(reset_random_prob)
        self.sensor_config = sensor
        # Default all off; enable is the master switch; when False none of the three terms takes effect (and no random numbers are consumed).
        self.perturb_cfg = dict(
            enable=False,
            angular_jitter_mrad=0.0,
            range_noise_cm=0.0,
            dropout_p=0.0,
            dropout_exempt_minority=False,
        )
        if sampler_perturb:
            unknown = set(sampler_perturb) - set(self.perturb_cfg)
            assert not unknown, f'sampler_perturb unknown fields: {sorted(unknown)}'
            self.perturb_cfg.update(sampler_perturb)
        self.consistency = consistency
        self.dense_train = dense_train
        self.d2skd = d2skd
        self.random_yaw = bool(random_yaw)
        self.test_max_points = int(test_max_points) if test_max_points else None
        self.dense_cap_range = tuple(int(v) for v in dense_cap_range) if dense_cap_range else None
        self.teacher_serve_max_points = int(teacher_serve_max_points) if teacher_serve_max_points else None
        assert teacher_serve_cap_scope in ('all', 'tokenizer'), teacher_serve_cap_scope
        self.teacher_serve_cap_scope = teacher_serve_cap_scope
        self.teacher_serve_cap_seed = None if teacher_serve_cap_seed is None else int(teacher_serve_cap_seed)
        self._teacher_cap_calls = 0
        self._teacher_serve_logged = False
        self.dynamic_loss_balance = dynamic_loss_balance if dynamic_loss_balance is not None else {}
        self.sample_repeat_count = 2
        # GradNorm-style dynamic multi-loss balancing (group-level)
        # Only the three supervised groups whose gradients are measurable are balanced; consistency does not take part -- its magnitude
        # is explicitly controlled by consistency_gain (static) × hook warmup (schedule), and its gradient, which passes through the
        # checkpoint graph, cannot be measured with autograd.grad under torch<=1.10.
        # task order is fixed to keep state deterministic.
        self._balance_task_names = ['point', 'voxel', 'sensor']
        self._balance_eps = 1e-6
        self._balance_enabled = bool(self.dynamic_loss_balance.get('enabled', True))
        self._balance_alpha = float(self.dynamic_loss_balance.get('alpha', 0.5))
        self._balance_step = float(self.dynamic_loss_balance.get('step', 0.05))
        self._balance_momentum = float(self.dynamic_loss_balance.get('momentum', 0.2))
        self._balance_min_weight = float(self.dynamic_loss_balance.get('min_weight', 0.2))
        self._balance_max_weight = float(self.dynamic_loss_balance.get('max_weight', 5.0))
        self._balance_ema_beta = float(self.dynamic_loss_balance.get('ema_beta', 0.9))
        # Gradient norms are measured once every interval calls (the weights evolve slowly with momentum, so per-iter
        # measurement is pure overhead: each time one partial backward for each of the 3 groups).
        self._balance_interval = max(1, int(self.dynamic_loss_balance.get('interval', 10)))
        # The L0 reference is the mean loss over the first l0_burnin_iters calls, to avoid initialization noise setting a wrong reference;
        # a group does not take part in weight updates before its burn-in completes.
        self._balance_l0_burnin = max(1, int(self.dynamic_loss_balance.get('l0_burnin_iters', 50)))
        # The voxel-rise guard is a compensation heuristic tuned under the old "sensor gradient always 0" pathology; disabled by default.
        self._voxel_guard_enabled = bool(self.dynamic_loss_balance.get('voxel_guard_enabled', False))
        self._voxel_guard_tol = float(self.dynamic_loss_balance.get('voxel_guard_tol', 0.02))
        self._voxel_guard_gain = float(self.dynamic_loss_balance.get('voxel_guard_gain', 0.6))
        base_weights_cfg = self.dynamic_loss_balance.get('base_weights', {})
        task_min_cfg = self.dynamic_loss_balance.get('task_min_weights', {})
        task_max_cfg = self.dynamic_loss_balance.get('task_max_weights', {})
        base_w = []
        task_min_w = []
        task_max_w = []
        for name in self._balance_task_names:
            base_w.append(float(base_weights_cfg.get(name, 1.0)))
            task_min_w.append(float(task_min_cfg.get(name, self._balance_min_weight)))
            task_max_w.append(float(task_max_cfg.get(name, self._balance_max_weight)))
        # All balancer state is persistent=False: it is adaptive state that quickly re-adapts on resume;
        # this also avoids shape conflicts caused by a change in the number of task groups (old checkpoints have 4 groups).
        num_tasks = len(self._balance_task_names)
        self.register_buffer(
            "_balance_weights",
            torch.ones(num_tasks, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_balance_initial_losses",
            torch.zeros(num_tasks, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_balance_base_weights",
            torch.tensor(base_w, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_balance_task_min_weights",
            torch.tensor(task_min_w, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_balance_task_max_weights",
            torch.tensor(task_max_w, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_balance_loss_ema",
            torch.zeros(num_tasks, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_balance_l0_sum",
            torch.zeros(num_tasks, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_balance_l0_count",
            torch.zeros(num_tasks, dtype=torch.float32), persistent=False,
        )
        self.register_buffer(
            "_balance_call_count",
            torch.zeros(1, dtype=torch.int64), persistent=False,
        )
        # Role-alternation counter of the alternate serial consistency mode (student/teacher swap every iter).
        # Every rank increments it on every consistency iter, so it is naturally synchronized across ranks.
        self.register_buffer(
            "_consistency_flip",
            torch.zeros(1, dtype=torch.int32),
            persistent=False,
        )
        # Gradient-ratio controller for the d2skd distillation auxiliary losses (independent of GradNorm).
        # Semantics: w_d = rho * g_seg / g_d, anchoring the gradient norm of each distillation loss on the shared reference
        # parameter to a fixed ratio rho of the average gradient of the segmentation group. rho is an equality, not an upper bound --
        # when the distillation loss value becomes smaller the weight grows automatically, so distillation is not suppressed by segmentation;
        # this is a ratio target on a single reference parameter, not a hard upper bound: EMA/clamping/gradient direction and the
        # other layers of the whole model are not covered by this ratio, nor is it guaranteed not to hurt segmentation learning. The two
        # distillation paths (inter = low-frequency features / vox = high-frequency logits) are controlled independently and do not suppress each other.
        # Not merged into GradNorm because GradNorm's "relative training rate" assumption does not hold for
        # teacher-matching losses (after a fast drop they plateau and would be pushed up without bound).
        self._distill_group_names = ['inter', 'vox']
        self._distill_key_map = {
            'inter': 'loss_inter_distill',   # low-frequency TPV feature distillation (detector level)
            'vox': 'loss_distill_vox',       # high-frequency voxel logits distillation (aggregator level)
        }
        distill_cfg = self.dynamic_loss_balance.get('distill', {})
        rho_cfg = distill_cfg.get('rho', {})
        # Dark-knowledge KD is brought under the rho regime (opt-in: only effective when 'kd' is explicitly written in rho).
        # Motivation, from a measurement (nusc epoch 1, 18 records after warmup): in hybrid mode
        # loss_kd_logit used a static λ=1.0 that bypasses the controller, and its g_kd/g_seg settled at 2.25,
        # while the two spectral paths together only had rho_inter+rho_vox=0.60 -- spectral 21% / dark knowledge 79%,
        # the opposite of the design intent "spectral 60 / dark 40". Moreover, while kd_loss dropped by 74%, g_kd
        # rose by 107% over the same period: under a static λ the KD share is an unmanaged free variable drifting upward during training.
        # Once included, w_kd tracks rho_kd * g_seg / g_kd; this does not make the whole-model share constant.
        # Why it must be opt-in: loss_kd_logit is also produced in the vanilla_kd baseline mode
        # (the vanilla_kd branch of TPVAggregator_Seg._forward_single); hard-coding it would also put the main loss of KDbaseline
        # under the controller and shift that baseline's existing reference result. Hence the config side makes it conditional on KDbaseline.
        # Side effect (kept deliberately): kd now shares the same warmup ramp with the spectral terms, instead of
        # the initial ratio "KD at full strength from iter0 while spectral is 0", which is the least favourable for the spectral terms.
        if 'kd' in rho_cfg:
            self._distill_group_names.append('kd')
            self._distill_key_map['kd'] = 'loss_kd_logit'
        self._distill_balance_enabled = bool(distill_cfg.get('enabled', True))
        self._distill_warmup_iters = max(1, int(distill_cfg.get('warmup_iters', 500)))
        self._distill_ema_beta = float(distill_cfg.get('ema_beta', 0.9))
        self._distill_w_min = float(distill_cfg.get('min_weight', 1e-3))
        self._distill_w_max = float(distill_cfg.get('max_weight', 1e5))
        # Gradient measurement basis of rho.
        # 'ref' (default = historical behaviour): measured only on the **single** parameter returned by _get_reference_param().
        #   That parameter is actually lidar_neck.fpn_convs.1.bn.bias (128-dim, 0.0004% of the 32,277,463 trainable
        #   parameters), a by-product of `for p in neck.parameters(): ref = p` taking the last one,
        #   not chosen for representativeness. Let c_g = ||∇_ref L_g|| / ||∇_full L_g||; then always
        #       (w_i·||∇_full L_i||) / (w_j·||∇_full L_j||) = (rho_i/rho_j) · (c_j/c_i)
        #   i.e. the actual budget ratio between paths is scaled as a whole by the ratio of c. LF is injected at the neck output and HF/KL at
        #   the aggregator output, and the probe sits exactly at LF's injection point: measured in a multi-GPU stress test, c_LF/c_HF = 2.79~4.26;
        #   LF actually gets 8~11% of the distillation budget (rho read as a whole-model share: 28.1%), HF actually gets 38~47% (26.3%).
        #   Note: the original statement of the gradient-ratio controller above is itself a "ratio target on the reference parameter", and by
        #   that statement 'ref' is faithful; but the argument above for bringing kd under rho uses model-level shares. Which reading applies is decided by the config.
        # 'full': measured on all trainable parameters. The teacher is the external model_t, not a submodule, and spectral_distiller
        #   only has buffers and no trainable parameters, so self.parameters() is exactly the clean set of student parameters.
        #   After switching the basis, the budget ratio between paths equals the ratio between the rhos (exact for a single measurement without EMA;
        #   under the EMA+interval closed loop the CPU reproduction deviates by about 10%~26%, still far below the systematic 2.8~4.3x deviation of 'ref').
        self._distill_budget_basis = str(distill_cfg.get('budget_basis', 'ref'))
        assert self._distill_budget_basis in ('ref', 'full'), self._distill_budget_basis
        # Segmentation reference of rho: 'mean_norms' (default = historical behaviour: arithmetic mean of the **unweighted**
        # gradient norms of point/voxel/sensor) / 'weighted_vector' (||Σ_g w_g·∇L_g||, only point+voxel,
        # i.e. the combined segmentation gradient the optimizer actually sees). The latter only changes the **total strength** of distillation
        # (measured +52%~+200%) and has almost no effect on the ratios between paths; it is a separate design question to be decided separately.
        self._distill_seg_reference = str(distill_cfg.get('seg_reference', 'mean_norms'))
        assert self._distill_seg_reference in ('mean_norms', 'weighted_vector')
        # In the 'full' basis each measurement requires len(groups) full-graph backward passes. A multiplier lowers the measurement frequency to
        # balance_interval * mult to amortize the overhead (CPU proxy: interval=10 → +27%, =50 → +5%).
        self._distill_measure_mult = max(1, int(distill_cfg.get('measure_interval_mult', 1)))
        self.register_buffer(
            "_distill_rho",
            torch.tensor([float(rho_cfg.get(n, 0.25)) for n in self._distill_group_names],
                         dtype=torch.float32),
            persistent=False,
        )
        if not bool(torch.isfinite(self._distill_rho).all()) or bool((self._distill_rho < 0).any()):
            raise ValueError("distill rho must be finite and nonnegative; zero disables that group")
        # 0 means "first gradient measurement not done yet"; the first measurement is assigned directly, then it evolves by EMA.
        self.register_buffer(
            "_distill_weights",
            torch.zeros(len(self._distill_group_names), dtype=torch.float32),
            persistent=False,
        )
        self.distill_mode = distill_mode
        self.enable_lf_distill = bool(enable_lf_distill)
        assert distill_mode in ('spectral', 'vanilla_kd', 'cwd', 'pvkd'), distill_mode
        if self.d2skd:
            if distill_mode == 'spectral':
                if self.enable_lf_distill:
                    self.spectral_distiller = TPVLowFreqSpectralDistiller(
                        **(lowfreq_distiller_cfg or {}))
                else:
                    # Ablation: no LF distiller is constructed, _lf_distill_losses returns an empty dict,
                    # and the key loss_inter_distill does not appear at all. Whether this key exists depends only
                    # on the config (identical on all ranks), not on values, so the DDP collectives stay aligned.
                    self.spectral_distiller = None
            elif distill_mode == 'vanilla_kd':
                from .vanillakd import VanillaKD
                self.vanilla_kd = VanillaKD(**(kd_cfg or {}))
            elif distill_mode == 'cwd':
                from .cwd import CWD
                self.cwd = CWD(**(cwd_cfg or {}))
            elif distill_mode == 'pvkd':
                from .pvkd import PVKD
                self.pvkd = PVKD(**(pvkd_cfg or {}))
        # Predefined bound parameters, to avoid creating new tensors on every call
        min_bound = torch.tensor([0, -3.1415926, -3.4], dtype=torch.float32)
        max_bound = torch.tensor([51.2, 3.1415926, 3], dtype=torch.float32)
        grid_size = torch.tensor([480, 360, 32], dtype=torch.float32)
        grid_size_vox = torch.tensor([480, 360, 32], dtype=torch.float32)
        grid_size_vox = torch.tensor([480, 360, 32], dtype=torch.float32)
        self.register_buffer("min_bound", min_bound)
        self.register_buffer("max_bound", max_bound)
        self.register_buffer("grid_size", grid_size)
        self.register_buffer("grid_size_vox", grid_size_vox)
        # build_model(..., test_cfg=...) passes via kwargs; super().__init__() does not
        # receive them, so keep detector-level cfg here for inference (e.g. save preds).
        self.train_cfg = kwargs.get('train_cfg', None)
        self.test_cfg = kwargs.get('test_cfg', None)

    @torch.no_grad()
    def _cap_teacher_scene(self, tg_t, gi_t, gv_t):
        """Truncate the teacher's serving input of a single scene to self.teacher_serve_max_points points.

        Default None => the same objects are returned unchanged, bit-identical; existing experiments are not affected at all.
        When enabled, the three tensors are truncated with the **same indices** to keep point-wise alignment:
          tg_t = 8-dim point features (tokenizer input) / gi_t = tokenizer fine-grid coordinates /
          gv_t = distillation-field support-set coordinates (distill_points of _query_distill_field).
        With scope='all', gv_t is truncated as well -- this also shrinks the HF/KL supervision support set;
        it is the cost of reproducing the teacher's training protocol, not a side effect, and must be stated in the experiment record.
        """
        cap = getattr(self, 'teacher_serve_max_points', None)
        if not cap or not torch.is_tensor(tg_t):
            return tg_t, gi_t, gv_t
        n = int(tg_t.shape[0])
        if n <= int(cap):
            return tg_t, gi_t, gv_t
        seed = getattr(self, 'teacher_serve_cap_seed', None)
        if seed is None:
            idx = torch.randperm(n, device=tg_t.device)[:int(cap)]
        else:
            g = torch.Generator().manual_seed(int(seed) + int(self._teacher_cap_calls))
            idx = torch.randperm(n, generator=g)[:int(cap)].to(tg_t.device)
        self._teacher_cap_calls += 1

        def _take(x):
            # The original implementation silently returned the original tensor on a shape mismatch, so the truncation range actually in
            # effect would not match the config while training kept running and logging -- such silent degradation is more dangerous than a crash.
            if x is None:
                raise ValueError(
                    'teacher_serve cap: expected three tensors truncated with the same indices, but got None. '
                    'Once point-wise alignment is broken the distillation supervision is misaligned as a whole; refusing to continue silently.')
            if not torch.is_tensor(x):
                raise TypeError(f'teacher_serve cap: expected a tensor, got {type(x)}')
            if x.shape[0] != n:
                raise ValueError(
                    f'teacher_serve cap: tensor leading dim {x.shape[0]} does not match the reference {n}; '
                    'cannot truncate with the same indices, refusing to continue silently.')
            return x[idx]

        if getattr(self, 'teacher_serve_cap_scope', 'all') == 'tokenizer':
            return _take(tg_t), _take(gi_t), gv_t
        return _take(tg_t), _take(gi_t), _take(gv_t)

    def _log_teacher_serve_density(self, tg_t):
        """Print, at the first distillation iter, the teacher's actual serving point count and the effective cap.

        The teacher-serving density issue stayed invisible for a long time precisely because nowhere along this chain was the teacher's
        real input point count written to the log. Printing it once lets anyone reading the training log verify the protocol immediately.
        """
        if getattr(self, '_teacher_serve_logged', False) or not torch.is_tensor(tg_t):
            return
        self._teacher_serve_logged = True
        # [Fix] This used to print getattr(self, 'dense_cap_range'), but self is the **student**;
        # the student config has no such key → always None → always printed "dense cap 500000", a value that was never read
        # from the teacher (the three SK/NU/WA teachers were in fact trained with a fixed 500,000-point cap, dense_cap_range=None,
        # but the student cannot verify that). The sole purpose of this log line is to let people verify the protocol,
        # and faking the protocol is worse than not printing. The real cap used in teacher training only exists in the teacher ckpt meta,
        # which the student side cannot access, so only what is known for certain is reported here, no longer pretending to know the teacher's value.
        print(f"[LiEQ] teacher serving input: {int(tg_t.shape[0])} pts/scene, "
              f"teacher_serve_max_points={getattr(self, 'teacher_serve_max_points', None)}, "
              f"scope={getattr(self, 'teacher_serve_cap_scope', 'all')}, "
              f"seed={getattr(self, 'teacher_serve_cap_seed', None)} "
              f"(teacher-side training cap is recorded in the teacher checkpoint meta, "
              f"not visible from the student)", flush=True)

    def _lf_distill_losses(self, feats_s, feats_t, logits_vox_t=None):
        """Feature-level distillation losses (same tap point as the LF spectral branch: the three TPV planes).

        The returned dict is merged directly into the loss dict:
          - 'spectral'  : {'loss_inter_distill': raw} -- raw magnitude in the dict; the weight is assigned uniformly by
            the rho gradient-ratio controller of _apply_dynamic_loss_balance (unchanged).
          - 'vanilla_kd': {'loss_kd_feat': λ_feat×raw, 'kd_raw_feat': raw.detach()}
            -- static weight (as specified); the key contains no point/voxel/sensor substring and is not in
            _distill_key_map, so it naturally bypasses the three GradNorm groups and the rho controller;
            kd_raw_feat contains no "loss" → logging only.

        logits_vox_t: the teacher distillation field t_out[3] ([B,K,240,180,16], 0 outside Ω_T).
        It is read only when an LF switch needs plane inputs (gate_domain='support' / gate_source='prob' /
        block_weight='support' / ungated_domain='support'), from which the teacher-supported plane projection and the teacher's
        per-plane class probabilities are built (no extra network forward); with the default switches it is not read and no extra computation is done, keeping the old behaviour bit for bit.
        ungated_domain is a non-gating switch that still takes effect when lf_gate=False: the no-gate ablation (final method +
        lf_gate=False, effective switches block_pad=fit, ungated_domain=support) needs this distillation field as well.
        """
        if self.distill_mode == 'vanilla_kd':
            raw = self.vanilla_kd.loss_feat(feats_s, feats_t)
            return {'loss_kd_feat': self.vanilla_kd.lambda_feat * raw,
                    'kd_raw_feat': raw.detach()}
        if self.distill_mode == 'cwd':
            # channel-wise softmax over the spatial dims, per plane and then averaged (see cwd.py)
            raw = self.cwd.loss_feat(feats_s, feats_t)
            return {'loss_cwd_feat': self.cwd.lambda_cwd * raw,
                    'cwd_raw_feat': raw.detach()}
        if self.distill_mode == 'pvkd':
            # PVKD inter-point affinity: SP on the TPV planes after difficulty-aware sampling
            raw, _ = self.pvkd.loss_aff_pts(feats_s, feats_t)
            return {'loss_pvkd_aff_pts': self.pvkd.lambda_aff_pts * self.pvkd._hard_clamp(raw),
                    'pvkd_raw_aff_pts': raw.detach()}
        if not self.enable_lf_distill:
            return {}          # ablation: the LF branch is off entirely (see the enable_lf_distill comment)
        sd = self.spectral_distiller
        if not sd.needs_plane_inputs:
            return {'loss_inter_distill': sd(feats_s, feats_t)}
        # A switch that needs plane inputs raises immediately when no teacher distillation field is available, instead of silently falling
        # back to the current gate -- that would make the method actually run differ from the config while training goes on (same principle as _cap_teacher_scene).
        if logits_vox_t is None:
            raise ValueError(
                f'LF distiller switches {sd.plane_input_switches()} need the teacher distillation field t_out[3] to build the plane support / '
                'teacher per-plane class probabilities, but this call did not provide it (the d2skd path with consistency=False or random=False '
                'only runs the teacher extract_feat, without the teacher aggregator distillation-field forward). Train on the consistency=True and '
                'random=True (CBA-CL alternate) path, or turn these switches off (ungated_domain / block_weight '
                'are non-gating switches: the no-gate ablation must use the same values as the final method, they must not be turned off only in the ablation arm).')
        plane_support, plane_prob = sd.plane_inputs_from_field(logits_vox_t)
        return {'loss_inter_distill': sd(feats_s, feats_t,
                                         plane_support=plane_support, plane_prob=plane_prob)}

    def _group_loss_keys(self, loss_dict):
        # Note: loss_consistency is deliberately not bucketed -- it does not take part in GradNorm;
        # its magnitude is explicitly controlled by consistency_gain × hook warmup, and the balancer passes it through unchanged.
        group_keys = {name: [] for name in self._balance_task_names}
        for key in loss_dict.keys():
            if not key.startswith('loss_'):
                continue
            if key.startswith('loss_point_'):
                group_keys['point'].append(key)
            elif key.startswith('loss_voxel_'):
                group_keys['voxel'].append(key)
            elif key == 'loss_sensor':
                group_keys['sensor'].append(key)
        return group_keys

    def _get_reference_param(self):
        # The reference parameter must lie on the shared trunk "through which all task gradients pass". Previously the
        # first parameter of tpv_aggregator (decoder weight) was used, but the sensor loss branches off from fused before the
        # decoder, so its gradient w.r.t. that parameter was always None→0, causing the balancer to push the sensor
        # weight to the upper bound. The last parameter of lidar_neck is used instead: all four loss groups point/voxel/sensor/
        # consistency flow through the TPV features output by the neck.
        neck = getattr(self, 'lidar_neck', None)
        if neck is not None:
            ref = None
            for p in neck.parameters():
                if p.requires_grad:
                    ref = p
            if ref is not None:
                return ref
        for p in self.tpv_aggregator.parameters():
            if p.requires_grad:
                return p
        return None

    def _get_budget_params(self):
        """Gradient measurement basis of the rho budget. 'ref' degenerates to the historical single reference parameter."""
        if self._distill_budget_basis == 'full':
            ps = [p for p in self.parameters() if p.requires_grad]
            return ps if ps else None
        ref = self._get_reference_param()
        return [ref] if ref is not None else None

    @staticmethod
    def _grad_concat_norm(loss, params, weights=None):
        """Compute gradients on params and return the L2 norm of the concatenated vector.
        When weights is not None, loss is a list: it is first linearly combined into Σ w_i·L_i and differentiated once,
        which is mathematically equivalent to "differentiate separately, weight-sum, then take the norm" (CPU-verified relative error 3.6e-16),
        but needs only one backward. With a single parameter and weights=None it uses .norm(p=2), bit-identical to the historical behaviour."""
        try:
            if weights is None:
                grads = torch.autograd.grad(
                    loss, params, retain_graph=True, create_graph=False,
                    allow_unused=True)
            else:
                combo = None
                for w, l in zip(weights, loss):
                    term = l * float(w)
                    combo = term if combo is None else combo + term
                if combo is None:
                    return None
                grads = torch.autograd.grad(
                    combo, params, retain_graph=True, create_graph=False,
                    allow_unused=True)
        except RuntimeError:
            return None
        if weights is None and len(grads) == 1 and grads[0] is not None:
            return grads[0].detach().norm(p=2)
        sq = None
        for g in grads:
            if g is None:
                continue
            s = g.detach().pow(2).sum()
            sq = s if sq is None else sq + s
        if sq is None:
            return None
        return sq.sqrt()

    def _apply_dynamic_loss_balance(self, loss_dict):
        if (not self._balance_enabled) or (not isinstance(loss_dict, dict)):
            return loss_dict
        if not any(k.startswith('loss_') for k in loss_dict.keys()):
            return loss_dict

        group_keys = self._group_loss_keys(loss_dict)
        grouped_losses = {}
        active_indices = []
        for idx, name in enumerate(self._balance_task_names):
            keys = group_keys.get(name, [])
            if len(keys) == 0:
                continue
            loss_sum = None
            for k in keys:
                v = loss_dict[k]
                if torch.is_tensor(v):
                    loss_sum = v if loss_sum is None else (loss_sum + v)
            if loss_sum is None or (not loss_sum.requires_grad):
                continue
            grouped_losses[name] = loss_sum
            active_indices.append(idx)

        # The distillation auxiliary groups (inter/vox) are not put in the GradNorm buckets; they are managed by a separate gradient-ratio controller.
        distill_losses = {}
        if self._distill_balance_enabled:
            for gi, gname in enumerate(self._distill_group_names):
                v = loss_dict.get(self._distill_key_map[gname])
                if torch.is_tensor(v) and v.requires_grad:
                    distill_losses[gname] = (gi, v)
                    # Record the unweighted raw magnitude: the loss_* in the log have been multiplied by the weight,
                    # and looking only at them would misread "warmup + weight ramp-up" as "distillation loss rising".
                    # The key must not contain the substring 'loss' -- mmdet's _parse_losses adds any
                    # key containing 'loss' to the total loss.
                    loss_dict[f'd2s_raw_{gname}'] = v.detach()

        # DDP does not broadcast these adaptive buffers. L0/EMA/relative rates must use the same global statistics,
        # otherwise each GPU would optimize a different weighted objective. Fixed task order; missing supervision participates with count=0.
        with torch.no_grad():
            task_stats = self._balance_weights.new_zeros((len(self._balance_task_names), 2))
            for idx in active_indices:
                task_stats[idx, 0] = grouped_losses[self._balance_task_names[idx]].detach().abs()
                task_stats[idx, 1] = 1.0
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(task_stats, op=dist.ReduceOp.SUM)
            active_indices = [i for i in range(len(self._balance_task_names))
                              if task_stats[i, 1] > 0]
            global_losses = {self._balance_task_names[i]: task_stats[i, 0] / task_stats[i, 1]
                             for i in active_indices}

        if len(active_indices) == 0:
            return loss_dict

        with torch.no_grad():
            self._balance_call_count += 1
            # L0 burn-in: the mean loss accumulated over the first l0_burnin_iters calls is the reference
            # (no gradient probing, negligible cost). A group does not take part in weight updates before its burn-in completes.
            for idx in active_indices:
                if self._balance_l0_count[idx] < self._balance_l0_burnin:
                    n = self._balance_task_names[idx]
                    self._balance_l0_sum[idx] += global_losses[n].to(self._balance_l0_sum.device)
                    self._balance_l0_count[idx] += 1
                    if self._balance_l0_count[idx] >= self._balance_l0_burnin:
                        self._balance_initial_losses[idx] = (
                            self._balance_l0_sum[idx] / self._balance_l0_count[idx].clamp_min(1.0)
                        ).clamp_min(self._balance_eps)

        # Gradient norms are measured once every interval calls; the other calls only apply the existing weights.
        do_measure = (
            len(active_indices) > 1
            and int(self._balance_call_count.item()) % self._balance_interval == 0
        )
        ref_param = self._get_reference_param() if do_measure else None

        update_indices = []
        grad_norms = {}
        if ref_param is not None:
            # Only groups whose L0 is ready and whose loss magnitude is large enough are measured. Unmeasurable groups (grad None or
            # autograd.grad failing because of the checkpoint graph) drop out of this round of updates and keep their weights --
            # never feed 0 into the formula as a gradient norm (otherwise the weight would be constantly pushed to the upper bound).
            no_update_loss_thresh = 1e-4
            for idx in active_indices:
                if self._balance_initial_losses[idx] <= 0:
                    continue
                n = self._balance_task_names[idx]
                if n not in grouped_losses:
                    continue
                l_cur = global_losses[n]
                if float(l_cur.item()) < no_update_loss_thresh:
                    continue
                try:
                    grad = torch.autograd.grad(
                        grouped_losses[n],
                        ref_param,
                        retain_graph=True,
                        create_graph=False,
                        allow_unused=True,
                    )[0]
                except RuntimeError:
                    grad = None
                if grad is None:
                    continue
                grad_norms[n] = grad.detach().norm(p=2)
                update_indices.append(idx)

        # Use the cross-rank mean of the local gradient norms (not the norm of the DDP-combined gradient), keeping the original
        # controller definition and only fixing the per-GPU independent updates. Together with the loss/L0 sync, the weights are identical on all GPUs.
        if ref_param is not None:
            stats = self._balance_weights.new_zeros((len(self._balance_task_names), 2))
            for idx in update_indices:
                stats[idx, 0] = grad_norms[self._balance_task_names[idx]]
                stats[idx, 1] = 1.0
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            update_indices = [i for i in active_indices if stats[i, 1] > 0]
            grad_norms = {self._balance_task_names[i]: stats[i, 0] / stats[i, 1]
                          for i in update_indices}

        # Gradient measurement of the distillation groups (shares the interval with GradNorm to amortize the overhead).
        # With budget_basis='ref' the measurement basis is ref_param, bit-identical to the historical behaviour; with ='full' it switches to
        # all trainable parameters, in which case the segmentation reference **must** switch to the same basis, otherwise the two norms have
        # different bases and their ratio is meaningless. The grad_norms and weight updates of the three GradNorm groups themselves are never affected.
        distill_grad_norms = {}
        budget_seg_norm = None
        budget_params = None
        if ref_param is not None and distill_losses:
            if int(self._balance_call_count.item()) % (
                    self._balance_interval * self._distill_measure_mult) == 0:
                budget_params = self._get_budget_params()
        if budget_params is not None:
            for gname, (gi, v) in distill_losses.items():
                n = self._grad_concat_norm(v, budget_params)
                if n is not None:
                    distill_grad_norms[gname] = n
            if (self._distill_budget_basis != 'ref'
                    or self._distill_seg_reference != 'mean_norms'):
                # update_indices here is already the cross-rank-consistent set after all_reduce.
                seg_names = [self._balance_task_names[i] for i in update_indices
                             if self._balance_task_names[i] in grouped_losses]
                if self._distill_seg_reference == 'weighted_vector':
                    ls, ws = [], []
                    for nm in seg_names:
                        if nm == 'sensor':
                            continue
                        ls.append(grouped_losses[nm])
                        ws.append(float(self._balance_weights[
                            self._balance_task_names.index(nm)].item()))
                    budget_seg_norm = (self._grad_concat_norm(ls, budget_params, ws)
                                       if ls else None)
                else:
                    vals = [self._grad_concat_norm(grouped_losses[nm], budget_params)
                            for nm in seg_names]
                    vals = [v for v in vals if v is not None]
                    budget_seg_norm = torch.stack(vals).mean() if vals else None

        with torch.no_grad():
            if len(update_indices) > 1:
                update_idx_tensor = torch.tensor(update_indices, device=self._balance_weights.device, dtype=torch.long)

                for idx in update_indices:
                    curr_loss = global_losses[self._balance_task_names[idx]].to(self._balance_weights.device)
                    if self._balance_loss_ema[idx] <= 0:
                        self._balance_loss_ema[idx] = curr_loss
                    else:
                        self._balance_loss_ema[idx] = (
                            self._balance_ema_beta * self._balance_loss_ema[idx]
                            + (1.0 - self._balance_ema_beta) * curr_loss
                        )

                grad_stack = []
                rel_stack = []
                for idx in update_indices:
                    n = self._balance_task_names[idx]
                    g = grad_norms[n].to(self._balance_weights.device)
                    grad_stack.append(g)
                    l_cur = global_losses[n].to(self._balance_weights.device)
                    l0 = self._balance_initial_losses[idx].clamp_min(self._balance_eps)
                    rel_stack.append(l_cur / l0)

                grad_stack = torch.stack(grad_stack)
                rel_stack = torch.stack(rel_stack)
                grad_avg = grad_stack.mean().clamp_min(self._balance_eps)
                rel_rate = rel_stack / rel_stack.mean().clamp_min(self._balance_eps)
                target_grad = grad_avg * torch.pow(rel_rate, self._balance_alpha)

                curr_w = self._balance_weights.index_select(0, update_idx_tensor)
                update_ratio = (target_grad / grad_stack.clamp_min(self._balance_eps)).clamp(0.2, 5.0)
                new_w = curr_w * torch.pow(update_ratio, self._balance_step)
                new_w = torch.clamp(new_w, min=self._balance_min_weight, max=self._balance_max_weight)
                new_w = new_w / new_w.mean().clamp_min(self._balance_eps)
                blended_w = (1.0 - self._balance_momentum) * curr_w + self._balance_momentum * new_w
                # Apply task priors and per-task bounds.
                base_w_active = self._balance_base_weights.index_select(0, update_idx_tensor)
                min_w_active = self._balance_task_min_weights.index_select(0, update_idx_tensor)
                max_w_active = self._balance_task_max_weights.index_select(0, update_idx_tensor)
                blended_w = blended_w * base_w_active
                blended_w = blended_w / blended_w.mean().clamp_min(self._balance_eps)
                blended_w = torch.max(torch.min(blended_w, max_w_active), min_w_active)

                # Voxel-rise guard: if voxel loss is rising above EMA, boost voxel weight.
                if self._voxel_guard_enabled and ('voxel' in global_losses):
                    voxel_idx = self._balance_task_names.index('voxel')
                    pos = (update_idx_tensor == voxel_idx).nonzero(as_tuple=False)
                    if pos.numel() > 0:
                        v_pos = int(pos[0].item())
                        voxel_cur = global_losses['voxel'].to(self._balance_weights.device)
                        voxel_ema = self._balance_loss_ema[voxel_idx].clamp_min(self._balance_eps)
                        if voxel_cur > voxel_ema * (1.0 + self._voxel_guard_tol):
                            rise_ratio = voxel_cur / voxel_ema
                            boost = 1.0 + self._voxel_guard_gain * (rise_ratio - 1.0)
                            boost = torch.clamp(boost, min=1.0, max=2.5)
                            boosted = blended_w[v_pos] * boost
                            blended_w[v_pos] = torch.clamp(boosted, min=min_w_active[v_pos], max=max_w_active[v_pos])
                            blended_w = blended_w / blended_w.mean().clamp_min(self._balance_eps)
                            blended_w = torch.max(torch.min(blended_w, max_w_active), min_w_active)

                self._balance_weights.index_copy_(0, update_idx_tensor, blended_w)

        # ---- Distillation gradient-ratio controller: weight update ----
        # w_d = rho * g_seg / g_d (the first measurement is assigned directly, then it evolves by EMA).
        # The entry condition (d2skd produces both distillation losses every iter + the interval counter) is identical on all
        # ranks, so all_reduce cannot mismatch; ranks whose measurement failed participate in the sum with count=0,
        # and dividing by the valid count gives the cross-rank mean.
        if ref_param is not None and distill_losses:
            with torch.no_grad():
                dev = self._distill_weights.device
                if budget_seg_norm is not None:
                    # Segmentation reference on the same measurement basis as the distillation norms; still averaged across ranks with the
                    # existing "local scalar sum / count" protocol, same communication volume (scalar all_reduce).
                    g_seg_sum = budget_seg_norm.to(dev).reshape(())
                    seg_cnt = torch.ones((), device=dev)
                else:
                    seg_vals = [g.to(dev).reshape(()) for g in grad_norms.values()]
                    if seg_vals:
                        g_seg_sum = torch.stack(seg_vals).sum()
                        seg_cnt = torch.tensor(float(len(seg_vals)), device=dev)
                    else:
                        g_seg_sum = torch.zeros((), device=dev)
                        seg_cnt = torch.zeros((), device=dev)
                stats = [g_seg_sum, seg_cnt]
                for gname in self._distill_group_names:
                    g = distill_grad_norms.get(gname)
                    if g is not None:
                        stats.append(g.to(dev).reshape(()))
                        stats.append(torch.ones((), device=dev))
                    else:
                        stats.append(torch.zeros((), device=dev))
                        stats.append(torch.zeros((), device=dev))
                stats = torch.stack(stats)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                g_seg = stats[0] / stats[1].clamp_min(1.0)
                for gi, gname in enumerate(self._distill_group_names):
                    if gname not in distill_losses:
                        continue
                    if self._distill_rho[gi] == 0:
                        self._distill_weights[gi] = 0
                        continue
                    cnt = stats[3 + 2 * gi]
                    if float(cnt.item()) < 0.5 or float(g_seg.item()) <= self._balance_eps:
                        continue
                    g_d = stats[2 + 2 * gi] / cnt
                    if float(g_d.item()) <= self._balance_eps:
                        continue
                    target_w = (self._distill_rho[gi] * g_seg / g_d).clamp(
                        self._distill_w_min, self._distill_w_max)
                    if self._distill_weights[gi] <= 0:
                        self._distill_weights[gi] = target_w
                    else:
                        self._distill_weights[gi] = (
                            self._distill_ema_beta * self._distill_weights[gi]
                            + (1.0 - self._distill_ema_beta) * target_w
                        )

        # Apply group weights to individual losses and log current weights.
        for idx, name in enumerate(self._balance_task_names):
            keys = group_keys.get(name, [])
            if len(keys) == 0:
                continue
            w = self._balance_weights[idx]
            for k in keys:
                if torch.is_tensor(loss_dict[k]):
                    loss_dict[k] = loss_dict[k] * w
            loss_dict[f'balancer_w_{name}'] = w.detach()

        # ---- Apply the distillation weights (with a linear warmup ramp) ----
        # Before the first measurement the weight is 0: the distillation loss is multiplied by 0 so that the key is always present (DDP collectives aligned)
        # but produces no gradient; during warmup it is released linearly, and kicks in fully once the GradNorm statistics are stable.
        if distill_losses:
            warmup = min(
                1.0,
                float(self._balance_call_count.item()) / float(self._distill_warmup_iters),
            )
            for gname, (gi, v) in distill_losses.items():
                key = self._distill_key_map[gname]
                w_eff = (self._distill_weights[gi] * warmup).detach()
                if self._distill_rho[gi] == 0:
                    w_eff = w_eff.new_zeros(())
                loss_dict[key] = loss_dict[key] * w_eff
                loss_dict[f'balancer_w_distill_{gname}'] = w_eff

        return loss_dict

    @staticmethod
    def _scene_item(container, b):
        """Take the b-th scene of a per-scene container (the *_orig before sampling), always returning an [N,3]/[N,C] tensor."""
        if container is None:
            return None
        if isinstance(container, (list, tuple)):
            return container[b] if b < len(container) else container[-1]
        if torch.is_tensor(container):
            if container.dim() >= 3:
                return container[b] if b < container.shape[0] else container[-1]
            return container
        return container

    def extract_feat(self, train_grid, voxel_label):
        """Extract features of points."""
        x_3view = self.lidar_tokenizer(train_grid, voxel_label)
        tpv_list = []
        x_tpv = self.lidar_backbone(x_3view)
        for x in x_tpv:
            x = self.lidar_neck(x)
            if not isinstance(x, torch.Tensor):
                x = x[0]
            tpv_list.append(x)
        return tpv_list

    def _per_sample_reset_random(self, reset_random, points):
        """Parse the per-sample reset_random flags set by the dataset into list[bool] (length = local batch).

        The old implementation skipped simulated sampling only if "the whole local batch is raw and all DDP ranks are all raw at the same time";
        with multiple GPUs the trigger probability ≈ p^(B*world_size) (p=0.2, about 1e-23 with 4 GPUs x8),
        which completely contradicts the per-sample probability semantics of the config; moreover, raw sparse samples in a mixed batch were
        treated as dense clouds, simulated-sampled again and corrupted. It is now a per-sample bypass (passed through frame by frame inside _sampling);
        all ranks take the same code path and the loss keys are constant, so no cross-rank synchronization is needed.
        """
        if isinstance(points, list):
            bs = len(points)
        elif torch.is_tensor(points):
            bs = points.shape[0] if points.dim() >= 3 else 1
        else:
            bs = 1
        if reset_random is None:
            return [False] * bs
        vals = reset_random
        if isinstance(vals, torch.Tensor):
            vals = vals.reshape(-1).tolist()
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        flags = []
        for v in vals:
            if isinstance(v, torch.Tensor):
                v = v.reshape(-1)[0].item() if v.numel() > 0 else 0
            flags.append(bool(v))
        if len(flags) == 1 and bs > 1:
            flags = flags * bs
        if len(flags) < bs:
            flags = flags + [False] * (bs - len(flags))
        return flags[:bs]

    @torch.no_grad()
    def _random_yaw_rotation(self, train_grid, grid_ind, grid_ind_vox, train_voxel_label):
        """Per-scene random whole-bin yaw rotation (in polar coordinates == cyclic shift along phi, exact, no interpolation).

        Motivation: the training pipeline has no global rotation augmentation, and the polar TPV grid is rigidly tied to the sensor heading,
        so the model would memorize a scene-layout prior of "some azimuth ≈ some class"; when deployed to scenes with a different layout
        this shows up as mis-segmentation in fixed sectors. Under whole-bin rotation, point features and voxel GT can be synchronized exactly:
          - phi column (col 4) + k*step, wrapped to [-pi, pi)
          - xy columns (col 6:8) multiplied by the rotation matrix
          - rel_xyz (col 0:3) unchanged (voxel centers rotate with the grid, the differences are unchanged)
          - phi index (col 1) of grid_ind / grid_ind_vox + k, modulo
          - train_voxel_label roll(k) along the phi dim
        Side effect: this also randomizes the azimuth zero phase of the simulated LiDAR (previously fixed at 0).
        Returns the containers as lists (downstream handles both list and tensor).
        """
        n_phi = int(self.grid_size[1].item())
        n_phi_vox = int(self.grid_size_vox[1].item())
        if n_phi != n_phi_vox or n_phi <= 1 or train_voxel_label is None:
            return train_grid, grid_ind, grid_ind_vox, train_voxel_label

        def _frames(x):
            if isinstance(x, list):
                return list(x)
            if torch.is_tensor(x) and x.dim() >= 3:
                return [x[b] for b in range(x.shape[0])]
            return [x]

        tg_f = _frames(train_grid)
        gi_f = _frames(grid_ind)
        giv_f = _frames(grid_ind_vox)
        # a single voxel-label frame is 3D [W,H,D], so the frame list needs a special dim check
        if isinstance(train_voxel_label, list):
            vl_f = list(train_voxel_label)
        elif torch.is_tensor(train_voxel_label) and train_voxel_label.dim() >= 4:
            vl_f = [train_voxel_label[b] for b in range(train_voxel_label.shape[0])]
        else:
            vl_f = [train_voxel_label]
        bs = len(tg_f)
        if not (len(gi_f) == bs and len(giv_f) == bs and len(vl_f) == bs):
            return train_grid, grid_ind, grid_ind_vox, train_voxel_label

        two_pi = 2.0 * math.pi
        step = two_pi / n_phi
        out_grid, out_gi, out_giv, out_vl = [], [], [], []
        for b in range(bs):
            k = random.randint(0, n_phi - 1)
            if k == 0 or not torch.is_tensor(tg_f[b]):
                out_grid.append(tg_f[b]); out_gi.append(gi_f[b])
                out_giv.append(giv_f[b]); out_vl.append(vl_f[b])
                continue
            theta = k * step
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            pts = tg_f[b].clone()
            x = pts[:, 6].clone()
            y = pts[:, 7].clone()
            pts[:, 6] = x * cos_t - y * sin_t
            pts[:, 7] = x * sin_t + y * cos_t
            pts[:, 4] = torch.remainder(pts[:, 4] + theta + math.pi, two_pi) - math.pi
            gi = gi_f[b].clone()
            gi[:, 1] = torch.remainder(gi[:, 1] + k, n_phi)
            giv = giv_f[b].clone()
            giv[:, 1] = torch.remainder(giv[:, 1] + k, n_phi_vox)
            # label grid [.., phi, z]: phi is the second-to-last dim (compatible with [W,H,D] and [1,W,H,D])
            vl = torch.roll(vl_f[b], shifts=k, dims=-2)
            out_grid.append(pts); out_gi.append(gi)
            out_giv.append(giv); out_vl.append(vl)
        return out_grid, out_gi, out_giv, out_vl

    def forward_train(self,
                train_grid=None,
                grid_ind=None,
                grid_ind_vox=None,
                train_voxel_label=None,
                train_pts_label=None,
                return_loss=True,
                dataset_flag= None,
                reset_random=None,
                model_t=None,
                **kwargs
        ):
        """Forward training function.
        """
        if self.d2skd and not self.dense_train and model_t is None:
            raise ValueError("d2skd student training requires a loaded teacher model")
        # initialize sensor_params to avoid it being undefined later
        sensor_params = None

        # Per-scene random yaw rotation (must happen before capturing *_orig and sampling, so that the teacher/all branches are consistent)
        if self.random and self.random_yaw:
            train_grid, grid_ind, grid_ind_vox, train_voxel_label = self._random_yaw_rotation(
                train_grid, grid_ind, grid_ind_vox, train_voxel_label)

        # keep the original grid_ind and grid_ind_vox (for the teacher)
        grid_ind_orig = grid_ind
        grid_ind_vox_orig = grid_ind_vox
        
        # initialize the teacher features to avoid them being undefined later
        x_lidar_tpv_t = None
        loss_distill = None
        
        train_grid_orig = train_grid
        # CBA-CL and d2skd are two orthogonal invariance axes and can be enabled at the same time:
        #   - CBA-CL: student ↔ student, across two random sparse configs → sensor/sampling invariance;
        #   - d2skd : student (sparse) ← teacher (dense) → density invariance.
        # In 'alternate' mode only the main view carries gradients (aux is entirely no_grad and only serves as the consistency target),
        # so distillation naturally acts on teacher ↔ main view; main/aux alternate every iter,
        # so across iterations both views are supervised and distilled.
        distill_active = (model_t is not None) and self.d2skd
        enable_model_random = self.random
        if enable_model_random:
            # Per-sample raw bypass: samples with reset_random=True (real sparse scans loaded on the data side with p=reset_random_prob)
            # are passed through frame by frame inside _sampling without mixup/simulated sampling;
            # their sensor parameters are a NaN sentinel meaning "no GT", and the aggregator takes the prediction-conditioned branch for them.
            raw_mask = self._per_sample_reset_random(reset_random, train_grid)
            train_grid, grid_ind, grid_ind_vox, train_pts_label, sensor_params = self._sampling(
                train_grid, grid_ind, grid_ind_vox, train_pts_label, raw_mask=raw_mask)

        # CBA-CL branch. Two modes (determined by aggregator.consistency_grad_mode):
        #  - 'alternate' (default, memory ≈ a single view): in this iter one view carries gradients and is supervised
        #    (student), the other view is entirely no_grad and only provides the consistency target (teacher);
        #    the roles swap in the next iter. Across iterations: equivalent bidirectional consistency + supervision of both views.
        #    With d2skd enabled, the dense-teacher forward and the two-level spectral distillation are also done on this path.
        #  - 'both' (old behaviour, memory ≈ 2x): both views go through the backbone as one batch, gradients flow back through
        #    both graphs, and pairing and consistency are left to the aggregator's internal branch (skip_consistency=False).
        # Entered only when random sampling produces 2x samples.
        if self.consistency and enable_model_random \
                and isinstance(grid_ind, list) and len(grid_ind) >= 2:
            repeat = max(1, getattr(self, "sample_repeat_count", 2))
            total_samples = len(grid_ind)
            if total_samples % repeat != 0:
                repeat = total_samples
            base_bs = total_samples // repeat

            # the aggregator indexes per view; points / point_labels / sensor are all per view
            points_for_agg = grid_ind_vox if grid_ind_vox else grid_ind

            # voxel_label / dataset_flag are per scene (not duplicated by sampling) and shared by the two views.
            per_scene_voxel = []
            per_scene_flag = []
            for b in range(base_bs):
                if isinstance(train_voxel_label, list):
                    vl = train_voxel_label[b]
                else:
                    vl = train_voxel_label[b:b+1] if train_voxel_label is not None else None
                if isinstance(vl, torch.Tensor) and vl.dim() == 3:
                    vl = vl.unsqueeze(0)
                if isinstance(dataset_flag, list):
                    df = dataset_flag[b]
                elif isinstance(dataset_flag, torch.Tensor):
                    df = dataset_flag[b:b+1] if b < dataset_flag.shape[0] else dataset_flag
                else:
                    df = dataset_flag
                per_scene_voxel.append(vl)
                per_scene_flag.append(df)

            grad_mode = getattr(self.tpv_aggregator, 'consistency_grad_mode', 'both')
            # At the first consistency iter, print the effective mode once, to confirm the memory path in the training log
            # ('alternate' = serial, memory-saving; 'both' = backprop through both graphs, memory ~2x).
            # Note: if the config lacks the consistency_grad_mode key it silently falls back to 'both'.
            if not getattr(self, '_consistency_mode_logged', False):
                self._consistency_mode_logged = True
                print(
                    f"[LiEQ] CBA-CL consistency_grad_mode={grad_mode}, "
                    f"checkpoint_consistency_loss="
                    f"{getattr(self.tpv_aggregator, 'checkpoint_consistency_loss', True)}, "
                    f"scenes_per_gpu={base_bs}, repeat={repeat}",
                    flush=True,
                )
            # Guard rail: the 'both' path has no distillation wired in (both graphs carry gradients, there is no place for the teacher).
            # If it is misconfigured to 'both' with d2skd enabled, distillation is silently dropped -- training converges normally,
            # but loss_inter_distill / loss_distill_vox do not exist at all and the whole experiment is wasted.
            # Better to raise immediately. Note that the config falls back to 'both' when the consistency_grad_mode key is missing.
            if distill_active and not (grad_mode == 'alternate' and repeat == 2):
                raise ValueError(
                    f"d2skd=True requires consistency_grad_mode='alternate' and repeat==2 "
                    f"(current grad_mode={grad_mode!r}, repeat={repeat}). "
                    f"The 'both' mode has no distillation wired in, so the teacher forward and the two-level spectral distillation would be silently skipped."
                )
            if grad_mode == 'alternate' and repeat == 2:
                # Restores the detector-level group-serial logic of the old version: the main view carries gradients and
                # is supervised, the aux view is entirely no_grad and only provides the consistency target, and the backbone never
                # holds the backward graphs of both views at the same time → memory ≈ single-view training with the same batch.
                # Two enhancements over the old version:
                #  1) main/aux roles alternate every iter (otherwise the aux view pulled towards extreme configs
                #     by maximin would never receive supervision or consistency gradients);
                #  2) the consistency loss goes through the checkpoint wrapper (pure function without parameters, DDP-safe),
                #     freeing its internal ~1.5GB/pair of dense intermediate activations.
                student_v = int(self._consistency_flip.item()) % 2
                self._consistency_flip += 1
                use_ckpt = bool(getattr(self.tpv_aggregator, 'checkpoint_consistency_loss', True))
                # effective scale = consistency_gain (static, config) × cl_weight (hook warmup schedule)
                # × raw (T² compensation inside the function). consistency does not take part in the GradNorm balancer.
                cl_weight = float(getattr(self.tpv_aggregator, 'consistency_loss_weight', 0.0))
                cons_gain = float(getattr(self.tpv_aggregator, 'consistency_gain', 40.0))

                # The backbone runs scene by scene (batch-1). Batching was measured to be only 1.06x faster, but it would change
                # the batch-statistics semantics of BN2d(track_running_stats=False) in the neck, so it is not worth it.
                agg_loss = None
                for b in range(base_bs):
                    idx_main = b * repeat + student_v
                    idx_aux = b * repeat + (1 - student_v)
                    voxel_main = per_scene_voxel[b]
                    flag_main = per_scene_flag[b]

                    pts_main = train_pts_label[idx_main] if isinstance(train_pts_label, list) \
                        else train_pts_label[idx_main:idx_main+1]
                    if isinstance(pts_main, torch.Tensor) and pts_main.dim() == 1:
                        pts_main = pts_main.unsqueeze(0)
                    sensor_main = None
                    sensor_aux = None
                    if sensor_params is not None and isinstance(sensor_params, torch.Tensor) \
                            and sensor_params.dim() > 1:
                        sensor_main = sensor_params[idx_main:idx_main+1]
                        sensor_aux = sensor_params[idx_aux:idx_aux+1]

                    # ---- aux view first: entirely no_grad, keeping only the consistency-target logits.
                    # The order matters: the transient working memory of the aux/teacher forward does not stack with the
                    # student's backward graph, so peak memory ≈ single-view training + two sets of logits. ----
                    with torch.no_grad():
                        x_aux = self.extract_feat(
                            [train_grid[idx_aux]], [grid_ind[idx_aux]])
                        out_aux = self.tpv_aggregator(
                            x_aux,
                            points=[points_for_agg[idx_aux]],
                            voxel_label=None,
                            point_labels=None,
                            dataset_flag=flag_main,
                            return_loss=False,
                            sensor_vec=sensor_aux,
                            skip_consistency=True,
                            return_logits_for_consistency=True,
                        )
                        logits_vox_aux = out_aux[0]
                        x_aux = out_aux = None

                    # ---- teacher (dense, omniscient view): likewise entirely no_grad ----
                    # Uses the original dense point cloud before sampling (random yaw was applied uniformly before sampling, so teacher and student views agree).
                    x_tpv_t = None
                    logits_vox_t_b = None
                    distill_pts_b = None
                    if distill_active:
                        with torch.no_grad():
                            tg_t = self._scene_item(train_grid_orig, b)
                            gi_t = self._scene_item(grid_ind_orig, b)
                            gv_t = self._scene_item(grid_ind_vox_orig, b)
                            # [teacher_serve_max_points] default None = no truncation, keeping the old behaviour bit for bit.
                            tg_t, gi_t, gv_t = self._cap_teacher_scene(tg_t, gi_t, gv_t)
                            self._log_teacher_serve_density(tg_t)
                            # The teacher's voxel support set is determined by this point cloud; the student side uses the same
                            # cloud for unique(floor(·/ds)), so the two spectral fields share the grid and the support set.
                            # The original code `gv_t if gv_t is not None else gi_t`
                            # was a silent fallback: with teacher_serve_cap_scope='tokenizer' gv_t should stay
                            # unchanged, and if it were unexpectedly None it would quietly degrade to gi_t,
                            # i.e. the scope actually run would differ from the config while training goes on normally.
                            # Such silent degradation is far more dangerous than a crash (in a scope ablation the two arms would become identical).
                            if gv_t is None:
                                raise ValueError(
                                    'distill support set gv_t is None: this should not happen with teacher_serve_cap_scope='
                                    f'{getattr(self, "teacher_serve_cap_scope", "all")!r}. '
                                    'Refusing to silently fall back to gi_t -- that would make the actual scope differ from the config.')
                            distill_pts_b = gv_t
                            x_tpv_t = model_t.extract_feat(
                                train_grid=[tg_t], voxel_label=[gi_t])
                            t_out = model_t.tpv_aggregator(
                                x_tpv_t,
                                points=distill_pts_b,
                                voxel_label=None,
                                point_labels=None,
                                dataset_flag=flag_main,
                                return_loss=False,
                                sensor_vec=None,
                                skip_consistency=True,
                                # the teacher runs the same distill_points code to produce the coarse-grid field,
                                # sharing the grid and the support set with the student's distillation field.
                                distill_points=distill_pts_b,
                                return_distill_field=True,
                            )
                            logits_vox_t_b = t_out[3]
                            t_out = None

                    # ---- main view: with gradients; supervised losses + consistency student + distillation student ----
                    # The aggregator forces return_loss=False when self.d2skd=True, so the main view would get
                    # no supervised loss and the loss-dict assembly below would crash; it is temporarily turned off here and restored.
                    x_main = self.extract_feat(
                        [train_grid[idx_main]], [grid_ind[idx_main]])
                    _saved_agg_d2skd = self.tpv_aggregator.d2skd
                    self.tpv_aggregator.d2skd = False
                    try:
                        out_main = self.tpv_aggregator(
                            x_main,
                            points=[points_for_agg[idx_main]],
                            voxel_label=voxel_main,
                            point_labels=pts_main,
                            dataset_flag=flag_main,
                            return_loss=True,
                            sensor_vec=sensor_main,
                            skip_consistency=True,
                            return_logits_for_consistency=True,
                            # high-frequency spectral distillation: teacher logits + query indices on the teacher support set;
                            # together they produce loss_distill_vox with the same support set inside the aggregator.
                            logits_vox_t=logits_vox_t_b,
                            distill_points=distill_pts_b,
                        )
                    finally:
                        self.tpv_aggregator.d2skd = _saved_agg_d2skd
                    loss_main_dict, logits_vox_main = out_main[0], out_main[1]

                    loss_main_dict['loss_consistency'] = cons_gain * cl_weight * cb_adc_consistency_loss_ckpt(
                        logits_vox_main,
                        logits_vox_aux.detach(),
                        voxel_main,
                        num_classes=self.tpv_aggregator.classes,
                        T=1.5,
                        ignore_index=0,
                        use_checkpoint=use_ckpt,
                    )

                    # ---- Low-frequency feature distillation: TPV features of the student main view ↔ the teacher's ----
                    # TPV features are dense convolutional maps (gradient path everywhere), so there is no support-set issue,
                    # and they are already at native resolution, no downsampling needed. The raw magnitude goes into the dict;
                    # the weight is assigned uniformly by the gradient-ratio controller.
                    # The teacher distillation field logits_vox_t_b is passed in as well: it is read only when an LF switch needs the plane support / teacher
                    # plane probabilities (see _lf_distill_losses); with the default switches it is not read.
                    if distill_active and x_tpv_t is not None:
                        loss_main_dict.update(self._lf_distill_losses(
                            x_main, x_tpv_t, logits_vox_t=logits_vox_t_b))

                    # Release the large tensors of this scene promptly: aux/teacher logits, teacher TPV features.
                    # The accumulated loss dict still holds the main graph (needed for backward), but these used-up
                    # temporaries can be returned to the memory pool immediately.
                    logits_vox_aux = x_main = out_main = None
                    x_tpv_t = logits_vox_t_b = distill_pts_b = None

                    if agg_loss is None:
                        agg_loss = {k: v for k, v in loss_main_dict.items()}
                    else:
                        for k in loss_main_dict:
                            agg_loss[k] = agg_loss.get(k, 0) + loss_main_dict[k]
                    loss_main_dict = logits_vox_main = None
                for k in agg_loss:
                    agg_loss[k] = agg_loss[k] / base_bs
                agg_loss = self._apply_dynamic_loss_balance(agg_loss)
                return agg_loss

            # ---- 'both' old path: extract features for the whole batch, gradients flow back through both graphs ----
            per_view_voxel = [per_scene_voxel[b] for b in range(base_bs) for _ in range(repeat)]
            per_view_flag = [per_scene_flag[b] for b in range(base_bs) for _ in range(repeat)]

            x_lidar_tpv = self.extract_feat(train_grid, grid_ind)

            agg_loss = self.tpv_aggregator(
                x_lidar_tpv,
                points=points_for_agg,
                voxel_label=per_view_voxel,
                point_labels=train_pts_label,
                dataset_flag=per_view_flag,
                return_loss=True,
                sensor_vec=sensor_params,
                skip_consistency=False,
            )
            agg_loss = self._apply_dynamic_loss_balance(agg_loss)
            return agg_loss

        if model_t is not None and self.d2skd:
            # [teacher_serve_max_points] This path (consistency=False) also uses
            # grid_ind_vox_orig as the teacher points later, so the three *_orig containers must be truncated together.
            if getattr(self, 'teacher_serve_max_points', None):
                if not (isinstance(train_grid_orig, list) and isinstance(grid_ind_orig, list)
                        and isinstance(grid_ind_vox_orig, list)):
                    raise ValueError(
                        'teacher_serve_max_points requires per-scene lists for '
                        'train_grid_orig/grid_ind_orig/grid_ind_vox_orig; '
                        'got non-list input, refusing to silently continue with untruncated teacher input.')
                _tg, _gi, _gv = [], [], []
                for _b in range(len(train_grid_orig)):
                    _a, _c, _d = self._cap_teacher_scene(
                        train_grid_orig[_b], grid_ind_orig[_b], grid_ind_vox_orig[_b])
                    _tg.append(_a); _gi.append(_c); _gv.append(_d)
                train_grid_orig, grid_ind_orig, grid_ind_vox_orig = _tg, _gi, _gv
                self._log_teacher_serve_density(train_grid_orig[0])
            # the teacher forward runs entirely under no_grad, avoiding building a computation graph that occupies memory
            with torch.no_grad():
                x_lidar_tpv_t = model_t.extract_feat(
                    train_grid=train_grid_orig,
                    voxel_label=grid_ind_orig
                )
            x_lidar_tpv = self.extract_feat(
                train_grid=train_grid,
                voxel_label=grid_ind
            )
            # dict form: spectral → {'loss_inter_distill': raw};
            # vanilla_kd → {'loss_kd_feat': λ×raw, 'kd_raw_feat': raw}
            # This path (consistency=False or random=False) has no teacher distillation field: when an LF switch needs plane inputs,
            # _lf_distill_losses raises on the first call.
            # Both the final method and the no-gate ablation (ungated_domain='support') raise here; the error message names the switches.
            loss_distill = self._lf_distill_losses(x_lidar_tpv, x_lidar_tpv_t)
        else:
            x_lidar_tpv = self.extract_feat(
                train_grid=train_grid,
                voxel_label=grid_ind
            )
        grid_ind_ = grid_ind_vox if grid_ind_vox is not None else grid_ind
        if isinstance(grid_ind_, list):
            if all(pl.shape == grid_ind_[0].shape for pl in grid_ind_):
                grid_ind_ = torch.cat([pl[None, ...] for pl in grid_ind_], dim=0)
            else:
                grid_ind_ = [pl[None, ...] for pl in grid_ind_]
        if isinstance(train_pts_label, list):
            if all(pl.shape == train_pts_label[0].shape for pl in train_pts_label):
                train_pts_label = torch.cat([pl[None, ...] for pl in train_pts_label], dim=0)
            else:
                train_pts_label = [pl[None, ...] for pl in train_pts_label]
        if isinstance(train_voxel_label, list):
            if all(pl.shape == train_voxel_label[0].shape for pl in train_voxel_label):
                train_voxel_label = torch.cat([pl[None, ...] for pl in train_voxel_label], dim=0)
            else:
                train_voxel_label = [pl[None, ...] for pl in train_voxel_label]
        if isinstance(grid_ind_vox_orig, list):
            if all(pl.shape == grid_ind_vox_orig[0].shape for pl in grid_ind_vox_orig):
                grid_ind_vox_orig = torch.cat([pl[None, ...] for pl in grid_ind_vox_orig], dim=0)
            else:
                grid_ind_vox_orig = [pl[None, ...] for pl in grid_ind_vox_orig]
        if isinstance(grid_ind_orig, list):
            if all(pl.shape == grid_ind_orig[0].shape for pl in grid_ind_orig):
                grid_ind_orig = torch.cat([pl[None, ...] for pl in grid_ind_orig], dim=0)
            else:
                grid_ind_orig = [pl[None, ...] for pl in grid_ind_orig]
        # When batch contains frames with different point counts, run per-sample and aggregate losses
        grid_is_list = isinstance(grid_ind_, list)
        voxel_is_list = isinstance(train_voxel_label, list)
        pts_is_list = isinstance(train_pts_label, list)
        flag_is_list = isinstance(dataset_flag, list) if dataset_flag is not None else False

        if grid_is_list or pts_is_list or voxel_is_list or flag_is_list:
            bs = x_lidar_tpv[0].shape[0]
            agg_loss = None
            for i in range(bs):
                x_i = [feat.index_select(0, torch.tensor([i], device=feat.device)) for feat in x_lidar_tpv]
                if grid_is_list:
                    points_i = grid_ind_[i]
                else:
                    points_i = grid_ind_[i:i+1]
                if voxel_is_list:
                    voxel_label_i = train_voxel_label[i]
                else:
                    voxel_label_i = train_voxel_label[i:i+1] if train_voxel_label is not None else None
                if pts_is_list:
                    pl_i = train_pts_label[i]
                    point_labels_i = pl_i
                else:
                    point_labels_i = train_pts_label[i:i+1] if train_pts_label is not None else None
                if flag_is_list:
                    dataset_flag_i = dataset_flag[i]
                elif isinstance(dataset_flag, torch.Tensor):
                    dataset_flag_i = dataset_flag[i:i+1]
                else:
                    dataset_flag_i = dataset_flag
                sensor_params_i = None
                if sensor_params is not None:
                    if isinstance(sensor_params, torch.Tensor):
                        if sensor_params.dim() > 1 and sensor_params.size(0) > 1:
                            sensor_params_i = sensor_params[min(i, sensor_params.size(0) - 1)]
                        else:
                            sensor_params_i = sensor_params
                    else:
                        sensor_params_i = sensor_params
                # if distillation is enabled, the teacher predictions are needed
                logits_vox_t = None
                logits_pts_t = None
                points_i_t = None
                if model_t is not None and self.d2skd:
                    x_i_t = [feat.index_select(0, torch.tensor([i], device=feat.device)) for feat in x_lidar_tpv_t]
                    # the teacher predicts with the original grid_ind (unsampled)
                    if isinstance(grid_ind_vox_orig, list):
                        points_i_t = grid_ind_vox_orig[i] if i < len(grid_ind_vox_orig) else grid_ind_orig[i]
                    elif grid_ind_vox_orig is not None:
                        if isinstance(grid_ind_vox_orig, torch.Tensor):
                            points_i_t = grid_ind_vox_orig[i:i+1] if i < grid_ind_vox_orig.shape[0] else grid_ind_orig[i:i+1]
                        else:
                            points_i_t = points_i  # fallback
                    else:
                        if isinstance(grid_ind_orig, list):
                            points_i_t = grid_ind_orig[i]
                        else:
                            points_i_t = grid_ind_orig[i:i+1]
                    
                    with torch.no_grad():
                        # the teacher predicts with the original input; distill_points makes it produce a coarse-grid distillation field
                        # sharing the grid and the support set with the student.
                        teacher_outs = model_t.tpv_aggregator(
                            x_i_t,
                            points=points_i_t,
                            voxel_label=None,
                            point_labels=None,
                            dataset_flag=dataset_flag_i,
                            return_loss=False,
                            sensor_vec=None,
                            distill_points=points_i_t,
                            return_distill_field=True,
                        )
                        _, logits_pts_t, _, logits_vox_t = teacher_outs
                        # print(logits_vox_t.shape)
                        # np.save("teacher_logits.npy", logits_vox_t.detach().float().cpu().numpy())
                
                loss_i = self.tpv_aggregator(
                    x_i,
                    points=points_i,
                    voxel_label=voxel_label_i,
                    point_labels=point_labels_i,
                    dataset_flag=dataset_flag_i,
                    return_loss=return_loss,
                    sensor_vec=sensor_params_i,
                    logits_vox_t=logits_vox_t,
                    logits_pts_t=logits_pts_t,
                    # with distillation enabled, single-view training: disable the aggregator-internal consistency pairing
                    skip_consistency=distill_active,
                    # spectral distillation field on the teacher's support set (the teacher's dense points)
                    distill_points=points_i_t if distill_active else None,
                )
                # the low-frequency feature distillation loss enters the dict at raw magnitude;
                # the weight is assigned uniformly by the gradient-ratio controller of _apply_dynamic_loss_balance.
                if distill_active and return_loss and loss_distill is not None:
                    if isinstance(loss_i, dict):
                        loss_i.update(loss_distill)
                if agg_loss is None:
                    agg_loss = {k: loss_i[k] for k in loss_i}
                else:
                    for k in loss_i:
                        agg_loss[k] = agg_loss.get(k, 0) + loss_i[k]
            for k in agg_loss:
                agg_loss[k] = agg_loss[k] / bs
            agg_loss = self._apply_dynamic_loss_balance(agg_loss)
            return agg_loss
        else:
            # make sure grid_ind_ is a tensor rather than a list
            # defensive check: if grid_ind_ is still a list, force conversion to a tensor
            if isinstance(grid_ind_, list):
                # if all elements have the same shape, they can be concatenated
                if all(pl.shape == grid_ind_[0].shape for pl in grid_ind_):
                    grid_ind_ = torch.cat([pl[None, ...] for pl in grid_ind_], dim=0)
                else:
                    # if the shapes differ, the list branch should have been taken but the check failed;
                    # in this case we would need to force the list branch or take the first element;
                    # for safety, we take the first element and add a batch dim
                    grid_ind_ = grid_ind_[0].unsqueeze(0) if len(grid_ind_) > 0 else grid_ind_
            
            # make sure train_voxel_label and train_pts_label are tensors as well
            if isinstance(train_voxel_label, list):
                if all(pl.shape == train_voxel_label[0].shape for pl in train_voxel_label):
                    train_voxel_label = torch.cat([pl[None, ...] for pl in train_voxel_label], dim=0)
                else:
                    train_voxel_label = train_voxel_label[0].unsqueeze(0) if len(train_voxel_label) > 0 else train_voxel_label
            
            if isinstance(train_pts_label, list):
                if all(pl.shape == train_pts_label[0].shape for pl in train_pts_label):
                    train_pts_label = torch.cat([pl[None, ...] for pl in train_pts_label], dim=0)
                else:
                    train_pts_label = train_pts_label[0].unsqueeze(0) if len(train_pts_label) > 0 else train_pts_label
            
            # if distillation is enabled, the teacher predictions are needed
            logits_vox_t = None
            logits_pts_t = None
            grid_ind_t = None
            if model_t is not None and self.d2skd:
                # the teacher predicts with the original grid_ind (unsampled)
                grid_ind_t = grid_ind_vox_orig if grid_ind_vox_orig is not None else grid_ind_orig
                if isinstance(grid_ind_t, list):
                    if all(pl.shape == grid_ind_t[0].shape for pl in grid_ind_t):
                        grid_ind_t = torch.cat([pl[None, ...] for pl in grid_ind_t], dim=0)
                    else:
                        grid_ind_t = grid_ind_t[0].unsqueeze(0) if len(grid_ind_t) > 0 else grid_ind_t
                
                with torch.no_grad():
                    # the teacher predicts with the original input; distill_points makes it produce a coarse-grid distillation field
                    # sharing the grid and the support set with the student.
                    teacher_outs = model_t.tpv_aggregator(
                        x_lidar_tpv_t,
                        points=grid_ind_t,
                        voxel_label=None,
                        point_labels=None,
                        dataset_flag=dataset_flag,
                        return_loss=False,
                        sensor_vec=None,
                        distill_points=grid_ind_t,
                        return_distill_field=True,
                    )
                    _, logits_pts_t, _, logits_vox_t = teacher_outs
                    # print(logits_vox_t.shape)
                    # np.save("teacher_logits.npy", logits_vox_t.detach().float().cpu().numpy())
            outs = self.tpv_aggregator(
                x_lidar_tpv,
                points=grid_ind_,
                voxel_label=train_voxel_label,
                point_labels=train_pts_label,
                dataset_flag=dataset_flag,
                return_loss=return_loss,
                sensor_vec=sensor_params if sensor_params is not None else None,
                logits_vox_t=logits_vox_t,
                logits_pts_t=logits_pts_t,
                # with distillation enabled, single-view training: disable the aggregator-internal consistency pairing
                skip_consistency=distill_active,
                # spectral distillation field on the teacher's support set (the teacher's dense points)
                distill_points=grid_ind_t if distill_active else None,
            )

            # the low-frequency feature distillation loss enters the dict at raw magnitude;
            # the weight is assigned uniformly by the gradient-ratio controller of _apply_dynamic_loss_balance.
            if distill_active and return_loss and loss_distill is not None:
                if isinstance(outs, dict):
                    outs.update(loss_distill)

            if isinstance(outs, dict):
                outs = self._apply_dynamic_loss_balance(outs)
            return outs

    def train_step(self, data, optimizer, model_t=None, **kwargs):
        """Override train_step to support the distillation teacher (model_t)."""
        losses = self.forward_train(model_t=model_t, **data)
        loss, log_vars = self._parse_losses(losses)

        # estimate the batch size from the batch dim of train_grid
        num_samples = 1
        if 'train_grid' in data:
            tg = data['train_grid']
            if isinstance(tg, torch.Tensor):
                num_samples = tg.shape[0]
            elif isinstance(tg, list) and len(tg) > 0 and isinstance(tg[0], torch.Tensor):
                num_samples = len(tg)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=num_samples)
        return outputs
    
    def simple_test(self,
                train_grid=None,
                grid_ind=None,
                grid_ind_vox=None,
                train_voxel_label=None,
                train_pts_label=None,
                dataset_flag= None,
                return_loss=False,
                **kwargs):
        """Forward testing function for semantic segmentation.
        
        Args:
            train_grid (torch.Tensor): Input point cloud data
            voxel_label (torch.Tensor): Ground truth labels
            grid_ind_vox (torch.Tensor): Grid indices for voxels
            dataset_flag (torch.Tensor): Dataset type flag
            visible_mask (torch.Tensor): Mask for visible points
            return_loss (bool): Whether to return loss
            
        Returns:
            dict: Test results containing predictions and metrics
        """
        # [test_max_points] Evaluation support-set truncation: aligned with the 500k uniformly truncated input of teacher **training**
        # (a density-consistent "serving quality" measure). Note that the distillation serving path is **not** truncated by default;
        # it is controlled separately by teacher_serve_max_points. A fixed seed keeps the val subset identical across epochs,
        # removing protocol noise; per-point quantities are truncated with the same indices to stay aligned; train_voxel_label is a
        # per-scene grid and is left untouched. Default None means zero behaviour change.
        if getattr(self, 'test_max_points', None) and isinstance(train_grid, list):
            cap = self.test_max_points
            for i in range(len(train_grid)):
                n = train_grid[i].shape[0]
                if n <= cap:
                    continue
                g = torch.Generator()
                g.manual_seed(0)
                sel = torch.randperm(n, generator=g)[:cap].to(train_grid[i].device)
                train_grid[i] = train_grid[i][sel]
                if isinstance(grid_ind, list) and i < len(grid_ind) \
                        and torch.is_tensor(grid_ind[i]) and grid_ind[i].shape[0] == n:
                    grid_ind[i] = grid_ind[i][sel]
                if isinstance(grid_ind_vox, list) and i < len(grid_ind_vox) \
                        and torch.is_tensor(grid_ind_vox[i]) and grid_ind_vox[i].shape[0] == n:
                    grid_ind_vox[i] = grid_ind_vox[i][sel]
                if isinstance(train_pts_label, list) and i < len(train_pts_label) \
                        and torch.is_tensor(train_pts_label[i]) and train_pts_label[i].shape[0] == n:
                    train_pts_label[i] = train_pts_label[i][sel]
        x_lidar_tpv = self.extract_feat(train_grid=train_grid, voxel_label=grid_ind)
        grid_ind_ = grid_ind_vox if grid_ind_vox is not None else grid_ind
        if isinstance(grid_ind_, list):
            if all(pl.shape == grid_ind_[0].shape for pl in grid_ind_):
                grid_ind_ = torch.cat([pl[None, ...] for pl in grid_ind_], dim=0)
            else:
                grid_ind_ = [pl[None, ...] for pl in grid_ind_]
        if isinstance(train_pts_label, list):
            if all(pl.shape == train_pts_label[0].shape for pl in train_pts_label):
                train_pts_label = torch.cat([pl[None, ...] for pl in train_pts_label], dim=0)
            else:
                train_pts_label = [pl[None, ...] for pl in train_pts_label]
        if isinstance(train_voxel_label, list):
            if all(pl.shape == train_voxel_label[0].shape for pl in train_voxel_label):
                train_voxel_label = torch.cat([pl[None, ...] for pl in train_voxel_label], dim=0)
            else:
                train_voxel_label = [pl[None, ...] for pl in train_voxel_label]
        pred_list = None
        # If variable points per frame, run per-sample and pad results to evaluate
        if (
            isinstance(grid_ind_, list)
            or isinstance(train_pts_label, list)
            or isinstance(train_voxel_label, list)
            or (isinstance(dataset_flag, list) if dataset_flag is not None else False)
        ):
            bs = x_lidar_tpv[0].shape[0]
            pred_list = []
            gt_list = []
            pred_sensor_output_list = []
            for i in range(bs):
                # Per-frame TPV features; keep a batch dimension of 1
                x_i = [feat.index_select(0, torch.tensor([i], device=feat.device)) for feat in x_lidar_tpv]
                # Ensure points have shape [1, N_i, 3]
                if isinstance(grid_ind_, list):
                    points_i = grid_ind_[i].unsqueeze(0)
                else:
                    points_i = grid_ind_[i:i+1]
                # Ensure voxel labels have shape [1, W, H, D] if provided
                if isinstance(train_voxel_label, list):
                    voxel_label_i = train_voxel_label[i].unsqueeze(0)
                else:
                    voxel_label_i = train_voxel_label[i:i+1] if train_voxel_label is not None else None
                # Ensure point labels have shape [1, N_i, 1, 1] if provided
                if isinstance(train_pts_label, list):
                    pl_i = train_pts_label[i]
                    point_labels_i = pl_i.unsqueeze(0) if pl_i.dim() == 1 else pl_i.unsqueeze(0) if pl_i.dim() == 2 else pl_i.unsqueeze(0) if pl_i.dim() == 3 else pl_i
                else:
                    point_labels_i = train_pts_label[i:i+1] if train_pts_label is not None else None
                if isinstance(dataset_flag, list):
                    dataset_flag_i = dataset_flag[i]
                elif isinstance(dataset_flag, torch.Tensor):
                    dataset_flag_i = dataset_flag[i:i+1]
                else:
                    dataset_flag_i = dataset_flag
                _, logits_pts_i, pred_sensor_output = self.tpv_aggregator(
                    x_i,
                    points=points_i,
                    voxel_label=voxel_label_i,
                    point_labels=point_labels_i,
                    dataset_flag=dataset_flag_i,
                    return_loss=return_loss,
                )
                logits_pts_i = logits_pts_i.squeeze(-1).squeeze(-1).squeeze(0)  # [classes, n_i]
                pred_i = torch.argmax(logits_pts_i, dim=0).detach().cpu()
                gt_i = point_labels_i.squeeze().detach().cpu()
                pred_list.append(pred_i)
                gt_list.append(gt_i)
                pred_sensor_output_list.append(pred_sensor_output)
            max_n = max(p.shape[0] for p in pred_list)
            predict_labels_pts = torch.zeros((bs, max_n), dtype=pred_list[0].dtype)
            val_pt_labs = torch.zeros((bs, max_n), dtype=gt_list[0].dtype)
            for i in range(bs):
                n_i = pred_list[i].shape[0]
                predict_labels_pts[i, :n_i] = pred_list[i]
                val_pt_labs[i, :n_i] = gt_list[i]
        else:
            _, predict_labels_pts, pred_sensor_output = self.tpv_aggregator(x_lidar_tpv, points=grid_ind_, voxel_label=train_voxel_label, point_labels=train_pts_label, dataset_flag= dataset_flag, return_loss=return_loss)
            predict_labels_pts = predict_labels_pts.squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            val_pt_labs = train_pts_label.squeeze(-1).detach().cpu() 
        
        tc = getattr(self, 'test_cfg', None)
        save_infer = False
        if tc is not None:
            if hasattr(tc, 'get'):
                save_infer = bool(tc.get('save_inference_preds', False))
            else:
                save_infer = bool(getattr(tc, 'save_inference_preds', False))
        if save_infer:
            test_output = {
                'pred': predict_labels_pts,
                'pred_sensor_output': pred_sensor_output,
            }
        else:
            IoU_metric = self.evaluation_semantic(
                predict_labels_pts, val_pt_labs, dataset_flag)
            test_output = {
                'SSC_metric': IoU_metric,
                'pred': predict_labels_pts,
                'pred_sensor_output': pred_sensor_output,
            }

        self._maybe_save_inference_preds(
            predict_labels_pts, pred_list, train_grid, kwargs)

        return test_output
    
    def forward_test(self,
                train_grid=None,
                grid_ind=None,
                grid_ind_vox=None,
                train_voxel_label=None,
                train_pts_label=None,
                dataset_flag= None,
                return_loss=False,
                **kwargs
        ):
        """
        Forward testing function.
        """
        return self.simple_test(
            train_grid, grid_ind, grid_ind_vox, train_voxel_label, train_pts_label,
            dataset_flag, return_loss, **kwargs)

    def _unwrap_train_grid_for_save(self, train_grid):
        """Strip DataContainer / single-element list wrappers from collate."""
        if train_grid is None:
            return None
        x = train_grid
        while isinstance(x, DataContainer):
            x = x.data
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        return x

    def _maybe_save_inference_preds(
            self, predict_labels_pts, pred_list, train_grid, kwargs):
        """Write .label (uint32) and optional .ply (train_grid xyz) when enabled."""
        tc = getattr(self, 'test_cfg', None)
        if tc is None:
            return
        if hasattr(tc, 'get'):
            save = tc.get('save_inference_preds', False)
            out_root = tc.get('inference_pred_dir', './work_dirs/realdata_preds')
            save_ply = tc.get('save_inference_ply', True)
        else:
            save = getattr(tc, 'save_inference_preds', False)
            out_root = getattr(tc, 'inference_pred_dir', './work_dirs/realdata_preds')
            save_ply = getattr(tc, 'save_inference_ply', True)
        if not save:
            return
        img_metas = kwargs.get('img_metas', None)
        if img_metas is None:
            return
        metas = img_metas.data[0] if hasattr(img_metas, 'data') else img_metas
        if not isinstance(metas, (list, tuple)):
            metas = [metas]
        os.makedirs(out_root, exist_ok=True)
        pred_np = predict_labels_pts.detach().cpu().numpy()
        bs = pred_np.shape[0]
        tg_in = train_grid if train_grid is not None else kwargs.get('train_grid')
        tg = self._unwrap_train_grid_for_save(tg_in)
        for b in range(bs):
            meta = metas[b] if b < len(metas) else metas[0]
            stem = None
            if isinstance(meta, dict):
                tid = meta.get('lidar_token')
                if tid is not None:
                    stem = str(tid)
                if stem is None:
                    for key in ('pts_filename', 'filename'):
                        p = meta.get(key)
                        if p:
                            stem = osp.splitext(osp.basename(str(p)))[0]
                            break
                if stem is None and meta.get('sample_idx') is not None:
                    stem = str(meta['sample_idx'])
            if stem is None:
                stem = f'frame_{b:06d}'
            if pred_list is not None:
                n_i = pred_list[b].shape[0]
                labels = pred_np[b, :n_i].astype(np.uint32)
            else:
                labels = pred_np[b].astype(np.uint32)
                n_i = int(labels.shape[0])
            out_path = osp.join(out_root, f'{stem}.label')
            labels.tofile(out_path)
            if save_ply and tg is not None:
                rows = None
                if isinstance(tg, torch.Tensor):
                    if tg.dim() == 3:
                        rows = tg[b, :n_i]
                    elif tg.dim() == 2 and bs == 1 and b == 0:
                        rows = tg[:n_i]
                elif isinstance(tg, (list, tuple)) and b < len(tg):
                    tgb = tg[b]
                    if isinstance(tgb, torch.Tensor) and tgb.dim() == 2:
                        rows = tgb[:n_i]
                    elif isinstance(tgb, torch.Tensor) and tgb.dim() == 3:
                        rows = tgb[0, :n_i] if tgb.shape[0] == 1 else tgb[b, :n_i]
                if rows is not None and rows.shape[0] >= n_i:
                    rows = rows[:n_i]
                    ply_path = osp.join(out_root, f'{stem}.ply')
                    save_tensor_as_ply(rows, ply_path)
    
    def evaluation_semantic(self, pred, gt, dataset_flag):
        """Evaluate semantic segmentation results.
        
        Args:
            pred (torch.Tensor): Predicted point cloud labels, shape [B, N, num_classes]
            gt (torch.Tensor): Ground truth point cloud labels, shape [B, N]
            dataset_flag (torch.Tensor): Dataset type flag
            visible_mask (torch.Tensor, optional): Mask for visible points
            
        Returns:
            tuple: (IoU histogram matrix, visible IoU histogram matrix)
        """
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        
        # Determine number of classes based on dataset
        if isinstance(dataset_flag, list):
            if len(dataset_flag) == 1:
                dataset_flag = dataset_flag[0].item()
            elif all(flag.item() == dataset_flag[0].item() for flag in dataset_flag):
                dataset_flag = dataset_flag[0].item()
            else:
                raise NotImplementedError("Different dataset_flag when eval")
        else:
            assert isinstance(dataset_flag, torch.Tensor)
            dataset_flag = dataset_flag.item()
            
        uniq_gt = np.unique(gt)
        if len(uniq_gt) >= 2 and np.sort(uniq_gt)[-2] <= 7:
            max_label = 8
        elif dataset_flag == 1:
            max_label = 17
        elif dataset_flag == 2:
            max_label = 20
        elif dataset_flag == 3:
            max_label = 23
        else:
            max_label = 8
            
        # Create mask for valid points (excluding noise/unknown)
        valid_mask = gt != 0
        
        # Compute IoU histogram for all valid points
        hist = fast_hist(pred[valid_mask], gt[valid_mask], max_label=max_label)
            
        return hist
    
    @torch.no_grad()
    def _mixup_minority(self, points, grid_ind, grid_ind_vox, labels, num_add=2):
        """
        Instance-aware minority class mixup with physical placement validation.
    
        Args:
            points:        [N, C]   layout: [:, 0:3]=rel_xyz, [:, 3:6]=xyz_pol(rho,phi,z),
                                            [:, 6:8]=xy(cart),  [:, 8:]=other features
            grid_ind:      [N, 3]   voxel idx in polar grid (float)
            grid_ind_vox:  [N, 3]   voxel idx in cubic grid  (float)
            labels:        [N]
            num_add:       how many copies per instance to *attempt* to place
                        (each copy is only kept if placement is valid)
    
        Returns:
            augmented points / grid_ind / grid_ind_vox / labels
        """
        device = points.device
        dtype = points.dtype
    
        MINORITY_CLASSES = (5, 6)
    
        # ---- Tunables ------------------------------------------------------
        # Cluster: BFS in polar voxel space; max neighbour distance in voxel idx
        CLUSTER_MAX_VOX_DIST = 1
        CLUSTER_MIN_PTS = 3                # ignore stray minority points
        # Per-attempt search:
        PLACEMENT_TRIES = 8                # how many random poses tried per copy
        ROT_RANGE_RAD = 0.25 * math.pi     # ±45° azimuthal rotation
        RHO_JITTER_REL = 0.20              # ±20% radial scale jitter
        Z_JITTER = 0.05                    # ±5 cm vertical jitter on top of ground snap
        # Placement constraints (in metres)
        NEAR_FRONT_TOL = 0.50              # if any existing point in target cell is
                                        # closer than (instance_min_depth - TOL),
                                        # placement is rejected (occlusion violation)
        GROUND_SEARCH_RADIUS = 0.6         # m, radial+azimuthal radius in metric space
        GROUND_SEARCH_DZ = 1.0             # max distance below instance bottom to look for ground
        MAX_GROUND_LIFT = 0.5              # max +z shift the snap may impose
        MAX_GROUND_DROP = 0.5              # max -z shift the snap may impose
        # ROI from class attributes
        min_bound = self.min_bound
        max_bound = self.max_bound
        grid_size = self.grid_size
        grid_size_vox = self.grid_size_vox
        crop_range = max_bound - min_bound
        intervals = crop_range / grid_size
        intervals_vox = crop_range / grid_size_vox
    
        # -----------------------------------------------------------------
        # 0. Select minority points
        # -----------------------------------------------------------------
        minority_mask = torch.zeros_like(labels, dtype=torch.bool)
        for c in MINORITY_CLASSES:
            minority_mask = minority_mask | (labels == c)
        if not minority_mask.any():
            return points, grid_ind, grid_ind_vox, labels
    
        minor_pts = points[minority_mask]
        minor_lbs = labels[minority_mask]
        minor_grid = grid_ind[minority_mask]      # polar voxel idx
        # ground reference: existing non-minority points
        bg_mask = ~minority_mask
        bg_xy = points[bg_mask][:, 6:8]            # [B, 2] cartesian xy
        bg_z = points[bg_mask][:, 5]               # [B] z
        bg_rho = points[bg_mask][:, 3]             # [B] rho
        bg_phi = points[bg_mask][:, 4]             # [B] phi
        bg_labels = labels[bg_mask]
        # rough "ground" candidates: classes 1 (Driveable Ground), 7 (Other Ground)
        # If you don't trust this, fall back to "lowest-z in column" later.
        ground_class_mask = (bg_labels == 1) | (bg_labels == 7)
        ground_xy = bg_xy[ground_class_mask]
        ground_z = bg_z[ground_class_mask]
    
        # -----------------------------------------------------------------
        # 1. Cluster minority points -> instances
        #    Use a hash on the polar voxel grid + 6-neighbour BFS.
        #    For typical KITTI frames this is < 50 minority points -> trivial cost.
        # -----------------------------------------------------------------
        N_minor = minor_pts.shape[0]
        if N_minor < CLUSTER_MIN_PTS:
            return points, grid_ind, grid_ind_vox, labels
    
        # Do the clustering on CPU (small N, host-side is faster and simpler)
        minor_grid_cpu = minor_grid.detach().to(torch.int64).cpu().numpy()
        # Build a hash table voxel -> point index
        vox_to_idx = {}
        for i in range(N_minor):
            key = (int(minor_grid_cpu[i, 0]),
                int(minor_grid_cpu[i, 1]),
                int(minor_grid_cpu[i, 2]))
            vox_to_idx.setdefault(key, []).append(i)
    
        visited = [False] * N_minor
        instances = []   # list of list-of-point-indices
        for seed in range(N_minor):
            if visited[seed]:
                continue
            stack = [seed]
            comp = []
            while stack:
                cur = stack.pop()
                if visited[cur]:
                    continue
                visited[cur] = True
                comp.append(cur)
                ck = minor_grid_cpu[cur]
                # 6-neighbour search (could widen with CLUSTER_MAX_VOX_DIST)
                r = CLUSTER_MAX_VOX_DIST
                for d0 in range(-r, r + 1):
                    for d1 in range(-r, r + 1):
                        for d2 in range(-r, r + 1):
                            if d0 == 0 and d1 == 0 and d2 == 0:
                                continue
                            nbr = (int(ck[0]) + d0,
                                int(ck[1]) + d1,
                                int(ck[2]) + d2)
                            if nbr in vox_to_idx:
                                for j in vox_to_idx[nbr]:
                                    if not visited[j]:
                                        stack.append(j)
            if len(comp) >= CLUSTER_MIN_PTS:
                instances.append(comp)
    
        if not instances:
            return points, grid_ind, grid_ind_vox, labels
    
        # -----------------------------------------------------------------
        # 2. For each instance, attempt num_add valid placements
        # -----------------------------------------------------------------
        add_points_list = []
        add_labels_list = []
        add_grid_list = []
        add_grid_vox_list = []
    
        # Pre-compute existing-cloud sorted-by-phi index for fast cell lookup
        all_xy = points[:, 6:8]                          # [N, 2]
        # -----------------------------------------------------------------
        # Pre-compute sorted-by-phi index over the WHOLE existing cloud,
        # used for occlusion-violation check. Sorting once per frame is O(N log N);
        # inside the placement loop each query is O(log N + window).
        # -----------------------------------------------------------------
        # [FIX occlusion ring] The old implementation put **all points** (including ground) into the occlusion candidate ring and compared only rho:
        # within the target azimuth window there is almost always a ground point with a smaller rho → every placement was judged "occluded";
        # the measured placement acceptance rate was 0 and the minority mixup silently did nothing (one of the hidden causes why Living Being/Objects
        # stayed stuck at 0.3-0.4 for a long time). Fix:
        #   1) the occlusion ring only takes non-ground points (physically, ground does not occlude a target standing on it);
        #   2) add a z gate: only closer points roughly within the instance's vertical extent count as occluders.
        occ_mask = (labels != 1) & (labels != 7)
        all_rho_full = points[occ_mask, 3].contiguous()   # rho (non-ground)
        all_phi_full = points[occ_mask, 4].contiguous()   # phi in [-pi, pi]
        all_z_full = points[occ_mask, 5].contiguous()     # z (for the height gate)

        # Sort by phi once. We will binary-search phi-windows inside the loop.
        phi_sorted, phi_sort_idx = torch.sort(all_phi_full)
        rho_sorted = all_rho_full[phi_sort_idx]
        z_sorted = all_z_full[phi_sort_idx]

        # To handle the wrap-around at ±pi without writing two-segment queries,
        # build a duplicated "ring" array: [phi - 2pi  |  phi  |  phi + 2pi].
        # This way any window centred near ±pi still maps to a single contiguous slice.
        TWO_PI = 2.0 * math.pi
        phi_ring = torch.cat([phi_sorted - TWO_PI, phi_sorted, phi_sorted + TWO_PI], dim=0)
        rho_ring = torch.cat([rho_sorted,           rho_sorted, rho_sorted          ], dim=0)
        z_ring_occ = torch.cat([z_sorted,           z_sorted,   z_sorted            ], dim=0)

        # Same idea for ground points, also keyed by phi for quick local lookup.
        if ground_xy.shape[0] > 0:
            ground_phi = torch.atan2(ground_xy[:, 1], ground_xy[:, 0])
            g_sorted_phi, g_sort_idx = torch.sort(ground_phi)
            g_sorted_z = ground_z[g_sort_idx]
            g_sorted_xy = ground_xy[g_sort_idx]
            g_phi_ring = torch.cat([g_sorted_phi - TWO_PI, g_sorted_phi, g_sorted_phi + TWO_PI], dim=0)
            g_z_ring = torch.cat([g_sorted_z, g_sorted_z, g_sorted_z], dim=0)
            g_xy_ring = torch.cat([g_sorted_xy, g_sorted_xy, g_sorted_xy], dim=0)
        else:
            g_phi_ring = None

        phi_tol = 1.5 * TWO_PI / 1800.0   # ~1.5 horizontal bins
        GROUND_PHI_TOL = max(phi_tol * 4, 0.02)  # wider window for ground lookup
    
        for inst in instances:
            inst_idx = torch.tensor(inst, device=device, dtype=torch.long)
            ip = minor_pts[inst_idx]                     # [K, C]
            il = minor_lbs[inst_idx]                     # [K]
    
            ip_xy = ip[:, 6:8]                           # cartesian xy
            ip_z = ip[:, 5]                              # z
            # Instance centroid (in cartesian, on xy plane)
            cx = ip_xy[:, 0].mean()
            cy = ip_xy[:, 1].mean()
            c_rho = torch.sqrt(cx * cx + cy * cy)
            c_phi = torch.atan2(cy, cx)
            # Per-instance points relative to centroid (so we can rotate as a rigid body)
            rel_xy = ip_xy - torch.stack([cx, cy])
            z_min_inst = ip_z.min()
            z_max_inst = ip_z.max()
    
            for _copy in range(num_add):
                placed = False
                for _try in range(PLACEMENT_TRIES):
                    # ---- propose a new pose ---------------------------------
                    d_phi = (torch.rand((), device=device) * 2 - 1) * ROT_RANGE_RAD
                    rho_scale = 1.0 + (torch.rand((), device=device) * 2 - 1) * RHO_JITTER_REL
                    new_rho = (c_rho * rho_scale).clamp(min=1.0)
                    new_phi = c_phi + d_phi
                    # wrap to [-pi, pi]
                    new_phi = (new_phi + math.pi) % (2 * math.pi) - math.pi
                    new_cx = new_rho * torch.cos(new_phi)
                    new_cy = new_rho * torch.sin(new_phi)
    
                    # rotate instance rigidly so it still "faces" the new bearing
                    cos_dphi = torch.cos(d_phi)
                    sin_dphi = torch.sin(d_phi)
                    R = torch.stack([
                        torch.stack([cos_dphi, -sin_dphi]),
                        torch.stack([sin_dphi,  cos_dphi]),
                    ])
                    # scale radial component to honour rho_scale (keeps wider object wider
                    # at new range — optional; we set scale=1 along radial too for safety)
                    new_rel_xy = rel_xy @ R.T
                    new_xy = new_rel_xy + torch.stack([new_cx, new_cy])
    
                    # ---- find ground at target xy (phi-window restricted) ---
                    if g_phi_ring is None:
                        target_ground_z = torch.tensor(0.0, device=device, dtype=dtype)
                    else:
                        # find the phi-window centred at new_phi
                        lo = torch.searchsorted(g_phi_ring, (new_phi - GROUND_PHI_TOL).reshape(1)).item()
                        hi = torch.searchsorted(g_phi_ring, (new_phi + GROUND_PHI_TOL).reshape(1)).item()
                        if hi <= lo:
                            continue
                        cand_xy = g_xy_ring[lo:hi]
                        cand_z = g_z_ring[lo:hi]
                        dxy = cand_xy - torch.stack([new_cx, new_cy])
                        d2 = (dxy * dxy).sum(dim=1)
                        nearest = d2.argmin()
                        if torch.sqrt(d2[nearest]) > GROUND_SEARCH_RADIUS:
                            continue
                        target_ground_z = cand_z[nearest]
    
                    z_shift = target_ground_z - z_min_inst
                    # clip shift to avoid pathological large jumps
                    z_shift = torch.clamp(z_shift, min=-MAX_GROUND_DROP, max=MAX_GROUND_LIFT)
                    # tiny vertical jitter on top of ground snap
                    z_shift = z_shift + (torch.rand((), device=device) * 2 - 1) * Z_JITTER
                    new_z = ip_z + z_shift
    
                    # ---- occlusion-violation check (phi-window binary search) ---
                    # Replaces the previous O(K * B_full) dense pairwise check, which was
                    # the cause of the 10 GB+ memory spikes per try.
                    new_rho_pts = torch.sqrt(new_xy[:, 0] ** 2 + new_xy[:, 1] ** 2)
                    new_phi_pts = torch.atan2(new_xy[:, 1], new_xy[:, 0])

                    # Per instance-point, find the slice of background rho values whose phi
                    # is within phi_tol. Then check if any of those is closer than
                    # (this point's rho - NEAR_FRONT_TOL).
                    lo_q = new_phi_pts - phi_tol
                    hi_q = new_phi_pts + phi_tol
                    # searchsorted is vectorised over K; output shape [K]
                    lo_idx = torch.searchsorted(phi_ring, lo_q.contiguous())
                    hi_idx = torch.searchsorted(phi_ring, hi_q.contiguous())

                    # [FIX occlusion test] vectorized + z gate + single sync:
                    #  - an occluder must both be closer in rho (difference > NEAR_FRONT_TOL) and have z near
                    #    the instance's vertical extent (a thin ground layer and high structures do not count as occluders).
                    #  - the old code did 3 device syncs per point (.item()x2 + .any()); K points x 16
                    #    tries could reach thousands of syncs; here the windows are tensorized and fused with the bounds check into
                    #    a single sync. OR is associative, so chunking does not change the result.
                    z_block_lo = new_z.min() + 0.2   # ignore the thin layer touching the ground
                    z_block_hi = new_z.max() + 0.5   # slightly above the instance top can still occlude
                    W = (hi_idx - lo_idx).clamp_(min=0)
                    Wmax = int(W.max().item())
                    conflict_t = torch.zeros((), dtype=torch.bool, device=device)
                    if Wmax > 0:
                        thr = new_rho_pts - NEAR_FRONT_TOL
                        last = rho_ring.numel() - 1
                        ar = torch.arange(Wmax, device=device)
                        K_inst = new_rho_pts.shape[0]
                        # limit the per-chunk peak to <= ~4M window elements to prevent memory spikes on large instances
                        step = max(1, min(K_inst, (1 << 22) // max(Wmax, 1)))
                        for s0 in range(0, K_inst, step):
                            e0 = min(s0 + step, K_inst)
                            off = (lo_idx[s0:e0].unsqueeze(1) + ar.unsqueeze(0)).clamp_(max=last)
                            # naming to avoid shadowing: c_rho/c_z would overwrite the outer instance centroid c_rho (this once caused a crash)
                            occ_rho = rho_ring[off]
                            occ_z = z_ring_occ[off]
                            blocking = ((occ_rho < thr[s0:e0].unsqueeze(1))
                                        & (occ_z > z_block_lo)
                                        & (occ_z < z_block_hi)
                                        & (ar.unsqueeze(0) < W[s0:e0].unsqueeze(1)))
                            conflict_t = conflict_t | blocking.any()

                    # ---- bounds check (polar ROI), fused with the occlusion test into a single sync ----
                    new_xyz_pol = torch.stack([new_rho_pts, new_phi_pts, new_z], dim=1)
                    in_bound = ((new_xyz_pol >= min_bound) &
                                (new_xyz_pol < (max_bound - 1e-3))).all(dim=1)
                    # must use a tensor bitwise or: python `or` would short-circuit and sync early
                    if bool(conflict_t | in_bound.all().logical_not()):
                        continue
    
                    # ---- valid placement: build output rows ------------------
                    # grid indices
                    new_grid_ind = torch.floor((new_xyz_pol - min_bound) / intervals).to(torch.int32)
                    new_grid_ind_vox_float = (new_xyz_pol - min_bound) / intervals_vox
                    # voxel centres for rel_xyz
                    voxel_centers = (new_grid_ind.float() + 0.5) * intervals + min_bound
                    new_rel_xyz = new_xyz_pol - voxel_centers
    
                    # carry-over features unchanged (intensity etc.)
                    feat = ip[:, 8:]
                    # assemble: [rel_xyz (3) | xyz_pol (3) | xy (2) | feat]
                    chunk_aug = torch.cat([new_rel_xyz, new_xyz_pol, new_xy, feat], dim=1)
    
                    add_points_list.append(chunk_aug)
                    add_labels_list.append(il)
                    add_grid_list.append(new_grid_ind.float())
                    add_grid_vox_list.append(new_grid_ind_vox_float)
                    placed = True
                    break
                # if not placed after PLACEMENT_TRIES, skip this copy
                if not placed:
                    continue
    
        if not add_points_list:
            return points, grid_ind, grid_ind_vox, labels
    
        final_points = torch.cat([points] + add_points_list, dim=0)
        final_labels = torch.cat([labels] + add_labels_list, dim=0)
        final_grid = torch.cat([grid_ind] + add_grid_list, dim=0)
        final_grid_vox = torch.cat([grid_ind_vox] + add_grid_vox_list, dim=0)
    
        return final_points, final_grid, final_grid_vox, final_labels

    @torch.no_grad()
    def _sampling(self, points, grid_ind, grid_ind_vox, labels, raw_mask=None):
        # raw_mask: list[bool]; frames with True are reset_random real scans, passed through frame by frame without simulation.
        # --------------------------------------------------
        # Dense training mode: uniform downsampling
        # --------------------------------------------------
        if self.dense_train:
            # keep the original input format
            was_single_tensor = not isinstance(points, list)
            
            # make sure the input is in list format and detached
            if isinstance(points, list):
                points = [p.detach() for p in points]
            else:
                points = [points.detach()]
            
            if isinstance(grid_ind, list):
                grid_ind = [g.detach() if isinstance(g, torch.Tensor) else g for g in grid_ind]
            else:
                grid_ind = [grid_ind.detach() if isinstance(grid_ind, torch.Tensor) else grid_ind]
            
            if isinstance(grid_ind_vox, list):
                grid_ind_vox = [g.detach() if isinstance(g, torch.Tensor) else g for g in grid_ind_vox]
            else:
                grid_ind_vox = [grid_ind_vox.detach() if isinstance(grid_ind_vox, torch.Tensor) else grid_ind_vox]
            
            if isinstance(labels, list):
                labels = [l.detach() if isinstance(l, torch.Tensor) else l for l in labels]
            else:
                labels = [labels.detach() if isinstance(labels, torch.Tensor) else labels]
            
            results_points = []
            results_grid_ind = []
            results_grid_ind_vox = []
            results_labels = []
            
            max_points = 500000
            # [Teacher v3 | density-jitter augmentation] The only input randomness of dense teacher training is the 500k uniform
            # truncation itself; making the cap a per-sample random U[lo,hi] is equivalent to density dropout on the input:
            # (1) it injects input randomness, mechanistically countering overfitting of the dense task (the direct cause of the post-peak
            # decline measured in v2); (2) the teacher becomes robust over the [lo,hi] density band. Default None = fixed 500k, bit-exact
            # compatible; during distillation the teacher is built from the student config (without this parameter), so serving behaviour is unaffected.
            cap_range = getattr(self, 'dense_cap_range', None)
            if cap_range:
                lo, hi = int(cap_range[0]), int(cap_range[1])
                max_points = int(torch.randint(lo, hi + 1, (1,)).item())

            for idx in range(len(points)):
                curr_points = points[idx]
                curr_labels = labels[idx]
                curr_grid_ind = grid_ind[idx]
                curr_grid_ind_vox = grid_ind_vox[idx]
                
                num_points = curr_points.shape[0]
                
                # if the point count is <= 500000, return the original data directly
                if num_points <= max_points:
                    results_points.append(curr_points)
                    results_grid_ind.append(curr_grid_ind)
                    results_grid_ind_vox.append(curr_grid_ind_vox)
                    results_labels.append(curr_labels)
                else:
                    # uniformly downsample to at most 500000 points
                    device = curr_points.device
                    indices = torch.randperm(num_points, device=device)[:max_points]
                    
                    sampled_points = curr_points[indices]
                    sampled_labels = curr_labels[indices]
                    sampled_grid_ind = curr_grid_ind[indices]
                    sampled_grid_ind_vox = curr_grid_ind_vox[indices]
                    
                    results_points.append(sampled_points)
                    results_grid_ind.append(sampled_grid_ind)
                    results_grid_ind_vox.append(sampled_grid_ind_vox)
                    results_labels.append(sampled_labels)
            
            # sensor_params_tensor is set to None
            sensor_params_tensor = None
            
            # if the original input was a single tensor, return a single tensor; otherwise return a list
            if was_single_tensor and len(results_points) == 1:
                return results_points[0], results_grid_ind[0], results_grid_ind_vox[0], results_labels[0], sensor_params_tensor
            else:
                return results_points, results_grid_ind, results_grid_ind_vox, results_labels, sensor_params_tensor
        
        # --------------------------------------------------
        # Original sampling logic (ray-casting based)
        # --------------------------------------------------
        points = [p.detach() for p in points] if isinstance(points, list) else points.detach()
        
        LiDAR_height = self.sensor_config.get('LiDAR_height', [1, 2])
        num_of_beams = self.sensor_config.get('num_of_beams', [16, 128])
        horizontal_angular_resolution = self.sensor_config.get('horizontal_angular_resolution', [900, 3600])
        lower_vertical_field_of_view_bound = self.sensor_config.get('lower_vertical_field_of_view_bound', [-40, -5])
        upper_vertical_field_of_view_bound = self.sensor_config.get('upper_vertical_field_of_view_bound', [0, 25])

        mixup_prob = 0.15

        def _sample_sensor_params_tuple():
            # Continuous sampling: rounding to 0.1m would leave stage0 (range 0.2m) with only 3 discrete target values,
            # degrading the height regression of sensor_head into classification; after the origin fix the height is continuously observable.
            lidar_height = random.uniform(LiDAR_height[0], LiDAR_height[1])
            beams = sample_beams(num_of_beams[0], num_of_beams[1])
            horizontal_resolution = random.choice(
                range(horizontal_angular_resolution[0], horizontal_angular_resolution[1] + 1, 100)
            )
            vertical_lower_angle = round(
                random.uniform(lower_vertical_field_of_view_bound[0],
                               lower_vertical_field_of_view_bound[1]), 1
            )
            vertical_upper_angle = round(
                random.uniform(upper_vertical_field_of_view_bound[0],
                               upper_vertical_field_of_view_bound[1]), 1
            )
            return (beams, lidar_height, horizontal_resolution,
                    vertical_lower_angle, vertical_upper_angle)

        def _normalize_sensor_params_tuple(params):
            beams, lidar_height, hres, vlow, vhigh = params
            beam_span = max(num_of_beams[1] - num_of_beams[0], 1)
            height_span = max(LiDAR_height[1] - LiDAR_height[0], 1e-6)
            hres_lo, hres_hi = horizontal_angular_resolution
            hres_span = max(hres_hi - hres_lo, 1)
            vlow_lo, vlow_hi = lower_vertical_field_of_view_bound
            vlow_span = max(vlow_hi - vlow_lo, 1e-6)
            vup_lo, vup_hi = upper_vertical_field_of_view_bound
            vup_span = max(vup_hi - vup_lo, 1e-6)
            return np.array([
                (beams - num_of_beams[0]) / beam_span,
                (lidar_height - LiDAR_height[0]) / height_span,
                (hres - hres_lo) / hres_span,
                (vlow - vlow_lo) / vlow_span,
                (vhigh - vup_lo) / vup_span,
            ], dtype=np.float64)

        def _sensor_params_distance(params_a, params_b):
            diff = _normalize_sensor_params_tuple(params_a) - _normalize_sensor_params_tuple(params_b)
            return float(np.linalg.norm(diff))

        def _sample_diverse_sensor_params(repeat_count, num_candidates=12):
            """Greedy maximin: each new sample maximizes min distance to prior ones."""
            if repeat_count <= 1:
                return [_sample_sensor_params_tuple()]
            selected = [_sample_sensor_params_tuple()]
            for _ in range(repeat_count - 1):
                best_candidate = None
                best_min_dist = -1.0
                for _ in range(num_candidates):
                    candidate = _sample_sensor_params_tuple()
                    min_dist = min(_sensor_params_distance(candidate, prev) for prev in selected)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_candidate = candidate
                selected.append(best_candidate)
            return selected

        def _run_single_sampling(sensor_params_tuple=None, frame_indices=None):
            # Perturbation suite: read once outside the frame loop, read-only inside it. pcfg['enable'] is the master switch.
            # getattr is used instead of accessing self.perturb_cfg directly: analysis scripts
            # construct a lightweight shell that bypasses __init__ and then bind
            # _sampling to it, in which case this attribute does not exist. Missing is treated as all off, identical to the behaviour
            # without the perturbation suite.
            pcfg = getattr(self, 'perturb_cfg', None) or dict(
                enable=False, angular_jitter_mrad=0.0, range_noise_cm=0.0,
                dropout_p=0.0, dropout_exempt_minority=False)
            _pt_on = bool(pcfg['enable'])
            # frame_indices: process only the given frames (for drawing a config independently per scene); None = all frames.
            # --------------------------------------------------
            # 0. Sample the sensor parameters
            # --------------------------------------------------
            if sensor_params_tuple is None:
                (beams, lidar_height, horizontal_resolution,
                 vertical_lower_angle, vertical_upper_angle) = _sample_sensor_params_tuple()
            else:
                (beams, lidar_height, horizontal_resolution,
                 vertical_lower_angle, vertical_upper_angle) = sensor_params_tuple
            # can be used for fixed-parameter debugging
            # lidar_height = 1.73
            # beams = 64
            # horizontal_resolution = 1800
            # vertical_lower_angle = -24.9
            # vertical_upper_angle = 2

            results_points = []
            results_grid_ind = []
            results_grid_ind_vox = []
            results_labels = []
        
            device = points[0].device
            dtype = points[0].dtype
            sensor_params = torch.tensor(
                [beams, lidar_height, horizontal_resolution,
                vertical_lower_angle, vertical_upper_angle],
                device=device
            )

            # Polar / voxel grid constants (exactly the same as PointsegMapping), used in step 10 to recompute
            # grid_ind / grid_ind_vox / relative coordinates after "points are snapped to rays + depth changed by ring smoothing",
            # keeping train_grid and the indices self-consistent.
            min_bound_s = self.min_bound.to(device=device, dtype=dtype)
            max_bound_s = self.max_bound.to(device=device, dtype=dtype)
            intervals_s = (max_bound_s - min_bound_s) / self.grid_size.to(device=device, dtype=dtype)
            intervals_vox_s = (max_bound_s - min_bound_s) / self.grid_size_vox.to(device=device, dtype=dtype)
        
            # --------------------------------------------------
            # Global hyperparameters (managed centrally)
            # --------------------------------------------------
            # basics
            MAX_RANGE = 80.0
            EPS = 1e-8
            LARGE_VAL = 1e10
        
            # =========================
            # [FIX #1] angular-domain candidate filtering
            # =========================
            # 0.65 -> 1.0: each bin uses its own full footprint. Under point->nearest-bin
            # (round) assignment the bins exactly partition the angular space, so 1.0 is the physically correct setting.
            ANGULAR_MARGIN_SCALE = 1.0
        
            # =========================
            # Occlusion envelope suppression (main occlusion parameters)
            # =========================
            OCC_AZ_WIN = 11
            # Meaning: window width (odd) in azimuth (horizontal direction) examined for the occlusion test.
            OCC_EL_WIN = 10
            # Meaning: how many layers of "upper beams" are examined for the occlusion test.
            OCC_SUPPORT_MIN = 3
            # Meaning: minimum number of support points in the upper neighbourhood that are "close to the nearest foreground depth".
            OCC_DEPTH_MARGIN = 0.5   # [FIX #3] was 0.35
            # Meaning: a point is considered occluded only if it is at least this much "deeper (metres)" than the foreground reference point.
            # Raised to 0.5 so that thin objects next to walls (poles, pedestrians, cyclists) are not wrongly removed.
            OCC_NEAR_BAND = 0.25
            OCC_REL_MARGIN = 0.02
            RING_KEEP_BAND = 0.20    # [FIX #3] was 0.12
            # Bandwidth of the "similar depth" test for same-ring neighbours, widened to protect thin objects more easily.
            RING_KEEP_MIN = 6        # [FIX #3] was 3
            # Number of similar neighbours required in the same-ring neighbourhood (6, out of the 2×RING_KEEP_WIN = 14
            # neighbours in the window).
            RING_KEEP_WIN = 7
            # Half-width of the same-ring check window (window 2×7+1 = 15 bins; the neighbourhood contains 14 neighbours, skipping the center).
        
            # =========================
            # Ring continuity (local median shrinkage)
            # =========================
            SMOOTH_WIN = 15
            SMOOTH_SUPPORT_MIN = 4
            SMOOTH_SPREAD_MAX = 1.6
        
            # distance-adaptive ring continuity (shrinkage strength)
            SMOOTH_REL_THRESH = 0.002
            SMOOTH_ABS_FLOOR = 0.02
            SMOOTH_BLEND_NEAR = 0.88
            SMOOTH_BLEND_FAR = 0.8   # [FIX #4] was 0.995
            # Maximum shrinkage ratio at long range. The original 0.995 almost completely replaced far outliers with the median,
            # which badly hurt far minority classes (e.g. distant poles/pedestrians); lowered to 0.8 as a global safety net,
            # plus the class bypass below as a second protection.
            SMOOTH_BLEND_REF = 8.0
        
            # additional "fit the local main trend" gate
            SMOOTH_TREND_REL_BAND = 0.010
            SMOOTH_TREND_ABS_BAND = 0.03
        
            # =========================
            # [NEW] Class awareness (minority-class protection)
            # =========================
            # uses the mapped compressed CL labels: 5=Living Being, 6=Objects
            MINORITY_CLASSES = (5, 6)
        
            # --------------------------------------------------
            # Utility functions
            # --------------------------------------------------
            def wrap_to_pi(angle):
                return (angle + torch.pi) % (2 * torch.pi) - torch.pi
        
            def make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox):
                device = curr_points.device
                dtype = curr_points.dtype
                return (
                    torch.empty((0, curr_points.shape[1]), device=device, dtype=dtype),
                    torch.empty((0,), device=device, dtype=curr_labels.dtype),
                    torch.empty((0, 3), device=device, dtype=curr_grid_ind.dtype),
                    torch.empty((0, 3), device=device, dtype=curr_grid_ind_vox.dtype),
                )
        
            def circular_median_filter_2d(depth_img, hit_img, win):
                """Batch version: process all beams at once. depth_img/hit_img: [B, H]."""
                B, H = depth_img.shape
                pad = win // 2
                depth_pad = torch.cat(
                    [depth_img[:, -pad:], depth_img, depth_img[:, :pad]], dim=1
                )  # [B, H+2*pad]
                valid_pad = torch.cat(
                    [hit_img[:, -pad:], hit_img, hit_img[:, :pad]], dim=1
                )
                depth_unf = depth_pad.unfold(1, win, 1)  # [B, H, win]
                valid_unf = valid_pad.unfold(1, win, 1)
                masked = torch.where(
                    valid_unf, depth_unf, torch.full_like(depth_unf, LARGE_VAL)
                )
                sorted_vals, _ = torch.sort(masked, dim=2)
                support = valid_unf.sum(dim=2)
                med_idx = torch.clamp(
                    torch.div(support - 1, 2, rounding_mode='floor'), min=0
                ).long().clamp(max=win - 1)
                max_idx = torch.clamp(support - 1, min=0).long().clamp(max=win - 1)
                med_row = sorted_vals.gather(2, med_idx.unsqueeze(2)).squeeze(2)
                min_vals = sorted_vals[:, :, 0]
                max_vals = sorted_vals.gather(2, max_idx.unsqueeze(2)).squeeze(2)
                spread_row = max_vals - min_vals
                return med_row, spread_row, support
        
            # --------------------------------------------------
            # Precomputation: convert to scalar/float constants to speed up the subsequent filtering
            # --------------------------------------------------
            # [For visualization] beam_concentration: concentrates the vertical angles towards 0° (the horizon).
            # Uniform angles are very non-uniform on the ground -- a beam at elevation θ lands at radius h/tan|θ|; with 41 beams over
            # -27.4°..3°, 78% of the downward beams are crammed within 15m and only 6 land in 15-50m,
            # which in BEV is an overly dense cluster of rings around the vehicle. Real spinning LiDARs (e.g. HDL-64E) have dense beams near
            # the horizon and sparse downward beams. A power warp |t|**c is applied on both sides; endpoints, beam count and
            # monotonicity are completely unchanged. Default None = the original linspace behaviour, bit-identical (the training path is unaffected).
            _beam_conc = getattr(self, 'beam_concentration', None)
            if _beam_conc is None:
                vertical_angles_rad = torch.deg2rad(
                    torch.linspace(vertical_lower_angle, vertical_upper_angle, beams, device=device, dtype=dtype)
                )
            else:
                _c = float(_beam_conc)
                _lo, _up = float(vertical_lower_angle), float(vertical_upper_angle)
                _n_dn = max(min(int(round(beams * abs(_lo) / max(abs(_lo) + abs(_up), 1e-6))),
                                beams - 1), 1)
                _n_up = beams - _n_dn
                _t_dn = torch.linspace(1.0, 0.0, _n_dn + 1, device=device, dtype=dtype)[:-1]
                _t_up = torch.linspace(0.0, 1.0, _n_up, device=device, dtype=dtype)
                _deg = torch.cat([_lo * _t_dn.pow(_c), _up * _t_up.pow(_c)])
                vertical_angles_rad = torch.deg2rad(torch.sort(_deg).values)
            horizontal_step = 2.0 * math.pi / horizontal_resolution
            # Random azimuth zero phase: the angular ticks of a real spinning LiDAR have an arbitrary phase relative to the scene;
            # a fixed phase-0 angular comb is a regularity specific to the simulator, which the model would exploit as a shortcut feature.
            az_phase = torch.rand((), device=device, dtype=dtype) * horizontal_step
            horizontal_angles_rad = torch.arange(horizontal_resolution, device=device, dtype=dtype) * horizontal_step + az_phase
            vertical_step = (vertical_angles_rad[-1] - vertical_angles_rad[0]).item() / max(1, beams - 1)
        
            vv, hh = torch.meshgrid(vertical_angles_rad, horizontal_angles_rad, indexing='ij')
            ray_directions = torch.stack([
                torch.cos(vv) * torch.cos(hh),
                torch.cos(vv) * torch.sin(hh),
                torch.sin(vv)
            ], dim=-1).reshape(-1, 3)
            # lidar_origin is computed per frame (see inside the frame loop): the point cloud is in the sensor frame (z=0 is the original
            # sensor, the ground is at z<0, about -2m). lidar_height means "mounting height above ground"; if it were used directly as z,
            # the effective mounting height would become lidar_height + real height ≈ 3~4m, and the ground-ring
            # geometry / near-field blind zone of all simulated scans would systematically deviate from real sensors. Correct: origin_z = ground_z + lidar_height.
        
            az_half = 0.5 * horizontal_step
            if beams > 1:
                if _beam_conc is None:
                    el_half = 0.5 * vertical_step
                else:
                    # With non-uniform spacing use the largest adjacent spacing, so that points at the sparsest place still fall in their own footprint
                    # (using the smallest spacing would make points at the sparse end be judged out of bounds and lost entirely).
                    el_half = 0.5 * float(torch.diff(vertical_angles_rad).max())
            else:
                el_half = 1e6
        
            # Scalar extraction (greatly speeds up the operations inside the loop)
            az_margin_val = az_half * ANGULAR_MARGIN_SCALE
            el_margin_val = el_half * ANGULAR_MARGIN_SCALE
            footprint_angle_val = math.sqrt(az_half ** 2 + el_half ** 2) * ANGULAR_MARGIN_SCALE
            cos_footprint = math.cos(footprint_angle_val)
            num_rays = beams * horizontal_resolution
        
            # --------------------------------------------------
            # 1. Main per-frame sampling loop
            # --------------------------------------------------
            frame_iter = frame_indices if frame_indices is not None else range(len(points))
            for idx in frame_iter:
                curr_points = points[idx]
                curr_labels = labels[idx]
                curr_grid_ind = grid_ind[idx]
                curr_grid_ind_vox = grid_ind_vox[idx]

                # --------------------------------------------------
                # 0. reset_random real scans: passed through unchanged (no mixup, no simulated sampling),
                #    serving as anchor samples of the real scan pattern.
                # --------------------------------------------------
                if raw_mask is not None and raw_mask[idx]:
                    results_points.append(curr_points)
                    results_labels.append(curr_labels)
                    results_grid_ind.append(curr_grid_ind)
                    results_grid_ind_vox.append(curr_grid_ind_vox)
                    continue

                # --------------------------------------------------
                # 1. Minority-class Copy / Mixup (kept)
                # --------------------------------------------------
                if random.random() < mixup_prob:
                    curr_points, curr_grid_ind, curr_grid_ind_vox, curr_labels = \
                        self._mixup_minority(
                            curr_points, curr_grid_ind, curr_grid_ind_vox, curr_labels, num_add=2
                        )
        
                # --------------------------------------------------
                # 2. Take xyz (according to the storage format)
                # --------------------------------------------------
                point_cloud_tensor = torch.cat(
                    (curr_points[:, 6:8], curr_points[:, 5].unsqueeze(1)),
                    dim=1
                )  # [N, 3], x y z

                # Estimate the ground height of this frame: prefer the median z of the COLA ground classes (1=Driveable Ground, 7=Other Ground,
                # same convention as _mixup_minority) at near range (rho<20m); without ground points fall back to
                # the 2% quantile of all z (kthvalue, avoiding torch.quantile's large-tensor limit).
                # origin_z = ground_z + lidar_height restores the "height above ground" semantics of lidar_height,
                # so that real sensor geometry (e.g. nuScenes 1.84m) falls within the training distribution.
                ground_mask = ((curr_labels == 1) | (curr_labels == 7)) & (curr_points[:, 3] < 20.0)
                if ground_mask.any():
                    ground_z = curr_points[ground_mask, 5].median()
                else:
                    z_all = curr_points[:, 5]
                    k = max(1, int(0.02 * z_all.numel()))
                    ground_z = z_all.kthvalue(k).values
                lidar_origin = torch.stack([
                    torch.zeros((), device=device, dtype=dtype),
                    torch.zeros((), device=device, dtype=dtype),
                    (ground_z + lidar_height).to(dtype),
                ])

                vec_all = point_cloud_tensor - lidar_origin
                range_all = torch.norm(vec_all, dim=1)
        
                # --------------------------------------------------
                # 3. Range filtering
                # --------------------------------------------------
                keep_mask = (range_all > EPS) & (range_all <= MAX_RANGE)
                if not keep_mask.any():
                    sampled_points, sampled_labels, sampled_grid_ind, sampled_grid_ind_vox = \
                        make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox)
                    results_points.append(sampled_points)
                    results_labels.append(sampled_labels)
                    results_grid_ind.append(sampled_grid_ind)
                    results_grid_ind_vox.append(sampled_grid_ind_vox)
                    continue
                kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
                kept_points_raw = curr_points[kept_idx]
                kept_labels = curr_labels[kept_idx]
                kept_grid_ind = curr_grid_ind[kept_idx]
                kept_grid_ind_vox = curr_grid_ind_vox[kept_idx]
        
                kept_xyz = point_cloud_tensor[kept_idx]   # [M, 3]
                kept_vec = vec_all[kept_idx]              # [M, 3]
                kept_range = range_all[kept_idx]          # [M]
                kept_range_safe = torch.clamp(kept_range, min=EPS)
        
                # --------------------------------------------------
                # 4. Map each point to the nearest ray center (ray_directions is precomputed)
                # --------------------------------------------------
                point_az = torch.atan2(kept_vec[:, 1], kept_vec[:, 0]) % (2.0 * torch.pi)
                point_el = torch.asin(
                    torch.clamp(kept_vec[:, 2] / kept_range_safe, -1.0, 1.0)
                )
                az_bin = torch.round((point_az - az_phase) / horizontal_step).long() % horizontal_resolution
                if beams > 1:
                    if _beam_conc is None:
                        el_bin = torch.round((point_el - vertical_angles_rad[0]) / max(vertical_step, EPS)).long()
                    else:
                        # With non-uniform vertical angles (beam_concentration on), inverting with a fixed step is invalid,
                        # so nearest-beam lookup is used instead. With uniform spacing both give the same result.
                        _eb = torch.searchsorted(
                            vertical_angles_rad.contiguous(), point_el.contiguous())
                        _eb = _eb.clamp(1, beams - 1)
                        _lo_closer = ((point_el - vertical_angles_rad[_eb - 1]).abs()
                                      <= (vertical_angles_rad[_eb] - point_el).abs())
                        el_bin = torch.where(_lo_closer, _eb - 1, _eb)
                else:
                    el_bin = torch.zeros_like(az_bin)
                el_bin.clamp_(0, beams - 1)
                flat_bin = el_bin * horizontal_resolution + az_bin
                ray_dir_for_points = ray_directions[flat_bin]
        
                # --------------------------------------------------
                # 6. Angular consistency filtering (drops the expensive arccos)
                #    [FIX #1] az_margin/el_margin are now the full az_half/el_half,
                #    so all points fall within the footprint of their nearest bin.
                # --------------------------------------------------
                unit_vec = kept_vec / kept_range.unsqueeze(1)
                cos_sim = torch.sum(unit_vec * ray_dir_for_points, dim=1).clamp_(-1.0, 1.0)
        
                assigned_az = horizontal_angles_rad[az_bin]
                assigned_el = vertical_angles_rad[el_bin]
        
                diff_az = torch.abs(point_az - assigned_az)
                delta_az = torch.min(diff_az, 2.0 * math.pi - diff_az)
                delta_el = torch.abs(point_el - assigned_el)
                depths = torch.sum(kept_vec * ray_dir_for_points, dim=1)
        
                valid_mask = (
                    (depths > 0.0) &
                    (delta_az <= az_margin_val) &
                    (delta_el <= el_margin_val) &
                    (cos_sim >= cos_footprint)
                )
                if not valid_mask.any():
                    sp, sl, sgi, sgiv = make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox)
                    results_points.append(sp); results_labels.append(sl)
                    results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
                    continue
                valid_bins = flat_bin[valid_mask]
                valid_depths = depths[valid_mask]
        
                # --------------------------------------------------
                # 7. First layer: each bin keeps only the nearest point (basic z-buffer)
                # --------------------------------------------------
                num_rays = beams * horizontal_resolution
                min_depth_flat, min_idx_in_valid = scatter_min(
                    valid_depths,
                    valid_bins,
                    dim=0,
                    dim_size=num_rays
                )
                hit_mask_flat = torch.isfinite(min_depth_flat)
                K = valid_depths.shape[0]
                idx_ok = (min_idx_in_valid >= 0) & (min_idx_in_valid < K)
                hit_mask_flat = hit_mask_flat & idx_ok
        
                # map indices of the valid subset back to indices of the kept subset
                valid_pos = valid_mask.nonzero(as_tuple=False).squeeze(1)
                chosen_kept_idx_flat = torch.full(
                    (num_rays,), -1, device=device, dtype=torch.long
                )
                chosen_kept_idx_flat[hit_mask_flat] = valid_pos[min_idx_in_valid[hit_mask_flat]]
        
                depth_img = min_depth_flat.view(beams, horizontal_resolution)          # [B, H]
                hit_img = hit_mask_flat.view(beams, horizontal_resolution)             # [B, H]
                idx_img = chosen_kept_idx_flat.view(beams, horizontal_resolution)      # [B, H]

                # Raised tolerance for minority classes. The MinorityDensePurify step of the data pipeline has already stripped hallucinated
                # minority points (completed by LiDiff, deviating >0.2m from the real body) to ignore, so the remaining minority candidates
                # are all real returns, and the tolerance can be widened from 0.05 to 0.3: this rescues boundary bins such as feet taken over by
                # ground-grazing / silhouette points (an audit measured 0.3 rescuing 2-10% more instance bins than 0.05, up to 20% under favorable
                # configurations), without again "seeing pedestrians through occluders" (the hallucinations have been purified).
                MINORITY_DEPTH_TOL = 0.3   # metres, tunable

                # take the subset of all valid candidates that belongs to minority classes
                valid_labels_all = kept_labels[valid_pos]            # [V], V = valid_mask.sum()
                is_minor_valid = torch.zeros_like(valid_labels_all, dtype=torch.bool)
                for _cls in MINORITY_CLASSES:
                    is_minor_valid = is_minor_valid | (valid_labels_all == _cls)

                if is_minor_valid.any():
                    minor_bins = valid_bins[is_minor_valid]
                    minor_depths = valid_depths[is_minor_valid]
                    minor_local_pos = is_minor_valid.nonzero(as_tuple=False).squeeze(1)   # index within valid_*

                    # run scatter_min again on the minority candidates to get "the nearest minority point of each bin"
                    minor_min_depth, minor_min_idx = scatter_min(
                        minor_depths, minor_bins, dim=0, dim_size=num_rays
                    )
                    minor_hit = torch.isfinite(minor_min_depth)
                    K_minor = minor_depths.shape[0]
                    minor_ok = (minor_min_idx >= 0) & (minor_min_idx < K_minor)
                    minor_hit = minor_hit & minor_ok

                    # compare with the current nearest depth of the bin: a minority point may not be farther than the original nearest point by more than TOL
                    promote_mask = (
                        minor_hit &
                        hit_mask_flat &
                        (minor_min_depth <= min_depth_flat + MINORITY_DEPTH_TOL)
                    )

                    # also allow bins "originally hit by nothing but having a minority point" (a safeguard; should not happen in theory since minority points are also in valid)
                    promote_only_minor = minor_hit & (~hit_mask_flat)
                    promote_mask = promote_mask | promote_only_minor

                    if promote_mask.any():
                        # replace chosen_kept_idx_flat and min_depth_flat
                        new_kept_idx = valid_pos[minor_local_pos[minor_min_idx[promote_mask]]]
                        chosen_kept_idx_flat[promote_mask] = new_kept_idx
                        min_depth_flat = torch.where(promote_mask, minor_min_depth, min_depth_flat)
                        hit_mask_flat = hit_mask_flat | promote_only_minor

                # refresh depth_img / hit_img / idx_img
                depth_img = min_depth_flat.view(beams, horizontal_resolution)
                hit_img   = hit_mask_flat.view(beams, horizontal_resolution)
                idx_img   = chosen_kept_idx_flat.view(beams, horizontal_resolution)
        
                # --------------------------------------------------
                # [NEW] 7.5  Semantic class of each bin + minority mask
                #     used for the minority protection in step 8 / step 9.
                # --------------------------------------------------
                safe_idx_img = torch.clamp(idx_img, min=0)        # temporarily replace -1 with 0 for gather
                labels_lookup = kept_labels[safe_idx_img]         # [B, H]; values at -1 are garbage
                minority_bin_mask = torch.zeros_like(hit_img)
                for _cls in MINORITY_CLASSES:
                    minority_bin_mask = minority_bin_mask | (labels_lookup == _cls)
                minority_bin_mask = minority_bin_mask & hit_img   # only valid on bins that actually have a point
        
                # --------------------------------------------------
                # 8. Second layer: improved occlusion suppression
                #    Goal: keep continuous far ground, only remove background points that "really look occluded by the foreground"
                #    [FIX #3] minority bins skip leak_mask entirely
                # --------------------------------------------------
                depth_for_occ = torch.where(hit_img, depth_img, depth_img.new_tensor(LARGE_VAL))
        
                # 8.1 collect occlusion references only from the "upper beams", no longer symmetric up/down
                az_pad = OCC_AZ_WIN // 2
                upper_stack_list = []
                for db in range(1, OCC_EL_WIN + 1):
                    src_b = torch.clamp(torch.arange(beams, device=device) + db, max=beams - 1)
                    upper_depth = depth_for_occ[src_b]
                    upper_pad = F.pad(
                        upper_depth.unsqueeze(0).unsqueeze(0),
                        (az_pad, az_pad, 0, 0), mode='circular'
                    ).squeeze(0).squeeze(0)
                    upper_unf = upper_pad.unfold(1, OCC_AZ_WIN, 1)
                    upper_stack_list.append(upper_unf)
        
                if upper_stack_list:
                    upper_stack = torch.cat(upper_stack_list, dim=2)
                    occ_local_min = upper_stack.amin(dim=2)
                    near_support = (upper_stack <= (occ_local_min.unsqueeze(2) + OCC_NEAR_BAND)).sum(dim=2)
                else:
                    occ_local_min = torch.full_like(depth_img, LARGE_VAL)
                    near_support = torch.zeros_like(depth_img, dtype=torch.long)
        
                # 8.2 same-ring continuity protection (vectorized unfold instead of multiple rolls)
                ring_pad = RING_KEEP_WIN
                depth_pad = F.pad(
                    depth_img.unsqueeze(0).unsqueeze(0),
                    (ring_pad, ring_pad, 0, 0), mode='circular'
                ).squeeze(0).squeeze(0)
                hit_pad = F.pad(
                    hit_img.unsqueeze(0).unsqueeze(0).float(),
                    (ring_pad, ring_pad, 0, 0), mode='circular'
                ).squeeze(0).squeeze(0).bool()
                win_size = 2 * RING_KEEP_WIN + 1
                depth_unf = depth_pad.unfold(1, win_size, 1)
                hit_unf = hit_pad.unfold(1, win_size, 1)
                # skip the center (index RING_KEEP_WIN). The old (0,1,2,4,5,6) was the layout from the win=7 era;
                # with win=15 it picks only neighbours to the left of the center, disabling same-ring thin-object protection.
                neighbor_idx = tuple(i for i in range(win_size) if i != RING_KEEP_WIN)
                depth_neighbors = depth_unf[:, :, neighbor_idx]
                hit_neighbors = hit_unf[:, :, neighbor_idx]
                depth_diff_ok = torch.abs(depth_neighbors - depth_img.unsqueeze(2)) <= RING_KEEP_BAND
                same_ring_support = (hit_neighbors & depth_diff_ok).sum(dim=2)
                ring_keep_mask = hit_img & (same_ring_support >= RING_KEEP_MIN)
        
                # 8.3 adaptive depth threshold
                adaptive_margin = OCC_DEPTH_MARGIN + OCC_REL_MARGIN * occ_local_min
        
                # 8.4 finally remove only points that "really look occluded by the foreground"
                #     [FIX #3] add (~minority_bin_mask) so that minority points are immune to occlusion suppression
                leak_mask = (
                    hit_img &
                    torch.isfinite(occ_local_min) &
                    (near_support >= OCC_SUPPORT_MIN) &
                    ((depth_img - occ_local_min) > adaptive_margin) &
                    (~ring_keep_mask) &
                    (~minority_bin_mask)   # [FIX #3] minority protection
                )
                # remove the points judged to "really penetrate to the background"
                hit_img = hit_img & (~leak_mask)
                idx_img = torch.where(hit_img, idx_img, torch.full_like(idx_img, -1))
                depth_img = torch.where(hit_img, depth_img, depth_img.new_tensor(LARGE_VAL))
                # minority_bin_mask needs no update: minority bins were not removed by leak_mask,
                # their hit/idx/labels are unchanged, so the flags are still correct.
        
                # --------------------------------------------------
                # 9. Third layer: distance-adaptive ring continuity (vectorized)
                #    [FIX #4] blend_factor of minority bins is forced to 0
                # --------------------------------------------------
                med_row, spread_row, support_row = circular_median_filter_2d(
                    depth_img, hit_img, SMOOTH_WIN
                )
                abs_diff = torch.abs(depth_img - med_row)
                trend_band = torch.clamp(depth_img * SMOOTH_TREND_REL_BAND, min=SMOOTH_TREND_ABS_BAND)
                trend_ok = abs_diff <= (3.0 * trend_band)
                smoothable = (
                    hit_img &
                    (support_row >= SMOOTH_SUPPORT_MIN) &
                    ((spread_row <= SMOOTH_SPREAD_MAX) | trend_ok)
                )
                adaptive_thresh = torch.maximum(
                    torch.full_like(depth_img, SMOOTH_ABS_FLOOR),
                    depth_img * SMOOTH_REL_THRESH
                )
                outlier_mask = smoothable & (abs_diff > adaptive_thresh)
                normal_mask = smoothable & (~outlier_mask)
                blend = SMOOTH_BLEND_NEAR + (
                    SMOOTH_BLEND_FAR - SMOOTH_BLEND_NEAR
                ) * torch.clamp(depth_img / SMOOTH_BLEND_REF, 0.0, 1.0)
                blend_factor = torch.where(
                    outlier_mask, blend,
                    torch.where(normal_mask, 0.5 * blend, torch.zeros_like(blend))
                )
                # [FIX #4] no ring smoothing for minority bins, so that they are not dragged to the background depth by the median
                blend_factor = torch.where(
                    minority_bin_mask, torch.zeros_like(blend_factor), blend_factor
                )
                new_row = torch.lerp(depth_img, med_row, blend_factor)
                final_depth_img = torch.where(hit_img, new_row, depth_img)
        
                # --------------------------------------------------
                # 10. Output: still "sampling on points"
                #     take the selected real points, but change their xyz to the corresponding position in final_depth_img
                # --------------------------------------------------
                final_hit_flat = hit_img.view(-1)
                final_idx_flat = idx_img.view(-1)
                final_depth_flat = final_depth_img.view(-1)
                valid_out_mask = final_hit_flat & (final_idx_flat >= 0) & torch.isfinite(final_depth_flat)
                if not valid_out_mask.any():
                    sampled_points, sampled_labels, sampled_grid_ind, sampled_grid_ind_vox = \
                        make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox)
                    results_points.append(sampled_points)
                    results_labels.append(sampled_labels)
                    results_grid_ind.append(sampled_grid_ind)
                    results_grid_ind_vox.append(sampled_grid_ind_vox)
                    continue
                chosen_kept_idx = final_idx_flat[valid_out_mask]                  # index into kept_*
                chosen_bins = valid_out_mask.nonzero(as_tuple=False).squeeze(1)   # flat bin ids
                chosen_depths = final_depth_flat[valid_out_mask].unsqueeze(1)     # [S, 1]

                # Random dropout: simulates missing/weak returns (0~8% per frame); minority points are exempt to
                # preserve the minority augmentation targets.
                drop_p = random.random() * 0.08
                if drop_p > 0 and chosen_kept_idx.numel() > 16:
                    cand_labels = kept_labels[chosen_kept_idx]
                    is_minor_out = torch.zeros_like(cand_labels, dtype=torch.bool)
                    for _cls in MINORITY_CLASSES:
                        is_minor_out = is_minor_out | (cand_labels == _cls)
                    keep_out = (torch.rand(chosen_kept_idx.shape[0], device=device) >= drop_p) | is_minor_out
                    chosen_kept_idx = chosen_kept_idx[keep_out]
                    chosen_bins = chosen_bins[keep_out]
                    chosen_depths = chosen_depths[keep_out]
                    if chosen_kept_idx.numel() == 0:
                        sp, sl, sgi, sgiv = make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox)
                        results_points.append(sp); results_labels.append(sl)
                        results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
                        continue

                # [Perturbation suite] extra random dropout: in series with the U[0,8%) term above (does not replace it).
                # The term above is data augmentation with minority exemption; this term physically models detection failures and
                # is class-agnostic by default (a real sensor does not know classes).
                if _pt_on and pcfg['dropout_p'] > 0 and chosen_kept_idx.numel() > 16:
                    p2 = pcfg['dropout_p']
                    keep2 = torch.rand(chosen_kept_idx.shape[0], device=device) >= p2
                    if pcfg['dropout_exempt_minority']:
                        cand_labels2 = kept_labels[chosen_kept_idx]
                        is_minor2 = torch.zeros_like(cand_labels2, dtype=torch.bool)
                        for _cls in MINORITY_CLASSES:
                            is_minor2 = is_minor2 | (cand_labels2 == _cls)
                        keep2 = keep2 | is_minor2
                    chosen_kept_idx = chosen_kept_idx[keep2]
                    chosen_bins = chosen_bins[keep2]
                    chosen_depths = chosen_depths[keep2]
                    if chosen_kept_idx.numel() == 0:
                        sp, sl, sgi, sgiv = make_empty_outputs(curr_points, curr_labels, curr_grid_ind, curr_grid_ind_vox)
                        results_points.append(sp); results_labels.append(sl)
                        results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
                        continue

                # [Provenance -- off by default; the default path is bit-identical]
                # Records which "input dense point" each sampled point comes from. Chain:
                #   kept_idx          : curr_points row -> kept_* row  (step 3)
                #   chosen_kept_idx   : rows of kept_* selected by a ray and surviving dropout
                #   composite kept_idx[chosen_kept_idx] = original curr_points row
                # This mapping allows "predictions of the same dense point under N sensor configurations" to be aligned,
                # which is the prerequisite of the output-level invariance metric.
                # Runs only when self.return_provenance is true; default getattr(...) -> False,
                # consumes no random numbers and reads/writes no output tensor, so the training path is bit-identical.
                if getattr(self, 'return_provenance', False):
                    if getattr(self, '_provenance_buffer', None) is None:
                        self._provenance_buffer = []
                    self._provenance_buffer.append(
                        (idx, kept_idx[chosen_kept_idx].clone()))

                # Range noise: sigma(r) = 1cm + 0.1%*r, truncated at 3sigma. The noise is injected before the grid/rel
                # recomputation and flows into all features together with the projected coordinates, keeping them self-consistent.
                RANGE_NOISE_BASE = 0.01
                RANGE_NOISE_REL = 0.001
                noise_sigma = RANGE_NOISE_BASE + RANGE_NOISE_REL * chosen_depths
                depth_noise = torch.randn_like(chosen_depths) * noise_sigma
                depth_noise = torch.maximum(torch.minimum(depth_noise, 3.0 * noise_sigma), -3.0 * noise_sigma)
                chosen_depths = chosen_depths + depth_noise

                # [Perturbation suite] extra range noise: added independently of the sigma(r)=1cm+0.1%r above
                # (total variance = existing^2 + new^2). The clamp_min(0.1) of the existing block is moved after this term
                # and done once, avoiding "clamp once -> add noise again -> negative again"; in the off state
                # clamp_min still runs exactly once on the same value, so the result is bit-identical and unaffected.
                if _pt_on and pcfg['range_noise_cm'] > 0:
                    sigma_r = pcfg['range_noise_cm'] * 0.01
                    n_extra = torch.randn_like(chosen_depths).clamp_(-3.0, 3.0) * sigma_r
                    chosen_depths = chosen_depths + n_extra
                chosen_depths = chosen_depths.clamp_min(0.1)

                sampled_points = kept_points_raw[chosen_kept_idx].clone()
                sampled_labels = kept_labels[chosen_kept_idx]
                chosen_ray_dirs = ray_directions[chosen_bins]                     # [S, 3]

                # [Perturbation suite] angular jitter: applied on the **output side** (point selection is already done), only affecting the
                # geometric position of the returned points. The global ray_directions are deliberately not perturbed -- that would make the
                # footprint filter cos_sim >= cos_footprint of step 6 additionally reject boundary points, and these losses would
                # be wrongly counted into the dropout term, confounding the three-term attribution; widening the margin could compensate,
                # but would break the "bins exactly partition the angular space" guaranteed by ANGULAR_MARGIN_SCALE=1.0.
                # Physical correspondence: the footprint effect of beam divergence is already covered by snap-to-ray (the observed azimuth
                # deviation is bounded by half an angular resolution, an order of magnitude above the divergence); this term models pointing/calibration errors.
                # sigma_theta is the std of each transverse axis; the total deviation follows a Rayleigh distribution
                # (mean 1.253 sigma). Depth unchanged -> the transverse displacement r*sigma_theta grows linearly with range
                # (1.6cm at 80m), which automatically satisfies "uncertainty grows with distance".
                if _pt_on and pcfg['angular_jitter_mrad'] > 0:
                    sigma_t = pcfg['angular_jitter_mrad'] * 1e-3           # mrad -> rad
                    d = chosen_ray_dirs                                    # [S,3] unit vectors
                    # orthonormal basis of the tangent plane: use the z axis as the auxiliary vector when |d_z| is small, otherwise the x axis, to avoid degeneracy
                    aux = torch.where(
                        (d[:, 2].abs() < 0.9).unsqueeze(1),
                        d.new_tensor([0., 0., 1.]).expand_as(d),
                        d.new_tensor([1., 0., 0.]).expand_as(d))
                    e1 = torch.cross(d, aux, dim=1)
                    e1 = e1 / e1.norm(dim=1, keepdim=True).clamp_min(EPS)
                    e2 = torch.cross(d, e1, dim=1)      # d, e1 orthogonal and unit -> e2 is already unit
                    g = torch.randn(d.shape[0], 2, device=d.device, dtype=d.dtype).clamp_(-3.0, 3.0)
                    d = d + sigma_t * (g[:, 0:1] * e1 + g[:, 1:2] * e2)
                    chosen_ray_dirs = d / d.norm(dim=1, keepdim=True).clamp_min(EPS)

                # WARNING, ordering constraint:
                # projected_xyz must be computed **after** the angular jitter -- that way the polar /
                # voxel index / relative-coordinate recomputation below automatically follows the jittered directions, with no extra
                # consistency handling. If the jitter were moved after this line, train_grid would become inconsistent with grid_ind /
                # grid_ind_vox (a train_grid/index consistency check catches this).
                projected_xyz = lidar_origin.unsqueeze(0) + chosen_depths * chosen_ray_dirs

                # After a point is snapped to the ray center and its depth is changed by ring smoothing, its absolute coordinates have changed,
                # so the polar coordinates / in-voxel relative coordinates / grid indices must be recomputed accordingly, otherwise train_grid is
                # inconsistent with grid_ind / grid_ind_vox. The formulas match PointsegMapping.
                proj_x = projected_xyz[:, 0]
                proj_y = projected_xyz[:, 1]
                proj_z = projected_xyz[:, 2]
                proj_rho = torch.sqrt(proj_x * proj_x + proj_y * proj_y)
                proj_phi = torch.atan2(proj_y, proj_x)
                xyz_pol = torch.stack([proj_rho, proj_phi, proj_z], dim=1)         # [S, 3]
                xyz_pol_grid = torch.maximum(
                    torch.minimum(xyz_pol, max_bound_s - 1e-3), min_bound_s
                )
                new_grid_ind = torch.floor((xyz_pol_grid - min_bound_s) / intervals_s)
                new_grid_ind_vox = (xyz_pol_grid - min_bound_s) / intervals_vox_s
                voxel_centers = (new_grid_ind + 0.5) * intervals_s + min_bound_s
                rel_xyz = xyz_pol - voxel_centers                                  # uses the unclipped values, same as PointsegMapping

                # write back all coordinate-dependent columns: [0:3]=rel_polar, 3=rho, 4=phi, 5=z, 6=x, 7=y
                sampled_points[:, 0:3] = rel_xyz
                sampled_points[:, 3] = proj_rho
                sampled_points[:, 4] = proj_phi
                sampled_points[:, 5] = proj_z
                sampled_points[:, 6] = proj_x
                sampled_points[:, 7] = proj_y
                sampled_grid_ind = new_grid_ind.to(kept_grid_ind.dtype)
                sampled_grid_ind_vox = new_grid_ind_vox.to(kept_grid_ind_vox.dtype)
                results_points.append(sampled_points)
                results_labels.append(sampled_labels)
                results_grid_ind.append(sampled_grid_ind)
                results_grid_ind_vox.append(sampled_grid_ind_vox)

            return results_points, results_grid_ind, results_grid_ind_vox, results_labels, sensor_params


        # --------------------------------------------------
        # consistency logic (keeps the original implementation)
        # --------------------------------------------------
        if self.consistency:
            # Draw an independent maximin-diverse pair of configs per scene (sharing one pair across the batch would reduce the per-step config
            # coverage by a factor of B and make the sensor regression target constant within the batch), and assemble in "interleaved order":
            # the two views of the same scene are adjacent, i.e. [scene0_v0, scene0_v1, scene1_v0, ...],
            # matching the per_view_voxel=[vl0,vl0,vl1,vl1,...] assembly in forward_train and
            # the (0,1), (2,3)... pairing assumption inside the aggregator.
            # Normalize to list-of-frames (defensive: uncollated single-frame [N,C] tensor input)
            if torch.is_tensor(points) and points.dim() == 2:
                points, grid_ind, grid_ind_vox, labels = [points], [grid_ind], [grid_ind_vox], [labels]
            num_frames = len(points) if isinstance(points, list) else points.shape[0]
            all_points, all_grid_ind, all_grid_ind_vox, all_labels = [], [], [], []
            sensor_params_entries = []
            for f in range(num_frames):
                pair = _sample_diverse_sensor_params(self.sample_repeat_count)
                for v in range(self.sample_repeat_count):
                    res_points, res_grid, res_grid_vox, res_labels, sensor_params = _run_single_sampling(
                        sensor_params_tuple=pair[v], frame_indices=[f])
                    all_points.append(res_points[0])
                    all_grid_ind.append(res_grid[0])
                    all_grid_ind_vox.append(res_grid_vox[0])
                    all_labels.append(res_labels[0])
                    # raw frames have no GT sensor parameters: NaN sentinel → the aggregator takes the prediction-conditioned branch
                    if raw_mask is not None and raw_mask[f]:
                        sensor_params_entries.append(torch.full_like(sensor_params, float('nan')))
                    else:
                        sensor_params_entries.append(sensor_params)

            sensor_params_tensor = torch.stack(sensor_params_entries, dim=0) if sensor_params_entries else None
            return all_points, all_grid_ind, all_grid_ind_vox, all_labels, sensor_params_tensor

        # non-consistency: likewise draw configs independently per frame
        if isinstance(points, list):
            all_points, all_grid_ind, all_grid_ind_vox, all_labels = [], [], [], []
            entries = []
            for f in range(len(points)):
                res_points, res_grid, res_grid_vox, res_labels, sensor_params = _run_single_sampling(
                    frame_indices=[f])
                all_points.append(res_points[0])
                all_grid_ind.append(res_grid[0])
                all_grid_ind_vox.append(res_grid_vox[0])
                all_labels.append(res_labels[0])
                # raw frames have no GT sensor parameters: NaN sentinel → the aggregator takes the prediction-conditioned branch
                if raw_mask is not None and raw_mask[f]:
                    entries.append(torch.full_like(sensor_params, float('nan')))
                else:
                    entries.append(sensor_params)
            sensor_params_tensor = torch.stack(entries, dim=0) if entries else None
            return all_points, all_grid_ind, all_grid_ind_vox, all_labels, sensor_params_tensor

        # single-tensor fallback path (rarely triggered): the whole batch shares one config
        res_points, res_grid, res_grid_vox, res_labels, sensor_params = _run_single_sampling()
        if res_points:
            entries = []
            for f in range(len(res_points)):
                if raw_mask is not None and raw_mask[f]:
                    entries.append(torch.full_like(sensor_params, float('nan')))
                else:
                    entries.append(sensor_params)
            sensor_params_tensor = torch.stack(entries, dim=0)
        else:
            sensor_params_tensor = None
        return res_points, res_grid, res_grid_vox, res_labels, sensor_params_tensor
    

    # ==================================================================
    # Risley-prism non-repetitive (solid-state) scan sampler
    # ------------------------------------------------------------------
    # Standalone prototype: demonstrates that the point-centric
    # formulation depends only on "a set of ray directions", not on the
    # regular ring grid of a spinning mechanical LiDAR.
    #
    # INFERENCE ONLY. Never called during training; does not touch
    # _run_single_sampling (the mechanical path) in any way, and returns
    # no sensor_params (solid-state parameters are NOT fed to S-Film).
    #
    # Reused verbatim from the mechanical path: ground/origin estimation,
    # range filter, z-buffer reduction (scatter_min), minority-class
    # promotion + exemption, range noise, random dropout, output/reprojection.
    # Rewritten: ray construction (Risley trajectory), point->ray assignment
    # (exact kd-tree NN instead of analytic binning), occlusion suppression
    # (ray kNN graph instead of range-image unfold).
    # Disabled: ring continuity median filter (no scanlines exist).
    # ==================================================================
    def sample_solid_state(self,
                           points,
                           grid_ind,
                           grid_ind_vox,
                           labels,
                           integration_time=0.5,
                           point_rate=240000.0,
                           fov_half_deg=35.2,
                           rho_deg=None,
                           f1_hz=7294.0,
                           freq_ratio=None,
                           beam_divergence_deg=0.28,
                           footprint_scale=1.0,
                           lidar_height=1.73,
                           occlusion_mode='knn',
                           occ_knn_k=48,
                           seed=None):
        """Sample a simulated Livox-Avia-style non-repetitive scan from dense points.

        Args:
            points/grid_ind/grid_ind_vox/labels: same list-of-frames layout as
                ``_sampling`` (each frame [N, 8] in PointsegMapping column order).
            integration_time (float): T in seconds. R = point_rate * T.
            point_rate (float): points/second (Avia first-return spec: 240k).
            fov_half_deg (float): half-angle of the circular FoV cone.
            rho_deg (float|None): per-prism deflection. None -> fov_half_deg/2,
                which forces rho1 == rho2 so theta reaches 0 (dense centre, no hole).
            f1_hz (float): first prism rotation frequency.
            freq_ratio (float|None): f2/f1. None -> -1/golden_ratio (counter-rotating,
                hardest-to-rationally-approximate ratio => longest pattern closure).
            beam_divergence_deg (float): lower bound of the footprint angle.
            footprint_scale (float): multiplies the local mean inter-ray spacing.
            lidar_height (float): mounting height above ground, same semantics as
                the mechanical path (origin_z = ground_z + lidar_height).
            occlusion_mode (str): 'knn' (option A, ray-neighbourhood reconstruction)
                or 'none' (option B, disabled).
            occ_knn_k (int): neighbourhood size for the occlusion graph.
            seed (int|None): if given, seeds prism phases for reproducibility.

        Returns:
            (results_points, results_grid_ind, results_grid_ind_vox, results_labels,
             info_dicts) — first four match ``_sampling``'s output contract exactly;
            info_dicts carries per-frame diagnostics (ray count, hit count, etc).
        """
        from scipy.spatial import cKDTree

        if isinstance(points, torch.Tensor) and points.dim() == 2:
            points = [points]
            grid_ind = [grid_ind]
            grid_ind_vox = [grid_ind_vox]
            labels = [labels]

        device = points[0].device
        dtype = points[0].dtype

        # ---- constants reused verbatim from the mechanical path ----
        MAX_RANGE = 80.0
        EPS = 1e-8
        LARGE_VAL = 1e10
        MINORITY_CLASSES = (5, 6)
        MINORITY_DEPTH_TOL = 0.3
        OCC_SUPPORT_MIN = 3
        OCC_DEPTH_MARGIN = 0.5
        OCC_NEAR_BAND = 0.25
        OCC_REL_MARGIN = 0.02
        RING_KEEP_BAND = 0.20
        RING_KEEP_MIN = 6
        RANGE_NOISE_BASE = 0.01
        RANGE_NOISE_REL = 0.001

        min_bound_s = self.min_bound.to(device=device, dtype=dtype)
        max_bound_s = self.max_bound.to(device=device, dtype=dtype)
        intervals_s = (max_bound_s - min_bound_s) / self.grid_size.to(device=device, dtype=dtype)
        intervals_vox_s = (max_bound_s - min_bound_s) / self.grid_size_vox.to(device=device, dtype=dtype)

        if rho_deg is None:
            rho_deg = fov_half_deg / 2.0          # rho1 == rho2 => theta in [0, fov_half]
        if freq_ratio is None:
            freq_ratio = -1.0 / ((1.0 + math.sqrt(5.0)) / 2.0)   # counter-rotating, golden

        num_rays = int(round(point_rate * integration_time))
        rho_rad = math.radians(rho_deg)
        f2_hz = f1_hz * freq_ratio

        results_points, results_grid_ind = [], []
        results_grid_ind_vox, results_labels = [], []
        info_dicts = []

        for idx in range(len(points)):
            curr_points = points[idx]
            curr_labels = labels[idx]
            curr_grid_ind = grid_ind[idx]
            curr_grid_ind_vox = grid_ind_vox[idx]

            def _empty():
                return (
                    torch.empty((0, curr_points.shape[1]), device=device, dtype=dtype),
                    torch.empty((0,), device=device, dtype=curr_labels.dtype),
                    torch.empty((0, 3), device=device, dtype=curr_grid_ind.dtype),
                    torch.empty((0, 3), device=device, dtype=curr_grid_ind_vox.dtype),
                )

            # --------------------------------------------------
            # RAY CONSTRUCTION (rewritten): Risley two-prism trajectory
            #   u(t) = r1 cos(2pi f1 t + psi1) + r2 cos(2pi f2 t + psi2)
            #   v(t) = r1 sin(2pi f1 t + psi1) + r2 sin(2pi f2 t + psi2)
            # Phases are redrawn every frame -> non-repetitive across frames.
            # --------------------------------------------------
            gen = np.random.default_rng(None if seed is None else seed + idx)
            psi1 = gen.uniform(0.0, 2.0 * np.pi)
            psi2 = gen.uniform(0.0, 2.0 * np.pi)

            t = np.linspace(0.0, integration_time, num_rays, endpoint=False)
            a1 = 2.0 * np.pi * f1_hz * t + psi1
            a2 = 2.0 * np.pi * f2_hz * t + psi2
            u = rho_rad * np.cos(a1) + rho_rad * np.cos(a2)
            v = rho_rad * np.sin(a1) + rho_rad * np.sin(a2)

            theta = np.sqrt(u * u + v * v)          # off-axis angle
            phi = np.arctan2(v, u)                  # roll angle about the axis
            # optical axis = +x; orthonormal basis (x_hat, y_hat, z_hat)
            ray_np = np.stack([
                np.cos(theta),
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
            ], axis=-1)
            ray_np /= np.linalg.norm(ray_np, axis=1, keepdims=True)
            ray_directions = torch.from_numpy(ray_np).to(device=device, dtype=dtype)

            # --------------------------------------------------
            # footprint threshold: max(beam divergence, scale * local ray spacing)
            # Local spacing = median NN angular distance over the ray set (k=6).
            # --------------------------------------------------
            ray_tree = cKDTree(ray_np)
            probe = gen.choice(num_rays, size=min(4096, num_rays), replace=False)
            nn_d, _ = ray_tree.query(ray_np[probe], k=7)      # k=7 incl. self
            # chord length -> angle (unit vectors): ang = 2*asin(chord/2)
            local_spacing = float(np.median(2.0 * np.arcsin(np.clip(nn_d[:, 1:], 0, 2) / 2.0)))
            footprint_angle = max(math.radians(beam_divergence_deg),
                                  footprint_scale * local_spacing)
            cos_footprint = math.cos(footprint_angle)

            # --------------------------------------------------
            # ground / origin / range filter  (reused verbatim)
            # --------------------------------------------------
            point_cloud_tensor = torch.cat(
                (curr_points[:, 6:8], curr_points[:, 5].unsqueeze(1)), dim=1)
            ground_mask = ((curr_labels == 1) | (curr_labels == 7)) & (curr_points[:, 3] < 20.0)
            if ground_mask.any():
                ground_z = curr_points[ground_mask, 5].median()
            else:
                z_all = curr_points[:, 5]
                k = max(1, int(0.02 * z_all.numel()))
                ground_z = z_all.kthvalue(k).values
            lidar_origin = torch.stack([
                torch.zeros((), device=device, dtype=dtype),
                torch.zeros((), device=device, dtype=dtype),
                (ground_z + lidar_height).to(dtype),
            ])

            vec_all = point_cloud_tensor - lidar_origin
            range_all = torch.norm(vec_all, dim=1)
            keep_mask = (range_all > EPS) & (range_all <= MAX_RANGE)
            if not keep_mask.any():
                sp, sl, sgi, sgiv = _empty()
                results_points.append(sp); results_labels.append(sl)
                results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
                info_dicts.append(dict(num_rays=num_rays, num_hit=0))
                continue
            kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
            kept_points_raw = curr_points[kept_idx]
            kept_labels = curr_labels[kept_idx]
            kept_vec = vec_all[kept_idx]
            kept_range = range_all[kept_idx]

            # --------------------------------------------------
            # POINT -> RAY ASSIGNMENT (rewritten): exact NN via kd-tree.
            # Unit-vector Euclidean distance is monotone in angle, so the
            # Euclidean nearest ray IS the angular nearest ray.
            # --------------------------------------------------
            unit_vec = kept_vec / kept_range.unsqueeze(1)
            unit_np = unit_vec.detach().cpu().numpy().astype(np.float64)
            _, nn_ray = ray_tree.query(unit_np, k=1)
            flat_bin = torch.from_numpy(np.asarray(nn_ray)).to(device=device).long()
            ray_dir_for_points = ray_directions[flat_bin]

            # --------------------------------------------------
            # footprint consistency filter (threshold redefined, logic same)
            # --------------------------------------------------
            cos_sim = torch.sum(unit_vec * ray_dir_for_points, dim=1).clamp_(-1.0, 1.0)
            depths = torch.sum(kept_vec * ray_dir_for_points, dim=1)
            valid_mask = (depths > 0.0) & (cos_sim >= cos_footprint)
            if not valid_mask.any():
                sp, sl, sgi, sgiv = _empty()
                results_points.append(sp); results_labels.append(sl)
                results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
                info_dicts.append(dict(num_rays=num_rays, num_hit=0,
                                       footprint_angle=footprint_angle))
                continue
            valid_bins = flat_bin[valid_mask]
            valid_depths = depths[valid_mask]

            # --------------------------------------------------
            # z-buffer reduction  (reused verbatim, bin index -> ray index)
            # --------------------------------------------------
            min_depth_flat, min_idx_in_valid = scatter_min(
                valid_depths, valid_bins, dim=0, dim_size=num_rays)
            hit_mask_flat = torch.isfinite(min_depth_flat)
            K = valid_depths.shape[0]
            idx_ok = (min_idx_in_valid >= 0) & (min_idx_in_valid < K)
            hit_mask_flat = hit_mask_flat & idx_ok

            valid_kept_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
            safe_min_idx = torch.clamp(min_idx_in_valid, 0, max(K - 1, 0))
            chosen_kept_idx_flat = torch.where(
                hit_mask_flat, valid_kept_idx[safe_min_idx],
                torch.full_like(safe_min_idx, -1))
            depth_flat = torch.where(hit_mask_flat, min_depth_flat,
                                     min_depth_flat.new_tensor(LARGE_VAL))

            # --------------------------------------------------
            # minority-class promotion  (reused verbatim)
            # --------------------------------------------------
            minor_pt_mask = torch.zeros_like(valid_depths, dtype=torch.bool)
            vlabels = kept_labels[valid_kept_idx]
            for _cls in MINORITY_CLASSES:
                minor_pt_mask = minor_pt_mask | (vlabels == _cls)
            if minor_pt_mask.any():
                minor_depths = valid_depths[minor_pt_mask]
                minor_bins = valid_bins[minor_pt_mask]
                minor_src = minor_pt_mask.nonzero(as_tuple=False).squeeze(1)
                minor_min_depth, minor_min_idx = scatter_min(
                    minor_depths, minor_bins, dim=0, dim_size=num_rays)
                Km = minor_depths.shape[0]
                minor_ok = (torch.isfinite(minor_min_depth) &
                            (minor_min_idx >= 0) & (minor_min_idx < Km))
                promote = minor_ok & hit_mask_flat & (
                    minor_min_depth <= depth_flat + MINORITY_DEPTH_TOL)
                safe_mi = torch.clamp(minor_min_idx, 0, max(Km - 1, 0))
                promoted_idx = valid_kept_idx[minor_src[safe_mi]]
                chosen_kept_idx_flat = torch.where(promote, promoted_idx,
                                                   chosen_kept_idx_flat)
                depth_flat = torch.where(promote, minor_min_depth, depth_flat)

            # --------------------------------------------------
            # minority mask per ray  (reused verbatim)
            # --------------------------------------------------
            safe_idx = torch.clamp(chosen_kept_idx_flat, min=0)
            labels_lookup = kept_labels[safe_idx]
            minority_ray_mask = torch.zeros_like(hit_mask_flat)
            for _cls in MINORITY_CLASSES:
                minority_ray_mask = minority_ray_mask | (labels_lookup == _cls)
            minority_ray_mask = minority_ray_mask & hit_mask_flat

            # --------------------------------------------------
            # OCCLUSION SUPPRESSION (rewritten, option A): ray kNN graph.
            # Keeps the mechanical path's asymmetry (only rays ABOVE the
            # current one act as occluders) and reuses every threshold.
            # The range-image unfold of the mechanical path cannot be used:
            # a non-repetitive pattern has no [beams, H] shape, and forcing
            # one would invent false adjacency.
            # --------------------------------------------------
            if occlusion_mode == 'knn' and int(hit_mask_flat.sum()) > occ_knn_k:
                hit_ray_idx = hit_mask_flat.nonzero(as_tuple=False).squeeze(1)
                hit_np = ray_np[hit_ray_idx.cpu().numpy()]
                hit_tree = cKDTree(hit_np)
                kq = min(occ_knn_k + 1, hit_np.shape[0])
                _, nbr = hit_tree.query(hit_np, k=kq)
                nbr = np.asarray(nbr)[:, 1:]                       # drop self
                nbr_t = torch.from_numpy(nbr).to(device=device).long()

                hit_depths = depth_flat[hit_ray_idx]               # [Hh]
                # elevation proxy: z component of the ray direction
                hit_elev = ray_directions[hit_ray_idx][:, 2]
                nbr_depths = hit_depths[nbr_t]                     # [Hh, K]
                nbr_elev = hit_elev[nbr_t]
                # only neighbours ABOVE self are occluder candidates
                upper = nbr_elev > hit_elev.unsqueeze(1)
                upper_depths = torch.where(upper, nbr_depths,
                                           nbr_depths.new_tensor(LARGE_VAL))
                occ_local_min = upper_depths.amin(dim=1)
                near_support = (upper_depths <=
                                (occ_local_min.unsqueeze(1) + OCC_NEAR_BAND)).sum(dim=1)
                # "same-ring protection" -> depth-similar neighbours anywhere
                same_support = (torch.abs(nbr_depths - hit_depths.unsqueeze(1))
                                <= RING_KEEP_BAND).sum(dim=1)
                ring_keep = same_support >= RING_KEEP_MIN
                adaptive_margin = OCC_DEPTH_MARGIN + OCC_REL_MARGIN * occ_local_min
                leak = (torch.isfinite(occ_local_min) &
                        (occ_local_min < LARGE_VAL) &
                        (near_support >= OCC_SUPPORT_MIN) &
                        ((hit_depths - occ_local_min) > adaptive_margin) &
                        (~ring_keep) &
                        (~minority_ray_mask[hit_ray_idx]))       # minority exempt
                if leak.any():
                    drop_rays = hit_ray_idx[leak]
                    hit_mask_flat = hit_mask_flat.clone()
                    hit_mask_flat[drop_rays] = False
                    chosen_kept_idx_flat = chosen_kept_idx_flat.clone()
                    chosen_kept_idx_flat[drop_rays] = -1

            # NOTE: ring continuity median filter (mechanical step 9) is
            # deliberately NOT applied — it smooths along a scanline, and a
            # non-repetitive pattern has no scanlines.

            # --------------------------------------------------
            # output + reprojection  (reused verbatim)
            # --------------------------------------------------
            valid_out = hit_mask_flat & (chosen_kept_idx_flat >= 0) & torch.isfinite(depth_flat)
            if not valid_out.any():
                sp, sl, sgi, sgiv = _empty()
                results_points.append(sp); results_labels.append(sl)
                results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
                info_dicts.append(dict(num_rays=num_rays, num_hit=0,
                                       footprint_angle=footprint_angle))
                continue
            chosen_kept_idx = chosen_kept_idx_flat[valid_out]
            chosen_bins = valid_out.nonzero(as_tuple=False).squeeze(1)
            chosen_depths = depth_flat[valid_out].unsqueeze(1)

            # random dropout, minority exempt  (reused verbatim)
            drop_p = random.random() * 0.08
            if drop_p > 0 and chosen_kept_idx.numel() > 16:
                cand_labels = kept_labels[chosen_kept_idx]
                is_minor_out = torch.zeros_like(cand_labels, dtype=torch.bool)
                for _cls in MINORITY_CLASSES:
                    is_minor_out = is_minor_out | (cand_labels == _cls)
                keep_out = (torch.rand(chosen_kept_idx.shape[0], device=device)
                            >= drop_p) | is_minor_out
                chosen_kept_idx = chosen_kept_idx[keep_out]
                chosen_bins = chosen_bins[keep_out]
                chosen_depths = chosen_depths[keep_out]
                if chosen_kept_idx.numel() == 0:
                    sp, sl, sgi, sgiv = _empty()
                    results_points.append(sp); results_labels.append(sl)
                    results_grid_ind.append(sgi); results_grid_ind_vox.append(sgiv)
                    info_dicts.append(dict(num_rays=num_rays, num_hit=0,
                                           footprint_angle=footprint_angle))
                    continue

            # range noise sigma(r) = 1cm + 0.1%*r, 3-sigma clipped
            noise_sigma = RANGE_NOISE_BASE + RANGE_NOISE_REL * chosen_depths
            depth_noise = torch.randn_like(chosen_depths) * noise_sigma
            depth_noise = torch.maximum(torch.minimum(depth_noise, 3.0 * noise_sigma),
                                        -3.0 * noise_sigma)
            chosen_depths = (chosen_depths + depth_noise).clamp_min(0.1)

            sampled_points = kept_points_raw[chosen_kept_idx].clone()
            sampled_labels = kept_labels[chosen_kept_idx]
            chosen_ray_dirs = ray_directions[chosen_bins]
            projected_xyz = lidar_origin.unsqueeze(0) + chosen_depths * chosen_ray_dirs

            proj_x = projected_xyz[:, 0]
            proj_y = projected_xyz[:, 1]
            proj_z = projected_xyz[:, 2]
            proj_rho = torch.sqrt(proj_x * proj_x + proj_y * proj_y)
            proj_phi = torch.atan2(proj_y, proj_x)
            xyz_pol = torch.stack([proj_rho, proj_phi, proj_z], dim=1)
            xyz_pol_grid = torch.maximum(
                torch.minimum(xyz_pol, max_bound_s - 1e-3), min_bound_s)
            new_grid_ind = torch.floor((xyz_pol_grid - min_bound_s) / intervals_s)
            new_grid_ind_vox = (xyz_pol_grid - min_bound_s) / intervals_vox_s
            voxel_centers = (new_grid_ind + 0.5) * intervals_s + min_bound_s
            rel_xyz = xyz_pol - voxel_centers

            sampled_points[:, 0:3] = rel_xyz
            sampled_points[:, 3] = proj_rho
            sampled_points[:, 4] = proj_phi
            sampled_points[:, 5] = proj_z
            sampled_points[:, 6] = proj_x
            sampled_points[:, 7] = proj_y

            results_points.append(sampled_points)
            results_labels.append(sampled_labels)
            results_grid_ind.append(new_grid_ind.to(curr_grid_ind.dtype))
            results_grid_ind_vox.append(new_grid_ind_vox.to(curr_grid_ind_vox.dtype))
            info_dicts.append(dict(
                num_rays=num_rays,
                num_hit=int(sampled_points.shape[0]),
                footprint_angle=footprint_angle,
                local_spacing=local_spacing,
                psi1=float(psi1), psi2=float(psi2),
                ray_directions=ray_np,
            ))

        return (results_points, results_grid_ind, results_grid_ind_vox,
                results_labels, info_dicts)


def train_grid_features_to_xyz_numpy(feature_2d):
    """XYZ from PointsegMapping ``train_grid`` layout (cols xy + z slot, same as legacy ply)."""
    if isinstance(feature_2d, torch.Tensor):
        arr = feature_2d.detach().float().cpu().numpy()
    else:
        arr = np.asarray(feature_2d, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.shape[1] < 8:
        if arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float64)
        raise ValueError(
            f'train_grid needs >=8 cols for polar layout, got {arr.shape[1]}')
    xyz = np.concatenate((arr[:, 6:8], arr[:, 5:6]), axis=1).astype(np.float64)
    return xyz


def save_tensor_as_ply(tensor, file_path):
    """Save ``train_grid``-style point features as PLY (xyz only)."""
    xyz = train_grid_features_to_xyz_numpy(tensor)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud(file_path, point_cloud)

# cache for sample_beams: (lower, upper) -> (beam_range, probability_density)
_sample_beams_cache = {}


def _beam_pmf(lower_beams_bound, upper_beams_bound):
    """Build the (beam_range, probability_density) of the beam sampling distribution.

    beams are not sampled uniformly but from a Gaussian mixture over several "canonical beam counts" (16/32/64/80/128).
    The pmf construction is factored out so that sampling and the statistics (mean/std) share it, keeping the normalization consistent.
    """
    key = (lower_beams_bound, upper_beams_bound)
    if key not in _sample_beams_cache:
        variance = 16
        std_dev = np.sqrt(variance)
        candidate_means = [16, 32, 64, 80, 128]
        valid_means = [m for m in candidate_means if lower_beams_bound <= m <= upper_beams_bound]
        beam_range = np.arange(lower_beams_bound, upper_beams_bound + 1, dtype=np.int64)
        probability_density = np.zeros(len(beam_range), dtype=np.float64)
        for mean in valid_means:
            probability_density += np.exp(-0.5 * ((beam_range - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
        total = probability_density.sum()
        if total <= 0:
            # if no canonical beam count lies in the interval, fall back to a uniform distribution to avoid nan from 0/0
            probability_density = np.ones(len(beam_range), dtype=np.float64)
            total = probability_density.sum()
        probability_density /= total
        _sample_beams_cache[key] = (beam_range, probability_density)
    return _sample_beams_cache[key]


def sample_beams(lower_beams_bound, upper_beams_bound):
    beam_range, probability_density = _beam_pmf(lower_beams_bound, upper_beams_bound)
    return int(np.random.choice(beam_range, p=probability_density))


def beam_distribution_mean_std(lower_beams_bound, upper_beams_bound):
    """Exact mean/std of the sample_beams distribution, used for the beam dim of the sensor-parameter normalization."""
    beam_range, probability_density = _beam_pmf(lower_beams_bound, upper_beams_bound)
    beam_range = beam_range.astype(np.float64)
    mean = float((beam_range * probability_density).sum())
    var = float((probability_density * (beam_range - mean) ** 2).sum())
    std = float(np.sqrt(max(var, 1e-12)))
    return mean, std

    
def get_nuScenes_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        nuScenesyaml = yaml.safe_load(stream)
    nuScenes_label_name = dict()
    for i in sorted(list(nuScenesyaml['learning_map'].keys()))[::-1]:
        val_ = nuScenesyaml['learning_map'][i]
        nuScenes_label_name[val_] = nuScenesyaml['labels_16'][val_]

    return nuScenes_label_name

def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred, minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)