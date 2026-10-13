# =============================================================================
# Dense teacher config (nuScenes), derived from pointtpv_nusc_seg_random.py.
# The values in this file are those used to train the dense teacher whose checkpoint the
# student loads (d2skd_load_from) for the main results.
# Layout used to train this teacher: 8 GPUs x samples_per_gpu=1, no gradient accumulation
# = effective batch 8. Warmup is counted in iterations (warmup_iters=2500), as are the other
# iteration-based schedules, so changing the layout changes the schedule. When changing the
# number of GPUs, adjust samples_per_gpu (and, if needed, gradient accumulation) together to
# keep the effective batch; otherwise the BatchNorm statistics and the iteration-based
# schedules no longer match and the result is not comparable with the teacher used in the
# paper.
# Differences from the student config:
#   1. dense_train=True (no beam simulation, uniform 500k-point cap; no FiLM/sensor branch is
#      built) and dense7sparse=False (pure dense input).
#   2. d2skd=False, consistency=False, sensor_film=False (pure supervised training).
#   3. No MinorityDensePurify in the training pipeline; dynamic loss balancing disabled.
#   4. val/test use the dense protocol (random=True + reset_random_prob=0.0 + loader
#      Random=True), so per-epoch validation monitors the teacher on its serving input.
#   5. Teacher recipe: repeat=2, lr=6e-4, warmup_iters=2500, min_lr=8e-5, no gradient
#      accumulation.
# =============================================================================
# ===================== Coordinate frame / ground height (measured) =====================
# **Native ground height** of this dataset's point clouds: median z of the Driveable Ground
# coarse-class points pooled over the full val split (1 mm histogram):
#   SemanticKITTI -1.751 · nuScenes -1.829 · Waymo -0.018   (metres)
#   Samples: SK val seq08, all 4071 frames / 93.91M points (raw 40,44,60)
#            NU val, all 6019 frames / 56.05M points (raw 24)
#            WA val, the 729 frames with semantic labels / 23.66M points (raw 18,19,20)
# An earlier value came from "the 5th percentile of z over all points of 20 frames", which is
# not a ground estimator: it is dragged down by below-ground returns and its bias differs per
# dataset. It is also very unstable -- on the same SK seq08 the pooled P5 over all 4071 frames
# is -1.9645, over the first 200 frames -2.2109, and over one 20-frame sample -2.524 (the
# origin of the old constant -2.529); the Driveable-median estimate stays near -1.75 under the
# same sampling changes (full split -1.7505 / first 200 frames -1.7723 / another 200-frame
# sample -1.751).
# SK's -1.751 agrees with KITTI's ~1.73 m Velodyne mounting height, and NU's -1.829 with its
# 1.84 m LIDAR_TOP mounting height. Waymo is stored in the vehicle/ground frame (ground ~0),
# SK/NU in the sensor frame, hence the ~1.7 m difference.
NATIVE_GROUND_Z = -1.829  # nuScenes (sensor frame)
#
# Ground height of the reference frame -- decides on which height datum this config places the
# point cloud:
#   · keep = NATIVE_GROUND_Z (default, shift=0): native frame.
#     **All existing weights were trained in the native frame**, so training/evaluation on the
#     same dataset is self-consistent; changing this line misaligns trained weights and data,
#     and evaluation results become invalid.
#   · set to -1.751 (SemanticKITTI datum): unified with the other datasets. Used for
#     (a) new training runs (all three datasets on the same datum);
#     (b) evaluating an SK-trained model on this dataset (cross-dataset evaluation).
REFERENCE_GROUND_Z = NATIVE_GROUND_Z
#
# shift_coors is subtracted (points -= shift) and applied **before** the range filter
# (see loading_voxels_sk.py), so that the vertical span kept relative to the ground is the
# same under any datum.
GROUND_ALIGN_SHIFT = [0, 0, round(NATIVE_GROUND_Z - REFERENCE_GROUND_Z, 3)]
# ====================================================================

_base_ = [
    './_base_/default_runtime.py',
    './_base_/custom_nus-3d.py',
]

dataset_type = 'NuscSegDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
DG = True
Random = True
dense7sparse = False  # pure dense input for the teacher
ResetRandomProb = 0.0
plugin = True
consistency = False
# d2skd needs a dense pretrained teacher checkpoint (d2skd_load_from); training refuses to
# start without one (no randomly initialised teacher).
d2skd = False
# ---- VanillaKD distillation baseline (comparison experiments; implementation in vanillakd.py,
# sharing exactly the same signal path as spectral distillation) ----
# With KDbaseline=True plain distillation (point-wise MSE on TPV features + temperature KL on
# the teacher support) replaces spectral distillation; only effective with d2skd=True and
# mutually exclusive with spectral distillation. The three comparison rows only change the
# two lambdas:
#   Soft-label-KL row:      (KD_lambda_kd, KD_lambda_feat) = (1.0, 0)
#   Feature-mimicking row:  (0, 0.5)
#   Both row:               (1.0, 0.5)
KDbaseline = False
KD_T = 4.0  # distillation temperature 4 across the project (also the official PVKD/CWD value)
KD_lambda_kd = 1.0
KD_lambda_feat = 0.5
dense_train = True
sensor_film = False  # no FiLM for the teacher (not built when dense_train=True anyway; disabled explicitly)
plugin_dir = "projects/unilidar_plugin/"

# occ_path = "./data/nuScenes-Occupancy"
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
# For nuScenes we usually do 10-class detection
class_names = ['noise', 'flat.driveable_surface', 'flat.sidewalk', 'flat.terrain', 
             'flat.other', 'static.manmade', 'static.vegetation', 'static.other', 
             'vehicle.ego']

# CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Dynamic Object', 'Static Object', 'Other Ground']

# NUSCENES_TO_CL =  {
#     0: 0,   # noise -> Other
#     24: 1,  # flat.driveable_surface -> Driveable Ground
#     25: 8,  # flat.sidewalk -> Structure
#     26: 3,  # flat.terrain -> Nature
#     27: 4,  # flat.other -> Other Ground
#     28: 2,  # static.manmade -> Structure
#     29: 4,  # static.vegetation -> Nature
#     30: 5,  # static.other -> Static Object
#     31: 3,  # vehicle.ego -> Vehicle
# }

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Objects', 'Other Ground']

# Mapping from the original nuScenes semantic ids to the coarse DG taxonomy.
# Each entry documents: fine label -> CLASS_NAMES_DG target.
NUSCENES_TO_CL =  {
        0: 0,   # void / ignore -> Other
        1: 6,   # barrier -> Objects
        2: 6,   # bicycle -> Objects
        3: 3,   # bus -> Vehicle
        4: 3,   # car -> Vehicle
        5: 3,   # construction_vehicle -> Vehicle
        6: 6,   # motorcycle -> Objects
        7: 5,   # pedestrian -> Living Being
        8: 6,   # traffic_cone -> Objects
        9: 3,   # trailer -> Vehicle
        10: 3,  # truck -> Vehicle
        11: 1,  # driveable_surface -> Driveable Ground
        12: 7,  # other_flat -> Other Ground
        13: 7,  # sidewalk -> Other Ground
        14: 7,  # terrain -> Other Ground (previously Nature)
        15: 2,  # manmade -> Structure
        16: 4,  # vegetation -> Nature
}

unilidar=False
cylinder=False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
model_empty_idx = 0  # noise 0-->255
empty_idx = 0  # noise 0-->255
num_cls_nu = 8 if DG else 17
# len(class_names)  # 0 free, 1-16 obj
visible_mask = True

sensor = dict(
    LiDAR_height=[1.5, 1.85],
    num_of_beams=[16, 80],
    horizontal_angular_resolution=[600, 1800],
    lower_vertical_field_of_view_bound=[-30, -15],
    upper_vertical_field_of_view_bound=[2, 15],
    # LiDAR_height=[1, 2],
    # num_of_beams=[16, 128],
    # horizontal_angular_resolution=[900, 3600],
    # lower_vertical_field_of_view_bound=[-40, -5],
    # upper_vertical_field_of_view_bound=[0, 25],
)

# [earlier curriculum switch] Measured over 24 epochs (same on nuScenes and SK): when stage 3
# arrived at ep16 the cosine LR had already decayed to 1.4e-4 and kept falling; train loss
# jumped at ep17 and val dropped with it, the remaining LR was too low to recover, and the
# last 5 epochs plateaued. The stages are moved to (0,6),(6,12),(12,24): the mIoU curve shows
# stage 1 already plateaus at ep5-7 (0.594-0.609), so switching earlier costs nothing; stage 3
# now arrives at LR~2.9e-4 with 12 epochs to absorb it. Total epochs unchanged, keeping the
# comparison with the baseline fair.
schedule=[(0, 6, {
                'num_of_beams': (32, 64),
                'horizontal_angular_resolution': (900, 2400),
                'lower_vertical_field_of_view_bound': (-35, -20),
                'upper_vertical_field_of_view_bound': (0, 10),
                'LiDAR_height': (1.7, 1.9)
            }),
                      (6, 12, {
                          'num_of_beams': (24, 80),
                          'horizontal_angular_resolution': (600, 3000),
                          'lower_vertical_field_of_view_bound': (-40, -15),
                          'upper_vertical_field_of_view_bound': (0, 15),
                          'LiDAR_height': (1.5, 2)
                      }),
                      (12, 24, {
                          'num_of_beams': (16, 96),
                          'horizontal_angular_resolution': (400, 4000),
                          'lower_vertical_field_of_view_bound': (-45, -10),
                          'upper_vertical_field_of_view_bound': (0, 20),
                          'LiDAR_height': (1.2, 2)
                      })]

cascade_ratio = 2 
dataset_flag_nu = 1 if dataset_type == 'NuscSegDataset' else 2


find_unused_parameters = False
unique_label = [0, 1, 2, 3, 4, 5, 6, 7] if DG else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
track_running_stats = False

_dim_ = 128 

tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2

coarse_ratio = cascade_ratio
sweeps_num = 10
grid_size = [480, 360, 32]
nbr_class = num_cls_nu

model = dict(
    type='PointTPV_Seg',
    tpv_aggregator=dict(
        type='TPVAggregator_Seg',
        tpv_h=tpv_h_,
        tpv_w=tpv_w_,
        tpv_z=tpv_z_,
        loss_weight=[1,0.2],
        nbr_classes=nbr_class,
        in_dims=_dim_,
        hidden_dims=2*_dim_,
        out_dims=_dim_,
        scale_h=scale_h,
        scale_w=scale_w,
        scale_z=scale_z,
        consistency = consistency,
        dense_train = dense_train,
        d2skd = d2skd,
        outside_d2skd = d2skd,  # high-frequency logit distiller (TPVHighFreqSpectralDistiller)
        distill_mode = 'vanilla_kd' if KDbaseline else 'spectral',
        kd_cfg = dict(T=KD_T, lambda_kd=KD_lambda_kd, lambda_feat=KD_lambda_feat),
        # [lower consistency while distilling] Measured over 24 epochs: consistency is 15-18% of the
        # total loss, ~15x the vox distillation path; it pulls the student towards the worse sparse aux
        # view, whereas the dense teacher is a strictly better invariance target, and the two overlap
        # or even conflict (L_cons jumps at every curriculum switch). With distillation on it is
        # halved, leaving the budget to supervision + distillation. The baseline (d2skd=False) is
        # unaffected and keeps 40.
        consistency_gain = 20.0 if d2skd else 40.0,
        # [HF distiller] gate_kappa was lowered from 1 to reduce the gate noise floor tau, so that
        # detail matching can also open where the teacher structure is weaker (raw_vox showed no floor
        # over 24 epochs, so the gate can be more aggressive).
        # class_weights: coarse-grid logit channels == classes; the bottleneck classes Living Being(5)/
        # Objects(6) get 2x weight (normalised internally to mean 1, overall magnitude unchanged).
        highfreq_distiller_cfg = dict(
        # moment_norm: see the note at lowfreq_distiller_cfg; the same moment-normalisation fix on the
        # HF side.
        moment_norm='joint',
            # Fix for a double-normalisation bug (in the old 'sum' path detail/suppress scaled as 1/N and
            # were measured to be constantly 0 -- the gate never took effect). mean_rms restores an O(1)
            # size-invariant quantity; in a 100-iter measurement the gated share went 0% -> 71% and the
            # gated terms became the main terms.
            gate_norm='mean_rms',
            gate_kappa=0.5,  # recalibrated from the 0.25 set while the gate was dead
            lambda_suppress=1.0,  # soft-threshold "suppress where the teacher is smooth" within the support; original design kept
            # eng/mom/ecc: 25% together in a 100-iter measurement, dropping quickly as alignment improves
            # (33% -> 25%), and falling into the 12-18% target band during convergence. mom = the "close to
            # the teacher's spectral statistics" term.
            lambda_energy=2.0, lambda_moment=2.0, lambda_ecc=4.0,
            lambda_flat=1.0,
            # Profile metric JS -> first-order L1; measured share 3.6% (inside the target band).
            profile_metric='l1', lambda_profile=5.0,
            class_weights=[1, 1, 1, 1, 1, 2, 2, 1],
            # Deeper high-pass group (enabled once the gate became active): 1-rho^32 responds 0.56 at
            # lambda=0.1, covering the lower measured aliasing band; measured iteration time +2.8%,
            # negligible.
            extra_scales=[16, 32]),
        sensor_film = sensor_film,
        # CBA-CL two-view memory/gradient mode: 'alternate' = sequential (memory ~ one view) /
        # 'both' = backpropagate through both graphs
        consistency_grad_mode = 'alternate',
        supervise_aux_views = False,
        checkpoint_consistency_loss = True,
    ),
    lidar_tokenizer=dict(
        type='CylinderEncoder_Seg',
        grid_size=grid_size,
        in_channels=8 if Random else 10,
        out_channels=256,
        fea_compre=None,
        base_channels=256,
        split=[16,16,16],
        track_running_stats=track_running_stats,
    ),
    lidar_backbone=dict(
        type='Swin',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        in_channels=256,
        patch_size=4,
        strides=[1,2,2,2],
        frozen_stages=-1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1,2,3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
          type='Pretrained',
          checkpoint='pretrain/swin_tiny_patch4_window7_224.pth'),
    ),
    lidar_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=_dim_,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(
          type='BN2d',
          requires_grad=True,
          track_running_stats=track_running_stats),
        act_cfg=dict(
          type='ReLU',
          inplace=True),
        upsample_cfg=dict(
          mode='bilinear',
          align_corners=False),
    ),
    empty_idx = 0,
    random = Random,
    reset_random_prob = ResetRandomProb,
    sensor = sensor,
    consistency = consistency,
    dense_train = dense_train,
    # Dense input cap of the teacher: fixed at 500k points. None uses the default
    # max_points=500000 of the dense branch of _sampling (pointtpv_seg.py); bit-exact, no
    # torch.randint call and no CPU RNG consumption.
    # Why fixed rather than the range U[350k,500k] (whose upper bound is also 500k): measured on
    # 150 nuScenes frames (after range + ego rho>=1.0 filtering) mean 505,124 / median 504,830;
    # only 54.7% of frames exceed 500k but 100% exceed 350k. With (350000, 500000) every NU frame
    # would be reduced to U[350k,500k] (mean ~425k) during teacher training, while distillation
    # serving caps only 54.7% of the frames at 500k -- the training density would be
    # systematically ~15% below the serving density (capped frames 100% vs 54.7%). With a fixed
    # 500000, teacher training / teacher evaluation (test_max_points) / distillation serving
    # (teacher_serve_max_points) follow exactly the same rule.
    # Cost: the density-amplitude jitter augmentation is lost. The "fresh subset every
    # iteration" randomness is fully kept for 100% of the SK/Waymo frames; it only vanishes for
    # the 45.3% of nuScenes frames below 500k. Estimated wall clock per iteration +10% to +18%
    # (425k -> 500k points, tokenizer/scatter roughly linear).
    # Whether the range's density jitter helps against post-peak decay was not disproven;
    # save_best plus distilling from the best-val teacher checkpoint serve as the safeguard.
    dense_cap_range = None,
    # Evaluation support capped uniformly at 500k (aligned with training / distillation serving
    # input); not comparable with uncapped evaluations, re-evaluate those with the same cap.
    test_max_points = 500000,
    d2skd = d2skd,
    distill_mode = 'vanilla_kd' if KDbaseline else 'spectral',
    kd_cfg = dict(T=KD_T, lambda_kd=KD_lambda_kd, lambda_feat=KD_lambda_feat),
    # [LF distiller] block-local response alignment: targets the ~0.5 high floor of raw_inter --
    # global normalisation averaged away the shape mismatch in small-object regions; (30,30)
    # blocks align each block independently (on the small yz/zx planes the short z axis is
    # covered by a single block). None falls back to the global behaviour.
    lowfreq_distiller_cfg = dict(
        # Two implementation details that did not match their own stated design are fixed and
        # explicitly enabled here:
        #  · moment_norm='joint': the shrinkage moments used to be normalised per scale along the
        #    channel axis separately, which quotients out the scale axis itself, so for a whole-class
        #    field that factorises as channel (x) space the loss and its gradient were identically 0
        #    (identically zero for any two fields when C=1). Joint-axis normalisation keeps only the
        #    single global-scaling invariance of Proposition 5(ii) in the paper. Measured on 8 real
        #    fields the magnitude changes by LF +2.99% / HF +0.15%, so lambda_moment needs no
        #    recalibration.
        #  · suppress_ref='teacher_floor': the old form normalised by the student's own RMS and was
        #    exactly 0-homogeneous in the student amplitude (scaling the amplitude 16x changed the loss
        #    by only 3e-5), and the gradient through the denominator was always parallel to +u,
        #    amplifying student high frequencies in teacher-structure regions (positive in all 18 real
        #    groups). With max(detach(E_s), E_t) it is 2-homogeneous in the student, so the claim
        #    "teacher and student both flat -> 0" actually holds. Total LF changes by -0.5% to -2.9%.
        #  Both can be set back to 'per_scale_channel' / 'self_rms' to reproduce the old behaviour
        #  bit for bit.
        moment_norm='joint', suppress_ref='teacher_floor',
        block_size=(30, 30),
        # LF gating (finalised from measurements): detail = gated alignment in teacher-structure
        # regions, suppress = penalise the student's local variation (L·g_s)^2 where the teacher is
        # flat (main battlefield against folded aliasing).
        # suppress acts on the local variation instead of the low-pass amplitude (the old form
        # suppressed legitimate DC and blew up as O(N) under an all-flat teacher; see
        # Spectraldistiller.py).
        # lambda_sup=0.25: with the old form at 0.5 its share was 43% in a 100-iter measurement, too
        # heavy, hence halved; the raw magnitude of the new form is ~0.83x the old one, lambda kept;
        # the yz/zx planes must not be suppressed heavily.
        lf_gate=True, gate_kappa=0.5, lambda_suppress=0.25,
        # Profile metric JS (dead second-order term) -> first-order L1; lambda=8 -> share in the 3-8%
        # target band.
        profile_metric='l1', lambda_profile=8.0),
    dynamic_loss_balance=dict(
        enabled=False,  # no loss balancer for the teacher
        alpha=0.1,          # GradNorm target strength
        step=0.001,          # per-step update size
        momentum=0.2,       # smoothing for weight updates
        min_weight=0.1,
        max_weight=2.0,
        ema_beta=0.8,       # EMA for loss trend
        # Interval of the gradient probing. Each measurement runs 5 autograd.grad calls (3 supervised
        # groups + 2 distillation groups) through the graph of base_bs scenes, measured at 0.20 s per
        # iteration when amortised (2.3%). The weights evolve slowly with momentum anyway, so measuring
        # less often does not affect convergence but saves 0.16 s/iter.
        interval=50,
        # Task grouping follows fixed order in model:
        # point / voxel / sensor / consistency
        base_weights=dict(
            point=1.0,
            voxel=0.9,
            sensor=1.2,
            consistency=0.4,
        ),

        task_min_weights=dict(
            point=0.9,
            voxel=0.65,
            sensor=0.8,
            consistency=0.15,
        ),
        task_max_weights=dict(
            point=2.5,
            voxel=1.6,
            sensor=3.0,
            consistency=0.8,
        ),
        voxel_guard_enabled=True,
        voxel_guard_tol=0.02,
        voxel_guard_gain=3.0,
        # Gradient-ratio controller for the d2skd distillation losses (independent of the GradNorm
        # groups): the gradient norm of each distillation loss is anchored to rho times the mean
        # gradient of the segmentation groups, so distillation and segmentation do not suppress each
        # other. inter = low-frequency feature distillation, vox = high-frequency logit distillation.
        distill=dict(
            enabled=True,
            # inter 0.25->0.35: over 24 epochs raw_inter stalled at a ~0.5 high floor while the weight
            # never hit its bound (the distillation gradient was held stably at the target ratio), which
            # suggests untapped LF teacher signal; vox stays at 0.25 (raw_vox kept decreasing slowly, no
            # floor).
            # Gradient measurement basis of rho: budget_basis='ref' (the default) is used. Both bases
            # are biased proxies: the bias of 'ref' comes from the probe topology, that of 'full' from
            # DDP coherence -- with 'full', LF reached only 21.0-21.4% in measurements, still short of
            # the nominal 28.1%. Measured under 'ref': the total drifts to 3.31 within a measurement
            # interval, and the closed-loop achievement error is +/-5% to 21%, the same order as 'full'.
            # 'ref' measures on a single 128-d tensor, lidar_neck.fpn_convs.1.bn.bias (0.0004% of the
            # 32,277,463 trainable parameters; a by-product of taking the last entry of parameters()).
            # Since LF is injected at the neck output and HF/KL at the aggregator output, the probe sits
            # right at the LF injection point; on 8 GPUs c_LF/c_HF = 2.79-4.26 was measured, so in
            # whole-model shares LF actually gets 8-11% and HF 38-47% (the rho ratio corresponds to
            # 28.1% : 26.3%).
            # Two readings are possible: the controller is specified as a ratio target on a single
            # reference parameter, under which the implementation is faithful; the argument for putting
            # kd under rho used model-level shares, under which the implementation does not deliver that
            # intent. This is a design choice, not a bug, and is left unchanged by default.
            # Note on scale: "w_LF would rise to about 2.6x" refers to the local neck scale (total
            # 2.85 -> 3.99-4.67); on the whole-model scale 4 of 6 sampled points go down or stay flat,
            # and only step16 rises by +34.7% to +36.7%.
            # To switch to the whole-model measurement basis, uncomment the two lines below (cost: 6
            # full-graph backward passes; with mult=5 an estimated +3% to +6% iteration time; w_LF
            # immediately rises to about 2.6x):
            # budget_basis='full',
            # measure_interval_mult=5,
            rho=dict(inter=0.35, vox=0.25),
            warmup_iters=500,
            ema_beta=0.9,
            min_weight=1e-3,
            max_weight=1e5,
        ),
    ),
)


bda_aug_conf = dict(
            # rot_lim=(-22.5, 22.5),
            rot_lim=(-0, 0),
            scale_lim=(0.95, 1.05),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5)


if DG:
    train_pipeline = [
        dict(type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=3 if Random else 5,
        use_dim=3 if Random else 5,
        shift_height=True,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=GROUND_ALIGN_SHIFT,
        Random = Random,
        dense7sparse = dense7sparse if Random else False),
        dict(type='PointsegMapping', 
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [51.2, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -3.4],
                cal_visible=False,
                RPR = RPR,
                restrict_pc_range = point_cloud_range),
        dict(type='VoxelClassMapping'),
        # MinorityDensePurify is not used for teacher training: its mismatch between training GT
        # (purified / densified) and val GT (raw), and the erosion of nearby ground GT by the densify
        # copies, were secondary suspects for the post-peak decay.
        dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'reset_random']),
    ]
    test_pipeline = [
        dict(type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3 if Random else 5,
        shift_height=True,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=GROUND_ALIGN_SHIFT,
        Random = True),
        dict(type='PointsegMapping', 
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [51.2, 3.1415926, 3],  # same as train (the former 50 caused a rho scale mismatch)
                min_volume_space = [0, -3.1415926, -3.4],  # same as train (the former -5 gave a z-interval of 0.25 vs 0.20, misplacing the ground by 5 bins)
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range),
        dict(type='VoxelClassMapping'),
        dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'dataset_flag'],
                meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]
else:
    train_pipeline = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=3 if Random else 5,
        use_dim=3 if Random else 5,
        shift_height=True,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=GROUND_ALIGN_SHIFT,
        Random = Random,
        dense7sparse = dense7sparse if Random else False),
    # dict(type='LoadPointsFromMultiSweeps_RPR',
    #     sweeps_num=10,
    #     RPR=False,
    #     point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,),
    dict(type='PointsegMapping', 
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [51.2, 3.1415926, 3],
                min_volume_space = [0, -3.1415926, -3.4],
                cal_visible=False,
                RPR = RPR,
                restrict_pc_range = point_cloud_range),
    dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'reset_random']),
]

    test_pipeline = [
        dict(
        type='LoadPointsFromFile_RPR',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3 if Random else 5,
        shift_height=True,
        RPR=RPR,
        point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
        dataset_flag=dataset_flag_nu,
        shift_coors=GROUND_ALIGN_SHIFT,
        Random = True),
        # dict(type='LoadPointsFromMultiSweeps_RPR',
        # sweeps_num=10,
        # RPR=False,
        # point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,),
        dict(type='PointsegMapping', 
                grid_size = grid_size,
                grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
                coarse_ratio = coarse_ratio,
                pc_range = ori_point_cloud_range,
                fill_label = 0,
                unique_label = unique_label,
                fixed_volume_space = True,
                max_volume_space = [51.2, 3.1415926, 3],  # same as train (the former 50 caused a rho scale mismatch)
                min_volume_space = [0, -3.1415926, -3.4],  # same as train (the former -5 gave a z-interval of 0.25 vs 0.20, misplacing the ground by 5 bins)
                cal_visible=visible_mask,
                RPR = RPR,
                restrict_pc_range = point_cloud_range),
        dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'dataset_flag'],
                meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']),
    ]
    

test_config=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=CLASS_NAMES_DG if DG else class_names,
    modality=input_modality,
    pc_range=point_cloud_range,
    seg_label_mapping= NUSCENES_TO_CL if DG else None,
    random=True,
    reset_random_prob=0.0,
)

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        classes=CLASS_NAMES_DG if DG else class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        pc_range=point_cloud_range,
        box_type_3d='LiDAR',
        seg_label_mapping= NUSCENES_TO_CL if DG else None,
        random=Random,
        reset_random_prob=ResetRandomProb,
        repeat=2,  # two passes over the data per epoch (24 epochs = 48 passes)
        dense7sparse = dense7sparse if Random else False)

# Memory-critical item: the peak memory grows with samples_per_gpu (a single mmcv backward
# must hold the graphs of all scenes on this GPU; under torch1.10+DDP the backbone cannot be
# broken up with checkpointing).
# No gradient accumulation for this teacher (8 GPUs x 1 = effective batch 8); on 24GB cards
# keep spg=1.
data = dict(
    samples_per_gpu=1,  # effective batch 8 (8 GPUs x 1); 24GB cards also only fit 1
    # workers_per_gpu: with 32 CPU cores, 4 GPUs x 8 workers = 32 worker processes + 4 main
    # processes saturate the CPU and starve the main-process threads that launch CUDA kernels ->
    # measured GPU utilisation only 47-63%. data_time is only 0.042s (0.4% of an iteration), so
    # prefetching has plenty of headroom; fewer workers do not make data loading the bottleneck
    # and give the CPU back to the main processes.
    workers_per_gpu=6,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)


optimizer = dict(
    type='AdamW',
    lr=6e-4,  # teacher recipe value
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
# No gradient accumulation for the nuScenes teacher (effective batch 8 = 8 GPUs x 1):
# accumulation shrinks the per-sample step size, which was one of the main causes of
# under-optimisation in an earlier nuScenes teacher.
optimizer_config = dict(grad_clip=dict(max_norm=30, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2500,  # teacher recipe value (gentle warmup)
    warmup_ratio=1.0 / 3,
    min_lr=8e-5)  # higher LR floor to avoid sinking into a sharp minimum during annealing

# Model-weight EMA.
#
# mmcv's built-in EMAHook initialises the EMA buffer with the *initial weights*; with
# momentum 2e-4 the time constant is 5000 iterations, while one epoch here has only
# 2400-3500 iterations -- at the end of epoch 1 the EMA still holds 50%-62% of the initial
# weights, and eval/checkpoints use the EMA weights, so the metrics of the first epochs are
# heavily underestimated (rare classes such as Living Being drop to 0). Its warm_up argument
# uses min(), which only makes the EMA slower and cannot fix the cold start.
#
# WarmupEMAHook fixes this: during the first start_epochs epochs the EMA strictly follows the
# current weights (equivalent to no EMA, zero contamination); afterwards it accumulates with
# bias-corrected momentum, so the EMA never contains the initial weights.
#
# [enabled] Unit test: at the end of epoch 1 EMA == current weights 100%, == initial weights 0%
# (the 54% cold-start contamination of the old mmcv EMAHook does not occur); from epoch 2 it
# lags and smooths normally; the mmcv parent class swaps the EMA weights in/out around val.
# Effect: smooths the +/-4-8 pp epoch-to-epoch swings of Living Being/Objects (coarse classes
# 5/6; Objects was formerly called Movable objects).
custom_hooks = [
    dict(type='WarmupEMAHook', momentum=0.0002, start_epochs=1, priority='ABOVE_NORMAL'),
]

# 24 epochs; with repeat=2 that is 48 passes over the data; decay is countered by the high
# min_lr floor, with save_best as a safeguard.
runner = dict(type='EpochBasedRunner', max_epochs=24) if not d2skd else dict(type='DistillEpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

# d2skd teacher weights (dense pretrained model), required when d2skd is on:
work_dir = './work_dirs/pointtpv_nusc_teacher_dense'
# KD baselines get their own work_dir automatically (derived from the KD variables, so the
# three variant configs differ from this file only in the three KD lines, and the
# spectral-distillation checkpoints are not overwritten)
if KDbaseline:
    work_dir = work_dir + f'_KDbaseline_kd{KD_lambda_kd}_feat{KD_lambda_feat}'    
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='LiDARCurriculumHook', schedule=schedule),
    ])

checkpoint_config = dict(interval=2)
