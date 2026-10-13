# =============================================================================
# LiEqualizer student config: Waymo, random-sampling training with cross-density
# distillation (d2skd) from a dense teacher.
# The values in this file are those of the main results (the LiEqualizer rows of the
# main results tables in the paper).
# Layout: 4 GPUs x samples_per_gpu=4 x cumulative_iters=2 = effective batch 32.
# When changing the number of GPUs, adjust samples_per_gpu and cumulative_iters together
# so that the effective batch stays 32; otherwise the BatchNorm statistics and the
# iteration-based schedules no longer match, and the results are not comparable with
# the paper.
# The teacher checkpoint (d2skd_load_from) is trained with pointtpv_waymo_teacher_dense.py.
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
NATIVE_GROUND_Z = -0.018  # Waymo (vehicle/ground frame)
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
]

dataset_type = 'CustomWaymoDataset_T'
data_root = 'data/waymo/kitti_format/'
file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

class_names = [
    'TYPE_UNDEFINED',
    'TYPE_CAR',
    'TYPE_TRUCK',
    'TYPE_BUS',
    'TYPE_OTHER_VEHICLE', # Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    'TYPE_MOTORCYCLIST',
    'TYPE_BICYCLIST',
    'TYPE_PEDESTRIAN',
    'TYPE_SIGN',
    'TYPE_TRAFFIC_LIGHT',
    'TYPE_POLE', # Lamp post, traffic sign pole etc.
    'TYPE_CONSTRUCTION_CONE', # Construction cone/pole.
    'TYPE_BICYCLE',
    'TYPE_MOTORCYCLE',
    'TYPE_BUILDING',
    'TYPE_VEGETATION', # Bushes, tree branches, tall grasses, flowers etc.
    'TYPE_TREE_TRUNK',
    'TYPE_CURB', # Curb on the edge of roads. This does not include road boundaries if there's no curb.
    'TYPE_ROAD', # Surface a vehicle could drive on. This include the driveway connecting parking lot and road over a section of sidewalk.
    'TYPE_LANE_MARKER', # Marking on the road that's specifically for defining lanes such as single/double white/yellow lines.
    'TYPE_OTHER_GROUND', # Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
    'TYPE_WALKABLE', # Most horizontal surface that's not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
    'TYPE_SIDEWALK', # Nicely paved walkable surface when pedestrians most likely to walk on.
]


labels_map = {
    0: 0,  # TYPE_UNDEFINED
    1: 1,  # TYPE_CAR
    2: 2,  # TYPE_TRUCK
    3: 3,  # TYPE_BUS
    4: 4,  # TYPE_OTHER_VEHICLE, Other small vehicles (e.g. pedicab) and large vehicles (e.g. construction vehicles, RV, limo, tram).
    5: 5,  # TYPE_MOTORCYCLIST
    6: 6,  # TYPE_BICYCLIST
    7: 7,  # TYPE_PEDESTRIAN
    8: 8,  # TYPE_SIGN
    9: 9,  # TYPE_TRAFFIC_LIGHT
    10: 10, # TYPE_POLE, Lamp post, traffic sign pole etc.
    11: 11, # TYPE_CONSTRUCTION_CONE, Construction cone/pole.
    12: 12, # TYPE_BICYCLE
    13: 13, # TYPE_MOTORCYCLE
    14: 14, # TYPE_BUILDING
    15: 15, # TYPE_VEGETATION, Bushes, tree branches, tall grasses, flowers etc.
    16: 16, # TYPE_TREE_TRUNK
    17: 17, # TYPE_CURB, Curb on the edge of roads. This does not include road boundaries if there's no curb.
    18: 18, # TYPE_ROAD, Surface a vehicle could drive on. This include the driveway connecting parking lot and road over a section of sidewalk.
    19: 19, # TYPE_LANE_MARKER, Marking on the road that's specifically for defining lanes such as single/double white/yellow lines.
    20: 20, # TYPE_OTHER_GROUND, Marking on the road other than lane markers, bumps, cateyes, railtracks etc.
    21: 21, # TYPE_WALKABLE, Most horizontal surface that's not drivable, e.g. grassy hill, pedestrian walkway stairs etc.
    22: 22, # TYPE_SIDEWALK, Nicely paved walkable surface when pedestrians most likely to walk on.
}

color_map = { #rgb value
    0: [32, 119, 181],
    1: [228, 119, 194],
    2: [43, 160, 43],
    3: [220, 220, 141],
    4: [197, 176, 213],
    5: [209, 255, 6],
    6: [248, 182, 210],
    7: [152,224, 137],
    8: [29, 190, 208],
    9: [21, 255, 92],
    10: [174, 199, 232],
    11: [172,127,127],
    12: [215, 39, 40],
    13: [12, 116, 255],
    14: [140, 86, 74],
    15: [255, 127, 25],
    16: [200, 200, 200],
    17: [255, 152, 149],
    18: [158, 218, 229],
    19: [196, 156, 148],
    20: [255, 187, 120],
    21: [188, 190, 33],
    22: [148, 103, 189],
}

CLASS_NAMES_DG = ['Other', 'Driveable Ground', 'Structure', 'Vehicle', 'Nature', 'Living Being', 'Objects', 'Other Ground']

WAYMO_TO_CL = {
    0: 0,   # TYPE_UNDEFINED -> Other
    1: 3,   # TYPE_CAR -> Vehicle
    2: 3,   # TYPE_TRUCK -> Vehicle
    3: 3,   # TYPE_BUS -> Vehicle
    4: 3,   # TYPE_OTHER_VEHICLE -> Vehicle
    5: 5,   # TYPE_MOTORCYCLIST -> Living Being
    6: 5,   # TYPE_BICYCLIST -> Living Being
    7: 5,   # TYPE_PEDESTRIAN -> Living Being
    8: 6,   # TYPE_SIGN -> Objects
    9: 6,   # TYPE_TRAFFIC_LIGHT -> Objects
    10: 6,  # TYPE_POLE -> Objects
    11: 6,  # TYPE_CONSTRUCTION_CONE -> Objects
    12: 6,  # TYPE_BICYCLE -> Objects
    13: 6,  # TYPE_MOTORCYCLE -> Objects
    14: 2,  # TYPE_BUILDING -> Structure
    15: 4,  # TYPE_VEGETATION -> Nature
    16: 4,  # TYPE_TREE_TRUNK -> Nature
    17: 7,  # TYPE_CURB -> Other Ground (previously Driveable Ground)
    18: 1,  # TYPE_ROAD -> Driveable Ground
    19: 1,  # TYPE_LANE_MARKER -> Driveable Ground
    20: 1,  # TYPE_OTHER_GROUND -> Driveable Ground (previously Other Ground)
    21: 7,  # TYPE_WALKABLE -> Other Ground
    22: 7,  # TYPE_SIDEWALK -> Other Ground
}

DG = True


#TODO: metainfo
if DG:
    meta_info = dict(
        classes=CLASS_NAMES_DG, seg_label_mapping=WAYMO_TO_CL, max_label=8, voxel_label_mapping=WAYMO_TO_CL)
else:
    meta_info = dict(
        classes=class_names, seg_label_mapping=labels_map, max_label=19, voxel_label_mapping=labels_map)
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

Random = True
dense7sparse = True
ResetRandomProb = 0.0
consistency = True
dense_train = False
# Note: runner and model.d2skd are resolved when the config is parsed, so
# --cfg-options d2skd=True cannot change them; edit this line instead.
d2skd = True
# VanillaKD distillation baseline switch (comparison experiments; see vanillakd.py). When True,
# plain distillation (feature MSE + temperature KL) replaces spectral distillation over exactly
# the same signal path; only effective with d2skd=True and mutually exclusive with spectral
# distillation. The three baseline rows only change the two lambdas:
# (1,0)=Soft-label-KL / (0,0.5)=Feature-mimicking / (1,0.5)=Both
KDbaseline = False
KD_T = 4.0  # distillation temperature 4 across the project (also the official PVKD/CWD value)
KD_lambda_kd = 1.0
KD_lambda_feat = 0.5
sensor_film = True
plugin = True
plugin_dir = "projects/unilidar_plugin/"
img_norm_cfg = None
occ_path = "data/Waymo-Occ"
train_ann_file = data_root + 'waymo_infos_train.pkl'
val_ann_file = data_root + 'waymo_infos_val.pkl'

unilidar = False
cylinder=False
RPR = True
coor_alignment = False
ori_point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# point_cloud_range = [0, -25.6, -3.4, 51.2, 25.6, 3.0] if unilidar else [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [256, 256, 32] # ground truth
final_occ_size =  [480, 360, 32]# model output
model_empty_idx = 0  # noise 0-->255
empty_idx = 0  # noise 0-->255
num_cls = 8 if DG else 23
visible_mask = True


sensor = dict(
    LiDAR_height=[1.5, 1.85],
    num_of_beams=[32, 64],
    horizontal_angular_resolution=[600, 1800],
    lower_vertical_field_of_view_bound=[-28, -22],
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
# last 5 epochs plateaued. The stages are moved to (0,6),(6,12),(12,24): stage 3 now arrives
# at LR~2.9e-4 with 12 epochs to absorb it. Total epochs unchanged, keeping the comparison
# with the baseline fair.
schedule=[(0, 6, {
                'num_of_beams': (32, 64),
                'horizontal_angular_resolution': (900, 2400),
                'lower_vertical_field_of_view_bound': (-35, -20),
                'upper_vertical_field_of_view_bound': (0, 15),
                'LiDAR_height': (1.5, 1.9)
            }),
                      (6, 12, {
                          'num_of_beams': (24, 80),
                          'horizontal_angular_resolution': (600, 3000),
                          'lower_vertical_field_of_view_bound': (-40, -15),
                          'upper_vertical_field_of_view_bound': (0, 20),
                          'LiDAR_height': (1.2, 2)
                      }),
                      (12, 24, {
                          'num_of_beams': (16, 96),
                          'horizontal_angular_resolution': (400, 4000),
                          'lower_vertical_field_of_view_bound': (-45, -10),
                          'upper_vertical_field_of_view_bound': (0, 25),
                          'LiDAR_height': (0.8, 2.2)
                      })]

cascade_ratio = 2 
dataset_flag = 3


cumulative_iters = 2
find_unused_parameters = False
unique_label = [0, 1, 2, 3, 4, 5, 6, 7] if DG else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17,18,19,20,21,22]
track_running_stats = False

_dim_ = 128 

tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2

grid_size = final_occ_size
grid_size_occ = occ_size
coarse_ratio = cascade_ratio
sweeps_num = 10
nbr_class = num_cls

input_sample_policy = {
    "type": "random interval",
    "fix interval": 5,
    "number": 7,
} # only for training
pose_file = occ_path + '/cam_infos.pkl'
val_pose_file = occ_path + '/cam_infos_vali.pkl'

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
        # Hybrid dark knowledge: with d2skd and not KDbaseline, the vox path = spectral + temperature-KL
        # soft labels (the soft-label term is computed as in vanillakd.py with lambda=KD_lambda_kd; it
        # forms its own 'kd' group of the gradient-ratio controller below via rho['kd']); the inter
        # path stays purely spectral.
        hybrid_dark_kd = not KDbaseline,
        # [lower consistency while distilling] see the same note in the nuScenes config; the baseline
        # (d2skd=False) keeps 40.
        consistency_gain = 20.0 if d2skd else 40.0,
        # [HF distiller] lower gate_kappa noise floor + 2x channel weight for Living Being(5)/Objects(6)
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
            extra_scales=[16, 32],
            # ---- final HF settings (main results) ----
            hf_operator='support',      # 6-neighbour support-subgraph Laplacian on the teacher support
            gate_domain='support',      # gate statistics only on the gated region R (= teacher support)
            gate_source='prob',         # gate from the high-pass residual of the teacher class probabilities
            gate_share='voxel',         # one gate per voxel, shared by all channels
            hf_norm='teacher',          # residuals in units of the teacher residual scale (no student-side denominator)
            suppress_target='teacher',  # suppress only penalises the part of the student above the teacher
            lambda_value=20,            # value matching L_V
            lambda_cont=20),            # continuity L_C on neighbouring voxel pairs where the teacher gate is closed
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
    # Teacher serving input cap, aligned with the teacher training protocol: the maximum point
    # cloud is 500k points both for training the teacher and for serving it during distillation;
    # larger inputs are randomly downsampled to 500k points at the input stage, preserving the
    # scene structure. Teacher training (500k cap, see dense_cap_range in the teacher configs)
    # and teacher evaluation (test_max_points=500000) already follow this; only the
    # distillation-serving side did not cap before (SK measured 0.68M-1.17M points per frame,
    # about 2.2x the training density).
    # scope='all': tg_t/gi_t/gv_t are cut with the same indices, fully reproducing the teacher
    #   training protocol. Measured cost: -12.2% of the coarse voxels supervised by HF/KL, but the
    #   lost voxels are the fringe of the occupied set (71.6% are single-point voxels, 93.7% still
    #   have a surviving voxel in their 26-neighbourhood), not holes.
    #   With 'tokenizer' the support set is unchanged, but the teacher must output logits on ~12%
    #   of voxels outside its own token support. These voxels are not unseen during teacher
    #   training (it resamples every iteration; the dense branch of _sampling has no fixed seed),
    #   they are expected to be supervised ~11 times over 24 epochs; the real risk of 'tokenizer'
    #   is a **conditional-distribution mismatch** -- the voxel was supervised, but its own points
    #   are absent from the token input this time and the teacher has to extrapolate.
    # seed=None (resample every iteration): a fixed seed would leave 15.8% of the single-point
    #   voxels of a frame unsupervised in all 24 epochs, a permanent frame-bound blind spot; 18.3%
    #   of the coarse voxels of the bottleneck class Living Being are single-point voxels, which
    #   makes this particularly unacceptable. Resampling acts as "supervision-set dropout" and is
    #   unbiased across epochs.
    # Structure preservation measured: per-point class ratio 1.000+/-0.003 (0.964-1.035 for cls5
    #   is Poisson noise); coarse-voxel retention per range band 92.7/90.6/86.6/83.7/79.9%
    #   (0-10/../40-52 m), no cliff.
    #   Measured (SK, 12 frames x 30 independent draws): of 19 connected cls5 clusters, one k=1
    #   cluster is emptied in 16/30 draws, i.e. 0.044 objects/frame expected; 15 of 94 cls6
    #   clusters can be emptied, 0.40 objects/frame expected, with rho spread over 15.6-46 m.
    #   Whether a cluster is emptied is decided by its point count k, **not by distance**. The
    #   magnitude is tiny (about 10 points out of 1.1M per frame), but it is not zero loss. The
    #   bottleneck class cls5 is not selectively hurt: its coarse-voxel retention is 84.8%, at the
    #   all-class mean.
    # The lost 12.2% of coarse voxels by origin: voxels containing real scan returns keep 95.96%,
    #   pure LiDiff-completion voxels keep 83.49%; of the total loss 88.7% are pure completion
    #   voxels and 11.3% contain real returns (1.4 pp of the whole support set). The selectivity
    #   exists and goes in the right direction, but it does not follow that the lost voxels are
    #   essentially never real data.
    teacher_serve_max_points = 500000,
    teacher_serve_cap_scope = 'all',
    teacher_serve_cap_seed = None,
    d2skd = d2skd,
    distill_mode = 'vanilla_kd' if KDbaseline else 'spectral',
    kd_cfg = dict(T=KD_T, lambda_kd=KD_lambda_kd, lambda_feat=KD_lambda_feat),
    # [LF distiller] block-local response alignment (targets the high floor of raw_inter under
    # global normalisation; see the nuScenes config)
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
        profile_metric='l1', lambda_profile=8.0,
        # ---- final LF settings (main results) ----
        block_pad='fit',                  # blocks follow the actual plane width (no replicate padding)
        gate_domain='support',            # gated-plane statistics on the teacher-support projection R
        gate_source='prob',               # gate from the teacher class-probability planes
        profile_domain='open',            # profile energy weighted by the gate actually used
        plane_gate=[True, False, False],  # gate only the xy plane (order: xy, yz, zx)
        suppress_target='zero',           # push student band-pass energy towards 0 where the gate is closed
        suppress_ref_mode='log_tail',     # suppress reference depends only on the teacher
        ref_floor_rel=1e-4,               # floor coefficient of the teacher reference in log_tail mode
        ungated_domain='support'),        # ungated planes (yz, zx) use the teacher support
    dynamic_loss_balance=dict(
        enabled=True,
        alpha=0.1,          # GradNorm target strength
        step=0.0002,          # per-step update size
        momentum=0.3,       # smoothing for weight updates
        min_weight=0.1,
        max_weight=2.0,
        ema_beta=0.8,       # EMA for loss trend
        # Task grouping follows fixed order in model:
        # point / voxel / sensor / consistency
        base_weights=dict(
            point=1.2,
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
            point=1.8,
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
            # rho reallocation at an equal total (same as the nuScenes/SK configs; the nuScenes config has
            # the full rationale). Summary: measured, the dark-knowledge KD term outside this controller
            # had rho_kd=2.25 while the two spectral paths summed to 0.60, i.e. an actual split of
            # "spectral 21% / dark knowledge 79%", the opposite of the design intent (spectral 60 / dark
            # 40). The total is kept at 2.85 and only the direction changes => spectral 1.55 / dark 1.30
            # = 54/46; within the spectral budget vox goes 42% -> 48% (HF shares the same pair of tensors
            # with dark knowledge and competes on KD's home ground while getting only 8.8%; it was the
            # most starved path).
            # Waymo-specific caveat: rho_kd=2.25 was measured on nuScenes and never separately on Waymo,
            # so here it is a **transferred assumption**, not a measured value. Noise on the data side
            # also changes g_seg, which is the denominator of every rho in this controller. If Waymo
            # results look abnormal, check these points before changing rho.
            # The kd key is the opt-in switch and must be absent in KDbaseline mode (otherwise the main
            # loss of the vanilla baseline would also be put under the controller).
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
            rho=dict(inter=0.80, vox=0.75,
                     **({} if KDbaseline else dict(kd=1.30))),
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

train_pipeline = [
dict(
    type='LoadPointsFromFile_RPR',
    coord_type='LIDAR',
    load_dim=3 if Random else 6,
    use_dim=3 if Random else 5,
    shift_height=True,
    RPR=RPR,
    point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    dataset_flag=dataset_flag,
    shift_coors=GROUND_ALIGN_SHIFT,
    Random = Random,
    dense7sparse = dense7sparse if Random else False),
dict(
    type='LoadVoxels',
    to_float32=True, 
    use_semantic=True, 
    cylinder=cylinder, 
    occ_path=occ_path, 
    grid_size=occ_size, 
    use_vel=False,
    unoccupied=empty_idx, 
    pc_range = ori_point_cloud_range,
    RPR = RPR,
    restrict_pc_range = point_cloud_range,
    cal_visible=visible_mask,
    random=Random,
    file_client_args=dict(backend='disk')),
dict(type='VoxelClassMapping'),
dict(type='PointsegMapping', 
            grid_size = grid_size,
            grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z],
            coarse_ratio = coarse_ratio,
            pc_range = ori_point_cloud_range,
            RPR = RPR,
            restrict_pc_range = point_cloud_range,
            fill_label = 0,
            unique_label = unique_label,
            fixed_volume_space = True,
            max_volume_space = [51.2, 3.1415926, 3],
            min_volume_space = [0, -3.1415926, -3.4],
            cal_visible=False,
            ),
dict(type='MinorityDensePurify',
     # ---- per-dataset calibration (measured) ----
     # Waymo baseline (median of ROAD/BUILDING/CAR/VEGETATION) = 0.158m -- the Waymo LiDAR itself is
     # dense, and LiDiff completion quality is better overall than on the other two datasets.
     #   cls5: PEDESTRIAN 0.124m = 0.78x  -> completed even more accurately than the road surface
     #         BICYCLIST  0.140m = 0.89x  -> same   => cls5 needs no purification at all
     #   cls6: POLE       0.179m = 1.13x  (66% of cls6 points) -> good
     #         TRAFFIC_LIGHT 1.04x, CONSTR_CONE 0.71x        -> good
     #         SIGN       0.585m = 3.70x  (30%)              -> indeed hallucinated
     #         => cls6 is mixed: purifying all of cls6 to fix SIGN (30%) would wrongly drop POLE
     #            (66%), not worth it.
     #   => no purification on Waymo, only re-densification (adds density with real points, a pure
     #      gain).
     #   (Fixing SIGN separately would require purifying by the *raw fine class* rather than the
     #   COLA coarse class.)
     enable=True,
     purify_classes=(),  # no purification: measured completion fidelity is already good enough
     densify_classes=(5, 6),
     purify_thresh=0.2, densify_k=5, densify_jitter=0.02, max_add_per_class=20000),
dict(type='Collect3Dinput', keys=['train_grid', 'grid_ind', 'grid_ind_vox', 'train_voxel_label', 'train_pts_label', 'reset_random']),
]

test_pipeline = [
    dict(
    type='LoadPointsFromFile_RPR',
    coord_type='LIDAR',
    load_dim=6,
    use_dim=3 if Random else 5,
    shift_height=True,
    RPR=RPR,
    point_cloud_range=point_cloud_range if RPR else ori_point_cloud_range,
    dataset_flag=dataset_flag,
    shift_coors=GROUND_ALIGN_SHIFT,),
    dict(
    type='LoadVoxels',
    to_float32=True, 
    use_semantic=True, 
    cylinder=cylinder, 
    occ_path=occ_path, 
    grid_size=occ_size, 
    use_vel=False,
    unoccupied=empty_idx, 
    pc_range = ori_point_cloud_range,
    RPR = RPR,
    restrict_pc_range = point_cloud_range,
    cal_visible=visible_mask,
    file_client_args=dict(backend='disk')),
    dict(type='VoxelClassMapping'),
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
    occ_root=occ_path,
    data_root=data_root,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    modality=input_modality,
    pose_file=val_pose_file,
    split='training',
    classes=CLASS_NAMES_DG if DG else class_names,
    occ_size=occ_size,
    seg_label_mapping=WAYMO_TO_CL if DG else labels_map,
    pc_range=point_cloud_range,
    filter_empty_gt=True,
    random=False,
)

train_config=dict(
        type=dataset_type,
        data_root=data_root,
        occ_root=occ_path,
        ann_file=train_ann_file,
        pipeline=train_pipeline,
        modality=input_modality,
        pose_file=pose_file,
        input_sample_policy=input_sample_policy,
        split='training',
        classes=CLASS_NAMES_DG if DG else class_names,
        test_mode=False,
        occ_size=occ_size,
        seg_label_mapping=WAYMO_TO_CL if DG else labels_map,
        pc_range=point_cloud_range,
        filter_empty_gt=True,
        random=Random,
        reset_random_prob=ResetRandomProb,
        repeat=1 if Random else 1,
        dense7sparse = dense7sparse if Random else False,
)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6, #12
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,  # with 5e-4 the actual LR in phases 2/3 (ep9+) is 2.5x that of nuScenes, linked to late val oscillation; 3e-4 measured lossless at ep1
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=cumulative_iters,
    grad_clip=dict(max_norm=30, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr=3e-5)  # the former floor was 40-60% of the peak, with no low-LR consolidation phase -> late val oscillation

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
# [enabled] Unit test: during warmup the EMA strictly follows the current weights (the 54%
# cold-start contamination does not occur); the EMA weights are swapped in/out around val;
# smooths the epoch-to-epoch swings of the minority classes.
custom_hooks = [
    dict(type='WarmupEMAHook', momentum=0.0002, start_epochs=1, priority='ABOVE_NORMAL'),
]

runner = dict(type='EpochBasedRunner', max_epochs=24) if not d2skd else dict(type='DistillEpochBasedRunner', max_epochs=24)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_mean',
    rule='greater',
)

load_from = None
# d2skd teacher weights (dense pretrained model), required when d2skd is on:
d2skd_load_from = 'ckpts/teacher_dense_waymo.pth'  # placeholder: dense teacher trained with pointtpv_waymo_teacher_dense.py
work_dir = './work_dirs/pointtpv_waymo_seg_random'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # Waymo-specific alpha strategy (conclusion after two failed measurements; see the comments in
        # curriculumhook.py):
        #  - Waymo's sensor MAE floor is ~0.5 sigma (the true height 2.36 m lies outside the
        #    randomisation range, theta_low is not identifiable), so MAE gating is unreliable:
        #    0.5/0.6 -> jitters throughout; 0.6/0.7 -> locks too early at ep2.
        #  - Use a progress floor instead: alpha=0 for the first 50% of training (the decoder sees
        #    clean GT conditioning), a linear climb to 1 from 50% to 85%, and alpha=1 throughout the
        #    last ~4 epochs (consistent with inference, adapting in the low-LR phase).
        #  - alpha_target_low below the MAE floor -> the MAE up-path is silent; alpha is monotonic,
        #    without jitter.
        # Lesson from a later measurement: the new schedule makes the sensor task easier and the MAE
        # floor dropped from ~0.52 to ~0.45-0.47, right at the old silence threshold 0.45 -> bang-bang
        # one-way ratcheting again, pushing alpha to 0.99 already at ep5 (by design the floor drives
        # it from ep12). The silence threshold must be well below the MAE floor of the current
        # schedule -> 0.30.
        dict(type='LiDARCurriculumHook', schedule=schedule,
             alpha_target_low=0.30, alpha_target_high=0.40,
             alpha_floor_start=0.5, alpha_floor_end=0.85),
    ])

checkpoint_config = dict(interval=2)
