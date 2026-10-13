"""Single source of truth for the coarse-class mapping.

8 coarse classes:
    0 Other / 1 Driveable Ground / 2 Structure / 3 Vehicle
    4 Nature / 5 Living Being / 6 Objects / 7 Other Ground
Coarse class 0 does not take part in mIoU (it is dropped at evaluation via
valid_mask = gt != 0); mIoU = sum over the other 7 classes divided by 7.

Background: the same mapping table used to be duplicated in many places across the
repository (configs, diagnostic scripts, packaged copies), and VoxelClassMapping builds its
lookup table with np.zeros -- **a key missed in an edit silently becomes coarse class 0
without any error**. This module gathers the three tables in one place.

To change the mapping: edit this file, then make sure every copy (the inline tables in the
training configs and label_mapping/nuscenes.yaml) is still identical to it.
"""

COARSE_NAMES = {0: 'Other', 1: 'Driveable Ground', 2: 'Structure', 3: 'Vehicle',
                4: 'Nature', 5: 'Living Being', 6: 'Objects', 7: 'Other Ground'}

# --- SemanticKITTI: raw -> coarse -------------------------------------------
SEMANTICKITTI_TO_CL = {
    0: 0, 1: 0,                                  # unlabeled / outlier
    # 52/99 changed from 2/6 -> 0.
    # Evidence 1 (official convention): the official SemanticKITTI learning_map itself
    #   maps these two catch-all classes to unlabeled (semantic-kitti.yaml:121 and :135),
    #   officially defined as "too few points or too much intra-class variance". The
    #   coarse table previously revived them, inconsistent with the fine-class benchmark.
    # Evidence 2 (measured scale, full statistics read directly from the .label files,
    #   train 19130 frames, 2.35 billion points): raw 99 makes up 71.14% of the Objects
    #   training signal (val 75.47%), raw 52 makes up 1.15% of Structure (val 1.84%).
    #   I.e. nearly three quarters of the Objects supervision came from miscellaneous
    #   leftovers that the official benchmark deems unusable.
    # Evidence 3 (consequence, paired confusion matrices with the same weights and the same
    #   mapping): class 6 became a sink defined by exclusion; 28.02% of Living Being is
    #   predicted as Objects already in the source domain and 63.73% cross-domain, a
    #   selectivity 4-6x higher than for any other class.
    # Note: this is a change of the label taxonomy; SK numbers obtained with the earlier
    #   mapping are not comparable and require retraining / re-evaluation.
    52: 0, 99: 0,                                # other-structure / other-object
    10: 3, 13: 3, 16: 3, 18: 3, 20: 3,           # car/bus/on-rails/truck/other-vehicle
    11: 6, 15: 6,                                # bicycle / motorcycle   [15 previously 3]
    30: 5, 31: 5, 32: 5,                         # person/bicyclist/motorcyclist
    40: 1, 44: 1, 60: 1,                         # road/parking/lane-marking
    48: 7, 49: 7, 72: 7,                         # sidewalk/other-ground/terrain  [72 previously 4]
    50: 2, 51: 2,                                # building/fence
    70: 4, 71: 4,                                # vegetation/trunk
    80: 6, 81: 6,                                # pole/traffic-sign
    252: 3, 256: 3, 257: 3, 258: 3, 259: 3,      # moving-* vehicles
    253: 5, 254: 5, 255: 5,                      # moving-bicyclist/person/motorcyclist
}

# --- nuScenes: two levels. raw(0-31) -> 16 classes -> coarse ---------------------
# The first level has the same origin as projects/configs/pointocc/label_mapping/nuscenes.yaml::learning_map
NUSCENES_LEARNING_MAP = {
    0: 0, 29: 0, 31: 0,                          # noise / static.other / vehicle.ego
    1: 7, 5: 7, 7: 7, 8: 7,                      # animal/personal_mobility/stroller/wheelchair  [previously 0]
    2: 7, 3: 7, 4: 7, 6: 7,                      # pedestrian adult/child/worker/police
    9: 1, 12: 8,                                 # barrier / trafficcone
    10: 8, 11: 8, 13: 8,                         # debris/pushable_pullable/bicycle_rack  [previously 0]
    14: 2, 21: 6,                                # bicycle / motorcycle (riders included; not separable in the official labels)
    15: 3, 16: 3, 17: 4, 18: 5, 22: 9, 23: 10,   # bus.bendy/bus.rigid/car/construction/trailer/truck
    19: 4, 20: 4,                                # emergency.ambulance / emergency.police  [previously 0]
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 30: 16,
}
NUSCENES_TO_CL = {                               # 16 classes -> coarse
    0: 0, 1: 6, 2: 6, 3: 3, 4: 3, 5: 3, 6: 6, 7: 5, 8: 6,
    9: 3, 10: 3, 11: 1, 12: 7, 13: 7, 14: 7, 15: 2, 16: 4,   # [14 previously 4]
}
NUSCENES_RAW_TO_CL = {r: NUSCENES_TO_CL[c] for r, c in NUSCENES_LEARNING_MAP.items()}

# --- Waymo: raw -> coarse ----------------------------------------------------
WAYMO_TO_CL = {
    0: 0,
    1: 3, 2: 3, 3: 3, 4: 3,                      # CAR/TRUCK/BUS/OTHER_VEHICLE
    5: 5, 6: 5, 7: 5,                            # MOTORCYCLIST/BICYCLIST/PEDESTRIAN
    8: 6, 9: 6, 10: 6, 11: 6, 12: 6, 13: 6,      # SIGN/TRAFFIC_LIGHT/POLE/CONE/BICYCLE/MOTORCYCLE
    14: 2,                                       # BUILDING
    15: 4, 16: 4,                                # VEGETATION/TREE_TRUNK
    18: 1, 19: 1, 20: 1,                         # ROAD/LANE_MARKER/OTHER_GROUND  [20 previously 7]
    17: 7, 21: 7, 22: 7,                         # CURB/WALKABLE/SIDEWALK         [17 previously 1]
}

# Known inherent granularity differences that no mapping can bridge (to be disclosed as-is in the paper)
INHERENT_GAPS = [
    "nuScenes has no separate pole/sign/traffic-light fine class: the official static.manmade definition includes them together with buildings and walls, and they cannot be separated -> Structure in NU, Objects in SK/WA",
    "nuScenes riders cannot be separated from their vehicle: the official annotation rule says 'If there is a rider and/or passenger, include them in the box' -> Objects in NU, Living Being in SK/WA",
    "Waymo has no fence fine class: its Structure contains only building",
]
