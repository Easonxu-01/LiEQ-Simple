"""Data access for the native (point-level) spectral statistics.

Pairs each sparse raw LiDAR scan with its dense completed cloud, maps the raw
semantic labels of both sides to the 7 coarse classes, and selects the frames.

Expected layout (both roots are command-line arguments; the dense root
defaults to the raw root when the completed clouds live inside the dataset
tree):

  SemanticKITTI  raw:   sequences/<seq>/velodyne/<id>.bin
                        sequences/<seq>/labels/<id>.label
                 dense: sequences/<seq>/refine/<id>.ply
                        sequences/<seq>/dense_labels/<id>.label
  nuScenes       raw:   samples/LIDAR_TOP/<name>.pcd.bin
                        lidarseg/v1.0-trainval/<token>_lidarseg.bin
                        nuscenes_infos_val.pkl  (mmdet3d info file, val split)
                        v1.0-trainval/sample_data.json  (only for old-format info files)
                 dense: samples/LIDAR_TOP/refine/<name>.pcd.ply
                        lidarseg_refine/v1.0-trainval/<token>_lidarseg.bin
  Waymo          raw:   kitti_format/training/velodyne/<stem>.bin  (6 floats per point)
                        kitti_format/training/velodyne/<stem>.label (int32 pairs, class in column 1)
                 dense: kitti_format/training/refine/<stem>.ply
                        kitti_format/training/dense_labels/<stem>.label (uint32)

Frame uid: SemanticKITTI "<seq>_<id>", nuScenes the lidar sample_data token,
Waymo the file stem. The frames analysed in the paper are listed in frames/.
"""
import glob
import hashlib
import json
import os
import pickle
from collections import defaultdict

import numpy as np

DATASETS = ("semantickitti", "nuscenes", "waymo")
DATASET_ALIASES = {
    "sk": "semantickitti", "semantickitti": "semantickitti", "kitti": "semantickitti",
    "nus": "nuscenes", "nuscenes": "nuscenes",
    "waymo": "waymo",
}
DISPLAY = {"semantickitti": "SemanticKITTI", "waymo": "Waymo", "nuscenes": "nuScenes"}

# Spherical crop: keep points with ||p|| <= radius (metres).
POINT_CLOUD_RADIUS = 50.0

CLASS_NAMES = {
    0: "Other",
    1: "Driveable Ground",
    2: "Structure",
    3: "Vehicle",
    4: "Nature",
    5: "Living Being",
    6: "Objects",
    7: "Other Ground",
}
# Class 0 (Other) is excluded from every statistic.
CLASS_IDS = (1, 2, 3, 4, 5, 6, 7)

# ---- raw label id -> coarse class (same mapping as the training configs) ----
SEMANTICKITTI_TO_COARSE = {
    0: 0, 1: 0, 10: 3, 11: 6, 13: 3, 15: 6, 16: 3, 18: 3, 20: 3, 30: 5, 31: 5,
    32: 5, 40: 1, 44: 1, 48: 7, 49: 7, 50: 2, 51: 2, 52: 0, 60: 1, 70: 4, 71: 4,
    72: 7, 80: 6, 81: 6, 99: 0, 252: 3, 253: 5, 254: 5, 255: 5, 256: 3, 257: 3, 258: 3,
    259: 3,
}

WAYMO_TO_COARSE = {
    0: 0, 1: 3, 2: 3, 3: 3, 4: 3, 5: 5, 6: 5, 7: 5, 8: 6, 9: 6, 10: 6,
    11: 6, 12: 6, 13: 6, 14: 2, 15: 4, 16: 4, 17: 7, 18: 1, 19: 1,
    20: 1, 21: 7, 22: 7,
}

# nuScenes: raw lidarseg id -> 16-class learning map -> coarse class.
NUSCENES_LEARNING_MAP = {
    0: 0, 1: 7, 2: 7, 3: 7, 4: 7, 5: 7, 6: 7, 7: 7, 8: 7, 9: 1, 10: 8,
    11: 8, 12: 8, 13: 8, 14: 2, 15: 3, 16: 3, 17: 4, 18: 5, 19: 4,
    20: 4, 21: 6, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
    28: 15, 29: 0, 30: 16, 31: 0,
}
NUSCENES_16_TO_COARSE = {
    0: 0, 1: 6, 2: 6, 3: 3, 4: 3, 5: 3, 6: 6, 7: 5, 8: 6, 9: 3, 10: 3,
    11: 1, 12: 7, 13: 7, 14: 7, 15: 2, 16: 4,
}
NUSCENES_TO_COARSE = {raw: NUSCENES_16_TO_COARSE[c16]
                      for raw, c16 in NUSCENES_LEARNING_MAP.items()}

DEFAULT_SK_SEQS = "08"          # SemanticKITTI validation sequence
NUSCENES_VERSION = "v1.0-trainval"
NUSCENES_INFO = "nuscenes_infos_val.pkl"
FRAME_LIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")


# =============================================================================
# Loading
# =============================================================================

def map_labels(ids, mapping):
    """LUT-mapped ids; ids without a mapping entry (or out of range) -> 0."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    size = int(max(mapping)) + 1
    lut = np.zeros(size, dtype=np.int64)
    for k, v in mapping.items():
        lut[int(k)] = int(v)
    safe = np.where((ids >= 0) & (ids < size), ids, 0)
    return lut[safe]


def load_points(path, bin_dim=4):
    """(N, 3) float64 xyz from a .ply or a float32 .bin; non-finite rows dropped."""
    path = str(path)
    if path.endswith(".ply"):
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError("open3d is required for reading .ply files.") from exc
        pcd = o3d.io.read_point_cloud(path)
        points = np.asarray(pcd.points, dtype=np.float64).reshape(-1, 3)
    elif path.endswith(".bin"):
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % bin_dim != 0:
            raise ValueError("{}: raw float count {} is not divisible by bin_dim={}.".format(
                path, raw.size, bin_dim))
        points = raw.reshape(-1, bin_dim)[:, :3].astype(np.float64)
    else:
        raise ValueError("Unsupported point-cloud extension: {}".format(path))
    points = points[np.isfinite(points).all(axis=1)]
    return points


def load_labels(path, label_format, expected_n=None):
    """Raw label file -> coarse class ids in [0, 7].

    semantickitti     : uint32 .label (& 0xFFFF) -> SEMANTICKITTI_TO_COARSE
    waymo_sparse      : int32 (-1, 2)[:, 1] raw Waymo ids -> WAYMO_TO_COARSE
    waymo_dense       : uint32 raw Waymo ids -> WAYMO_TO_COARSE
    nuscenes_lidarseg : uint8 raw lidarseg ids -> NUSCENES_TO_COARSE
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if label_format == "semantickitti":
        raw = np.fromfile(path, dtype=np.uint32).reshape(-1)
        labels = map_labels(raw & 0xFFFF, SEMANTICKITTI_TO_COARSE)
    elif label_format == "waymo_sparse":
        raw = np.fromfile(path, dtype=np.int32)
        if raw.size % 2 != 0:
            raise ValueError("{}: Waymo sparse label count {} not divisible by 2".format(path, raw.size))
        labels = map_labels(raw.reshape(-1, 2)[:, 1], WAYMO_TO_COARSE)
    elif label_format == "waymo_dense":
        labels = map_labels(np.fromfile(path, dtype=np.uint32), WAYMO_TO_COARSE)
    elif label_format == "nuscenes_lidarseg":
        labels = map_labels(np.fromfile(path, dtype=np.uint8), NUSCENES_TO_COARSE)
    else:
        raise ValueError("Unsupported label_format={}".format(label_format))
    labels = np.clip(labels, 0, 7).astype(np.int64)
    if expected_n is not None and labels.shape[0] != expected_n:
        n = min(labels.shape[0], expected_n)
        print("[labels] count mismatch: {} vs {}; truncating to {}.".format(
            labels.shape[0], expected_n, n))
        labels = labels[:n]
    return labels


def align_points_and_labels(points, labels):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = min(points.shape[0], labels.shape[0])
    if points.shape[0] != labels.shape[0]:
        print("[align] points/labels mismatch: {} vs {}; truncating to {}.".format(
            points.shape[0], labels.shape[0], n))
    return points[:n], labels[:n]


def remove_invalid_points(points, labels):
    mask = np.isfinite(points).all(axis=1)
    return points[mask], labels[mask]


def filter_points_by_radius(points, labels, radius=POINT_CLOUD_RADIUS):
    """Keep points with Euclidean range ||p|| <= radius (spherical crop)."""
    if radius is None:
        return points, labels
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    mask = np.linalg.norm(points, axis=1) <= float(radius)
    return points[mask], labels[mask]


def deterministic_subsample(points, labels, max_points, seed):
    """Uniform random subsample without replacement to max_points (index order kept)."""
    if max_points is None or len(points) <= max_points:
        return points, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx], labels[idx]


def load_side(points_path, labels_path, bin_dim, label_format, radius=POINT_CLOUD_RADIUS):
    """Load one side of a frame: points + coarse labels, aligned, finite, cropped."""
    points = load_points(points_path, bin_dim=bin_dim)
    labels = load_labels(labels_path, label_format, expected_n=points.shape[0])
    points, labels = align_points_and_labels(points, labels)
    points, labels = remove_invalid_points(points, labels)
    points, labels = filter_points_by_radius(points, labels, radius)
    return points, labels


# =============================================================================
# Frame discovery and pairing
# =============================================================================

def _glob_stems(directory, ext):
    return {os.path.basename(p)[: -len(ext)]: p
            for p in glob.glob(os.path.join(directory, "*" + ext))}


def discover_semantickitti(raw_root, dense_root, seqs):
    """Paired frames: velodyne/*.bin, labels/*.label <-> refine/*.ply, dense_labels/*.label."""
    raw_base = os.path.join(raw_root, "sequences")
    dense_base = os.path.join(dense_root, "sequences")
    tasks, skipped = [], []
    for seq in seqs:
        sparse = _glob_stems(os.path.join(raw_base, seq, "velodyne"), ".bin")
        dense = _glob_stems(os.path.join(dense_base, seq, "refine"), ".ply")
        if not sparse and not dense:
            skipped.append(("semantickitti", seq, "*", "no files found for sequence"))
            continue
        for stem in sorted(set(sparse) | set(dense)):
            if stem not in sparse:
                skipped.append(("semantickitti", seq, stem, "missing sparse points"))
                continue
            if stem not in dense:
                skipped.append(("semantickitti", seq, stem, "missing dense points"))
                continue
            s_lab = os.path.join(raw_base, seq, "labels", stem + ".label")
            d_lab = os.path.join(dense_base, seq, "dense_labels", stem + ".label")
            if not os.path.exists(s_lab):
                skipped.append(("semantickitti", seq, stem, "missing sparse labels"))
                continue
            if not os.path.exists(d_lab):
                skipped.append(("semantickitti", seq, stem, "missing dense labels"))
                continue
            tasks.append(dict(
                dataset="semantickitti", seq=seq, frame=stem,
                uid="{}_{}".format(seq, stem),
                sparse_points=sparse[stem], dense_points=dense[stem],
                sparse_labels=s_lab, dense_labels=d_lab,
                sparse_bin_dim=4, dense_bin_dim=4,
                sparse_label_format="semantickitti",
                dense_label_format="semantickitti",
            ))
    return tasks, skipped


def discover_waymo(raw_root, dense_root):
    """The dense clouds define the universe (only the segmentation-annotated
    frames were completed). Stratification key = stem[:4]."""
    raw_base = os.path.join(raw_root, "kitti_format", "training")
    dense_base = os.path.join(dense_root, "kitti_format", "training")
    tasks, skipped = [], []
    dense = _glob_stems(os.path.join(dense_base, "refine"), ".ply")
    for stem in sorted(dense):
        s_pts = os.path.join(raw_base, "velodyne", stem + ".bin")
        s_lab = os.path.join(raw_base, "velodyne", stem + ".label")
        d_lab = os.path.join(dense_base, "dense_labels", stem + ".label")
        if not os.path.exists(s_pts):
            skipped.append(("waymo", stem[:4], stem, "missing sparse points"))
            continue
        if not os.path.exists(s_lab):
            skipped.append(("waymo", stem[:4], stem, "missing sparse labels"))
            continue
        if not os.path.exists(d_lab):
            skipped.append(("waymo", stem[:4], stem, "missing dense labels"))
            continue
        tasks.append(dict(
            dataset="waymo", seq=stem[:4], frame=stem, uid=stem,
            sparse_points=s_pts, dense_points=dense[stem],
            sparse_labels=s_lab, dense_labels=d_lab,
            sparse_bin_dim=6, dense_bin_dim=4,
            sparse_label_format="waymo_sparse",
            dense_label_format="waymo_dense",
        ))
    return tasks, skipped


def discover_nuscenes(raw_root, dense_root, info_pkl=None, version=NUSCENES_VERSION):
    """Val keyframes from the mmdet3d info file. Lidarseg files are keyed by
    the lidar sample_data token, point files by file name. Stratification
    key = log prefix of the lidar file name."""
    tasks, skipped = [], []
    pkl_path = info_pkl or os.path.join(raw_root, NUSCENES_INFO)
    if not os.path.exists(pkl_path):
        skipped.append(("nuscenes", "*", "*", "info file missing: {}".format(pkl_path)))
        return tasks, skipped
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    records = data["data_list"] if isinstance(data, dict) and "data_list" in data \
        else data.get("infos", []) if isinstance(data, dict) else data
    sd_token_by_fname = None   # old info format: lazily built file name -> token map
    for rec in records:
        try:
            lidar_name = rec["lidar_points"]["lidar_path"]
            mask_name = os.path.basename(rec["pts_semantic_mask_path"])
        except (KeyError, TypeError):
            # Old mmdet3d info format (top-level lidar_path, no mask path): the
            # lidarseg file is <sample_data_token>_lidarseg.bin, resolved through
            # the file name -> token map of sample_data.json. Both formats give
            # identical task fields.
            lp = rec.get("lidar_path") if isinstance(rec, dict) else None
            if not lp:
                continue
            lidar_name = os.path.basename(lp)
            if sd_token_by_fname is None:
                sd_json = os.path.join(raw_root, version, "sample_data.json")
                if not os.path.exists(sd_json):
                    skipped.append(("nuscenes", "*", "*",
                                    "old-format info file but sample_data.json missing: %s" % sd_json))
                    return tasks, skipped
                with open(sd_json, "r", encoding="utf-8") as f:
                    sd_token_by_fname = {
                        os.path.basename(r["filename"]): r["token"]
                        for r in json.load(f)
                        if "LIDAR_TOP" in r.get("filename", "")}
            token = sd_token_by_fname.get(lidar_name)
            if token is None:
                skipped.append(("nuscenes", "*", lidar_name, "no sample_data token"))
                continue
            mask_name = token + "_lidarseg.bin"
        seq = lidar_name.split("__")[0]
        stem = lidar_name[:-len(".pcd.bin")] if lidar_name.endswith(".pcd.bin") else lidar_name
        token = mask_name[:-len("_lidarseg.bin")] if mask_name.endswith("_lidarseg.bin") else mask_name
        s_pts = os.path.join(raw_root, "samples", "LIDAR_TOP", lidar_name)
        d_pts = os.path.join(dense_root, "samples", "LIDAR_TOP", "refine",
                             lidar_name.replace(".bin", ".ply"))
        s_lab = os.path.join(raw_root, "lidarseg", version, mask_name)
        d_lab = os.path.join(dense_root, "lidarseg_refine", version, mask_name)
        if not os.path.exists(s_pts):
            skipped.append(("nuscenes", seq, stem, "missing sparse points"))
            continue
        if not os.path.exists(d_pts):
            skipped.append(("nuscenes", seq, stem, "missing dense points"))
            continue
        if not os.path.exists(s_lab):
            skipped.append(("nuscenes", seq, stem, "missing sparse labels"))
            continue
        if not os.path.exists(d_lab):
            skipped.append(("nuscenes", seq, stem, "missing dense labels"))
            continue
        tasks.append(dict(
            dataset="nuscenes", seq=seq, frame=stem, uid=token,
            sparse_points=s_pts, dense_points=d_pts,
            sparse_labels=s_lab, dense_labels=d_lab,
            sparse_bin_dim=5, dense_bin_dim=4,
            sparse_label_format="nuscenes_lidarseg",
            dense_label_format="nuscenes_lidarseg",
        ))
    return tasks, skipped


def discover_frames(dataset, raw_root, dense_root, sk_seqs=DEFAULT_SK_SEQS, nusc_info=None):
    if dataset == "semantickitti":
        seqs = [s.strip() for s in sk_seqs.split(",") if s.strip()]
        return discover_semantickitti(raw_root, dense_root, seqs)
    if dataset == "waymo":
        return discover_waymo(raw_root, dense_root)
    if dataset == "nuscenes":
        return discover_nuscenes(raw_root, dense_root, info_pkl=nusc_info)
    raise ValueError("Unknown dataset: {}".format(dataset))


# =============================================================================
# Frame selection
# =============================================================================

def _stable_permutation(n, key):
    seed64 = int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(seed64).permutation(n)


def _round_robin(groups):
    out = []
    i = 0
    while True:
        emitted = False
        for g in groups:
            if i < len(g):
                out.append(g[i])
                emitted = True
        if not emitted:
            break
        i += 1
    return out


def stratified_sample_order(tasks, seed, dataset):
    """Deterministic per-sequence shuffle followed by a round-robin interleave.

    Frames are grouped by sequence key, sorted by frame id inside each group,
    shuffled with a permutation seeded from md5("<seed>:<dataset>:<seq>"), and
    the groups are interleaved one frame at a time in sorted sequence order.
    The first N entries are the stratified sample of size N.
    """
    by_seq = defaultdict(list)
    for t in tasks:
        by_seq[t["seq"]].append(t)
    groups = []
    for seq in sorted(by_seq):
        g = sorted(by_seq[seq], key=lambda t: t["frame"])
        perm = _stable_permutation(len(g), "{}:{}:{}".format(seed, dataset, seq))
        groups.append([g[i] for i in perm])
    return _round_robin(groups)


def read_frame_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def paper_frame_list(dataset):
    return read_frame_list(os.path.join(FRAME_LIST_DIR, "{}.txt".format(dataset)))


def select_frames(tasks, dataset, spec, seed=0):
    """spec: 'paper' (shipped list), 'sample:N' (stratified rule, first N) or a uid-list file."""
    if spec.startswith("sample:"):
        n = int(spec.split(":", 1)[1])
        return stratified_sample_order(tasks, seed, dataset)[:n]
    uids = paper_frame_list(dataset) if spec == "paper" else read_frame_list(spec)
    by_uid = {t["uid"]: t for t in tasks}
    missing = [u for u in uids if u not in by_uid]
    if missing:
        raise SystemExit("[{}] {} requested frame(s) not found under the given roots, e.g. {}".format(
            dataset, len(missing), ", ".join(missing[:3])))
    return [by_uid[u] for u in uids]


def sk_seqs_for_spec(spec, default=DEFAULT_SK_SEQS):
    """SemanticKITTI sequences needed by an explicit uid list (uids are '<seq>_<id>')."""
    if spec.startswith("sample:"):
        return default
    uids = paper_frame_list("semantickitti") if spec == "paper" else read_frame_list(spec)
    return ",".join(sorted({u.split("_", 1)[0] for u in uids}))


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "{}.tmp.{}".format(path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
