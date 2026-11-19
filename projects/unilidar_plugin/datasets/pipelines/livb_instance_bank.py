'''
Cross-frame Living-Being instance bank paste (LivBInstanceBankPaste)

Background: on SemanticKITTI, Living Being (coarse class 5 in coarse_mapping.py; the mapping is not
changed) accounts for only 0.053% of the labelled training points; 32% of the frames contain LivB, and
the whole training set has about 103 (seq, instance) instances. The in-model _mixup_minority only
re-places instances **within the same frame** (the 68% of frames without LivB never get any LivB), and
the dense teacher branch skips it altogether. This transform, on the dataloader side, takes K instances
from an offline instance bank (built from the training sequences only, see the README for its
construction and file format; sequence 08 never enters the bank) and pastes them into the current training frame with
probability p, giving these rare instances new contexts.

Placement rules (each instance is tried at most max_tries times; it is accepted only if all rules hold):
  1. Rotation about the sensor z axis (default ±180°): the horizontal range rho and the elevation
     angle of every point are unchanged, so the sampling density of the instance is exactly that of the
     original scan (same sensor model, HDL-64E) and the side that is seen is also the same as in the
     original observation. Optional range_jitter radial shift; when moved farther away the points are
     thinned by (r0/r)^2 to match the density.
  2. Ground snapping: the local ground of the target frame at the new position (median z of the points
     of coarse classes 1/7, with thresholds on the number of support points and on the p90-p10 flatness)
     is aligned with the original ground stored in the bank by a global shift dz (|dz| <= max_ground_shift).
  3. No collision: within collision_radius (horizontally) of the instance at the new position, in the
     height interval from clearance above the ground up to the top of the instance, there must be no
     non-ground points of the target frame (ground-class bumps higher than ground_bump_max also count);
     instances pasted several times into the same frame must not overlap each other either (their angular
     cells must be disjoint).
  4. Line of sight and occlusion: on the sensor (az, el) angular grid, the placement is rejected if target
     non-ground points in front of the instance block more than los_max_frac of its cells (the raw instance
     points in blocked cells are removed); target points behind the instance in the same angular cells that
     belong to "scan rows" are removed (keeping the scan self-consistent: in a real scan, points hidden
     behind a person return no echo).
     "Scan rows" are set by occlusion_rows: 'scan' (default) = the real sparse prefix of a concatenated
     cloud / all rows of a pure raw-scan frame; dense completion rows are not removed (the completion is
     scene geometry, not a scan; removing them would leave shadow holes in the dense teacher input that
     exist only where something was pasted -- a shortcut feature).

Row bookkeeping (same convention as MinorityDensePurify and as the nuScenes/Waymo boundary fix
shrink_num_sparse_points in PointoccMapping.py):
  row order of a concatenated cloud = [real sparse | dense completion], results['num_sparse_points'] =
  prefix length. This transform first removes rows with the keep mask (new boundary = the number of rows
  kept within the old prefix, the same formula as shrink_num_sparse_points), then inserts the raw part of
  the pasted instances at the end of the sparse prefix and appends the dense part at the very end;
  boundary += number of pasted raw rows. Downstream, MinorityDensePurify therefore treats the pasted raw
  points as real points (exempt from purification, used as the purification reference, copied by
  re-densification) and purifies the pasted dense points by the usual rules -- exactly the same treatment
  as native instances. Pure dense frames (the teacher, dense7sparse=False) do not write num_sparse_points.

Pipeline position: must be placed after VoxelClassMapping and before PointsegMapping (at that point the
labels are already COLA-8, and train_voxel_label / grid_ind / train_grid are all computed by
PointsegMapping from the points and labels, so the pasted instances automatically enter the voxel
supervision; the points added by the in-model _mixup_minority have no voxel labels).
A wrong position raises an error immediately instead of silently doing nothing.

With prob=0 (default) the first line of __call__ returns the same results object: the bank is not read
and no RNG is consumed (this transform only uses a private RandomState; even with p>0 it does not disturb
the global np.random / random / torch streams), so the output is bit-for-bit identical to a pipeline
without this entry.
'''
import json
import math
import os
import re
import warnings

import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

_SEQ_FRAME_RE = re.compile(r'sequences/(\d+)/(?:velodyne|refine|labels|dense_labels)/(\d+)\.')


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def shrink_boundary(n_sparse, keep):
    """Same formula as shrink_num_sparse_points in PointoccMapping.py: removing rows with a boolean
    mask preserves the row order, the real sparse points are still a prefix, and the new boundary =
    the number of rows kept within the old prefix."""
    keep = np.asarray(keep).reshape(-1).astype(bool)
    if keep.shape[0] < n_sparse:
        raise ValueError(f'[LivBInstanceBankPaste] mask length {keep.shape[0]} is smaller than the boundary {n_sparse}')
    return int(np.count_nonzero(keep[:n_sparse]))


class _Bank(object):
    """Offline instance bank (read-only mmap; forked dataloader workers share the page cache)."""
    FILES = ('raw_xyz', 'raw_sem', 'raw_rem', 'dense_xyz', 'dense_sem')

    def __init__(self, path):
        self.path = path
        meta_p = os.path.join(path, 'meta.json')
        if not os.path.exists(meta_p):
            raise FileNotFoundError(f'[LivBInstanceBankPaste] instance bank not found: {meta_p}')
        with open(meta_p) as f:
            self.meta = json.load(f)
        seqs = [str(s) for s in self.meta.get('sequences', [])]
        bad = [s for s in seqs if s in ('08',)]
        if bad or str(self.meta.get('split')) != 'train':
            raise ValueError(f'[LivBInstanceBankPaste] instance bank contains non-training data (split={self.meta.get("split")}, '
                             f'sequences={seqs}): 08=val must never be used for training.')
        for k in self.FILES:
            p = os.path.join(path, k + '.npy')
            if not os.path.exists(p):
                raise FileNotFoundError(f'[LivBInstanceBankPaste] instance bank is missing a file: {p}')
        self.obs = {}
        for k in self.meta['obs_fields']:
            self.obs[k] = np.load(os.path.join(path, f'obs_{k}.npy'))
        if np.any(self.obs['seq'] == 8):
            raise ValueError('[LivBInstanceBankPaste] instance bank contains observations from seq 08.')
        self.n_obs = int(self.obs['frame'].shape[0])
        self._arrays = None

    @property
    def arrays(self):
        if self._arrays is None:
            self._arrays = {k: np.load(os.path.join(self.path, k + '.npy'), mmap_mode='r')
                            for k in self.FILES}
        return self._arrays

    def raw_sem_ids(self):
        a = self.arrays
        return sorted(set(np.unique(np.asarray(a['raw_sem'])).tolist())
                      | set(np.unique(np.asarray(a['dense_sem'])).tolist()))

    def get(self, i):
        a, o = self.arrays, self.obs
        r0, nr = int(o['raw_off'][i]), int(o['n_raw'][i])
        d0, nd = int(o['dense_off'][i]), int(o['n_dense'][i])
        return dict(
            raw_xyz=np.asarray(a['raw_xyz'][r0:r0 + nr], np.float64),
            raw_sem=np.asarray(a['raw_sem'][r0:r0 + nr], np.int64),
            raw_rem=np.asarray(a['raw_rem'][r0:r0 + nr], np.float64),
            dense_xyz=np.asarray(a['dense_xyz'][d0:d0 + nd], np.float64),
            dense_sem=np.asarray(a['dense_sem'][d0:d0 + nd], np.int64),
            ground_z=float(o['ground_z'][i]), cx=float(o['cx'][i]), cy=float(o['cy'][i]),
            rho=float(o['rho'][i]), phi=float(o['phi'][i]), radius_xy=float(o['radius_xy'][i]),
            seq=int(o['seq'][i]), frame=int(o['frame'][i]), inst_index=int(o['inst_index'][i]))


@PIPELINES.register_module()
class LivBInstanceBankPaste(object):
    def __init__(self,
                 bank_path=None,
                 prob=0.0,                    # per-frame trigger probability; 0 = off (bit-for-bit zero difference)
                 num_instances=2,             # K: number of instances to try to paste per trigger; int or (lo, hi)
                 livb_class=5,                # COLA Living Being
                 ground_classes=(1, 7),       # COLA Driveable / Other Ground
                 sample_by='instance',        # 'instance': draw an instance uniformly, then one of its observations; 'observation': uniform over observations
                 max_tries=10,                # placement attempts per instance
                 yaw_range=180.0,             # range of the rotation about the sensor, ±deg
                 range_jitter=0.0,            # relative amplitude of the radial shift; 0 = pure rotation (density strictly identical)
                 ground_radius_pad=0.3,       # target ground search radius = instance horizontal radius + pad
                 ground_radius_min=0.6,
                 ground_min_points=5,
                 ground_spread_max=0.35,      # upper limit of the target ground p90-p10, m
                 max_ground_shift=0.4,        # upper limit of the ground-snapping shift |dz|, m
                 collision_radius=0.15,       # horizontal collision radius, m
                 collision_clearance=0.15,    # only non-ground points higher than this above the ground count as collisions, m
                 collision_top_margin=0.2,    # points up to this far above the instance top still count as collisions, m
                 ground_bump_max=0.3,         # ground-class points higher than this above the local ground also count as collisions, m
                 collision_tolerance=0,       # number of colliding points allowed
                 occ_az_res=0.2,              # azimuth resolution of the occlusion angular grid, deg (HDL-64E ≈0.17°)
                 occ_el_res=0.4,              # elevation resolution of the occlusion angular grid, deg (HDL-64E average ≈0.42°)
                 occ_margin=0.15,             # a point must be this much farther than the instance to count as occluded, m
                 los_margin=0.3,              # a point must be this much nearer than the instance to count as blocking it, m
                 los_max_frac=0.05,           # upper limit of the fraction of blocked angular cells
                 occlusion_rows='scan',       # 'scan' | 'all' | 'none': which target rows are removed when occluded
                 dense_only_parts='dense',    # which parts are pasted into pure dense frames: 'dense' | 'raw+dense'
                 min_part_points=10,          # minimum number of points of the parts to paste (otherwise another observation is drawn)
                 same_seq_exclude_frames=50,  # do not paste observations within ±N frames of the same sequence (avoids duplicating a person already in the frame)
                 sensor_origin=(0.0, 0.0, 0.0),
                 pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 polar_rho_max=51.2,          # PointsegMapping max_volume_space[0]
                 polar_z_range=(-3.4, 3.0),   # PointsegMapping min/max_volume_space[2]
                 input_kind='auto',           # 'auto' | 'union' | 'dense' | 'scan'
                 seed=None,
                 enable=True,
                 verbose=True):
        self.bank_path = bank_path
        self.prob = float(prob)
        if isinstance(num_instances, (list, tuple)):
            lo, hi = int(num_instances[0]), int(num_instances[1])
            assert 0 <= lo <= hi
            self.num_instances = (lo, hi)
        else:
            self.num_instances = (int(num_instances), int(num_instances))
        self.livb_class = int(livb_class)
        self.ground_classes = tuple(int(c) for c in ground_classes)
        assert sample_by in ('instance', 'observation')
        self.sample_by = sample_by
        self.max_tries = int(max_tries)
        self.yaw_range = math.radians(float(yaw_range))
        self.range_jitter = float(range_jitter)
        self.ground_radius_pad = float(ground_radius_pad)
        self.ground_radius_min = float(ground_radius_min)
        self.ground_min_points = int(ground_min_points)
        self.ground_spread_max = float(ground_spread_max)
        self.max_ground_shift = float(max_ground_shift)
        self.collision_radius = float(collision_radius)
        self.collision_clearance = float(collision_clearance)
        self.collision_top_margin = float(collision_top_margin)
        self.ground_bump_max = float(ground_bump_max)
        self.collision_tolerance = int(collision_tolerance)
        self.occ_az_res = math.radians(float(occ_az_res))
        self.occ_el_res = math.radians(float(occ_el_res))
        self.occ_margin = float(occ_margin)
        self.los_margin = float(los_margin)
        self.los_max_frac = float(los_max_frac)
        assert occlusion_rows in ('scan', 'all', 'none')
        self.occlusion_rows = occlusion_rows
        assert dense_only_parts in ('dense', 'raw+dense')
        self.dense_only_parts = dense_only_parts
        self.min_part_points = int(min_part_points)
        self.same_seq_exclude_frames = int(same_seq_exclude_frames)
        self.sensor_origin = np.asarray(sensor_origin, np.float64).reshape(3)
        self.pc_range = np.asarray(pc_range, np.float64).reshape(6)
        self.polar_rho_max = float(polar_rho_max)
        self.polar_z_range = (float(polar_z_range[0]), float(polar_z_range[1]))
        assert input_kind in ('auto', 'union', 'dense', 'scan')
        self.input_kind = input_kind
        self.seed = seed
        self.enable = bool(enable)
        self.verbose = bool(verbose)

        self._n_az = int(round(2.0 * math.pi / self.occ_az_res))
        self._n_el = int(math.ceil(math.pi / self.occ_el_res)) + 1
        self._n_bucket = 1440            # 0.25° azimuth buckets, used to fetch target rows by angular window
        self._rng = None
        self._rng_pid = None
        self._bank = None
        self._valid = None
        self._mapping_checked = False
        self._warned = set()
        self._logged_ok = False

        active = self.enable and self.prob > 0.0
        if active:
            if not bank_path:
                raise ValueError('[LivBInstanceBankPaste] prob>0 but no bank_path given.')
            self._bank = _Bank(bank_path)   # fail early: the bank is validated before training starts
        if self.verbose:
            if active:
                m = self._bank.meta
                print(f'[LivBInstanceBankPaste] effective config: p={self.prob} K={self.num_instances} '
                      f'sample_by={self.sample_by} yaw=±{math.degrees(self.yaw_range):.0f}° '
                      f'range_jitter={self.range_jitter} occlusion_rows={self.occlusion_rows} '
                      f'dense_only_parts={self.dense_only_parts} max_tries={self.max_tries} '
                      f'| bank={bank_path} obs={self._bank.n_obs} inst={m.get("n_instances")} '
                      f'seqs={m.get("sequences")} mapping_md5={m.get("coarse_mapping_md5")}', flush=True)
            else:
                print(f'[LivBInstanceBankPaste] disabled (enable={self.enable}, p={self.prob}): '
                      f'output is bit-for-bit identical to not having this transform.', flush=True)

    # ------------------------------------------------------------------ utils
    def _warn_once(self, msg):
        if msg in self._warned:
            return
        self._warned.add(msg)
        warnings.warn(f'[LivBInstanceBankPaste] {msg}', RuntimeWarning)

    def _get_rng(self):
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            if self.seed is None:
                s = int.from_bytes(os.urandom(4), 'little') ^ (pid & 0xFFFFFFFF)
            else:
                wid = 0
                try:
                    wi = torch.utils.data.get_worker_info()
                    wid = wi.id if wi is not None else 0
                except Exception:
                    wid = 0
                s = (int(self.seed) + 1000003 * wid) % (2 ** 32)
            self._rng = np.random.RandomState(s)
            self._rng_pid = pid
        return self._rng

    def _check_mapping(self, results):
        if self._mapping_checked:
            return
        seg_map = results.get('seg_label_mapping')
        if not isinstance(seg_map, dict):
            raise RuntimeError('[LivBInstanceBankPaste] results has no seg_label_mapping; cannot validate the instance bank labels.')
        ids = self._bank.raw_sem_ids()
        mapped = {int(k): int(seg_map.get(int(k), 0)) for k in ids}
        bad = {k: v for k, v in mapped.items() if v != self.livb_class}
        if bad:
            raise ValueError(f'[LivBInstanceBankPaste] after the current seg_label_mapping, the raw labels of the instance bank are not all '
                             f'{self.livb_class}: {bad} (mapping differs from the one used to build the bank? bank md5='
                             f'{self._bank.meta.get("coarse_mapping_md5")})')
        self._mapping_checked = True

    def _valid_obs(self, kind):
        if self._valid is None:
            self._valid = {}
        if kind not in self._valid:
            o = self._bank.obs
            if kind == 'dense' and self.dense_only_parts == 'dense':
                ok = o['n_dense'] >= self.min_part_points
            else:
                ok = o['n_raw'] >= self.min_part_points
            idx = np.nonzero(ok)[0]
            inst = o['inst_index'][idx]
            by_inst = {}
            for i, ii in zip(idx.tolist(), inst.tolist()):
                by_inst.setdefault(ii, []).append(i)
            self._valid[kind] = (idx, [np.asarray(v) for _, v in sorted(by_inst.items())])
        return self._valid[kind]

    def _sample_obs(self, rng, kind, tgt_seq, tgt_frame, used):
        idx, groups = self._valid_obs(kind)
        if idx.shape[0] == 0:
            return None
        o = self._bank.obs
        for _ in range(20):
            if self.sample_by == 'instance':
                g = groups[rng.randint(len(groups))]
                i = int(g[rng.randint(g.shape[0])])
            else:
                i = int(idx[rng.randint(idx.shape[0])])
            if int(o['inst_index'][i]) in used:   # never paste the same instance twice into one frame
                continue
            if (tgt_seq is not None and int(o['seq'][i]) == tgt_seq
                    and abs(int(o['frame'][i]) - tgt_frame) <= self.same_seq_exclude_frames):
                continue
            return i
        return None

    # ------------------------------------------------------------------ geometry
    def _angles(self, xyz):
        d = xyz - self.sensor_origin
        rxy = np.hypot(d[:, 0], d[:, 1])
        r3 = np.sqrt(rxy * rxy + d[:, 2] * d[:, 2])
        az = np.arctan2(d[:, 1], d[:, 0])
        el = np.arctan2(d[:, 2], rxy)
        return r3, az, el

    def _bin_keys(self, az, el):
        ia = np.floor((az + math.pi) / self.occ_az_res).astype(np.int64) % self._n_az
        ie = np.clip(np.floor((el + 0.5 * math.pi) / self.occ_el_res).astype(np.int64), 0, self._n_el - 1)
        return ia, ie

    def _shadow(self, P):
        """Angular-cell occupancy of the instance: returns (sorted keys, nearest range per cell). Each azimuth
        column is filled in elevation over [min, max] (closing the gaps between scan lines); filled cells take
        the nearest range of that column (conservative)."""
        r3, az, el = self._angles(P)
        ia, ie = self._bin_keys(az, el)
        order = np.lexsort((ie, ia))
        ia_s, ie_s, r_s = ia[order], ie[order], r3[order]
        cols, start = np.unique(ia_s, return_index=True)
        ie_min = np.minimum.reduceat(ie_s, start)
        ie_max = np.maximum.reduceat(ie_s, start)
        r_col = np.minimum.reduceat(r_s, start)
        cnt = ie_max - ie_min + 1
        col_rep = np.repeat(cols, cnt)
        off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ie_all = np.repeat(ie_min, cnt) + off
        r_all = np.repeat(r_col, cnt)
        keys = col_rep * self._n_el + ie_all
        o2 = np.argsort(keys, kind='stable')
        return keys[o2], r_all[o2], (ia, ie, r3)

    def _gather(self, F, az_c, half):
        """Indices of the target rows within the azimuth window [az_c-half, az_c+half] (0.25° buckets, with wrap-around)."""
        nb = self._n_bucket
        if half >= math.pi:
            return np.arange(F['N'])
        w = 2.0 * math.pi / nb
        b0 = int(math.floor((az_c - half + math.pi) / w))
        b1 = int(math.floor((az_c + half + math.pi) / w))
        bs = np.arange(b0, b1 + 1) % nb
        st, order = F['bucket_start'], F['bucket_order']
        parts = [order[st[b]:st[b + 1]] for b in np.unique(bs)]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def _frame_cache(self, xyz, lab):
        x, y = xyz[:, 0], xyz[:, 1]
        az = np.arctan2(y - self.sensor_origin[1], x - self.sensor_origin[0])
        nb = self._n_bucket
        b = np.clip(np.floor((az + math.pi) / (2.0 * math.pi) * nb).astype(np.int64), 0, nb - 1)
        order = np.argsort(b.astype(np.int16), kind='stable')
        cnt = np.bincount(b, minlength=nb)
        start = np.concatenate([[0], np.cumsum(cnt)])
        return dict(N=xyz.shape[0], xyz=xyz, lab=lab, az=az,
                    is_ground=np.isin(lab, self.ground_classes),
                    bucket_order=order, bucket_start=start)

    def _try_place(self, rng, F, ob, parts, removable, pasted_keys, pasted_xy, stats):
        from scipy.spatial import cKDTree
        Q_raw, Q_dn = ob['raw_xyz'], ob['dense_xyz']
        n_r = Q_raw.shape[0]
        Q = np.concatenate([Q_raw, Q_dn], 0)
        c0 = np.array([ob['cx'], ob['cy']])
        R_inst = ob['radius_xy']
        R_g = max(self.ground_radius_min, R_inst + self.ground_radius_pad)
        xyz, lab, is_ground = F['xyz'], F['lab'], F['is_ground']
        for _t in range(self.max_tries):
            stats['tries'] += 1
            dyaw = rng.uniform(-self.yaw_range, self.yaw_range) if self.yaw_range > 0 else 0.0
            sc = 1.0 + rng.uniform(-self.range_jitter, self.range_jitter) if self.range_jitter > 0 else 1.0
            cs, sn = math.cos(dyaw), math.sin(dyaw)
            P = Q.copy()
            P[:, 0] = cs * Q[:, 0] - sn * Q[:, 1]
            P[:, 1] = sn * Q[:, 0] + cs * Q[:, 1]
            c_new = np.array([cs * c0[0] - sn * c0[1], sn * c0[0] + cs * c0[1]])
            keep_pts = np.ones(P.shape[0], bool)
            if sc != 1.0:
                P[:, :2] += (sc - 1.0) * c_new
                c_new = c_new * sc
                if sc > 1.0:   # moved farther away: thin by (r0/r)^2, approximating the density under the same scan pattern
                    keep_pts = rng.rand(P.shape[0]) < (1.0 / (sc * sc))
            rho_c = float(np.hypot(c_new[0], c_new[1]))
            az_c = math.atan2(c_new[1], c_new[0])
            # ---- target ground
            half = math.asin(min(1.0, (R_inst + R_g + 0.3) / max(rho_c, 1e-3))) + 2.0 * math.pi / self._n_bucket
            W = self._gather(F, az_c, half)
            dxy = xyz[W, :2] - c_new
            d2 = (dxy * dxy).sum(1)
            gm = is_ground[W] & (d2 <= R_g * R_g)
            if int(gm.sum()) < self.ground_min_points:
                stats['rej_ground_support'] += 1; continue
            gz = xyz[W[gm], 2]
            q10, q50, q90 = np.quantile(gz, [0.1, 0.5, 0.9])
            if q90 - q10 > self.ground_spread_max:
                stats['rej_ground_spread'] += 1; continue
            dz = float(q50) - ob['ground_z']
            if abs(dz) > self.max_ground_shift:
                stats['rej_ground_shift'] += 1; continue
            P[:, 2] += dz
            # ---- bounds (RPR crop box + PointsegMapping polar volume, so the instance is not deformed by clipping)
            pr = self.pc_range
            prho = np.hypot(P[:, 0], P[:, 1])
            inb = ((P[:, 0] >= pr[0]) & (P[:, 0] < pr[3]) & (P[:, 1] >= pr[1]) & (P[:, 1] < pr[4])
                   & (P[:, 2] >= max(pr[2], self.polar_z_range[0])) & (P[:, 2] < min(pr[5], self.polar_z_range[1]) - 1e-3)
                   & (prho < self.polar_rho_max - 1e-3) & (prho >= 1.0))
            if not inb.all():
                stats['rej_bounds'] += 1; continue
            # ---- collision
            z_top = float(P[:, 2].max())
            lim = R_inst + self.collision_radius
            box = (np.abs(dxy[:, 0]) <= lim) & (np.abs(dxy[:, 1]) <= lim)
            zW = xyz[W, 2]
            obst = box & (((~is_ground[W]) & (zW >= q50 + self.collision_clearance))
                          | (is_ground[W] & (zW > q50 + self.ground_bump_max)))
            obst &= zW <= z_top + self.collision_top_margin
            n_hit = 0
            tree = None
            if obst.any():
                tree = cKDTree(P[:, :2])
                dd, _ = tree.query(xyz[W[obst], :2], k=1, distance_upper_bound=self.collision_radius)
                n_hit = int(np.isfinite(dd).sum())
            if n_hit > self.collision_tolerance:
                stats['rej_collision'] += 1; continue
            if pasted_xy is not None and pasted_xy.shape[0]:
                if tree is None:
                    tree = cKDTree(P[:, :2])
                dd, _ = tree.query(pasted_xy, k=1, distance_upper_bound=self.collision_radius + 0.2)
                if np.isfinite(dd).any():
                    stats['rej_collision_pasted'] += 1; continue
            # ---- line of sight / occlusion (angular cells)
            # The silhouette is "what the scan sees": when pasting raw, use the raw points (the dense completion
            # extends up to dense_assoc_radius=0.4 m beyond the raw points, which can be several degrees at close
            # range; using it would enlarge the shadow about 2.5x); when pasting dense only (pure dense teacher
            # frames), use the dense points.
            if 'raw' in parts and n_r > 0 and keep_pts[:n_r].any():
                sil = P[:n_r][keep_pts[:n_r]]
            else:
                sil = P[n_r:][keep_pts[n_r:]] if keep_pts[n_r:].any() else P[keep_pts]
            keys, rmin, _ = self._shadow(sil)
            if pasted_keys is not None and pasted_keys.shape[0]:
                if np.isin(keys, pasted_keys).any():
                    stats['rej_overlap_pasted'] += 1; continue
            _, az_p, _ = self._angles(P)
            az_mid = math.atan2(np.sin(az_p).mean(), np.cos(az_p).mean())
            half_a = float(np.abs(_wrap(az_p - az_mid)).max()) + 2.0 * self.occ_az_res
            A = self._gather(F, az_mid, half_a)
            r3_t, az_t, el_t = self._angles(xyz[A])
            ia_t, ie_t = self._bin_keys(az_t, el_t)
            key_t = ia_t * self._n_el + ie_t
            pos = np.searchsorted(keys, key_t)
            pos_c = np.minimum(pos, keys.shape[0] - 1)
            in_occ = keys[pos_c] == key_t
            rmin_t = np.where(in_occ, rmin[pos_c], np.inf)
            blocked = in_occ & (~is_ground[A]) & (r3_t < rmin_t - self.los_margin)
            blocked_keys = np.unique(key_t[blocked])
            if blocked_keys.shape[0] > self.los_max_frac * keys.shape[0]:
                stats['rej_line_of_sight'] += 1; continue
            occluded = in_occ & (r3_t > rmin_t + self.occ_margin)
            rm_rows = A[occluded & removable[A]]
            # ---- accepted: assemble the parts to paste
            raw_keep = keep_pts[:n_r].copy()
            if blocked_keys.shape[0]:
                # in cells blocked by target points in front, the raw instance points cannot have returns -> remove them
                ia_all, ie_all = self._bin_keys(*self._angles(P[:n_r])[1:])
                raw_keep &= ~np.isin(ia_all * self._n_el + ie_all, blocked_keys)
            out = dict(rm_rows=rm_rows, keys=keys, xy=P[:, :2],
                       raw=P[:n_r][raw_keep] if 'raw' in parts else np.zeros((0, 3)),
                       raw_rem=ob['raw_rem'][raw_keep] if 'raw' in parts else np.zeros(0),
                       raw_sem=ob['raw_sem'][raw_keep] if 'raw' in parts else np.zeros(0, np.int64),
                       dense=P[n_r:][keep_pts[n_r:]] if 'dense' in parts else np.zeros((0, 3)),
                       dense_sem=ob['dense_sem'][keep_pts[n_r:]] if 'dense' in parts else np.zeros(0, np.int64),
                       dyaw=dyaw, dz=dz, scale=sc, ground_target=float(q50), n_blocked_bins=int(blocked_keys.shape[0]))
            if out['raw'].shape[0] + out['dense'].shape[0] < self.min_part_points:
                stats['rej_too_few_points'] += 1; continue
            return out
        return None

    # ------------------------------------------------------------------ main
    @staticmethod
    def _unwrap_points(pts):
        container = None
        if hasattr(pts, 'data') and not torch.is_tensor(pts) and not isinstance(pts, np.ndarray) \
                and not hasattr(pts, 'tensor'):
            container = pts
            pts = pts.data
        if hasattr(pts, 'tensor'):
            return pts.tensor, ('basepoints', pts, container)
        if torch.is_tensor(pts):
            return pts, ('tensor', None, container)
        return torch.from_numpy(np.asarray(pts)), ('numpy', None, container)

    @staticmethod
    def _rewrap_points(t, info):
        kind, obj, container = info
        if kind == 'basepoints':
            out = obj.new_point(t)
        elif kind == 'tensor':
            out = t
        else:
            out = t.numpy()
        if container is not None:
            from mmcv.parallel import DataContainer
            out = DataContainer(out, **{k: getattr(container, '_' + k) for k in ('stack', 'padding_value', 'cpu_only')
                                        if hasattr(container, '_' + k)})
        return out

    @staticmethod
    def _select_cat(arr, idx_parts, new_parts):
        """arr: numpy or torch; reassembled as [arr[idx0], new0, arr[idx1], new1, ...], keeping type/dtype/ndim."""
        if torch.is_tensor(arr):
            chunks = []
            for idx, new in zip(idx_parts, new_parts):
                chunks.append(arr[torch.from_numpy(idx)])
                if new is not None and new.shape[0]:
                    chunks.append(torch.from_numpy(np.ascontiguousarray(new)).to(arr.dtype).reshape(
                        (-1,) + tuple(arr.shape[1:])))
            return torch.cat(chunks, 0)
        arr = np.asarray(arr)
        chunks = []
        for idx, new in zip(idx_parts, new_parts):
            chunks.append(arr[idx])
            if new is not None and new.shape[0]:
                chunks.append(np.asarray(new).astype(arr.dtype).reshape((-1,) + arr.shape[1:]))
        return np.concatenate(chunks, 0)

    def __call__(self, results):
        if not self.enable or self.prob <= 0.0:
            return results          # bit-for-bit zero difference: bank not read, no RNG touched
        rng = self._get_rng()
        if rng.rand() >= self.prob:
            return results

        if results.get('train_grid') is not None:
            raise RuntimeError('[LivBInstanceBankPaste] must be placed before PointsegMapping (train_grid detected).')
        if results.get('train_pts_label') is None:
            raise RuntimeError('[LivBInstanceBankPaste] must be placed after VoxelClassMapping (train_pts_label missing, '
                               'labels are not COLA-8 yet).')
        self._check_mapping(results)

        pts_t, pts_info = self._unwrap_points(results['points'])
        N = int(pts_t.shape[0])
        lab_obj = results['pts_semantic_mask']
        lab_np = (lab_obj.detach().cpu().numpy() if torch.is_tensor(lab_obj) else np.asarray(lab_obj))
        lab = lab_np.reshape(-1).astype(np.int64)
        if lab.shape[0] != N:
            self._warn_once(f'number of points {N} and number of labels {lab.shape[0]} differ, skipping paste.')
            return results
        tpl = results['train_pts_label']
        if int(tpl.shape[0]) != N:
            self._warn_once(f'train_pts_label row count {int(tpl.shape[0])} differs from the number of points {N}, skipping paste.')
            return results

        # ---- input kind and row partition
        kind = self.input_kind
        if kind == 'auto':
            if results.get('dense7sparse'):
                kind = 'union'
            else:
                fn = str(results.get('pts_filename', ''))
                kind = 'dense' if ('/refine/' in fn or fn.endswith('.ply') or fn.endswith('.npy')) else 'scan'
        n_sparse = None
        if kind == 'union':
            n_sparse = results.get('num_sparse_points')
            if n_sparse is None or not (0 < int(n_sparse) <= N):
                self._warn_once(f'invalid boundary of the concatenated cloud (num_sparse_points={n_sparse}, N={N}), skipping paste -- '
                                f'rows are not changed while the boundary is unknown, so as not to break the real/dense bookkeeping of MinorityDensePurify.')
                return results
            n_sparse = int(n_sparse)
            parts = ('raw', 'dense')
        elif kind == 'dense':
            parts = ('dense',) if self.dense_only_parts == 'dense' else ('raw', 'dense')
        else:
            parts = ('raw',)
        removable = np.zeros(N, bool)
        if self.occlusion_rows == 'all':
            removable[:] = True
        elif self.occlusion_rows == 'scan':
            if kind == 'union':
                removable[:n_sparse] = True
            elif kind == 'scan':
                removable[:] = True

        m = _SEQ_FRAME_RE.search(str(results.get('pts_filename', '')))
        tgt_seq, tgt_frame = (int(m.group(1)), int(m.group(2))) if m else (None, None)

        xyz = pts_t[:, :3].detach().cpu().numpy().astype(np.float64)
        F = self._frame_cache(xyz, lab)
        K = int(rng.randint(self.num_instances[0], self.num_instances[1] + 1))
        stats = dict(kind=kind, K=K, attempted=0, accepted=0, tries=0, rej_no_obs=0,
                     rej_ground_support=0, rej_ground_spread=0, rej_ground_shift=0, rej_bounds=0,
                     rej_collision=0, rej_collision_pasted=0, rej_overlap_pasted=0,
                     rej_line_of_sight=0, rej_too_few_points=0, removed_rows=0,
                     pasted_raw=0, pasted_dense=0, obs=[], placements=[])
        used = set()
        accepted = []
        pasted_keys = np.zeros(0, np.int64)
        pasted_xy = np.zeros((0, 2))
        removed = np.zeros(N, bool)
        for _k in range(K):
            stats['attempted'] += 1
            i = self._sample_obs(rng, kind, tgt_seq, tgt_frame, used)
            if i is None:
                stats['rej_no_obs'] += 1
                continue
            ob = self._bank.get(i)
            used.add(ob['inst_index'])
            pl = self._try_place(rng, F, ob, parts, removable & ~removed, pasted_keys, pasted_xy, stats)
            if pl is None:
                continue
            removed[pl['rm_rows']] = True
            pasted_keys = np.union1d(pasted_keys, pl['keys'])
            pasted_xy = np.concatenate([pasted_xy, pl['xy']], 0)
            accepted.append(pl)
            stats['obs'].append(i)
            stats['placements'].append(dict(obs=i, dyaw=pl['dyaw'], dz=pl['dz'],
                                            scale=pl['scale'], ground_target=pl['ground_target'],
                                            n_raw=int(pl['raw'].shape[0]), n_dense=int(pl['dense'].shape[0]),
                                            n_removed=int(pl['rm_rows'].shape[0]),
                                            n_blocked_bins=pl['n_blocked_bins']))
        if not accepted:
            results['livb_bank_stats'] = stats
            return results

        # ---- assemble: remove rows first (keeping the row order), then insert
        keep = ~removed
        raw_new = np.concatenate([a['raw'] for a in accepted], 0)
        raw_rem = np.concatenate([a['raw_rem'] for a in accepted], 0)
        dense_new = np.concatenate([a['dense'] for a in accepted], 0)
        C = int(pts_t.shape[1])

        def _rows(xyz_new, rem):
            r = np.zeros((xyz_new.shape[0], C), np.float64)
            r[:, :3] = xyz_new
            if C > 3:
                if rem is not None:
                    r[:, 3] = rem
                self._warn_once(f'point cloud has {C} columns (>3); extra feature columns of the pasted rows: raw rows get the remission in column 4, all other columns are 0.')
            return r
        raw_rows = _rows(raw_new, raw_rem) if raw_new.shape[0] else np.zeros((0, C))
        dense_rows = _rows(dense_new, None) if dense_new.shape[0] else np.zeros((0, C))
        lab_raw = np.full(raw_rows.shape[0], self.livb_class, np.int64)
        lab_dense = np.full(dense_rows.shape[0], self.livb_class, np.int64)

        if kind == 'union':
            idx_sp = np.nonzero(keep[:n_sparse])[0]
            idx_dn = n_sparse + np.nonzero(keep[n_sparse:])[0]
            idx_parts = [idx_sp, idx_dn]
            pt_new = [raw_rows, dense_rows]
            lb_new = [lab_raw, lab_dense]
        else:
            idx_parts = [np.nonzero(keep)[0]]
            pt_new = [np.concatenate([raw_rows, dense_rows], 0)]
            lb_new = [np.concatenate([lab_raw, lab_dense], 0)]

        new_pts = self._select_cat(pts_t, idx_parts, pt_new)
        results['points'] = self._rewrap_points(new_pts, pts_info)
        results['pts_semantic_mask'] = self._select_cat(lab_obj, idx_parts, lb_new)
        results['train_pts_label'] = self._select_cat(tpl, idx_parts, lb_new)
        if kind == 'union':
            ns = shrink_boundary(n_sparse, keep) + int(raw_rows.shape[0])
            assert ns == idx_parts[0].shape[0] + raw_rows.shape[0]
            results['num_sparse_points'] = ns
        stats['accepted'] = len(accepted)
        stats['removed_rows'] = int(removed.sum())
        stats['pasted_raw'] = int(raw_rows.shape[0])
        stats['pasted_dense'] = int(dense_rows.shape[0])
        results['livb_bank_stats'] = stats
        if self.verbose and not self._logged_ok:
            self._logged_ok = True
            print(f'[LivBInstanceBankPaste] active (pid={os.getpid()}): kind={kind} first frame pasted '
                  f'{len(accepted)}/{K} instances raw={stats["pasted_raw"]} dense={stats["pasted_dense"]} '
                  f'removed_by_occlusion={stats["removed_rows"]}', flush=True)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(p={self.prob}, K={self.num_instances}, bank={self.bank_path}, '
                f'yaw=±{math.degrees(self.yaw_range):.0f}, range_jitter={self.range_jitter}, '
                f'occlusion_rows={self.occlusion_rows})')
