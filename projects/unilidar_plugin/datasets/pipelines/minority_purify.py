'''
Minority-class supervision purification + real-point re-densification (MinorityDensePurify)

Background / root cause:
    Random-sampling training uses the concatenation of "real sparse scan + LiDiff dense
    completion cloud" (dense7sparse). Measurements on real nuScenes data showed that the
    LiDiff completion has poor fidelity on small dynamic objects such as pedestrians /
    movable objects: dense points labelled as a minority class (COLA 5=Living Being /
    6=Objects) lie at a median distance of ~0.64 m from the nearest real point of the same
    class, and 48% deviate by >0.5 m (a pedestrian itself is only ~0.4 m wide), while ~96%
    of the minority-class candidate points in the concatenated cloud come from the dense
    part. Every step was therefore supervising the minority classes with a "hallucinated
    noise cloud 0.5-2 m around the real object", and the minority x5 weight of CBA-CL put
    5x weight on those hallucinations. Major classes are unaffected (~0.4 m of noise is far
    smaller than their spatial extent); training on the raw point cloud is fine (labels sit
    exactly on real returns). This explains why minority-class IoU stayed pinned at a <0.1
    plateau.

This stage (placed after all label mappings and before Collect, where train_pts_label is
always COLA-8):
    1. Purify: dense points with label in {5,6} whose distance to the nearest "real sparse
       point of the same class" exceeds thresh get label 0 (ignore). The points themselves
       are kept (they still take part in geometry / occlusion / sampling); only their
       minority-class supervision is removed. If a minority class has no real sparse point
       to refer to, all dense points of that class are treated as hallucinations -> ignore.
    2. Re-densify: real sparse minority-class points are replicated xK (with small jitter)
       and injected, restoring candidate density with real geometry that sits entirely on
       the real object; this offsets the point-count drop caused by purification.
    3. Rebuild train_voxel_label (majority vote in COLA space), so that the voxel loss and
       the CBA-CL minority mask automatically act only on real minority-class voxels.

Active only during training (dense7sparse=True and not a reset_random sample); test /
inference skip it naturally.
'''
import warnings

import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from .PointoccMapping import nb_process_label


def _cart2polar_np(xyz):
    rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.stack((rho, phi, xyz[:, 2]), axis=1)


@PIPELINES.register_module()
class MinorityDensePurify(object):
    def __init__(self,
                 minority_classes=(5, 6),   # legacy: default used when purify/densify classes are not given
                 purify_classes=None,       # [per dataset] COLA classes to purify; see the measured table at the config call site
                 densify_classes=None,      # [per dataset] COLA classes to re-densify (decoupled from purification)
                 purify_thresh=0.2,         # m, max allowed distance from a dense minority point to a real point of the same class
                 densify_k=5,               # number of copies of each real sparse minority point (including the original, i.e. k-1 extra)
                 densify_jitter=0.02,       # m, Gaussian xyz jitter sigma applied to the copies
                 max_add_per_class=20000,
                 enable=True):
        base = tuple(int(c) for c in minority_classes)
        self.minority_classes = base
        self.purify_classes = base if purify_classes is None else tuple(int(c) for c in purify_classes)
        self.densify_classes = base if densify_classes is None else tuple(int(c) for c in densify_classes)
        self.purify_thresh = float(purify_thresh)
        self.densify_k = int(densify_k)
        self.densify_jitter = float(densify_jitter)
        self.max_add_per_class = int(max_add_per_class)
        self.enable = bool(enable)
        self._warned = set()
        self._logged_ok = False

    def _warn_once(self, msg):
        """Warn only once per cause (warning on every frame in dataloader workers would flood the log)."""
        if msg in self._warned:
            return
        self._warned.add(msg)
        warnings.warn(f'[MinorityDensePurify] minority-class purification is NOT active! {msg}', RuntimeWarning)

    def _log_ok_once(self, n_sparse, n_total, purified, added):
        """Print one line on the first successful frame to confirm the stage is actually running (guards against silently dead code)."""
        if self._logged_ok:
            return
        self._logged_ok = True
        print(f'[MinorityDensePurify] active: sparse={n_sparse} dense={n_total - n_sparse} '
              f'| first frame purified (set to ignore)={purified} re-densified (real points added)={added} '
              f'| purify={self.purify_classes}@{self.purify_thresh}m '
              f'densify={self.densify_classes}@k={self.densify_k}',
              flush=True)

    def __call__(self, results):
        if not self.enable:
            return results
        # Only applies to the concatenated cloud (sparse+dense); reset_random real scans /
        # pure test paths do not carry this flag. This is a normal skip (non-concatenated
        # samples need no purification), so no warning.
        if not results.get('dense7sparse'):
            return results

        if not self.purify_classes and not self.densify_classes:
            return results   # this dataset needs no purification/re-densification (e.g. Waymo, where measured completion fidelity is already good)

        # --- Any skip below means "should have purified but did not": warn loudly ---
        # sparse/dense boundary: prefer num_sparse_points (never popped downstream),
        # fall back to RPR_indices_sparse (it survives in the nuScenes pipeline).
        n_sparse = results.get('num_sparse_points', None)
        if n_sparse is None:
            rpr_sparse = results.get('RPR_indices_sparse', None)
            if rpr_sparse is not None:
                n_sparse = int(len(rpr_sparse))
        if n_sparse is None:
            self._warn_once(
                "cannot find the sparse/dense boundary (both num_sparse_points and RPR_indices_sparse are missing). "
                "Minority-class purification is skipped -- check that LoadPointsFromFile_RPR writes num_sparse_points "
                "and that no upstream stage pops it.")
            return results
        n_sparse = int(n_sparse)

        missing = [k for k in ('train_grid', 'train_pts_label', 'grid_ind', 'grid_ind_vox')
                   if results.get(k) is None]
        if missing:
            self._warn_once(
                f"missing required keys {missing} (should be produced by PointsegMapping). Minority-class purification is skipped.")
            return results
        tg = results['train_grid']
        tg = tg.data if hasattr(tg, 'data') else tg
        tg = tg.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(tg) else np.asarray(tg, np.float32)
        lab = results['train_pts_label']
        lab = lab.data if hasattr(lab, 'data') else lab
        lab = lab.detach().cpu().numpy() if torch.is_tensor(lab) else np.asarray(lab)
        lab = lab.reshape(-1).astype(np.int64)
        gi = results['grid_ind']
        gi = gi.detach().cpu().numpy() if torch.is_tensor(gi) else np.asarray(gi, np.float32)
        giv = results['grid_ind_vox']
        giv = giv.detach().cpu().numpy() if torch.is_tensor(giv) else np.asarray(giv, np.float32)

        N = tg.shape[0]
        if N != lab.shape[0] or n_sparse <= 0 or n_sparse >= N:
            self._warn_once(
                f"invalid sparse/dense boundary (n_sparse={n_sparse}, N_points={N}, "
                f"N_labels={lab.shape[0]}). Minority-class purification is skipped.")
            return results

        # train_grid column layout: [0:3]=rel_polar, 3=rho,4=phi,5=z, 6=x,7=y, 8:=feat
        xyz = np.stack([tg[:, 6], tg[:, 7], tg[:, 5]], axis=1)   # [N,3] x,y,z
        is_sparse = np.zeros(N, dtype=bool)
        is_sparse[:n_sparse] = True

        try:
            from scipy.spatial import cKDTree
        except Exception:
            cKDTree = None

        # ---------------- 1. Purify hallucinated dense minority points ----------------
        purified = 0
        changed_idx = []          # points whose label changed (for the incremental voxel-label rebuild)
        for cls in self.purify_classes:
            dense_cls = (~is_sparse) & (lab == cls)
            n_dense_cls = int(dense_cls.sum())
            if n_dense_cls == 0:
                continue
            sparse_cls_xyz = xyz[is_sparse & (lab == cls)]
            if sparse_cls_xyz.shape[0] == 0:
                # the real scan has no point of this class at all -> every dense point of it is a hallucination
                lab[dense_cls] = 0
                changed_idx.append(np.where(dense_cls)[0])
                purified += n_dense_cls
                continue
            q = xyz[dense_cls]
            if cKDTree is not None:
                dist, _ = cKDTree(sparse_cls_xyz).query(q, k=1)
            else:
                # fallback: chunked brute-force nearest neighbour, avoiding one huge matrix
                dist = np.empty(q.shape[0], np.float32)
                for s in range(0, q.shape[0], 4096):
                    e = min(s + 4096, q.shape[0])
                    d2 = ((q[s:e, None, :] - sparse_cls_xyz[None, :, :]) ** 2).sum(-1)
                    dist[s:e] = np.sqrt(d2.min(1))
            far = dist > self.purify_thresh
            idx_dense = np.where(dense_cls)[0]
            lab[idx_dense[far]] = 0
            changed_idx.append(idx_dense[far])
            purified += int(far.sum())

        # ---------------- 2. Re-densify with real sparse minority points ----------------
        add_tg, add_lab, add_gi, add_giv = [], [], [], []
        _gs_raw = results.get('_pointseg_grid_size')
        _gsv_raw = results.get('_pointseg_grid_size_vox')
        _maxb_raw = results.get('_pointseg_max_volume_space')
        _minb_raw = results.get('_pointseg_min_volume_space')
        if self.densify_k > 1 and all(x is not None for x in (_gs_raw, _gsv_raw, _maxb_raw, _minb_raw)):
            gs = np.asarray(_gs_raw, np.float64)
            gsv = np.asarray(_gsv_raw, np.float64)
            maxb = np.asarray(_maxb_raw, np.float64)
            minb = np.asarray(_minb_raw, np.float64)
            if True:
                itv = (maxb - minb) / gs
                itv_v = (maxb - minb) / gsv
                rng = np.random.RandomState()  # independent per frame is enough
                for cls in self.densify_classes:
                    src = np.where(is_sparse & (lab == cls))[0]
                    if src.shape[0] == 0:
                        continue
                    budget = self.max_add_per_class
                    for _k in range(self.densify_k - 1):
                        if budget <= 0:
                            break
                        take = src if src.shape[0] <= budget else src[rng.choice(src.shape[0], budget, replace=False)]
                        budget -= take.shape[0]
                        rows = tg[take].copy()
                        jit = rng.normal(0.0, self.densify_jitter, size=(take.shape[0], 3)).astype(np.float32)
                        nx = rows[:, 6] + jit[:, 0]
                        ny = rows[:, 7] + jit[:, 1]
                        nz = rows[:, 5] + jit[:, 2]
                        pol = _cart2polar_np(np.stack([nx, ny, nz], 1))
                        pol_g = np.clip(pol, minb, maxb - 1e-3)
                        g_ind = np.floor((pol_g - minb) / itv).astype(np.float32)
                        g_ind_v = ((pol_g - minb) / itv_v).astype(np.float32)
                        ctr = (g_ind.astype(np.float64) + 0.5) * itv + minb
                        rel = (pol - ctr).astype(np.float32)
                        rows[:, 0:3] = rel
                        rows[:, 3] = pol[:, 0]; rows[:, 4] = pol[:, 1]; rows[:, 5] = pol[:, 2]
                        rows[:, 6] = nx; rows[:, 7] = ny
                        add_tg.append(rows)
                        add_lab.append(np.full(take.shape[0], cls, np.int64))
                        add_gi.append(g_ind)
                        add_giv.append(g_ind_v)

        n_added_pts = int(sum(a.shape[0] for a in add_lab)) if add_lab else 0
        if add_tg:
            tg = np.concatenate([tg] + add_tg, axis=0)
            lab = np.concatenate([lab] + add_lab, axis=0)
            gi = np.concatenate([gi] + add_gi, axis=0)
            giv = np.concatenate([giv] + add_giv, axis=0)

        # ---------------- 3. Incrementally update train_voxel_label ----------------
        # [performance-critical] Purification / re-densification only touch very few voxels
        # (those holding minority classes), but a full-grid rebuild (lexsort + numba over
        # ~1.2M points) costs 364 ms/frame on Waymo -- a repeat of the one already done in
        # PointsegMapping, which makes the dataloader the bottleneck. Instead only the
        # "affected voxels" are recomputed: take the voxels containing purified points and
        # added points, redo the majority vote only for points falling in those voxels, and
        # write the result back into the label grid already produced by PointsegMapping.
        # All other voxels stay bit-identical.
        gsv = results.get('_pointseg_grid_size_vox')
        base_vl = results.get('train_voxel_label')
        if gsv is None or base_vl is None:
            self._warn_once(
                "_pointseg_grid_size_vox or train_voxel_label is missing, voxel labels were not updated -- "
                "voxel supervision and the CBA-CL minority mask will still act on unpurified hallucinated minority voxels.")
        else:
            gsv_i = np.asarray(gsv, np.int32)
            gv_int = np.clip(np.floor(giv).astype(np.int32), 0, gsv_i - 1)

            # affected points: purified + added (added points are all at the end of the array)
            touched = []
            if changed_idx:
                touched.append(np.concatenate(changed_idx))
            if n_added_pts > 0:
                touched.append(np.arange(N, N + n_added_pts))
            if touched:
                touched = np.concatenate(touched)
                # linear voxel key -> boolean lookup table (O(N) direct indexing, much faster than np.isin)
                key = ((gv_int[:, 0].astype(np.int64) * gsv_i[1] + gv_int[:, 1])
                       * gsv_i[2] + gv_int[:, 2])
                n_vox = int(gsv_i[0]) * int(gsv_i[1]) * int(gsv_i[2])
                vox_mask = np.zeros(n_vox, dtype=bool)
                vox_mask[key[touched]] = True
                sel = vox_mask[key]                      # ALL points falling in affected voxels
                gv_sel = gv_int[sel]
                lab_sel = lab[sel].reshape(-1, 1).astype(np.int32)
                if gv_sel.shape[0] > 0:
                    pair = np.concatenate([gv_sel, lab_sel], axis=1)
                    pair = pair[np.lexsort((gv_sel[:, 0], gv_sel[:, 1], gv_sel[:, 2])), :]
                    pl = base_vl.numpy() if torch.is_tensor(base_vl) else np.asarray(base_vl)
                    pl = np.ascontiguousarray(pl.astype(np.uint8))
                    # nb_process_label only writes voxels present in pair; the rest keep their values
                    pl = nb_process_label(pl, pair)
                    results['train_voxel_label'] = torch.from_numpy(pl).type(torch.LongTensor)

        # ---------------- Write back ----------------
        results['train_grid'] = torch.from_numpy(tg).to(torch.float32)
        results['train_pts_label'] = torch.from_numpy(lab)
        results['grid_ind'] = torch.from_numpy(gi).to(torch.float32)
        results['grid_ind_vox'] = torch.from_numpy(giv).to(torch.float32)
        results['minority_purify_stats'] = (int(purified), n_added_pts)
        self._log_ok_once(n_sparse, N, int(purified), n_added_pts)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(purify={self.purify_classes}@{self.purify_thresh}m, '
                f'densify={self.densify_classes}@k={self.densify_k})')
