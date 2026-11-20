"""Plane-cell semantic categories and per-plane point counts.

Planes: xy [C, 240 (rho), 180 (phi)], yz [C, 180 (phi), 16 (z)], zx [C, 16 (z), 240 (rho)], obtained from the fine grid
(480, 360, 32) by integer division by 2. Distillation-field voxel (a, b, c) in [240, 180, 16] -> xy = (a, b),
yz = (b, c), zx = (c, a).
"""
import numpy as np
import torch

PL = {'xy': (0, 1), 'yz': (1, 2), 'zx': (2, 0)}          # fine-grid axis pair -> plane (row, column)


def counts(gi, shapes):
    """gi: [N, 3] fine-grid indices (480, 360, 32); returns the number of points per plane cell for each plane."""
    c = (torch.floor(gi).long() // 2)
    out = {}
    for k, (r, q) in PL.items():
        H, W = shapes[k]
        rr = c[:, r].clamp(0, H - 1); qq = c[:, q].clamp(0, W - 1)
        m = torch.zeros(H * W, dtype=torch.int32, device=gi.device)
        m.index_add_(0, rr * W + qq, torch.ones_like(rr, dtype=torch.int32))
        out[k] = m.view(H, W).cpu().numpy()
    return out


def neighbor_stack(a, fill):
    """4-neighbour stack [4, H, W] (outside the grid filled with fill)."""
    p = np.pad(a, 1, constant_values=fill)
    return np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])


def cell_categories(cls_plane, ignore_classes=(0,)):
    """Semantic boundary / interior / deep interior (two rings of the same class) / occupancy boundary / empty.
    cls_plane: [H, W], -1 = empty. Classes in ignore_classes carry no semantics: such a neighbour does not form a
    semantic boundary and such a cell takes no part in the semantic statistics."""
    occ = cls_plane >= 0
    nb = neighbor_stack(cls_plane, -1)
    nb_occ = nb >= 0
    sem = occ & ~np.isin(cls_plane, ignore_classes)
    nb_sem = nb_occ & ~np.isin(nb, ignore_classes)
    sem_boundary = sem & (nb_sem & (nb != cls_plane[None])).any(0)
    occ_boundary = occ & (~nb_occ).any(0)
    interior = sem & ~sem_boundary & ~occ_boundary
    # deep interior: the cell and its 4 neighbours are all interior
    inter_nb = neighbor_stack(interior.astype(np.int8), 0).astype(bool)
    deep_interior = interior & inter_nb.all(0)
    empty = ~occ
    n_empty_nb = (~nb_occ).sum(0)
    return dict(occ=occ, sem_boundary=sem_boundary, interior=interior, deep_interior=deep_interior,
                occ_boundary=occ_boundary, empty=empty, n_empty_nb=n_empty_nb)
