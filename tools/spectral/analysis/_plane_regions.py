"""Plane class maps and class-interior regions (private copy of the two functions the gate test needs; same code as the
summary stage, so the regions are the ones the published statistics were computed on)."""
import numpy as np


def plane_majority_classes(grid_ind, labels, H, W, which, ncls=8):
    """Point labels (0..ncls-1, not coarsened) -> majority class per plane cell; cells without points -> -1."""
    g = np.floor(grid_ind).astype(np.int64) // 2
    i, j = {'xy': (g[:, 0], g[:, 1]), 'yz': (g[:, 1], g[:, 2]), 'zx': (g[:, 2], g[:, 0])}[which]
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W) & (labels >= 0) & (labels < ncls)
    lin = i[ok] * W + j[ok]
    tab = np.zeros((H * W, ncls), np.int64)
    np.add.at(tab, (lin, labels[ok].astype(np.int64)), 1)
    cls = tab.argmax(1)
    cls[tab.sum(1) == 0] = -1
    return cls.reshape(H, W)


def interior_radius(cls, r=2):
    """Cells with a class that have no class boundary within Chebyshev radius r (any class change counts as a boundary,
    class 0 included; conservative)."""
    from scipy.ndimage import binary_dilation
    occ = cls >= 0
    p = np.pad(cls, 1, constant_values=-1)
    bd = np.zeros_like(occ)
    for di, dj in ((0, 1), (2, 1), (1, 0), (1, 2)):
        nb = p[di:di + cls.shape[0], dj:dj + cls.shape[1]]
        bd |= occ & (nb >= 0) & (nb != cls)
    near = binary_dilation(bd, structure=np.ones((2 * r + 1, 2 * r + 1), bool))
    return occ & ~near
