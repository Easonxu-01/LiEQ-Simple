"""Held-out decomposition of the nested ridge recovery (private helper of crossterm.py).

ridge_nested_decomp is a line-by-line copy of logits_core.ridge_r2_nested (same lambda selection, same outer two folds
on even / odd phi sectors, same inner leave-one-sector-out search, same reference model) that additionally returns the
decomposition terms on the held-out folds. With Y the model's band signal, Yhat its prediction from the reference band
signal X, P = Yhat - mean, E = Y - Yhat:
    SST = sum (Y - mean)^2 = pred2 + res2 + cross2,   pred2 = sum P^2,   res2 = sum E^2,   cross2 = 2 sum P * E,
so a + b + c = 1 with a = pred2/SST, b = res2/SST, c = cross2/SST, and the forward recovery R^2 = 1 - b = a + c.
"""
import numpy as np
import torch


def ridge_nested_decomp(LC, X, Y, sector, grid=None):
    grid = LC.RIDGE_GRID if grid is None else grid
    X = X.double(); Y = Y.double()
    outer = (sector % 2) == 0
    sse = sst = 0.0
    chosen = []
    acc = dict(pred2=0.0, res2=0.0, cross2=0.0, SST=0.0, Y2=0.0)
    for test in (outer, ~outer):
        tr = ~test
        if int(tr.sum()) < X.shape[1] + 2 or int(test.sum()) < 2:
            return float('nan'), 0.0, chosen, None
        secs = torch.unique(sector[tr]).tolist()
        inner = []
        for s in secs:
            ti = tr & (sector == s); fi = tr & (sector != s)
            if int(ti.sum()) < 2 or int(fi.sum()) < X.shape[1] + 2:
                continue
            inner.append((ti, LC._ridge_stats(X[fi], Y[fi])))
        best, best_lam = float('inf'), grid[-1]
        for lam in grid:
            s_in = 0.0
            for ti, (mx, my, G, B, tr_c) in inner:
                W = torch.linalg.solve(G + lam * tr_c * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), B)
                s_in += float(((Y[ti] - ((X[ti] - mx) @ W + my)) ** 2).sum())
            if s_in < best:
                best, best_lam = s_in, lam
        mx, my, G, B, tr_c = LC._ridge_stats(X[tr], Y[tr])
        W = torch.linalg.solve(G + best_lam * tr_c * torch.eye(X.shape[1], dtype=X.dtype, device=X.device), B)
        pred = (X[test] - mx) @ W + my
        sse += float(((Y[test] - pred) ** 2).sum())
        sst += float(((Y[test] - my) ** 2).sum())
        chosen.append(best_lam)
        # decomposition terms (do not change any quantity above)
        Rres = Y[test] - pred
        Sc = pred - my
        Yc = Y[test] - my
        acc['pred2'] += float((Sc ** 2).sum())
        acc['res2'] += float((Rres ** 2).sum())
        acc['cross2'] += 2.0 * float((Sc * Rres).sum())
        acc['SST'] += float((Yc ** 2).sum())
        acc['Y2'] += float((Y[test] ** 2).sum())
    r2 = (1.0 - sse / sst) if sst > 0 else float('nan')
    return r2, sst, chosen, acc


def decomp_bands(LC, bands, ref, sector, nb):
    """bands / ref: [nb, C, N]. Per band: R^2, chosen lambdas and decomposition terms (Y = model, X = reference, the
    forward direction of the recovery). The copy is checked against logits_core.ridge_r2_nested on every band."""
    out = dict(r2=[], lam=[], pred2=[], res2=[], cross2=[], SST=[], Y2=[])
    for k in range(nb):
        X, Y = ref[k].T, bands[k].T
        r2, sst, lam, acc = ridge_nested_decomp(LC, X, Y, sector)
        r2_orig, sst_orig, lam_orig = LC.ridge_r2_nested(X, Y, sector)
        assert (r2 == r2_orig or (np.isnan(r2) and np.isnan(r2_orig))) and sst == sst_orig and lam == lam_orig, \
            ('decomposition copy differs from logits_core.ridge_r2_nested', k, r2, r2_orig)
        assert abs(acc['SST'] - sst) <= 1e-12 * max(sst, 1.0), ('SST mismatch', k)
        out['r2'].append(r2); out['lam'].append(lam)
        for key in ('pred2', 'res2', 'cross2', 'SST', 'Y2'):
            out[key].append(acc[key])
    return out
