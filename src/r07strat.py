"""R07 §6: stratified target labelling. The m seconds are split equally over the classes (m / K); the share of a class that falls short is distributed over the remaining classes;
from each class, consecutive 10-s slices are chosen at random from the class's uninterrupted chunks (non-overlapping slices with a random shift per chunk)."""
import numpy as np

SL = 10


def make_runs(rows, t, y, K):
    """rows: row indices in time order; t: second (consecutive = difference 1); y: class. Returns {k: [row arrays]} (uninterrupted, single class)."""
    rows = np.asarray(rows); out = {k: [] for k in range(K)}
    if len(rows) == 0: return out
    brk = np.r_[0, np.where((np.diff(t) != 1) | (np.diff(y) != 0))[0] + 1, len(rows)]
    for a, b in zip(brk[:-1], brk[1:]):
        k = int(y[a])
        if 0 <= k < K and b - a >= SL: out[k].append(rows[a:b])
    return out


def merge_runs(lst, K):
    out = {k: [] for k in range(K)}
    for r in lst:
        for k in range(K): out[k] += r[k]
    return out


def draw_strat(runs, K, m, rng):
    """Returns (row array, number of slices per usable class)."""
    slices = {}
    for k in range(K):
        sl = []
        for r in runs[k]:
            L = len(r); off = int(rng.integers(0, L % SL + 1))
            for s in range(off, L - SL + 1, SL): sl.append(r[s:s + SL])
        slices[k] = sl
    need = np.full(K, (m // K) // SL); cap = np.array([len(slices[k]) for k in range(K)])
    take = np.minimum(need, cap); deficit = int(need.sum() - take.sum())
    while deficit > 0:
        room = np.where(cap - take > 0)[0]
        if len(room) == 0: break
        for k in rng.permutation(room):
            if deficit == 0: break
            take[k] += 1; deficit -= 1
    sel = []
    for k in range(K):
        if take[k] > 0:
            idx = rng.permutation(len(slices[k]))[:take[k]]; sel += [slices[k][i] for i in idx]
    return (np.concatenate(sel) if sel else np.array([], int)), cap
