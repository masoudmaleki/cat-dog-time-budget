"""DM: distribution matching (multiclass version of HDy). Own implementation (no QuaPy). Task 08 §2.
Notation: P (n, K) posteriors; hist[k, j, :] = normalised 10-bin histogram of P(k|x) over the out-of-fold seconds whose true class is j.
"""
import numpy as np
from scipy.optimize import minimize

NB = 10


def _bins(v):
    """10 equal bins on [0, 1]; 1.0 is included in the last bin."""
    return np.minimum((np.asarray(v) * NB).astype(int), NB - 1)


def hist_norm(v):
    h = np.bincount(_bins(v), minlength=NB).astype(float)
    return h / h.sum() if h.sum() > 0 else h


def class_cond_hists(oof, y, K):
    """Returns (hist (K, K, NB), present (K,) bool). present[j] = whether there are out-of-fold seconds of class j."""
    oof = np.asarray(oof, float); y = np.asarray(y)
    H = np.zeros((K, K, NB)); present = np.zeros(K, bool)
    for j in range(K):
        m = y == j
        if m.any():
            present[j] = True
            for k in range(K): H[k, j] = hist_norm(oof[m, k])
    return H, present


def bag_hists(P):
    """t_k: histogram of P(k|x) over the seconds in the bag. (K, NB)"""
    P = np.asarray(P, float)
    return np.stack([hist_norm(P[:, k]) for k in range(P.shape[1])])


def hd(a, b):
    return np.sqrt(((np.sqrt(a) - np.sqrt(b)) ** 2).sum(-1))


def objective(p, T, H):
    """L(p) = 1/K sum_k HD(t_k, m_k(p)), m_k = sum_j p_j h_{k,j}."""
    M = np.einsum("j,kji->ki", p, H)
    return float(hd(T, M).mean())


def dm(P, H, present, starts):
    """P: (n, K) bag posteriors; H, present: output of class_cond_hists; starts: list of start vectors (K,) ([uniform, training prior, CC]).
    Classes known to be absent (present=False) are fixed at p = 0. Returns (p_hat (K,), at_boundary, L)."""
    K = H.shape[0]; T = bag_hists(P); act = np.where(present)[0]; ka = len(act)
    Ha = H[:, act, :]

    def fun(q): return objective(q, T, Ha)
    best = None
    for s in starts:
        s = np.asarray(s, float)[act]; s = s / s.sum() if s.sum() > 0 else np.full(ka, 1.0 / ka)
        r = minimize(fun, s, method="SLSQP", bounds=[(0.0, 1.0)] * ka, constraints=[{"type": "eq", "fun": lambda q: q.sum() - 1.0}],
                     options={"maxiter": 300, "ftol": 1e-12})
        q = np.clip(r.x, 0.0, None); q = q / q.sum(); L = fun(q)
        if best is None or L < best[1]: best = (q, L)
    q, L = best
    p = np.zeros(K); p[act] = q
    return p, bool((q < 1e-6).any()), L
