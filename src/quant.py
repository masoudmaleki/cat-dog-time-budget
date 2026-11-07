"""Quantification estimators: CC, ACC, PCC, PACC, EMQ. Own implementation (no QuaPy).
Notation: K classes; y true, ŷ predicted. C[i, j] = P(ŷ=i | y=j) (columns sum to 1).
"""
import numpy as np
from scipy.optimize import nnls


def class_counts_to_prev(counts):
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    return counts / s if s > 0 else np.full(len(counts), 1.0 / len(counts))


def cc(pred, K):
    """Classify & Count: proportions of predicted labels."""
    return np.bincount(np.asarray(pred, dtype=int), minlength=K) / max(len(pred), 1)


def pcc(P):
    """Probabilistic CC: mean of the posterior probabilities."""
    return np.asarray(P).mean(axis=0)


def conf_matrix(y, pred, K):
    """C[i, j] = P(ŷ=i | y=j). For a class j never seen, the column = unit vector (no correction)."""
    C = np.zeros((K, K))
    for j in range(K):
        m = np.asarray(y) == j
        if m.sum() > 0:
            C[:, j] = np.bincount(np.asarray(pred)[m], minlength=K) / m.sum()
        else:
            C[j, j] = 1.0
    return C


def soft_conf_matrix(y, P, K):
    """Class-conditional mean posterior: Cp[i, j] = E[P(ŷ=i | x) | y=j]."""
    P = np.asarray(P); C = np.zeros((K, K))
    for j in range(K):
        m = np.asarray(y) == j
        if m.sum() > 0:
            C[:, j] = P[m].mean(axis=0)
        else:
            C[j, j] = 1.0
    return C


def solve_simplex(C, q, w=1e3):
    """argmin_p ||C p - q||^2, p >= 0, sum p = 1. NNLS (a sum=1 row with weight w is appended) + renormalisation.
    Defined even when C is not invertible."""
    K = C.shape[1]
    A = np.vstack([C, w * np.ones((1, K))])
    b = np.concatenate([q, [w]])
    p, _ = nnls(A, b, maxiter=50 * K)
    s = p.sum()
    return p / s if s > 0 else np.full(K, 1.0 / K)


def acc(q_cc, C):
    return solve_simplex(C, q_cc)


def pacc(q_pcc, Cp):
    return solve_simplex(Cp, q_pcc)


def emq(P, train_prior, max_iter=1000, tol=1e-6):
    """Saerens–Latinne–Decaestecker EM. P: (n, K) posteriors, train_prior: (K,). Returns (prev, n_iter)."""
    P = np.asarray(P, dtype=float); pi = np.asarray(train_prior, dtype=float)
    pi = np.clip(pi, 1e-12, None); pi = pi / pi.sum()
    p = pi.copy()
    for it in range(1, max_iter + 1):
        adj = P * (p / pi)
        s = adj.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        adj = adj / s
        p_new = adj.mean(axis=0)
        if np.abs(p_new - p).max() < tol:
            p = p_new; break
        p = p_new
    return p, it


# ---- metrics
def ae(p_hat, p):
    return float(np.mean(np.abs(np.asarray(p_hat) - np.asarray(p))))


def rae(p_hat, p, n):
    """Smoothed RAE: eps = 1/(2n) is added to p and p̂, then renormalised."""
    eps = 1.0 / (2.0 * n)
    p_s = np.asarray(p) + eps; p_s = p_s / p_s.sum()
    h_s = np.asarray(p_hat) + eps; h_s = h_s / h_s.sum()
    return float(np.mean(np.abs(h_s - p_s) / p_s))
