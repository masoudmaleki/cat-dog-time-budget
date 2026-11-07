"""DR: direct compositional regression (bag descriptor -> ilr(proportions), PLS). Own implementation. Task 08 §3.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold

DECILES = np.arange(1, 10) / 10.0
PLS_TOL = 1e-26   # the sklearn comparison is on the squared norm; 1e-26 -> weight error ~1e-13


# ------------------------------------------------------------------ composition
def zero_replace(p, N):
    """eps = 1/(2N); p_j = 0 -> eps; the others p_j (1 - z eps). Sums to 1."""
    p = np.asarray(p, float); z = int((p == 0).sum()); eps = 1.0 / (2.0 * N)
    out = np.where(p == 0, eps, p * (1.0 - z * eps))
    return out


def pivot_basis(K):
    """Psi (K-1, K): pivot ilr basis; z = Psi @ ln x. Row i: sqrt((K-i)/(K-i+1)) * [0.., 1, -1/(K-i), ..]."""
    Psi = np.zeros((K - 1, K))
    for i in range(1, K):
        c = np.sqrt((K - i) / (K - i + 1.0)); Psi[i - 1, i - 1] = c; Psi[i - 1, i:] = -c / (K - i)
    return Psi


def ilr(x):
    """x: (K,) or (n, K) positive compositions (need not sum to 1). Returns (K-1,) or (n, K-1)."""
    x = np.asarray(x, float); K = x.shape[-1]
    return np.log(x) @ pivot_basis(K).T


def ilr_inv(z, K=None):
    """ilr -> clr -> exp -> closure."""
    z = np.asarray(z, float); K = z.shape[-1] + 1
    clr = z @ pivot_basis(K)
    e = np.exp(clr - clr.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ------------------------------------------------------------------ descriptors
def descr_q(X, rows):
    """The 9 decile cut points of each feature over the seconds in the bag (numpy.quantile, linear). X: (n, F) -> (9F,)"""
    return np.quantile(X[rows], DECILES, axis=0).ravel()


def descr_h(P, rows, nb=10):
    """Histogram of P(k|x) for each class k (nb bins, normalised). P: (n, K) -> (K*nb,)"""
    from src.quant_dm import bag_hists
    return bag_hists(P[rows]).ravel()


# ------------------------------------------------------------------ PLS
def pls_preds_by_a(Xtr, Ztr, Xte, amax):
    """Fit once and predict for a = 1..amax components (NIPALS components are successive; the first a components do not depend on amax).
    X must be standardised; Z is centred inside the PLS. Returns (list: (n_te, K-1) for each a), the really usable amax."""
    m = PLSRegression(n_components=amax, scale=False, tol=PLS_TOL, max_iter=20000).fit(Xtr, Ztr)
    na = m.x_rotations_.shape[1]
    out = []
    xm = np.asarray(Xtr, float).mean(0); ym = np.asarray(Ztr, float).mean(0); Xc = Xte - xm
    for a in range(1, na + 1):
        B = m.x_rotations_[:, :a] @ m.y_loadings_[:, :a].T
        out.append(Xc @ B + ym)
    return out


def _std(Xtr):
    mu = Xtr.mean(0); sd = Xtr.std(0, ddof=1); keep = sd > 0
    return mu, sd, keep


def _ae(Ph, Pt):
    return np.abs(Ph - Pt).mean(1)


class DR:
    """fit(Xb, pb, Nb, groups): Xb (nbag, d) descriptors, pb (nbag, K) proportions, Nb (nbag,) second counts, groups: individual.
    predict(Xnew) -> proportions (n, K) (>0). Selected number of components: self.a."""

    def __init__(self, amax=20, n_splits=4):
        self.amax = amax; self.n_splits = n_splits

    @staticmethod
    def target(pb, Nb):
        return np.stack([ilr(zero_replace(p, n)) for p, n in zip(pb, Nb)])

    def _prep(self, Xtr):
        mu, sd, keep = _std(Xtr)
        return mu, sd, keep

    def fit(self, Xb, pb, Nb, groups):
        Xb = np.asarray(Xb, float); pb = np.asarray(pb, float); Z = self.target(pb, Nb); n, d = Xb.shape
        mu, sd, keep = self._prep(Xb); self.mu, self.sd, self.keep = mu, sd, keep
        Xs = (Xb[:, keep] - mu[keep]) / sd[keep]
        amax = int(min(self.amax, keep.sum(), n - 1)); self.amax_eff = amax
        errs = np.zeros(amax); cnt = np.zeros(amax)
        for tr, va in GroupKFold(self.n_splits).split(Xs, Z, groups):
            mu_, sd_, kp = _std(Xb[tr]); Xt = (Xb[tr][:, kp] - mu_[kp]) / sd_[kp]; Xv = (Xb[va][:, kp] - mu_[kp]) / sd_[kp]
            am = int(min(amax, kp.sum(), len(tr) - 1))
            preds = pls_preds_by_a(Xt, Z[tr], Xv, am)
            for a, zh in enumerate(preds):
                errs[a] += _ae(ilr_inv(zh), pb[va]).sum(); cnt[a] += len(va)
            # too many components requested but not fittable (am < amax): fill with the last prediction
            for a in range(len(preds), amax):
                errs[a] += _ae(ilr_inv(preds[-1]), pb[va]).sum(); cnt[a] += len(va)
        cv = errs / cnt; self.cv_ae = cv
        best = int(np.argmin(cv)); mn = cv[best]
        a = int(np.where(cv <= mn + 1e-4)[0][0]) + 1       # on a tie within 1e-4 take fewer components
        self.a = a
        self.model = PLSRegression(n_components=a, scale=False, tol=PLS_TOL, max_iter=20000).fit(Xs, Z)
        return self

    def predict(self, Xnew):
        Xs = (np.asarray(Xnew, float)[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        zh = self.model.predict(Xs)
        return ilr_inv(zh)
