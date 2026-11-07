r"""R05 shared: data loaders (L1, D17), source model + inner CV, quantification estimators, bags.
Reads: <data root>/interim/features/*.parquet, smit_* (D.load). Writes nothing.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from config import paths as P
from src import smit_data as D, quant as Q, features_smit as F

SEED = P.SEED
L1 = D.L1; K3 = 3
M_ALL = (60, 300, 900, 1800)


def load_smit():
    """lab (Cat_id, sec, usable, y2), y1 (L1 index, -1 = excluded), XC32, XH32 (DataFrame), D17 collar, D17 harness"""
    lab, XC, XH = D.load()
    y1 = np.where(lab.y2.values >= 0, D.L1_IDX[np.clip(lab.y2.values, 0, None)], -1)
    return lab, y1, XC[D.FEAT32].astype(np.float64), XH[D.FEAT32].astype(np.float64), F.d17(XC).astype(np.float64), F.d17(XH).astype(np.float64)


def load_dunford(drop_edge=False, v2=False):
    df = pd.read_parquet(P.DUNFORD_FEAT_V2 if v2 else P.DUNFORD_FEAT)
    if drop_edge: df = df[~df.edge]
    df = df[df.y1 >= 0].reset_index(drop=True)
    return df, F.d17(df).astype(np.float64)


def load_dog(only_mapped=True):
    df = pd.read_parquet(P.DOG_FEAT)
    if only_mapped: df = df[df.y1 >= 0].reset_index(drop=True)
    return df, F.d17(df).astype(np.float64)


def rf(n, seed):
    return RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=seed)


def pad_proba(model, X, K):
    """Pad a model with fewer than K classes (model.classes_) to K columns."""
    p = model.predict_proba(X); out = np.zeros((len(X), K)); out[:, model.classes_.astype(int)] = p
    return out


def source_fit(X, y, groups, K, seed, n_outer=500, n_inner=300, strat=False):
    """On the source: outer model (all data, n_outer trees) + inner-CV out-of-fold predictions (GroupKFold(4) or StratifiedKFold(4)).
    Returns dict(model, C, Cp, prior_oof, oof). C[i,j] = P(ŷ=i | y=j) from the inner-CV out-of-fold hard predictions."""
    model = rf(n_outer, seed).fit(X, y)
    oof = np.zeros((len(X), K))
    if strat:
        sp = StratifiedKFold(4, shuffle=True, random_state=P.SEED).split(X, y)
    else:
        sp = GroupKFold(4).split(X, y, groups)
    for tr, te in sp:
        m = rf(n_inner, seed + 1).fit(X[tr], y[tr]); oof[te] = pad_proba(m, X[te], K)
    return dict(model=model, C=Q.conf_matrix(y, oof.argmax(1), K), Cp=Q.soft_conf_matrix(y, oof, K), prior_oof=oof.mean(0), oof=oof, K=K)


def estimates(P_, src, K):
    """CC, PCC, ACC, PACC, EMQ (source C/Cp/prior)."""
    cc = Q.cc(P_.argmax(1), K); pc = Q.pcc(P_)
    return {"CC": cc, "PCC": pc, "ACC": Q.acc(cc, src["C"]), "PACC": Q.pacc(pc, src["Cp"]), "EMQ": Q.emq(P_, src["prior_oof"])[0]}


# ------------------------------------------------------------------ records, bootstrap, bags
class Rec:
    """Bag-level result records. wide(): scenario, variant, level, arm, m, repeat, target_individual, bag, N, p0..p2, h0..h2, AE, RAE."""
    def __init__(self): self.rows = []

    def add(self, scen, variant, level, arm, m, rep, ind, bag, n, p, ph, extra=None):
        K = len(p); pp = np.full(3, np.nan); hh = np.full(3, np.nan); pp[:K] = p; hh[:K] = ph[:K]
        self.rows.append((scen, variant, level, arm, int(m), int(rep), str(ind), str(bag), int(n), K, *pp, *hh, Q.ae(ph[:K], p), Q.rae(ph[:K], p, n)))

    def wide(self):
        cols = ["scen", "variant", "level", "arm", "m", "rep", "ind", "bag", "N", "K", "p0", "p1", "p2", "h0", "h1", "h2", "AE", "RAE"]
        return pd.DataFrame(self.rows, columns=cols)


def boot_mean(vals, ids, nboot=1000):
    """Individual-level bootstrap (same method as R02/R03): mean, 2.5%, 97.5%. ids: individual ids (one per bag)."""
    vals = np.asarray(vals, float); u, inv = np.unique(np.asarray(ids).astype(str), return_inverse=True)
    S = np.bincount(inv, weights=vals, minlength=len(u)); N = np.bincount(inv, minlength=len(u)).astype(float)
    B = np.random.default_rng(SEED).integers(0, len(u), size=(nboot, len(u)))
    n = N[B].sum(1); ok = n > 0; b = S[B].sum(1)[ok] / n[ok]
    return float(S.sum() / N.sum()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def smit_bags(lab, y1, minutes=30, min_cov=0.5, mask=None):
    """Natural Smit bags (R02 window rule): [(cat, no, row indices)]; rows have y1>=0 (and mask)."""
    win, sel = D.make_windows(lab, minutes, min_cov); cat = lab.Cat_id.values; ok = (y1 >= 0) if mask is None else ((y1 >= 0) & mask)
    return [(c, int(w), np.where((cat == c) & (win == w) & ok)[0]) for (c, w) in sel if ((cat == c) & (win == w) & ok).any()]


def pad3(ph, K=3):
    o = np.zeros(K); o[:len(ph)] = ph; return o


def truth(y, K):
    return np.bincount(y, minlength=K) / len(y)


def target_C(yl, Pl, Cs, soft):
    """Target C: yl is the true class of the labelled target seconds, Pl the posteriors (soft) or the hard prediction (hard, 1-d). A missing class column is taken from the source Cs.
    Returns (C, present_class_mask)."""
    K = Cs.shape[1]; C = Cs.copy(); present = np.zeros(K, bool)
    for j in range(K):
        mj = yl == j
        if mj.any():
            present[j] = True
            C[:, j] = Pl[mj].mean(0) if soft else np.bincount(Pl[mj], minlength=K) / mj.sum()
    return C, present


def shrink(Ct, Cs, present, lam):
    C = Cs.copy(); C[:, present] = lam * Ct[:, present] + (1 - lam) * Cs[:, present]; return C
