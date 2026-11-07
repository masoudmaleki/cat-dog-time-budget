"""R07 §4: source-C arms with HGB and LR. Same inner-CV structure as R05 source_fit (GroupKFold(4)); only the classifier differs.
HGB: HistGradientBoostingClassifier defaults, random_state = SEED. LR: StandardScaler (in each training fold) + LogisticRegression(C=1, max_iter=2000)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from config import paths as P
from src import quant as Q, r05lib as R

SEED = P.SEED


def make(kind):
    if kind == "HGB": return HistGradientBoostingClassifier(random_state=SEED)
    if kind == "LR": return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    raise ValueError(kind)


def classes_of(model):
    return model.classes_


def pad(model, X, K):
    p = model.predict_proba(X); out = np.zeros((len(X), K)); out[:, model.classes_.astype(int)] = p; return out


def source_fit(kind, X, y, groups, K):
    """Outer model (all data) + inner-CV out-of-fold (GroupKFold(4)). C[i,j] = P(ŷ=i|y=j) (hard), Cp (soft), prior_oof."""
    model = make(kind).fit(X, y)
    oof = np.zeros((len(X), K))
    for tr, te in GroupKFold(4).split(X, y, groups):
        m = make(kind).fit(X[tr], y[tr]); oof[te] = pad(m, X[te], K)
    return dict(model=model, C=Q.conf_matrix(y, oof.argmax(1), K), Cp=Q.soft_conf_matrix(y, oof, K), prior_oof=oof.mean(0), K=K)


def ece_top(P_, y, nb=10):
    """Top-label ECE (10 equal-width bins)."""
    conf = P_.max(1); ok = (P_.argmax(1) == y).astype(float); e = 0.0
    b = np.minimum((conf * nb).astype(int), nb - 1)
    for k in range(nb):
        s = b == k
        if s.any(): e += s.mean() * abs(ok[s].mean() - conf[s].mean())
    return float(e)


def largest_remainder(p, size):
    x = p * size; n = np.floor(x).astype(int); r = size - n.sum()
    if r > 0: n[np.argsort(-(x - n))[:r]] += 1
    return n


def app_draw(cls_rows, K, nbag, size, rng, maxtry=1000):
    """Same as 21_r05_F.py (same seed -> same bags)."""
    out = []
    for b in range(nbag):
        ok = False
        for _ in range(maxtry):
            p = rng.dirichlet(np.ones(K)); n = largest_remainder(p, size)
            if all(n[k] <= len(cls_rows[k]) for k in range(K)): ok = True; break
        out.append((np.concatenate([rng.choice(cls_rows[k], n[k], replace=False) for k in range(K)]), n / size) if ok else None)
    return out
