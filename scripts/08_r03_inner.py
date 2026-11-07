r"""08_r03_inner.py — for R03 C and E: inner-CV out-of-fold posteriors (stored) and matched/invariant feature-set models. One checkpoint per fold.
Sets: raw (same as R02; only the inner-CV OOF posteriors are stored), map (harness->collar axis mapping), inv (14 orientation-insensitive features).
Reads: smit_* parquet. Writes: <data root>/interim/r03/inner_<cat>.pkl. Restartable.
"""
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from config import paths as P
from src import evalkit as E, smit_data as D, axismap as AM

K, SEED = E.K, P.SEED
OUT = P.INTERIM / "r03"; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(P.LOGS / "r03_inner.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


XC_all, XH_all = E.features()
XC = XC_all[D.FEAT32].values.astype(np.float64)
XH = XH_all[D.FEAT32].values.astype(np.float64)
XHm = AM.harness_to_collar(XH_all[D.FEAT32]).values.astype(np.float64)
inv_idx = [D.FEAT32.index(c) for c in AM.INV_COLS]
XCi, XHmi = XC[:, inv_idx], XHm[:, inv_idx]
Y, CATROW = E.Y, E.CATROW
idx_all = np.where(Y >= 0)[0]


def rf(n, off=0):
    return RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=SEED + off)


def full(model, X):
    p = model.predict_proba(X); out = np.zeros((len(X), K)); out[:, model.classes_] = p; return out


def inner(tr, Xs, Xother=None):
    """Same as inner_cv in R02: GroupKFold(4, group = cat), 300 trees, seed+1. Returns the OOF posteriors."""
    y = Y[tr]; g = CATROW[tr]; oof = np.zeros((len(tr), K), np.float32); oofx = np.zeros((len(tr), K), np.float32) if Xother is not None else None
    for a, b in GroupKFold(4).split(tr, y, g):
        m = rf(300, 1).fit(Xs[tr[a]], y[a])
        oof[b] = full(m, Xs[tr[b]])
        if Xother is not None: oofx[b] = full(m, Xother[tr[b]])
    return oof, oofx


for n_fold, c in enumerate(E.CATS):
    f = OUT / f"inner_{c}.pkl"
    if f.exists():
        log("atla", c); continue
    t0 = time.time()
    te = idx_all[CATROW[idx_all] == c]; tr = idx_all[CATROW[idx_all] != c]
    R = dict(cat=c, tr=tr.astype(np.int32), te=te.astype(np.int32))
    # raw (same seed/order as R02): only the OOF posteriors
    R["raw_collar_oof"], _ = inner(tr, XC)
    R["raw_harness_oof"], R["raw_harness_oofx"] = inner(tr, XH, XC)
    # map: harness->collar matched; the model is trained on the matched harness and applied to the collar test rows in S2
    m = rf(500).fit(XHm[tr], Y[tr])
    R["map_P_hh"] = full(m, XHm[te]); R["map_P_hc"] = full(m, XC[te])
    R["map_oof"], R["map_oofx"] = inner(tr, XHm, XC)
    # inv: 14 orientation-insensitive features
    mc = rf(500).fit(XCi[tr], Y[tr]); R["inv_P_cc"] = full(mc, XCi[te])
    mh = rf(500).fit(XHmi[tr], Y[tr]); R["inv_P_hh"] = full(mh, XHmi[te]); R["inv_P_hc"] = full(mh, XCi[te])
    R["inv_collar_oof"], _ = inner(tr, XCi)
    R["inv_harness_oof"], R["inv_harness_oofx"] = inner(tr, XHmi, XCi)
    with open(f, "wb") as fh: pickle.dump(R, fh)
    log(f"{n_fold+1}/12 {c} süre {time.time()-t0:.0f}s acc_map_hc={(R['map_P_hc'].argmax(1)==Y[te]).mean():.3f} acc_inv_cc={(R['inv_P_cc'].argmax(1)==Y[te]).mean():.3f} acc_inv_hc={(R['inv_P_hc'].argmax(1)==Y[te]).mean():.3f}")
log("BİTTİ inner")
