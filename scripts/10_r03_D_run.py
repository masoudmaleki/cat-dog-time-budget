r"""10_r03_D_run.py — R03 D: m labelled seconds from the target cat (consecutive 60-s blocks), 20 repeats; predictions of the retrain and target-only arms.
Checkpoint per (placement, cat): <data root>/interim/r03/D_<placement>_<cat>.pkl. Restartable.
Number of trees: 200 in the retrain and target-only arms (allowed by the task text). T-ACC arms are in analysis script 11 (from the R02 posteriors).
"""
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config import paths as P
from src import evalkit as E, smit_data as D

K, SEED = E.K, P.SEED
OUT = P.INTERIM / "r03"; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(P.LOGS / "r03_D.log", "a", encoding="utf-8")
MS = (60, 300, 900, 1800); REPS = 20


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


XC_all, XH_all = E.features()
XS = {"Collar": XC_all[D.FEAT32].values.astype(np.float64), "Harness": XH_all[D.FEAT32].values.astype(np.float64)}
Y, CATROW = E.Y, E.CATROW
SEC = E.LAB.sec.values.astype("datetime64[s]").astype(np.int64)
idx_all = np.where(Y >= 0)[0]


def pick_blocks(te_sorted, m, rng):
    """te_sorted: global indices, sorted by time. m/60 non-overlapping, consecutive 60-s blocks (all with L2 labels). Returns global indices or None."""
    t = SEC[te_sorted]; n = len(t)
    starts = np.where(t[59:] - t[:-59] == 59)[0] if n >= 60 else np.array([], int)   # [i, i+59] uninterrupted
    need = m // 60; chosen = []
    for s in rng.permutation(starts):
        if all(abs(s - q) >= 60 for q in chosen):
            chosen.append(int(s))
            if len(chosen) == need: break
    if len(chosen) < need: return None
    return np.concatenate([te_sorted[s:s + 60] for s in sorted(chosen)])


def rf200(seed):
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)


def pred_labels(model, X):
    return model.classes_[model.predict_proba(X).argmax(1)].astype(np.uint8)


for pos in ("Collar", "Harness"):
    X = XS[pos]
    for ci, c in enumerate(E.CATS):
        f = OUT / f"D_{pos}_{c}.pkl"
        if f.exists():
            log("atla", pos, c); continue
        t0 = time.time()
        te = idx_all[CATROW[idx_all] == c]; tr = idx_all[CATROW[idx_all] != c]
        order = np.argsort(SEC[te], kind="stable"); te_sorted = te[order]
        present = set(np.unique(Y[te]).tolist())
        res = dict(pos=pos, cat=c, te=te.astype(np.int32), runs={})
        Xtr, ytr = X[tr], Y[tr]
        for m in MS:
            for rep in range(REPS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, ci, m, rep]))   # placement-independent: same blocks
                lab = pick_blocks(te_sorted, m, rng)
                if lab is None:
                    res["runs"][(m, rep)] = None; continue
                rem = te[~np.isin(te, lab)]
                r = dict(lab=lab.astype(np.int32), n_rem=len(rem))
                mr = rf200(SEED).fit(np.vstack([Xtr, X[lab]]), np.concatenate([ytr, Y[lab]]))
                r["pred_retrain"] = pred_labels(mr, X[rem])
                miss = sorted(present - set(np.unique(Y[lab]).tolist()))
                r["target_missing_classes"] = miss
                if m == 60 and miss:
                    r["pred_target"] = None            # task rule: skip if a class is missing at m=60
                else:
                    mt = rf200(SEED).fit(X[lab], Y[lab]); r["pred_target"] = pred_labels(mt, X[rem])
                res["runs"][(m, rep)] = r
            log(f"{pos} {c} m={m} tamam ({time.time()-t0:.0f}s)")
        with open(f, "wb") as fh: pickle.dump(res, fh)
        log(f"KAYDEDİLDİ {pos} {c} süre {time.time()-t0:.0f}s")
log("BİTTİ D")
