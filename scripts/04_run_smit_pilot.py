r"""04_run_smit_pilot.py — R02 training/prediction stage (posteriors + inner-CV estimates for S0, S1-collar, S1-harness, S2).
Usage: python -I scripts/04_run_smit_pilot.py --variant a|b --feat 32|36 [--only-loco]
Reads: interim/smit_*.parquet. Writes: <data root>/interim/r02/raw_<tag>.pkl, logs/r02_<tag>.log
Output structure: S0 single fold (collar, random 70%/30% of seconds); LOCO for each cat: collar-collar, harness-harness, harness->collar posteriors;
inner GroupKFold(4, group = cat) estimates: C, Cp, prior; for the harness model additionally cross inner-CV on the collar features (T-ACC-oracle).
"""
import sys, time, pickle, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from config import paths as P
from src import smit_data as D, quant as Q

ap = argparse.ArgumentParser()
ap.add_argument("--variant", choices=["a", "b"], default="a")
ap.add_argument("--feat", choices=["32", "36"], default="32")
ap.add_argument("--only-loco", action="store_true")
ap.add_argument("--max-folds", type=int, default=0, help="zamanlama testi için")
args = ap.parse_args()
TAG = f"v{args.variant}_f{args.feat}"
K = len(D.L2)
SEED = P.SEED
OUT = P.INTERIM / "r02"; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(P.LOGS / f"r02_{TAG}.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


lab, XC_all, XH_all = D.load()
FEATS = D.FEAT32 if args.feat == "32" else D.FEAT36
mask = (lab.y2 >= 0).values
idx_all = np.where(mask)[0]
cats_row = lab.Cat_id.values
y_all = lab.y2.values
XC = XC_all[FEATS].values; XH = XH_all[FEATS].values
cats = sorted(lab.Cat_id.unique())
log("TAG", TAG, "L2 satır", len(idx_all), "kediler", len(cats))


def rf(n, seed_off=0):
    return RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=SEED + seed_off)


def cap(idx, rng):
    """Training variant (b): at most 7000 seconds per class (random)."""
    if args.variant == "a":
        return idx
    keep = []
    for k in range(K):
        ik = idx[y_all[idx] == k]
        keep.append(rng.choice(ik, 7000, replace=False) if len(ik) > 7000 else ik)
    return np.sort(np.concatenate(keep))


def full_proba(model, X):
    """A class may be missing -> widen to K columns."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), K)); out[:, model.classes_] = p
    return out


def inner_cv(tr, groups_src, primary, other=None):
    """tr: training row indices. primary: feature matrix (the one the model is trained on). other: cross placement (optional).
    Returns C, Cp, prior_oof, prior_train (+ cross C, Cp)."""
    y = y_all[tr]; g = groups_src[tr]
    oofP = np.zeros((len(tr), K)); oofX = np.zeros((len(tr), K)) if other is not None else None
    for a, b in GroupKFold(4).split(tr, y, g):
        m = rf(300, 1).fit(primary[tr[a]], y[a])
        oofP[b] = full_proba(m, primary[tr[b]])
        if other is not None:
            oofX[b] = full_proba(m, other[tr[b]])
    res = dict(C=Q.conf_matrix(y, oofP.argmax(1), K), Cp=Q.soft_conf_matrix(y, oofP, K),
               prior_oof=oofP.mean(0), prior_train=np.bincount(y, minlength=K) / len(y), oof_acc=float((oofP.argmax(1) == y).mean()))
    if other is not None:
        res.update(Cx=Q.conf_matrix(y, oofX.argmax(1), K), Cpx=Q.soft_conf_matrix(y, oofX, K), prior_oof_x=oofX.mean(0))
    return res


results = dict(tag=TAG, variant=args.variant, feat=args.feat, L2=D.L2, loco=[], s0=None, times={})
rng = np.random.default_rng(SEED)

# ---------------- LOCO (S1 and S2)
t_all = time.time()
for n_fold, c in enumerate(cats):
    if args.max_folds and n_fold >= args.max_folds:
        break
    t0 = time.time()
    te = idx_all[cats_row[idx_all] == c]
    tr = cap(idx_all[cats_row[idx_all] != c], np.random.default_rng(SEED + n_fold))
    fold = dict(cat=c, test_idx=te, n_train=len(tr))
    # tasma
    m_c = rf(500).fit(XC[tr], y_all[tr])
    fold["P_cc"] = full_proba(m_c, XC[te])
    fold["inner_collar"] = inner_cv(tr, cats_row, XC)
    # harness (+ S2: harness model tested on collar; collar features for the cross inner-CV oracle)
    m_h = rf(500).fit(XH[tr], y_all[tr])
    fold["P_hh"] = full_proba(m_h, XH[te])
    fold["P_hc"] = full_proba(m_h, XC[te])
    fold["inner_harness"] = inner_cv(tr, cats_row, XH, other=XC)
    results["loco"].append(fold)
    dt = time.time() - t0
    log(f"LOCO {n_fold+1}/{len(cats)} {c} n_tr={len(tr)} n_te={len(te)} acc_cc={(fold['P_cc'].argmax(1)==y_all[te]).mean():.3f} "
        f"acc_hh={(fold['P_hh'].argmax(1)==y_all[te]).mean():.3f} acc_hc={(fold['P_hc'].argmax(1)==y_all[te]).mean():.3f} süre {dt:.0f}s")
    results["times"][f"loco_{c}"] = dt

# ---------------- S0 (leaky reference): collar, random 70% of seconds
if not args.only_loco and not args.max_folds:
    t0 = time.time()
    r = np.random.default_rng(SEED)
    perm = r.permutation(idx_all)
    ntr = int(0.7 * len(perm))
    tr_full, te = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    tr = cap(tr_full, np.random.default_rng(SEED + 99))
    m = rf(500).fit(XC[tr], y_all[tr])
    results["s0"] = dict(test_idx=te, n_train=len(tr), P_cc=full_proba(m, XC[te]), inner_collar=inner_cv(tr, cats_row, XC))
    results["times"]["s0"] = time.time() - t0
    log(f"S0 n_tr={len(tr)} n_te={len(te)} acc={(results['s0']['P_cc'].argmax(1)==y_all[te]).mean():.3f} süre {time.time()-t0:.0f}s")

results["times"]["toplam"] = time.time() - t_all
suffix = "_timing" if args.max_folds else ""
with open(OUT / f"raw_{TAG}{suffix}.pkl", "wb") as f:
    pickle.dump(results, f)
log("BİTTİ toplam süre (dk):", round((time.time() - t_all) / 60, 1))
