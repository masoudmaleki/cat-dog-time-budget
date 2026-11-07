r"""18_r05_ladder.py — R05 C: ladder references (L1, D17, 500 trees; source C/Cp/prior from inner-CV out-of-fold, 300 trees).
S0 (random 70%/30% of seconds; inner CV StratifiedKFold(4)), S1 (Smit collar LOCO, D17 and 32 features), S2 (Smit harness 11 cats -> collar; D17), R-D (Dunford LOCO), K (dog ANeck GroupKFold(5)).
Reads: Smit table, features/dunford_1s.parquet, dog_aneck_1s.parquet. Writes: <data root>/interim/r05/ladder_<scenario>_<individual>.pkl (posteriors + source C),
r05_ladder_bags.parquet, results/tables/r05_classifier_by_ind.csv, r05_K_dog.json
Bags: Smit 30 min >= 50% (68) and cat bag; Dunford cat bag (N < 60 excluded from evaluation).
"""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from config import paths as P
from src import r05lib as R, quant as Q

SEED = P.SEED; OUT = P.R05; OUT.mkdir(parents=True, exist_ok=True); T = P.TABLES
LOG = open(P.LOGS / "r05_ladder.log", "a", encoding="utf-8")
t0 = time.time()


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


lab, y1, XC, XH, XC17, XH17 = R.load_smit()
ok = y1 >= 0; idx_all = np.where(ok)[0]; cat = lab.Cat_id.values; CATS = sorted(set(cat))
A = {"c17": XC17.values, "h17": XH17.values, "c32": XC.values}
BAGS30 = R.smit_bags(lab, y1, 30, 0.5)
rec = R.Rec(); clf = []


def add_bags(scen, te, P_, src, bagdefs=("30dk_ge50", "kedi"), cname=None):
    """te: global row indices (sorted), P_: posteriors (len(te), 3)."""
    pos = {"30dk_ge50": [(c, w, ix) for c, w, ix in BAGS30 if c == cname], "kedi": [(c, 0, te) for c in [cname]]}
    for bd in bagdefs:
        for c, w, ix in pos[bd]:
            ii = np.searchsorted(te, ix[np.isin(ix, te)]);
            if len(ii) == 0: continue
            p = R.truth(y1[te[ii]], 3)
            for arm, ph in R.estimates(P_[ii], src, 3).items():
                rec.add(scen, bd, "-", arm, 0, 0, c, w, len(ii), p, ph)


def clfrow(scen, c, yt, pred):
    lbl = np.unique(yt)
    clf.append(dict(scen=scen, ind=c, macroF1=f1_score(yt, pred, labels=lbl, average="macro", zero_division=0), balacc=balanced_accuracy_score(yt, pred), acc=float((yt == pred).mean())))


# ---------------------------------------------------------------- S0
perm = np.random.default_rng(SEED).permutation(idx_all); ntr = int(0.7 * len(perm)); tr0, te0 = perm[:ntr], np.sort(perm[ntr:])
src = R.source_fit(A["c17"][tr0], y1[tr0], None, 3, SEED, strat=True)
P0 = src["model"].predict_proba(A["c17"][te0]); del src["model"]
for bd in ("30dk_ge50", "kedi"):
    if bd == "30dk_ge50":
        for c, w, ix in BAGS30:
            ii = np.searchsorted(te0, ix[np.isin(ix, te0)])
            if len(ii) == 0: continue
            p = R.truth(y1[te0[ii]], 3)
            for arm, ph in R.estimates(P0[ii], src, 3).items(): rec.add("S0", bd, "-", arm, 0, 0, c, w, len(ii), p, ph)
    else:
        for c in CATS:
            ii = np.where(cat[te0] == c)[0]; p = R.truth(y1[te0[ii]], 3)
            for arm, ph in R.estimates(P0[ii], src, 3).items(): rec.add("S0", bd, "-", arm, 0, 0, c, 0, len(ii), p, ph)
clfrow("S0", "havuz", y1[te0], P0.argmax(1))
log("S0 tamam", "%.0f s" % (time.time() - t0))

# ---------------------------------------------------------------- S1 (D17, 32), S2 (D17): LOCO
for ci, c in enumerate(CATS):
    te = idx_all[cat[idx_all] == c]; tr = idx_all[cat[idx_all] != c]
    for scen, trk, tek in (("S1", "c17", "c17"), ("S1-32", "c32", "c32"), ("S2", "h17", "c17")):
        f = OUT / f"ladder_{scen}_{c}.pkl"
        if f.exists():
            d = pickle.load(open(f, "rb"))
        else:
            src = R.source_fit(A[trk][tr], y1[tr], cat[tr], 3, SEED + ci)
            Pm = src["model"].predict_proba(A[tek][te]); del src["model"]
            d = dict(te=te.astype(np.int32), P=Pm.astype(np.float32), src=src)
            pickle.dump(d, open(f, "wb"))
        add_bags(scen, d["te"], d["P"].astype(np.float64), d["src"], cname=c)
        clfrow(scen, c, y1[d["te"]], d["P"].argmax(1))
    log("LOCO", c, "%.0f s" % (time.time() - t0))

# ---------------------------------------------------------------- R-D: Dunford LOCO (8 cats train, 1 test); cat bag
dun, Xd = R.load_dunford(); dcat = dun.cat.values; DC = sorted(set(dcat)); Xd = Xd.values; yd = dun.y1.values
skipped_N = []
for ci, c in enumerate(DC):
    tr = np.where(dcat != c)[0]; te = np.where(dcat == c)[0]
    src = R.source_fit(Xd[tr], yd[tr], dcat[tr], 3, SEED + ci); Pm = src["model"].predict_proba(Xd[te]) if len(src["model"].classes_) == 3 else R.pad_proba(src["model"], Xd[te], 3)
    del src["model"]
    pickle.dump(dict(te=te.astype(np.int32), P=Pm, src=src), open(OUT / f"ladder_RD_{c}.pkl", "wb"))
    clfrow("R-D", c, yd[te], Pm.argmax(1))
    if len(te) < 60: skipped_N.append((c, len(te))); continue
    p = R.truth(yd[te], 3)
    for arm, ph in R.estimates(Pm, src, 3).items(): rec.add("R-D", "kedi", "-", arm, 0, 0, c, 0, len(te), p, ph)
log("R-D tamam", skipped_N, "%.0f s" % (time.time() - t0))

# ---------------------------------------------------------------- K: dog ANeck, GroupKFold(5), 2 classes (Inaktif/Lokomosyon)
dog, Xg = R.load_dog(); keep = dog.y1.values < 2; dog = dog[keep].reset_index(drop=True); Xg = Xg.values[keep]; yg = dog.y1.values; gg = dog.DogID.values
per = []; pooled_true, pooled_pred = [], []
for k, (tr, te) in enumerate(GroupKFold(5).split(Xg, yg, gg)):
    m = R.rf(500, SEED + k).fit(Xg[tr], yg[tr]); pr = m.predict(Xg[te]); pooled_true.append(yg[te]); pooled_pred.append(pr)
    for dg in np.unique(gg[te]):
        s = gg[te] == dg
        per.append(dict(scen="K", ind=int(dg), macroF1=f1_score(yg[te][s], pr[s], labels=np.unique(yg[te][s]), average="macro", zero_division=0),
                        balacc=balanced_accuracy_score(yg[te][s], pr[s]), acc=float((yg[te][s] == pr[s]).mean()), n=int(s.sum())))
pt, pp_ = np.concatenate(pooled_true), np.concatenate(pooled_pred)
kd = pd.DataFrame(per)
kres = dict(n_dog=int(kd.shape[0]), macroF1_per_dog=R.boot_mean(kd.macroF1.values, kd.ind.values), balacc_per_dog=R.boot_mean(kd.balacc.values, kd.ind.values),
            macroF1_pooled=float(f1_score(pt, pp_, average="macro")), balacc_pooled=float(balanced_accuracy_score(pt, pp_)), n_windows=int(len(pt)))
json.dump(kres, open(T / "r05_K_dog.json", "w"), indent=1)
clf = pd.concat([pd.DataFrame(clf), kd.drop(columns="n")], ignore_index=True); clf.to_csv(T / "r05_classifier_by_ind.csv", index=False)
rec.wide().to_parquet(OUT / "r05_ladder_bags.parquet")
log("K", kres, "BİTTİ ladder", "%.0f s" % (time.time() - t0))
