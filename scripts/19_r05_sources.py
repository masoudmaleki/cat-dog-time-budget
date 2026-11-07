r"""19_r05_sources.py — R05 D: S3 and S4 source-only. Source models (trained on everything; 500 trees; C/Cp/prior from inner CV GroupKFold(4), 300 trees):
 S3: Smit collar 12 cats -> Dunford 9 cats (cat bag). S4: dog ANeck 45 dogs (2 classes: Inaktif/Lokomosyon; S4-eating skipped) -> Smit collar (30 min >= 50% and cat bag) and Dunford (cat bag).
 S4-closed: the target's Bakim (care) seconds are removed, K=2. S4-open: Bakim stays in the target, the source predicts 2 classes (p_hat_Bakim = 0), true budget has K=3.
Reads: Smit table, features/*.parquet. Writes: <data root>/interim/r05/sources.pkl, r05_sources_bags.parquet, results/tables/r05_classifier_src.csv
"""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
from config import paths as P
from src import r05lib as R, quant as Q

SEED = P.SEED; OUT = P.R05; T = P.TABLES; t0 = time.time()
lab, y1, XC, XH, XC17, XH17 = R.load_smit()
idx = np.where(y1 >= 0)[0]; cat = lab.Cat_id.values; XS = XC17.values
dun, Xd = R.load_dunford(); Xd = Xd.values; yd = dun.y1.values; dcat = dun.cat.values
dog, Xg = R.load_dog(); keep = dog.y1.values < 2; dog = dog[keep].reset_index(drop=True); Xg = Xg.values[keep]; yg = dog.y1.values; gid = dog.DogID.values

src3 = R.source_fit(XS[idx], y1[idx], cat[idx], 3, SEED + 100); P3 = src3.pop("model").predict_proba(Xd)
print("S3 kaynak", "%.0f s" % (time.time() - t0), flush=True)
src4 = R.source_fit(Xg, yg, gid, 2, SEED + 101); m4 = src4.pop("model")
P4s = m4.predict_proba(XS[idx]); P4d = m4.predict_proba(Xd)
print("S4 kaynak", "%.0f s" % (time.time() - t0), flush=True)
pickle.dump(dict(idx=idx.astype(np.int32), P3=P3, src3=src3, P4s=P4s, P4d=P4d, src4=src4), open(OUT / "sources.pkl", "wb"))

rec = R.Rec(); clf = []; skipN = []
pos_idx = {int(g): i for i, g in enumerate(idx)}


def clfrow(scen, var, c, yt, pred):
    lbl = np.unique(yt)
    clf.append(dict(scen=scen, variant=var, ind=c, macroF1=f1_score(yt, pred, labels=lbl, average="macro", zero_division=0), balacc=balanced_accuracy_score(yt, pred), acc=float((yt == pred).mean())))


# ---- S3
for c in sorted(set(dcat)):
    ii = np.where(dcat == c)[0]
    clfrow("S3", "-", c, yd[ii], P3[ii].argmax(1))
    if len(ii) < 60: skipN.append(("S3", c, len(ii))); continue
    p = R.truth(yd[ii], 3)
    for arm, ph in R.estimates(P3[ii], src3, 3).items(): rec.add("S3", "kedi", "-", arm, 0, 0, c, 0, len(ii), p, ph)
# ---- S4-Dunford (closed / open)
for var in ("kapali", "acik"):
    for c in sorted(set(dcat)):
        ii = np.where((dcat == c) & ((yd < 2) if var == "kapali" else True))[0]
        Kt = 2 if var == "kapali" else 3
        if len(ii) == 0: continue
        clfrow("S4-Dunford", var, c, yd[ii], P4d[ii].argmax(1))
        if len(ii) < 60: skipN.append(("S4-Dunford-" + var, c, len(ii))); continue
        p = R.truth(yd[ii], Kt)
        for arm, ph in R.estimates(P4d[ii], src4, 2).items(): rec.add("S4-Dunford", var, "-", arm, 0, 0, c, 0, len(ii), p, R.pad3(ph, Kt))
# ---- S4-Smit (closed / open; 30 min >= 50% and cat bag)
for var in ("kapali", "acik"):
    Kt = 2 if var == "kapali" else 3
    bags = R.smit_bags(lab, y1, 30, 0.5, mask=(y1 < 2) if var == "kapali" else None)
    for c, w, ix in bags:
        ii = np.array([pos_idx[int(g)] for g in ix]); p = R.truth(y1[ix], Kt)
        for arm, ph in R.estimates(P4s[ii], src4, 2).items(): rec.add("S4-Smit", "30dk_ge50-" + var, "-", arm, 0, 0, c, w, len(ii), p, R.pad3(ph, Kt))
    for c in sorted(set(cat)):
        ix = np.where((cat == c) & (y1 >= 0) & ((y1 < 2) if var == "kapali" else True))[0]
        ii = np.array([pos_idx[int(g)] for g in ix]); p = R.truth(y1[ix], Kt)
        clfrow("S4-Smit", var, c, y1[ix], P4s[ii].argmax(1))
        for arm, ph in R.estimates(P4s[ii], src4, 2).items(): rec.add("S4-Smit", "kedi-" + var, "-", arm, 0, 0, c, 0, len(ii), p, R.pad3(ph, Kt))
rec.wide().to_parquet(OUT / "r05_sources_bags.parquet")
pd.DataFrame(clf).to_csv(T / "r05_classifier_src.csv", index=False)
json.dump(dict(skipped_N_lt60=skipN, sure_s=time.time() - t0), open(T / "r05_sources_info.json", "w"), indent=1)
print("BİTTİ sources", skipN, "%.0f s" % (time.time() - t0), flush=True)
