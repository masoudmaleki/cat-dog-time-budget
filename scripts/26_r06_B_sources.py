r"""26_r06_B_sources.py — R06 B: Dunford v2 with source-only S3, S4-Dunford (closed/open), R-D. Same source models/seeds as R05 (S3: SEED+100, S4 dog: SEED+101, R-D LOCO: SEED+ci).
Reads: Smit table, features/dunford_v2_1s.parquet, dog_aneck_1s.parquet. Writes: <data root>/interim/r06/sources_v2.pkl, RD_v2_<cat>.pkl, B_v2_bags.parquet
"""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R

SEED = P.SEED; OUT = P.R06; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); idx = np.where(y1 >= 0)[0]; cat = lab.Cat_id.values; XS = XC17.values
dun, Xd = R.load_dunford(v2=True); Xd = Xd.values; yd = dun.y1.values; dcat = dun.cat.values; DC = sorted(set(dcat))
dog, Xg = R.load_dog(); keep = dog.y1.values < 2; Xg = Xg.values[keep]; yg = dog.y1.values[keep]; gid = dog.DogID.values[keep]
f = OUT / "sources_v2.pkl"
if f.exists(): S = pickle.load(open(f, "rb"))
else:
    src3 = R.source_fit(XS[idx], y1[idx], cat[idx], 3, SEED + 100); P3 = src3.pop("model").predict_proba(Xd); print("S3 kaynak", "%.0f s" % (time.time() - t0), flush=True)
    src4 = R.source_fit(Xg, yg, gid, 2, SEED + 101); P4d = src4.pop("model").predict_proba(Xd)
    S = dict(P3=P3, src3=src3, P4d=P4d, src4=src4); pickle.dump(S, open(f, "wb"))
rec = R.Rec(); skip = []
for c in DC:
    ii = np.where(dcat == c)[0]
    if len(ii) < 60: skip.append(("S3", c, len(ii))); continue
    p = R.truth(yd[ii], 3)
    for arm, ph in R.estimates(S["P3"][ii], S["src3"], 3).items(): rec.add("S3", "kedi", "-", arm, 0, 0, c, 0, len(ii), p, ph)
for var in ("kapali", "acik"):
    Kt = 2 if var == "kapali" else 3
    for c in DC:
        ii = np.where((dcat == c) & ((yd < 2) if var == "kapali" else True))[0]
        if len(ii) < 60: skip.append(("S4-Dunford-" + var, c, len(ii))); continue
        p = R.truth(yd[ii], Kt)
        for arm, ph in R.estimates(S["P4d"][ii], S["src4"], 2).items(): rec.add("S4-Dunford", var, "-", arm, 0, 0, c, 0, len(ii), p, R.pad3(ph, Kt))
print("kaynak-yalniz tamam", "%.0f s" % (time.time() - t0), skip, flush=True)
for ci, c in enumerate(DC):
    tr = np.where(dcat != c)[0]; te = np.where(dcat == c)[0]
    src = R.source_fit(Xd[tr], yd[tr], dcat[tr], 3, SEED + ci); mdl = src.pop("model"); Pm = R.pad_proba(mdl, Xd[te], 3)
    pickle.dump(dict(te=te.astype(np.int32), P=Pm, src=src), open(OUT / f"RD_v2_{c}.pkl", "wb"))
    if len(te) < 60: skip.append(("R-D", c, len(te))); continue
    p = R.truth(yd[te], 3)
    for arm, ph in R.estimates(Pm, src, 3).items(): rec.add("R-D", "kedi", "-", arm, 0, 0, c, 0, len(te), p, ph)
    print("RD", c, "%.0f s" % (time.time() - t0), flush=True)
rec.wide().to_parquet(OUT / "B_v2_bags.parquet")
print("BITTI B sources", skip, flush=True)
