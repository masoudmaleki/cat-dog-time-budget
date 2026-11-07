r"""41_r07_4_analyze.py — R07 §4: source-C arms − CC for HGB and LR (and RF for comparison); F rule; family = scenario x classifier (Holm over 4 tests: PCC, ACC, PACC, EMQ).
APP bridge (S1-kopru: APP120, dogal2dk): same rule, family = bag type x classifier. Macro-F1 and ECE (top-label, 10 bins; pooled).
Reads: r05/r05_ladder_bags, r05_sources_bags, F_bags, r06/B_v2_bags, r07/bags_{HGB,LR}.parquet, r07_4_cls_*.csv, RF posterior pickles. Writes: results/tables/r07_4_tests.csv, r07_4_cls_summary.csv"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from config import paths as P
from src import r07lib as L, r07clf as C, r05lib as R, stats06 as S6

O5, O6, O7, T = P.R05, P.R06, P.INTERIM / "r07", P.TABLES
rd = L.rd
RF = pd.concat([rd(O5 / "r05_ladder_bags.parquet"), rd(O5 / "r05_sources_bags.parquet"), rd(O5 / "F_bags.parquet"), rd(O6 / "B_v2_bags.parquet")], ignore_index=True)
# RF Dunford: v2 (R06); drop the v1 rows
RF = RF[~((RF.scen.isin(["S3", "S4-Dunford", "R-D"])) & (RF.index < len(RF) - len(rd(O6 / "B_v2_bags.parquet"))))]
CL = {"RF": RF, "HGB": rd(O7 / "bags_HGB.parquet"), "LR": rd(O7 / "bags_LR.parquet")}
ARMS = ("PCC", "ACC", "PACC", "EMQ")
SC = [("S1", "30dk_ge50"), ("S2", "30dk_ge50"), ("S3", "kedi"), ("S4-Smit", "30dk_ge50-acik"), ("S4-Smit", "30dk_ge50-kapali"), ("S4-Dunford", "acik"), ("S4-Dunford", "kapali"),
      ("S1-kopru", "APP120"), ("S1-kopru", "dogal2dk")]
rows = []
for clf, df in CL.items():
    for scen, var in SC:
        g = df[(df.scen == scen) & (df.variant == var) & (df.m == 0)]
        if g.empty: continue
        cc = g[g.arm == "CC"][["ind", "bag", "AE"]]; st = []
        for a in ARMS:
            x = g[g.arm == a][["ind", "bag", "AE"]]; j = cc.merge(x, on=["ind", "bag"], suffixes=("_cc", "_a")); j["d"] = j.AE_cc - j.AE_a
            if j.empty: continue   # RF APP/natural 2-min: PCC was not stored in R05 -> family of 3 tests
            st.append(dict(clf=clf, scen=scen, variant=var, kol=a, n_torba=len(j), AE_CC=float(j.AE_cc.mean()), AE_kol=float(j.AE_a.mean()), **S6.stat(j.groupby("ind").d.mean().values)))
        rows += S6.family(st).to_dict("records")
TT = pd.DataFrame(rows); TT.to_csv(T / "r07_4_tests.csv", index=False)

# ---- makro-F1, ECE
cls = [pd.read_csv(T / f"r07_4_cls_{k}.csv") for k in ("HGB", "LR")]
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); cat = lab.Cat_id.values; CATS = sorted(set(cat)); idx = np.where(y1 >= 0)[0]
dun, Xd = R.load_dunford(v2=True); yd = dun.y1.values; dcat = dun.cat.values; DC = sorted(set(dcat))
rf = []; store = {}


def add(scen, var, ind, yt, P_):
    pred = P_.argmax(1); rf.append(dict(clf="RF", scen=scen, variant=var, ind=ind, n=len(yt), macroF1=f1_score(yt, pred, labels=np.unique(yt), average="macro", zero_division=0), acc=float((pred == yt).mean())))
    store.setdefault((scen, var), []).append((yt, P_))


for c in CATS:
    for scen in ("S1", "S2"):
        d = pickle.load(open(O5 / f"ladder_{scen}_{c}.pkl", "rb")); te = d["te"].astype(int); add(scen, "-", c, y1[te], d["P"].astype(np.float64))
S = pickle.load(open(O6 / "sources_v2.pkl", "rb")); S5 = pickle.load(open(O5 / "sources.pkl", "rb"))
for c in DC:
    ii = np.where(dcat == c)[0]; add("S3", "-", c, yd[ii], S["P3"][ii])
for var in ("kapali", "acik"):
    for c in DC:
        ii = np.where((dcat == c) & ((yd < 2) if var == "kapali" else True))[0]
        if len(ii): add("S4-Dunford", var, c, yd[ii], np.hstack([S["P4d"][ii], np.zeros((len(ii), 1))]) if var == "acik" else S["P4d"][ii])
    pos = {int(g): i for i, g in enumerate(S5["idx"])}
    for c in CATS:
        ix = np.where((cat == c) & (y1 >= 0) & ((y1 < 2) if var == "kapali" else True))[0]; ii = np.array([pos[int(g)] for g in ix])
        add("S4-Smit", var, c, y1[ix], np.hstack([S5["P4s"][ii], np.zeros((len(ii), 1))]) if var == "acik" else S5["P4s"][ii])
rfd = pd.DataFrame(rf); rfd["ECE_pooled"] = [C.ece_top(np.vstack([p for _, p in store[(r.scen, r.variant)]]), np.concatenate([y for y, _ in store[(r.scen, r.variant)]])) for r in rfd.itertuples()]
allc = pd.concat([rfd] + cls, ignore_index=True)
summ = allc.groupby(["scen", "variant", "clf"]).agg(n_birey=("ind", "size"), macroF1=("macroF1", "mean"), acc=("acc", "mean"), ECE_pooled=("ECE_pooled", "first")).reset_index()
summ.to_csv(T / "r07_4_cls_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
print(TT[["clf", "scen", "variant", "kol", "n_torba", "n", "AE_CC", "AE_kol", "d", "lo", "hi", "p_holm", "kategori"]].to_string()); print(summ.to_string())
