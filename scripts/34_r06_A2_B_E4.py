r"""34_r06_A2_B_E4.py — R06 A2 (S2: D17 - matched-32, same 68 bags), B (Dunford v1 - v2 headline counts), E4 (S1-closed - S4-Smit-closed).
Reads: r05/ladder_S2_<cat>.pkl, r06/A2_S2esl_<cat>.pkl, r05/r05_ladder_bags.parquet, r05_sources_bags.parquet, r06/B_v2_bags.parquet, r06/E4_S1kapali_bags.parquet
Writes: results/tables/r06_A2.csv, r06_B_headline.csv, r06_E4.csv
"""
import sys, pickle, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, stats06 as S6

T = P.TABLES; O5 = P.R05; O6 = P.R06
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); cat = lab.Cat_id.values; CATS = sorted(set(cat)); BAGS30 = R.smit_bags(lab, y1, 30, 0.5)
ESTS = ("CC", "ACC", "PACC")

# ------------------------------------------------------------------ A2
rec = R.Rec()
for scen, pref, d17 in (("S2-D17", O5 / "ladder_S2_{}.pkl", True), ("S2-esl32", O6 / "A2_S2esl_{}.pkl", False)):
    for c in CATS:
        d = pickle.load(open(str(pref).format(c), "rb")); te = d["te"].astype(int); Pm = d["P"].astype(np.float64); src = d["src"]
        for cc, w, ix in BAGS30:
            if cc != c: continue
            ii = np.searchsorted(te, ix); assert (te[ii] == ix).all()
            p = R.truth(y1[ix], 3)
            for arm, ph in R.estimates(Pm[ii], src, 3).items():
                if arm in ESTS: rec.add(scen, "30dk_ge50", "-", arm, 0, 0, c, w, len(ii), p, ph)
W = rec.wide(); W.to_parquet(O6 / "A2_bags.parquet")
rows = []
for e in ESTS:
    a = W[(W.scen == "S2-D17") & (W.arm == e)].set_index(["ind", "bag"]).AE; b = W[(W.scen == "S2-esl32") & (W.arm == e)].set_index(["ind", "bag"]).AE
    j = pd.concat([a, b], axis=1, keys=["d17", "esl"]); dd = (j.esl - j.d17).groupby(level=0).mean()      # positive = D17 better
    s = S6.stat(dd.values); rows.append(dict(est=e, n_bag=len(j), AE_D17=float(j.d17.mean()), AE_esl32=float(j.esl.mean()), **s))
A2 = S6.family(rows); A2.to_csv(T / "r06_A2.csv", index=False); print(A2.to_string())

# ------------------------------------------------------------------ B headline
v1 = pd.concat([pd.read_parquet(O5 / "r05_ladder_bags.parquet"), pd.read_parquet(O5 / "r05_sources_bags.parquet")], ignore_index=True)
v2 = pd.read_parquet(O6 / "B_v2_bags.parquet")
hb = []
for ver, df in (("v1", v1), ("v2", v2)):
    for sc, var in (("S3", "kedi"), ("S4-Dunford", "kapali"), ("S4-Dunford", "acik"), ("R-D", "kedi")):
        g = df[(df.scen == sc) & (df.variant == var)]
        for e in ESTS:
            x = g[g.arm == e]
            if x.empty: continue
            m_, lo, hi = R.boot_mean(x.AE.values, x.ind.values)
            hb.append(dict(surum=ver, scen=sc, var=var, est=e, n_kedi=x.ind.nunique(), AE=m_, lo=lo, hi=hi))
HB = pd.DataFrame(hb); HB.to_csv(T / "r06_B_headline.csv", index=False)
pd.set_option("display.width", 250, "display.float_format", "{:.4f}".format)
print(HB.pivot_table(index=["scen", "var", "est"], columns="surum", values=["AE", "n_kedi"]).to_string())

# ------------------------------------------------------------------ E4
f = O6 / "E4_S1kapali_bags.parquet"
if f.exists():
    e4 = pd.read_parquet(f); s4 = pd.read_parquet(O5 / "r05_sources_bags.parquet"); s4 = s4[(s4.scen == "S4-Smit") & (s4.variant == "30dk_ge50-kapali")]
    rows = []
    for e in ESTS:
        a = e4[e4.arm == e].set_index(["ind", "bag"]).AE; b = s4[s4.arm == e].set_index(["ind", "bag"]).AE
        j = pd.concat([a, b], axis=1, keys=["s1k", "s4k"]); dd = (j.s4k - j.s1k).groupby(level=0).mean()       # positive = S1-closed better (cost of the study/species shift)
        s = S6.stat(dd.values); rows.append(dict(est=e, n_bag=len(j), AE_S1kapali=float(j.s1k.mean()), AE_S4Smitkapali=float(j.s4k.mean()), **s))
    E4 = S6.family(rows); E4.to_csv(T / "r06_E4.csv", index=False); print(E4.to_string())
