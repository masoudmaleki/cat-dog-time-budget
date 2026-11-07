r"""32_r06_C_naive.py — R06 C: naive baselines (no classifier). N-source: class proportions of the training set; N-target: mean budget of the non-test individuals (individuals equally weighted).
Primary-set bags (Smit 68; Dunford v2 cat bag N>=60). d = AE_N-target - AE_CC (d>0: the classifier is better). SDs of the true budget.
Reads: Smit table, dog_aneck_1s.parquet, features/dunford_v2_1s.parquet, r05/r05_ladder_bags.parquet, r05/r05_sources_bags.parquet, r06/B_v2_bags.parquet, results/tables/r02_bag_level.csv
Writes: results/tables/r06_C_naive.csv, r06_C_naive_bag.parquet (bag-level AE), r06_C_budget_sd.csv
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, smit_data as D, stats06 as S6

T = P.TABLES; O5 = P.R05; O6 = P.R06
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); cat = lab.Cat_id.values; CATS = sorted(set(cat)); ok1 = y1 >= 0
dun, _ = R.load_dunford(v2=True); yd = dun.y1.values; dcat = dun.cat.values; DC = sorted(set(dcat))
dog, _ = R.load_dog(); yg = dog.y1.values[dog.y1.values < 2]
def prop(y, K): return np.bincount(y, minlength=K) / max(len(y), 1)
def rm_bakim(p3): q = np.asarray(p3, float)[:2]; return q / q.sum()
smit_b = {c: prop(y1[(cat == c) & ok1], 3) for c in CATS}
dun_b = {c: prop(yd[dcat == c], 3) for c in DC}
pi_dog = np.r_[prop(yg, 2), 0.0]
L1b = pd.concat([pd.read_parquet(O5 / "r05_ladder_bags.parquet"), pd.read_parquet(O5 / "r05_sources_bags.parquet")], ignore_index=True)
L1b = pd.concat([L1b[~L1b.scen.isin(["S3", "S4-Dunford", "R-D"])], pd.read_parquet(O6 / "B_v2_bags.parquet")], ignore_index=True)   # Dunford: v2 only
PVEC = ["p0", "p1", "p2"]; HVEC = ["h0", "h1", "h2"]


def rows_for(scen, var, level="-"):
    return L1b[(L1b.scen == scen) & (L1b.variant == var) & (L1b.m == 0)]


def nvecs(scen, var, c, Kt):
    """(N-source, N-target) vectors (length Kt)."""
    if scen in ("S1", "S2"):
        oth = [x for x in CATS if x != c]; ns = prop(y1[ok1 & (cat != c)], 3); nh = np.mean([smit_b[x] for x in oth], axis=0)
    elif scen == "S3":
        ns = prop(y1[ok1], 3); oth = [x for x in DC if x != c]; nh = np.mean([dun_b[x] for x in oth], axis=0)
    elif scen == "S4-Smit":
        oth = [x for x in CATS if x != c]; nh = np.mean([smit_b[x] for x in oth], axis=0); ns = pi_dog
        if Kt == 2: nh = rm_bakim(nh)
    elif scen == "S4-Dunford":
        oth = [x for x in DC if x != c]; nh = np.mean([dun_b[x] for x in oth], axis=0); ns = pi_dog
        if Kt == 2: nh = rm_bakim(nh)
    elif scen == "R-D":
        oth = [x for x in DC if x != c]; ns = prop(yd[dcat != c], 3); nh = np.mean([dun_b[x] for x in oth], axis=0)
    return ns[:Kt], nh[:Kt]


SCEN = [("S1", "30dk_ge50", 3), ("S2", "30dk_ge50", 3), ("S3", "kedi", 3), ("S4-Smit", "30dk_ge50-acik", 3), ("S4-Smit", "30dk_ge50-kapali", 2),
        ("S4-Dunford", "acik", 3), ("S4-Dunford", "kapali", 2), ("R-D", "kedi", 3)]
out = []; bagrows = []; fam = []
for scen, var, Kt in SCEN:
    g = rows_for(scen, var)
    arms = {a: gg.set_index(["ind", "bag"]) for a, gg in g.groupby("arm")}
    cc = arms["CC"]; p = cc[PVEC].values[:, :Kt]; hc = cc[HVEC].values[:, :Kt]
    ae_cc = np.abs(hc - p).mean(1)
    ns_ae = []; nh_ae = []
    for (ind, bag), pv in zip(cc.index, p):
        ns, nh = nvecs(scen, var, ind, Kt); ns_ae.append(np.abs(ns - pv).mean()); nh_ae.append(np.abs(nh - pv).mean())
    ns_ae = np.array(ns_ae); nh_ae = np.array(nh_ae)
    df = pd.DataFrame(dict(scen=scen, var=var, ind=[i for i, _ in cc.index], bag=[b for _, b in cc.index], N=cc.N.values, AE_CC=ae_cc, AE_Nk=ns_ae, AE_Nh=nh_ae))
    for a in ("PCC", "ACC", "PACC", "EMQ"): df["AE_" + a] = arms[a].AE.reindex(cc.index).values
    bagrows.append(df)
    d_h = df.assign(d=df.AE_Nh - df.AE_CC).groupby("ind").d.mean(); d_k = df.assign(d=df.AE_Nk - df.AE_CC).groupby("ind").d.mean()
    sh = S6.stat(d_h.values); sk = S6.stat(d_k.values)
    means = {a: float(df["AE_" + a].mean()) for a in ("CC", "PCC", "ACC", "PACC", "EMQ")}; best = min(means, key=means.get)
    fam.append(dict(scen=scen, var=var, K=Kt, n_bag=len(df), n_ind=df.ind.nunique(), AE_Nk=float(df.AE_Nk.mean()), AE_Nh=float(df.AE_Nh.mean()), AE_CC=means["CC"], en_iyi=best, AE_en_iyi=means[best],
                    d=sh["d"], lo=sh["lo"], hi=sh["hi"], p=sh["p"], d_Nk=sk["d"], lo_Nk=sk["lo"], hi_Nk=sk["hi"]))
res = S6.family(fam)
# each scenario is a separate single question -> family size 1 (no correction)
res["p_holm"] = res.p; res["aile_n"] = 1; res["kategori"] = [S6.category(r.d, r.lo, r.hi, r.p) for r in res.itertuples()]
# --- S1 L2 (with the CC in r02_bag_level.csv)
r2 = pd.read_csv(T / "r02_bag_level.csv", dtype={"torba": str})
r2 = r2[(r2.varyant == "va_f32") & (r2.bagdef == "30dk_ge50") & (r2.level == "L2") & (r2.basamak == "S1-tasma") & (r2.tahminci == "CC")]
y2 = lab.y2.values; K2 = len(D.L2)
b2 = {c: np.bincount(y2[(cat == c) & (y2 >= 0)], minlength=K2) / ((cat == c) & (y2 >= 0)).sum() for c in CATS}
l2 = []
for (ind, bag), g in r2.groupby(["kedi", "torba"]):
    g = g.set_index("sinif").reindex(D.L2); pv = g.p.values; hv = g.p_hat.values
    oth = [x for x in CATS if x != ind]; ns = np.bincount(y2[(cat != ind) & (y2 >= 0)], minlength=K2); ns = ns / ns.sum(); nh = np.mean([b2[x] for x in oth], axis=0)
    l2.append(dict(ind=ind, bag=bag, AE_CC=np.abs(hv - pv).mean(), AE_Nk=np.abs(ns - pv).mean(), AE_Nh=np.abs(nh - pv).mean()))
l2 = pd.DataFrame(l2); dh = (l2.AE_Nh - l2.AE_CC).groupby(l2.ind).mean(); dk = (l2.AE_Nk - l2.AE_CC).groupby(l2.ind).mean()
sh = S6.stat(dh.values); sk = S6.stat(dk.values)
res = pd.concat([res, pd.DataFrame([dict(scen="S1-L2", var="30dk_ge50", K=K2, n_bag=len(l2), n_ind=l2.ind.nunique(), AE_Nk=l2.AE_Nk.mean(), AE_Nh=l2.AE_Nh.mean(), AE_CC=l2.AE_CC.mean(), en_iyi="CC", AE_en_iyi=l2.AE_CC.mean(),
                                         d=sh["d"], lo=sh["lo"], hi=sh["hi"], p=sh["p"], d_Nk=sk["d"], lo_Nk=sk["lo"], hi_Nk=sk["hi"], p_holm=sh["p"], aile_n=1, kategori=S6.category(sh["d"], sh["lo"], sh["hi"], sh["p"]))])], ignore_index=True)
res.to_csv(T / "r06_C_naive.csv", index=False)
pd.concat(bagrows, ignore_index=True).to_parquet(O6 / "C_naive_bag.parquet")
# --- true-budget SD (primary set): between individuals and within individual
sd = []
for scen, var, Kt in SCEN:
    g = rows_for(scen, var); g = g[g.arm == "CC"]
    for k in range(Kt):
        pm = g.groupby("ind")[f"p{k}"].mean(); within = g.groupby("ind")[f"p{k}"].var(ddof=1).dropna()
        sd.append(dict(scen=scen, var=var, sinif=["Inaktif", "Lokomosyon", "Bakim"][k], ort=float(g[f"p{k}"].mean()), sd_arasi=float(pm.std(ddof=1)), sd_ici=(float(np.sqrt(within.mean())) if len(within) else np.nan), n_ind=int(len(pm)), n_bag=int(len(g))))
pd.DataFrame(sd).to_csv(T / "r06_C_budget_sd.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
print(res.drop(columns=["lo_Nk", "hi_Nk", "p_holm", "aile_n"]).to_string()); print(pd.DataFrame(sd).to_string())
