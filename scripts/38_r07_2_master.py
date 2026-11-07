r"""38_r07_2_master.py — R07 §2: consolidated results table (no new model). d = AE_CC − AE_arm (d>0: arm better); CC is the reference.
Families (Holm): 'source' = {PCC, ACC, PACC, EMQ, N-target} (L2: no N-target) per scenario; 'target-m1800' = target-labelled arms (m = 1,800, random block) per scenario.
Reads: r05/*bags.parquet, r06/*bags.parquet, E1_*.parquet, results/tables/r02_bag_level.csv, r02/raw_va_f32.pkl, r03/inner_*.pkl (S2-matched L2 recomputation).
Writes: results/tables/r07_master.csv, r07_L2_S2esl_bags.csv"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r07lib as L, stats06 as S6, evalkit as E, quant as Q

O5, O6, T = P.R05, P.R06, P.TABLES
rd = L.rd
lad = rd(O5 / "r05_ladder_bags.parquet"); src5 = rd(O5 / "r05_sources_bags.parquet"); v2 = rd(O6 / "B_v2_bags.parquet"); e4 = rd(O6 / "E4_S1kapali_bags.parquet")
nv = rd(O6 / "C_naive_bag.parquet")
DU, SM = L.load_E()
LD = L.cat_(rd(O5 / "E_ladder_extra.parquet"), *[rd(p) for p in sorted(O6.glob("E1_ladder_fit_m*.parquet")) if not p.name.endswith("_miss.parquet")]).drop_duplicates(L.KEYS, keep="last")

DK = "Dunford v2 kedi torbası, N>=60"; SK = "Smit 68 torba"
SPEC = [
    ("S0 (sızıntılı)", lad, ("S0", "30dk_ge50"), None, SK, None),
    ("S1", lad, ("S1", "30dk_ge50"), ("S1", "30dk_ge50"), SK, (LD, "S1", "30dk_ge50", "birey")),
    ("S2", lad, ("S2", "30dk_ge50"), ("S2", "30dk_ge50"), SK, (LD, "S2", "30dk_ge50", "birey")),
    ("S3 [keşifsel]", v2, ("S3", "kedi"), ("S3", "kedi"), DK, (DU, "S3", "-", "calisma")),
    ("S4-Smit-açık", src5, ("S4-Smit", "30dk_ge50-acik"), ("S4-Smit", "30dk_ge50-acik"), SK, (SM, "S4-Smit", "acik", "birey")),
    ("S4-Smit-kapalı", src5, ("S4-Smit", "30dk_ge50-kapali"), ("S4-Smit", "30dk_ge50-kapali"), SK, (SM, "S4-Smit", "kapali", "birey")),
    ("S4-Dunford-açık [keşifsel]", v2, ("S4-Dunford", "acik"), ("S4-Dunford", "acik"), DK, (DU, "S4-Dunford", "acik", "calisma")),
    ("S4-Dunford-kapalı [keşifsel]", v2, ("S4-Dunford", "kapali"), ("S4-Dunford", "kapali"), DK, (DU, "S4-Dunford", "kapali", "calisma")),
    ("R-D [keşifsel]", v2, ("R-D", "kedi"), ("R-D", "kedi"), DK, None),
    ("S1-kapalı", e4, ("S1-kapali", "30dk_ge50"), None, SK, None),
]
rows = []


def add_family(fam, sc, uzay, feat, kume, items):
    """items: (arm, dind Series, n_bag, AE_cc, AE_arm)"""
    st = []
    for arm, dind, nb, acc_, aarm in items:
        st.append(dict(senaryo=sc, uzay=uzay, oznitelik=feat, kol=arm, aile=fam, torba_kumesi=kume, n_torba=nb, AE_CC=acc_, AE_kol=aarm, **S6.stat(dind.values)))
    df = S6.family(st)
    for r in df.to_dict("records"): rows.append(r)


def ref_row(sc, uzay, feat, kume, n, ccm, nind):
    rows.append(dict(senaryo=sc, uzay=uzay, oznitelik=feat, kol="CC", aile="-", torba_kumesi=kume, n_torba=n, AE_CC=ccm, AE_kol=ccm, n=nind,
                     d=np.nan, lo=np.nan, hi=np.nan, p=np.nan, p_holm=np.nan, aile_n=np.nan, kategori="REFERANS"))


def src_items(df, scen, var, arms, naive=None):
    g = df[(df.scen == scen) & (df.variant == var) & (df.m == 0)]; cc = g[g.arm == "CC"][["ind", "bag", "AE"]]; out = []
    for a in arms:
        x = g[g.arm == a][["ind", "bag", "AE"]]; j = cc.merge(x, on=["ind", "bag"], suffixes=("_cc", "_a")); j["d"] = j.AE_cc - j.AE_a
        out.append((a, j.groupby("ind").d.mean(), len(j), float(j.AE_cc.mean()), float(j.AE_a.mean())))
    if naive is not None:
        n = nv[(nv.scen == naive[0]) & (nv["var"] == naive[1])]; j = cc.rename(columns={"AE": "AE_cc"}).merge(n[["ind", "bag", "AE_Nh"]], on=["ind", "bag"]); j["d"] = j.AE_cc - j.AE_Nh
        out.append(("N-hedef", j.groupby("ind").d.mean(), len(j), float(j.AE_cc.mean()), float(j.AE_Nh.mean())))
    return out


for lab_, df, (scen, var), nvk, kume, tg in SPEC:
    its = src_items(df, scen, var, ("PCC", "ACC", "PACC", "EMQ"), nvk)
    ref_row(lab_, "L1", "D17", kume, its[0][2], its[0][3], len(its[0][1]))
    add_family("kaynak", lab_, "L1", "D17", kume, its)
    if tg is not None:
        dfT, ts, tv, tl = tg; its = []
        for a in ("T-ACC", "PACC-hedef", "PACC-buzme", "Yeniden-egitim", "Yeniden-egitim-agirlikli", "Yalniz-hedef-esl"):
            pr = L.pairs(dfT, ts, tv, tl, a, "CC", (1800,))
            if 1800 in pr:
                dind, nb, a_kol, a_cc = pr[1800]; its.append((a + " (m=1800, rastgele blok)", dind, nb, a_cc, a_kol))
        add_family("hedef-m1800", lab_, "L1", "D17", kume + " (hedef saniyeleri çıkarılmış torbalar)", its)

# ---- L2, 32 features (R02 stored bag level, 30dk_ge50, va_f32)
b2 = pd.read_csv(T / "r02_bag_level.csv"); b2 = b2[(b2.varyant == "va_f32") & (b2.bagdef == "30dk_ge50") & (b2.level == "L2")].copy()
b2["ae"] = (b2.p - b2.p_hat).abs()
A2 = b2.groupby(["basamak", "tahminci", "kedi", "torba"]).ae.mean().reset_index()


def l2_items(df, arms=("PCC", "ACC", "PACC", "EMQ")):
    cc = df[df.tahminci == "CC"][["kedi", "torba", "ae"]]; out = []
    for a in arms:
        x = df[df.tahminci == a][["kedi", "torba", "ae"]]; j = cc.merge(x, on=["kedi", "torba"], suffixes=("_cc", "_a")); j["d"] = j.ae_cc - j.ae_a
        out.append((a, j.groupby("kedi").d.mean(), len(j), float(j.ae_cc.mean()), float(j.ae_a.mean())))
    return out


raw = pickle.load(open(P.INTERIM / "r02" / "raw_va_f32.pkl", "rb")); LOCO = {f["cat"]: f for f in raw["loco"]}
units = []
for c in E.CATS:
    f = LOCO[c]; I = pickle.load(open(P.INTERIM / "r03" / f"inner_{c}.pkl", "rb")); tr = I["tr"].astype(int); ytr = E.Y[tr]
    oof = I["map_oof"].astype(np.float64)
    inner = dict(C=Q.conf_matrix(ytr, oof.argmax(1), E.K), Cp=Q.soft_conf_matrix(ytr, oof, E.K), prior_oof=oof.mean(0)); Pm = I["map_P_hc"]; te = f["test_idx"]
    units.append(dict(scen="S2-esl", cat=c, te=te, fn=(lambda m, Pm=Pm, inner=inner: E.base_estimates(Pm[m], inner))))
dfE = E.evaluate(units, "30dk_ge50").rename(columns={"est": "tahminci", "cat": "kedi", "bag": "torba", "AE": "ae"})[["tahminci", "kedi", "torba", "ae"]]
dfE.to_csv(T / "r07_L2_S2esl_bags.csv", index=False)
for lab_, df in (("S1-tasma", A2[A2.basamak == "S1-tasma"]), ("S1-koşum", A2[A2.basamak == "S1-kosum"]), ("S2-eşlenmiş", dfE)):
    its = l2_items(df); ccm = float(df[df.tahminci == "CC"].ae.mean())
    ref_row(lab_, "L2", "32", "Smit 68 torba (R02)", its[0][2], ccm, len(its[0][1]))
    add_family("kaynak", lab_, "L2", "32", "Smit 68 torba (R02)", its)
M = pd.DataFrame(rows); M.to_csv(T / "r07_master.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
print(M[["senaryo", "uzay", "kol", "aile", "n_torba", "n", "AE_CC", "AE_kol", "d", "lo", "hi", "p_holm", "aile_n", "kategori"]].to_string())
