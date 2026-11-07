r"""46_r07_6_analyze.py — R07 §6: (1) proportion of draws with >=20 s from every class and proportion with a missing C column: random block (r06_E3_counts.csv) vs stratified; (2) F rule: stratified arms − CC-strat; random − stratified AE difference; target-only stratified.
d = AE_first-named counterpart − AE_arm (d>0: arm better). Family = scenario x variant x level x comparison (Holm over m).
Reads: <data root>/interim/r07/T_*.parquet, _cnt, r05/r06 random-block bag parquet files, results/tables/r06_E3_counts.csv. Writes: results/tables/r07_6_counts.csv, r07_6_tests.csv, r07_6_arms.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r07lib as L

O7 = P.INTERIM / "r07"; T = P.TABLES; rd = L.rd; MS = (60, 300, 900, 1800)
NAMES = ["ladder", "S4-Smit_acik", "S4-Smit_kapali", "S3_-", "S4-Dunford_kapali", "S4-Dunford_acik"]
ST = L.cat_(*[rd(O7 / f"T_{n}.parquet") for n in NAMES]); CN = pd.concat([rd(O7 / f"T_{n}_cnt.parquet") for n in NAMES], ignore_index=True)
MI = pd.concat([rd(O7 / f"T_{n}_miss.parquet") for n in NAMES], ignore_index=True)
DU, SM = L.load_E()
LD = L.cat_(rd(P.R05 / "E_ladder_extra.parquet"), *[rd(p) for p in sorted(P.R06.glob("E1_ladder_fit_m*.parquet")) if not p.name.endswith("_miss.parquet")],
            *[rd(p) for p in sorted(P.R06.glob("E1_ladder_e3_m*.parquet")) if not p.name.endswith("_miss.parquet")]).drop_duplicates(L.KEYS, keep="last")
RB = L.cat_(DU, SM, LD)

# ---- 1. class coverage: >=20 s and missing C column
e3 = pd.read_csv(T / "r06_E3_counts.csv"); e3 = e3[e3.sinif == "HEPSI>=20"]
rows = []
for (sc, var, lv, m), g in CN.groupby(["scen", "var", "level", "m"]):
    Km = 2 if sc == "S4-Smit" or sc == "S4-Dunford" else 3
    cm = g[[f"c{k}" for k in range(Km)]].values
    # unique draws at the study level (scen,var,level,m,unit,rep)
    rows.append(dict(scen=sc, var=var, level=lv, m=m, yontem="tabakali", n_cekilis=len(g), hepsi_ge20=float((cm.min(1) >= 20).mean()), eksik_C=float((cm.min(1) == 0).mean()), n_lab_ort=float(g.n_lab.mean())))
for r in e3.itertuples():
    sc = r.scen
    rows.append(dict(scen=sc, var=("30dk_ge50" if sc in ("S1", "S2") else r.var), level=r.level, m=r.m, yontem="rastgele_blok", n_cekilis=r.n_cekilis, hepsi_ge20=r.hepsi_ge20_orani, eksik_C=r.eksik_C_orani, n_lab_ort=r.n_lab_ort))
CT = pd.DataFrame(rows); CT.to_csv(T / "r07_6_counts.csv", index=False)

# ---- 2. testler
tests = []


def fam(df, scen, var, lv, a, b, tag, ms=MS):
    pr = L.pairs(df, scen, var, lv, a, b, ms)
    if not pr: return
    f = L.family(f"R07-6|{tag}|{scen}|{var}|{lv}", [(dict(scen=scen, var=var, level=lv, karsilastirma=tag, m=m), pr[m]) for m in ms if m in pr]); tests.append(f)


ALLD = L.cat_(ST, RB)
cells = sorted(set(map(tuple, ST[["scen", "variant", "level"]].drop_duplicates().values)))
for sc, var, lv in cells:
    for arm in ("T-ACC", "PACC-hedef"):
        fam(ALLD, sc, var, lv, arm + "-tab", "CC-tab", f"{arm}-tab_vs_CC-tab")
        fam(ALLD, sc, var, lv, arm + "-tab", arm, f"{arm}_rastgele_vs_tab")
    fam(ALLD, sc, var, lv, "Yalniz-hedef-tab", "CC-tab", "Yalniz-hedef-tab_vs_CC-tab", (300, 1800))
    fam(ALLD, sc, var, lv, "Yalniz-hedef-tab", "Yalniz-hedef-esl", "Yalniz-hedef_rastgele_vs_tab", (300, 1800))
TT = pd.concat(tests, ignore_index=True); TT.to_csv(T / "r07_6_tests.csv", index=False)
arms = ALLD.groupby(["scen", "variant", "level", "arm", "m"]).AE.agg(["mean", "size"]).reset_index().rename(columns={"mean": "AE", "size": "n_satir"}); arms.to_csv(T / "r07_6_arms.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 600, "display.float_format", "{:.4f}".format)
print(CT.pivot_table(index=["scen", "var", "level", "m"], columns="yontem", values=["hepsi_ge20", "eksik_C"]).to_string())
print(TT[["scen", "var", "level", "karsilastirma", "m", "n", "AE_a", "AE_b", "d", "lo", "hi", "p_holm", "aile_n", "kategori"]].to_string())
