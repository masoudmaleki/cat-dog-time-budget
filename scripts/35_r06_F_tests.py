r"""35_r06_F_tests.py — R06 E1/E2/E3 results and reclassification with the F decision rule. Family = hypothesis x scenario (x variant x level); Holm within the family (over m or over arms).
Reads: r05/*.parquet (bag-level), r06/B_v2_bags.parquet, E1_*.parquet. Writes: results/tables/r06_F_tests.csv, r06_E_arms.csv (arm AE summary), r06_bag_level.csv
Positive d = the first-named arm (a) is better (AE_b - AE_a). Pairing key (m, repeat, individual, bag); mean per individual; exact sign-flip test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, stats06 as S6

T = P.TABLES; O5 = P.R05; O6 = P.R06
KEYS = ["scen", "variant", "level", "arm", "m", "rep", "ind", "bag"]


def rd(p): return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def cat_(*ps):
    xs = [x for x in ps if len(x)]
    return pd.concat(xs, ignore_index=True) if xs else pd.DataFrame(columns=KEYS + ["AE"])


# ---- datasets
v2 = rd(O6 / "B_v2_bags.parquet")
src5 = rd(O5 / "r05_sources_bags.parquet"); lad5 = rd(O5 / "r05_ladder_bags.parquet")
DU = cat_(*[rd(O6 / f"E1_{s}_{v}.parquet") for s, v in (("S3", "-"), ("S4-Dunford", "kapali"), ("S4-Dunford", "acik"))])
SM = cat_(*[rd(O5 / f"E_S4-Smit_{v}.parquet") for v in ("acik", "kapali")], *[rd(O6 / f"E1_S4-Smit_{v}.parquet") for v in ("acik", "kapali")]).drop_duplicates(KEYS, keep="last")
LD = cat_(rd(O5 / "E_ladder_extra.parquet"), *[rd(p) for p in sorted(O6.glob("E1_ladder_fit_m*.parquet")) if not p.name.endswith("_miss.parquet")],
          *[rd(p) for p in sorted(O6.glob("E1_ladder_e3_m*.parquet")) if not p.name.endswith("_miss.parquet")]).drop_duplicates(KEYS, keep="last")
tests = []


def pairs(df, scen, var, level, a, b, m=None):
    """a, b arm names. No level filter for CC (same key). Returns a dict, by m, of the per-individual mean d (AE_b - AE_a)."""
    d = df[(df.scen == scen) & (df.variant == var)]
    def sel(arm): return d[(d.arm == arm)] if arm == "CC" else d[(d.arm == arm) & (d.level == level)]
    A_ = sel(a).drop_duplicates(["m", "rep", "ind", "bag"]); B_ = sel(b).drop_duplicates(["m", "rep", "ind", "bag"])
    j = A_[["m", "rep", "ind", "bag", "AE"]].merge(B_[["m", "rep", "ind", "bag", "AE"]], on=["m", "rep", "ind", "bag"], suffixes=("_a", "_b"))
    j["d"] = j.AE_b - j.AE_a
    out = {}
    for mm, g in j.groupby("m"):
        if m is not None and mm not in m: continue
        out[int(mm)] = (g.groupby("ind").d.mean(), len(g), float(g.AE_a.mean()), float(g.AE_b.mean()))
    return out


def run_family(name, items):
    """items: list of (labels dict, pairs result (dind, n_bag, AE_a, AE_b)). Holm within the family."""
    rows = []
    for lab_, (dind, nb, aa, ab) in items:
        s = S6.stat(dind.values); rows.append(dict(**lab_, n_bag=nb, AE_a=aa, AE_b=ab, **s))
    df = S6.family(rows)
    if len(df): df.insert(0, "aile", name); tests.append(df)
    return df


def over_m(df, scen, var, level, a, b, ms, fam):
    pr = pairs(df, scen, var, level, a, b, ms)
    items = [(dict(scen=scen, var=var, level=level, kol_a=a, kol_b=b, m=m), pr[m]) for m in ms if m in pr]
    return run_family(f"{fam}|{scen}|{var}|{level}|{a}-{b}", items)


MS4 = (60, 300, 900, 1800); MS2 = (300, 1800)
# ---------------------------------------------------------------- F1: H5 (source-C arms ACC/PACC/EMQ - CC), 3 tests per scenario/variant
def h5(df, scen, var, tag):
    g = df[(df.scen == scen) & (df.variant == var) & (df.m == 0)]
    cc = g[g.arm == "CC"]; items = []
    for e in ("ACC", "PACC", "EMQ"):
        x = g[g.arm == e]; j = x[["ind", "bag", "AE"]].merge(cc[["ind", "bag", "AE"]], on=["ind", "bag"], suffixes=("_a", "_b")); j["d"] = j.AE_b - j.AE_a
        items.append((dict(scen=scen, var=var + tag, level="-", kol_a=e, kol_b="CC", m=0), (j.groupby("ind").d.mean(), len(j), float(j.AE_a.mean()), float(j.AE_b.mean()))))
    run_family(f"F1-H5|{scen}|{var}{tag}", items)


for sc, var in (("S3", "kedi"), ("S4-Dunford", "kapali"), ("S4-Dunford", "acik")):
    h5(v2, sc, var, "|v2")
    h5(src5, sc, var, "|v1")
for var in ("30dk_ge50-acik", "30dk_ge50-kapali"): h5(src5, "S4-Smit", var, "")

# ---------------------------------------------------------------- E1/E2/E3 families (Dunford)
for sc, var, ms in (("S3", "-", MS4), ("S4-Dunford", "acik", MS4), ("S4-Dunford", "kapali", MS4)):
    lv = "calisma"
    if DU.empty: break
    over_m(DU, sc, var, lv, "Yeniden-egitim", "CC", ms, "F2-RT")
    over_m(DU, sc, var, lv, "Yeniden-egitim-agirlikli", "CC", ms, "F2-RTw")
    over_m(DU, sc, var, lv, "Yeniden-egitim-agirlikli", "Yeniden-egitim", ms, "F3-w-u")
    over_m(DU, sc, var, lv, "Yeniden-egitim-agirlikli", "Yalniz-hedef-esl", ms, "E1-w-YH")
    over_m(DU, sc, var, lv, "Yalniz-hedef-esl", "CC", ms, "E2-YH-CC")
    over_m(DU, sc, var, lv, "PACC-hedef-g", "CC", ms, "E3-g-CC")
    over_m(DU, sc, var, lv, "PACC-hedef", "CC", ms, "H6-PACChedef-CC")
    over_m(DU, sc, var, lv, "PACC-buzme", "CC", ms, "H6-PACCbuzme-CC")
# ---------------------------------------------------------------- Smit S4
for var in ("acik", "kapali"):
    for lv in ("birey", "calisma"):
        if SM.empty: break
        sub = SM[SM.variant == var]
        if var == "acik":
            over_m(SM, "S4-Smit", var, lv, "Yeniden-egitim", "CC", MS4, "F2-RT")
            over_m(SM, "S4-Smit", var, lv, "Yeniden-egitim-agirlikli", "CC", MS4, "F2-RTw")
            over_m(SM, "S4-Smit", var, lv, "Yeniden-egitim-agirlikli", "Yeniden-egitim", MS4, "F3-w-u")
            over_m(SM, "S4-Smit", var, lv, "Yeniden-egitim-agirlikli", "Yalniz-hedef-esl", MS4, "E1-w-YH")
        over_m(SM, "S4-Smit", var, lv, "Yalniz-hedef-esl", "CC", MS4, "E2-YH-CC")
        over_m(SM, "S4-Smit", var, lv, "PACC-hedef-g", "CC", MS4, "E3-g-CC")
        over_m(SM, "S4-Smit", var, lv, "PACC-hedef", "CC", MS4, "H6-PACChedef-CC")
        over_m(SM, "S4-Smit", var, lv, "PACC-buzme", "CC", MS4, "H6-PACCbuzme-CC")
# ---------------------------------------------------------------- ladder (S1, S2) individual
for sc in ("S1", "S2"):
    if LD.empty: break
    ms = tuple(sorted(set(LD[LD.arm == "Yeniden-egitim-agirlikli"].m.unique()))) or (1800,)
    over_m(LD, sc, "30dk_ge50", "birey", "Yeniden-egitim", "CC", MS2, "F2-RT")
    over_m(LD, sc, "30dk_ge50", "birey", "Yeniden-egitim-agirlikli", "CC", MS2, "F2-RTw")
    over_m(LD, sc, "30dk_ge50", "birey", "Yeniden-egitim-agirlikli", "Yeniden-egitim", MS2, "F3-w-u")
    over_m(LD, sc, "30dk_ge50", "birey", "PACC-hedef-g", "CC", MS2, "E3-g-CC")
    over_m(LD, sc, "30dk_ge50", "birey", "PACC-hedef", "CC", MS2, "H6-PACChedef-CC")

# ---------------------------------------------------------------- R05 H8 reclassification: retrain - best arm (1-3), same m (300, 1800); the best arm is chosen by AE (as in R05)
def h8(df, scen, var, lv, tag=""):
    d = df[(df.scen == scen) & (df.variant == var)]
    items = []
    for m in MS2:
        c = d[(d.level == lv) & d.arm.isin(["T-ACC", "PACC-hedef", "PACC-buzme"]) & (d.m == m)]
        if c.empty: continue
        best = c.groupby("arm").AE.mean().idxmin(); pr = pairs(d, scen, var, lv, "Yeniden-egitim", best, [m])
        if m in pr: items.append((dict(scen=scen, var=var, level=lv, kol_a="Yeniden-egitim", kol_b=best, m=m), pr[m]))
    run_family(f"F2-H8|{scen}|{var}|{lv}{tag}", items)


for sc, var, lv, df in [("S3", "-", "calisma", DU), ("S4-Dunford", "acik", "calisma", DU), ("S4-Dunford", "kapali", "calisma", DU), ("S4-Smit", "acik", "birey", SM), ("S4-Smit", "acik", "calisma", SM),
                        ("S4-Smit", "kapali", "birey", SM), ("S4-Smit", "kapali", "calisma", SM), ("S1", "30dk_ge50", "birey", LD), ("S2", "30dk_ge50", "birey", LD)]:
    if len(df): h8(df, sc, var, lv)

DU1 = cat_(*[rd(O5 / f"E_{s}_{v}.parquet") for s, v in (("S3", "-"), ("S4-Dunford", "kapali"), ("S4-Dunford", "acik"))])
for sc, var in (("S3", "-"), ("S4-Dunford", "acik"), ("S4-Dunford", "kapali")):
    if len(DU1): h8(DU1, sc, var, "calisma", "|v1")

TT = pd.concat(tests, ignore_index=True); TT.to_csv(T / "r06_F_tests.csv", index=False)
# ---- arm summary (AE, per m and level): E tables
summ = []
for nm, df in (("Dunford", DU), ("Smit", SM), ("merdiven", LD)):
    if df.empty: continue
    g = df.groupby(["scen", "variant", "level", "arm", "m"]).AE.agg(["mean", "size"]).reset_index().rename(columns={"mean": "AE", "size": "n_satir"}); g["kume"] = nm; summ.append(g)
pd.concat(summ).to_csv(T / "r06_E_arms.csv", index=False)
pd.set_option("display.width", 260, "display.max_columns", 30, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
print(TT.drop(columns=["pmin"]).to_string())
