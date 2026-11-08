r"""59_r15_strategy_own_cc.py — R15 A1: labelling strategy (stratified vs random block) compared on gain over its OWN CC.
gain(row) = AE_CC - AE_arm, CC and arm taken from the same draw (m, rep, ind, bag). Random block: CC / arm; stratified: CC-tab / arm-tab.
Individual level: mean gain per individual and strategy; diff(i) = gain_strat(i) - gain_rand(i) (>0: stratified gains more over its own CC).
Statistics: src.stats06 (same bootstrap, exact sign-flip, Holm, |d|>=0.01 rule as 46_r07_6_analyze.py). No model is trained.
Reads: interim/r07/T_*.parquet, r05/r06 random-block bag parquets (as 46_r07_6_analyze.py), results/tables/{r07_6_tests,r07_6_arms,r06_F_tests,r07_master}.csv
Writes: results/tables/r15_strategy_vs_own_cc.csv (+ r15_strategy_gate.csv)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r07lib as L, stats06 as S6

O7 = P.INTERIM / "r07"; T = P.TABLES; rd = L.rd; MS = (60, 300, 900, 1800)
NAMES = ["ladder", "S4-Smit_acik", "S4-Smit_kapali", "S3_-", "S4-Dunford_kapali", "S4-Dunford_acik"]
ST = L.cat_(*[rd(O7 / f"T_{n}.parquet") for n in NAMES])
DU, SM = L.load_E()
LD = L.cat_(rd(P.R05 / "E_ladder_extra.parquet"), *[rd(p) for p in sorted(P.R06.glob("E1_ladder_fit_m*.parquet")) if not p.name.endswith("_miss.parquet")],
            *[rd(p) for p in sorted(P.R06.glob("E1_ladder_e3_m*.parquet")) if not p.name.endswith("_miss.parquet")]).drop_duplicates(L.KEYS, keep="last")
ALLD = L.cat_(ST, L.cat_(DU, SM, LD))

# (output arm, random arm, stratified arm, CC_strat, m set)
PAIRS = [("T-ACC", "T-ACC", "T-ACC-tab", MS), ("PACC-hedef", "PACC-hedef", "PACC-hedef-tab", MS), ("Yalniz-hedef", "Yalniz-hedef-esl", "Yalniz-hedef-tab", (300, 1800))]
T6 = pd.read_csv(T / "r07_6_tests.csv"); T6 = T6[T6.karsilastirma.str.endswith("_rastgele_vs_tab")]
ARMS = pd.read_csv(T / "r07_6_arms.csv"); RF = pd.read_csv(T / "r06_F_tests.csv"); RM = pd.read_csv(T / "r07_master.csv")
STAB = pd.read_csv(T / "r07_6_tests.csv")


def gains(d, scen, var, lv, ccarm, arm):
    """Row-level gain AE_CC - AE_arm per (m, rep, ind, bag). Returns (joined frame, number of arm rows with no CC partner)."""
    d = d[(d.scen == scen) & (d.variant == var)]
    cc = d[d.arm == ccarm]; cc = cc if ccarm == "CC" else cc[cc.level == lv]
    a = d[(d.arm == arm) & (d.level == lv)]
    cc = cc.drop_duplicates(["m", "rep", "ind", "bag"]); a = a.drop_duplicates(["m", "rep", "ind", "bag"])
    j = a[["m", "rep", "ind", "bag", "AE"]].merge(cc[["m", "rep", "ind", "bag", "AE"]], on=["m", "rep", "ind", "bag"], how="left", suffixes=("_arm", "_cc"))
    miss = int(j.AE_cc.isna().sum()); j = j.dropna(subset=["AE_cc"]); j["g"] = j.AE_cc - j.AE_arm
    return j, miss


MASTER_KEY = {("S1", "30dk_ge50"): "S1", ("S2", "30dk_ge50"): "S2", ("S3", "-"): "S3 [keşifsel]", ("S4-Smit", "acik"): "S4-Smit-açık", ("S4-Smit", "kapali"): "S4-Smit-kapalı",
              ("S4-Dunford", "acik"): "S4-Dunford-açık [keşifsel]", ("S4-Dunford", "kapali"): "S4-Dunford-kapalı [keşifsel]"}


def stored_rand(scen, var, lv, armr, m):
    x = RF[(RF.scen == scen) & (RF["var"] == var) & (RF.level == lv) & (RF.kol_a == armr) & (RF.kol_b == "CC") & (RF.m == m)]
    if len(x): return float(x.d.iloc[0]), "r06_F_tests"
    # r07_master.csv holds T-ACC random block only for m=1800; its level is birey (Smit, S1, S2) or calisma (Dunford)
    if armr == "T-ACC" and m == 1800 and lv == ("calisma" if scen in ("S3", "S4-Dunford") else "birey"):
        y = RM[(RM.senaryo == MASTER_KEY[(scen, var)]) & RM.kol.str.startswith("T-ACC")]
        if len(y): return float(y.d.iloc[0]), "r07_master"
    return np.nan, ""


cells = sorted(set(map(tuple, T6[["scen", "var", "level"]].drop_duplicates().values)))
out, gate, problems = [], [], []
for sc, var, lv in cells:
    for arm, ar, at, ms in PAIRS:
        tt = T6[(T6.scen == sc) & (T6["var"] == var) & (T6.level == lv) & (T6.karsilastirma == f"{arm}_rastgele_vs_tab")]
        if tt.empty: continue
        jr, mr = gains(ALLD, sc, var, lv, "CC", ar); js, ms_ = gains(ALLD, sc, var, lv, "CC-tab", at)
        if mr or ms_: problems.append(f"{sc}|{var}|{lv}|{arm}: CC partner missing rand={mr} strat={ms_}")
        items = []; fam_ms = [int(m) for m in tt.m]
        for m in fam_ms:
            gr = jr[jr.m == m]; gs = js[js.m == m]
            ir = gr.groupby("ind").g.mean(); is_ = gs.groupby("ind").g.mean()
            ix = ir.index.intersection(is_.index); diff = (is_.loc[ix] - ir.loc[ix])
            # gate (i): stratified gain vs stored <arm>-tab_vs_CC-tab ; (ii): random gain vs stored random-arm vs CC
            s_ = STAB[(STAB.scen == sc) & (STAB["var"] == var) & (STAB.level == lv) & (STAB.karsilastirma == f"{arm}-tab_vs_CC-tab") & (STAB.m == m)]
            d_t = float(s_.d.iloc[0]) if len(s_) else np.nan; d_r, src_r = stored_rand(sc, var, lv, ar, m)
            e1 = abs(is_.mean() - d_t) if not np.isnan(d_t) else np.nan; e2 = abs(ir.mean() - d_r) if not np.isnan(d_r) else np.nan
            ok = bool(e1 < 1e-9 and e2 < 1e-9)
            gate.append(dict(scen=sc, var=var, level=lv, arm=arm, m=m, gain_tab=is_.mean(), d_tab_stored=d_t, err_tab=e1, gain_rand=ir.mean(), d_rand_stored=d_r, src_rand=src_r, err_rand=e2, gecti=int(ok)))
            ar_ = ARMS[(ARMS.scen == sc) & (ARMS.variant == var) & (ARMS.level == lv) & (ARMS.m == m)]
            def A(a_):
                x = ar_[ar_.arm == a_]
                if a_ == "CC" and not len(x): x = ARMS[(ARMS.scen == sc) & (ARMS.variant == var) & (ARMS.m == m) & (ARMS.arm == "CC")]  # CC takes no level filter (as in L.pairs)
                return float(x.AE.iloc[0]) if len(x) else np.nan
            lab = dict(scen=sc, var=var, level=lv, arm=arm, m=m, n_bag=len(gs), AE_CC_rand=A("CC"), AE_arm_rand=A(ar), AE_CC_strat=A("CC-tab"), AE_arm_strat=A(at))
            items.append((lab, (diff, ok)))
        # Holm over the m values of this comparison (same as R07-6); only gate-passing cells are written
        rows = [dict(**lab, **S6.stat(dd.values)) for lab, (dd, ok) in items]
        df = S6.family(rows)
        df["_ok"] = [ok for _, (_, ok) in items]
        out.append(df)
R = pd.concat(out, ignore_index=True); G = pd.DataFrame(gate)
G.to_csv(T / "r15_strategy_gate.csv", index=False)
print("gate passed:", int(G.gecti.sum()), "/", len(G)); print("problems:", problems)
print(G[G.gecti == 0].to_string())
R = R[R._ok].drop(columns="_ok")
cols = "scen var level arm m n n_bag AE_CC_rand AE_arm_rand AE_CC_strat AE_arm_strat d lo hi p pmin p_holm aile_n kategori".split()
cols = ["scen", "var", "level", "arm", "m", "n", "n_bag", "AE_CC_rand", "AE_arm_rand", "AE_CC_strat", "AE_arm_strat", "d", "lo", "hi", "p", "pmin", "p_holm", "aile_n", "kategori"]
R[cols].to_csv(T / "r15_strategy_vs_own_cc.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 200, "display.float_format", "{:.4f}".format); print(R[cols].to_string())
