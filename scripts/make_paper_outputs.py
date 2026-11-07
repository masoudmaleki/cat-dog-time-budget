r"""make_paper_outputs.py — R09 §C, §D, §E: produces the paper tables, figures and the number check from the stored intermediates with one command. NO model training.
Run:  python -I scripts/make_paper_outputs.py
Reads (read-only): results/tables/r0{2..9}_*.csv|json, <data root>/interim/{r05,r06}/*.parquet|pkl (for fig4), Smit/dog/Dunford feature tables (label counts only, Table 1).
Writes: results/tables/paper/*, results/figures/paper/*  (does not overwrite older figures). Rule 2: the fixed names given by the task were used; the script writes only to these new directories.
"""
import sys, json, time, shutil, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import paths as P
from src import r05lib as R, smit_data as D, stats06 as S6

sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time(); T = P.TABLES; TP = T / "paper"; FP = P.FIGURES / "paper"; TP.mkdir(parents=True, exist_ok=True); FP.mkdir(parents=True, exist_ok=True)
EN = {"KAZANIR": "Wins", "KAYBEDER": "Loses", "ESDEGER": "Equivalent", "BELIRSIZ": "Inconclusive", "REFERANS": "reference"}
rd = lambda n: pd.read_csv(T / n)


def md(df, path, title=None, note=None):
    cols = list(df.columns); L = []
    if title: L.append(f"**{title}**\n")
    L.append("| " + " | ".join(map(str, cols)) + " |"); L.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in df.itertuples(index=False): L.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    if note: L.append("\n" + note)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def save(df, stem, title, note=None):
    df.to_csv(TP / f"{stem}.csv", index=False, encoding="utf-8"); md(df, TP / f"{stem}.md", title, note)


def fmt(d, lo, hi, k):
    return "" if pd.isna(d) else f"{d:+.4f} [{lo:+.4f}; {hi:+.4f}] {EN.get(k, k)}"


# ================================================================== data
M = rd("r07_master.csv"); R8 = rd("r08_tests.csv"); R8m = R8[R8.aile == "birincil"]; R8main = rd("r08_main.csv")
T7_4 = rd("r07_4_tests.csv"); T7_5 = rd("r07_5_tests.csv"); C7_6 = rd("r07_6_counts.csv"); T7_6 = rd("r07_6_tests.csv"); F6 = rd("r06_F_tests.csv"); YH = rd("r07_1_yh.csv")

# ================================================================== TABLE 1 — datasets
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); ok = y1 >= 0
sm1 = np.bincount(y1[ok], minlength=3); sm2 = np.bincount(lab.y2.values[lab.y2.values >= 0], minlength=7)
NK = pd.read_parquet(P.DOG_FEAT); DOGL2 = ["Lying chest", "Sitting", "Standing", "Walking", "Trotting"]
dy1 = np.where(NK.y1.values == 2, -1, NK.y1.values); dy2 = np.array([DOGL2.index(b) if b in DOGL2 else -1 for b in NK.beh.values])
dg1 = np.bincount(dy1[dy1 >= 0], minlength=2); dg2 = np.bincount(dy2[dy2 >= 0], minlength=5)
dun, _ = R.load_dunford(v2=True); du1 = np.bincount(dun.y1.values, minlength=3)
n_dog1 = NK.DogID[dy1 >= 0].nunique(); n_dog2 = NK.DogID[dy2 >= 0].nunique()
t1 = pd.DataFrame([
    dict(Dataset="Smit 2023", Individuals_analysed=f"{lab.Cat_id.nunique()} cats", Sensor="ActiGraph wGT3X-BT", Rate_Hz=30, Placement="collar (ventral) + harness (left scapula)", Label_spaces="L1 (3 classes), L2 (7 classes)",
         Seconds_per_class_L1="Inactive %d; Locomotion %d; Care %d" % tuple(sm1), Seconds_per_class_L2="; ".join(f"{n} {v}" for n, v in zip(D.L2, sm2)), Source="Sensors 23:7165; FigShare doi:10.6084/m9.figshare.23605842"),
    dict(Dataset="Dunford 2024 [exploratory]", Individuals_analysed=f"{dun.cat.nunique()} cats (8 with >=60-s bags; v2, block repeats removed)", Sensor="Daily Diary", Rate_Hz=40, Placement="collar (under chin)", Label_spaces="L1 (3 classes)",
         Seconds_per_class_L1="Inactive %d; Locomotion %d; Care %d" % tuple(du1), Seconds_per_class_L2="–", Source="Ecol Evol 14:e11380; Dryad doi:10.5061/dryad.q2bvq83sx"),
    dict(Dataset="Vehkaoja 2022 (dogs)", Individuals_analysed=f"{n_dog1} dogs (K-L1), {n_dog2} dogs (K-L2); 44 in within-dog evaluation", Sensor="ActiGraph GT9X Link", Rate_Hz=100, Placement="collar (ventral) + harness (back)", Label_spaces="K-L1 (2 classes), K-L2 (5 classes)",
         Seconds_per_class_L1="Inactive %d; Locomotion %d (no Care class)" % tuple(dg1), Seconds_per_class_L2="; ".join(f"{n} {v}" for n, v in zip(DOGL2, dg2)), Source="Data in Brief 40:107822; Mendeley Data vxhx934tbn"),
])
save(t1, "table1_datasets", "Table 1. Datasets", "Seconds are 1-s windows with a single mapped label (labels outside the common space are excluded). Collar features (ANeck, D17) are used for dogs.")

# ================================================================== TABLE 2 — transfer scenarios
def nb(sc, kume=None, kol="CC"):
    g = M[(M.senaryo == sc) & (M.kol == kol)]; return (int(g.n_torba.iloc[0]), int(g.n.iloc[0])) if len(g) else ("", "")
rows2 = []
def r2(code, name, train, test, feat, kume, nbag, n, note=""): rows2.append(dict(Code=code, Scenario=name, Training=train, Test=test, Feature_set=feat, Bag_set=kume, N_bags=nbag, n_individuals=n, Note=note))
def NB(sc): return nb(sc)
r2("S0", "random seconds (leaky reference)", "Smit collar, random 70 % of seconds", "random 30 % of seconds", "D17", "Smit 30-min bags", *NB("S0 (sızıntılı)"), "leaky by definition; not an evaluation of transfer")
r2("S1", "new individual", "Smit collar, 11 cats", "held-out cat", "D17", "Smit 30-min bags (>=50 % coverage)", *NB("S1"), "also natural 2-min (1083 bags) and APP 120-s (600 bags)")
r2("S1-L2", "new individual, 7-class space", "Smit collar, 11 cats", "held-out cat", "32 features", "Smit 30-min bags", *NB("S1-tasma"))
r2("S2", "harness → collar", "Smit harness, 11 cats", "held-out cat (collar)", "D17", "Smit 30-min bags", *NB("S2"))
r2("S3", "cross-study (Smit → Dunford) [exploratory]", "Smit collar, 12 cats", "Dunford cats", "D17", "Dunford cat bags (N>=60)", *NB("S3 [keşifsel]"))
r2("S4-Smit-open", "dog → cat, open set", "45 dogs, collar (2 classes)", "Smit cats (3 classes; Care unseen)", "D17", "Smit 30-min bags", *NB("S4-Smit-açık"))
r2("S4-Smit-closed", "dog → cat, closed set", "45 dogs, collar (2 classes)", "Smit cats, Care seconds removed", "D17", "Smit 30-min bags", *NB("S4-Smit-kapalı"))
r2("S4-Dunford-open / closed", "dog → cat [exploratory]", "45 dogs, collar", "Dunford cats", "D17", "Dunford cat bags (N>=60)", f"{NB('S4-Dunford-açık [keşifsel]')[0]} / {NB('S4-Dunford-kapalı [keşifsel]')[0]}", f"{NB('S4-Dunford-açık [keşifsel]')[1]} / {NB('S4-Dunford-kapalı [keşifsel]')[1]}")
r2("K-L1, K-L2", "within-dog (new dog)", "dogs, 5-fold GroupKFold", "held-out dogs", "32 features (D17 as sensitivity)", "dog × recording (61 bags) and natural 2-min (130 bags)", "61 / 130", "44 / 35")
r2("R-D", "Dunford leave-one-cat-out [exploratory]", "Dunford cats", "held-out cat", "D17", "Dunford cat bags (N>=60)", *NB("R-D [keşifsel]"))
r2("S1-closed", "new individual, Care removed", "Smit collar, 11 cats", "held-out cat", "D17", "Smit 30-min bags", *NB("S1-kapalı"))
save(pd.DataFrame(rows2), "table2_transfer_scenarios", "Table 2. Transfer scenarios", "Bag sets follow the primary bag-set rule of the project; Dunford rows are exploratory.")

# ================================================================== TABLE 3 — consolidated results
SRCARMS = ["N-hedef", "PCC", "ACC", "PACC", "EMQ"]; ANAME = {"N-hedef": "N-target", "PCC": "PCC", "ACC": "ACC", "PACC": "PACC", "EMQ": "EMQ", "DM": "DM", "DR-APP-q": "DR-APP", "DR-dogal-q": "DR-natural"}
R8MAP = {"S1": ("S1", "30dk"), "S1-L2 (collar)": ("S1-L2", "30dk"), "S2": ("S2", "30dk"), "S4-Smit-kapalı": ("S4-Smit-kapali", "30dk"), "S3 [keşifsel]": ("S3", "kedi"), "S4-Dunford-kapalı [keşifsel]": ("S4-Dunford-kapali", "kedi")}
rows3 = []; long3 = []
def add3(label, kume, n, nbag, aecc, cells, expl=False):
    r = dict(Scenario=label + (" [exploratory]" if expl and "keşifsel" not in label else ""), Bag_set=kume, n=n, N_bags=nbag, AE_CC=round(aecc, 4))
    for a in list(ANAME.values()):
        c = cells.get(a); r[f"AE_{a}"] = round(c[4], 4) if c else "–"; r[f"d_{a} [95% CI] category"] = fmt(*c[:4]) if c else "–"
        if c: long3.append(dict(Scenario=r["Scenario"], Bag_set=kume, Arm=a, AE_arm=c[4], d=c[0], lo=c[1], hi=c[2], category=c[3], n=n, N_bags=nbag))
    rows3.append(r)
EXP = lambda s: "keşifsel" in s
LBL = {"S0 (sızıntılı)": "S0 (leaky)", "S1": "S1", "S2": "S2", "S3 [keşifsel]": "S3 [exploratory]", "S4-Smit-açık": "S4-Smit-open", "S4-Smit-kapalı": "S4-Smit-closed",
       "S4-Dunford-açık [keşifsel]": "S4-Dunford-open [exploratory]", "S4-Dunford-kapalı [keşifsel]": "S4-Dunford-closed [exploratory]", "R-D [keşifsel]": "R-D [exploratory]", "S1-kapalı": "S1-closed",
       "S1-tasma": "S1-L2 (collar)", "S1-koşum": "S1-L2 (harness)", "S2-eşlenmiş": "S2-L2 (matched)"}
for sc in M.senaryo.unique():
    g = M[(M.senaryo == sc) & (M.aile == "kaynak")]; ref = M[(M.senaryo == sc) & (M.kol == "CC")].iloc[0]; cells = {ANAME[r.kol]: (r.d, r.lo, r.hi, r.kategori, r.AE_kol) for r in g.itertuples() if r.kol in ANAME}
    lb = LBL[sc]
    if lb in R8MAP:
        a8 = R8m[(R8m.senaryo == R8MAP[lb][0]) & (R8m.torba_turu == R8MAP[lb][1])]
        for r in a8.itertuples(): cells[ANAME[r.kol]] = (r.d, r.lo, r.hi, r.kategori, r.AE_kol)
    add3(lb, ref.torba_kumesi.replace("Smit 68 torba", "Smit 30-min bags (68)").replace("Dunford v2 kedi torbası, N>=60", "Dunford cat bags (N>=60)").replace(" (R02)", ""), int(ref.n), int(ref.n_torba), ref.AE_CC, cells)
# S1 extra bag types (R07 §4 RF + R08)
for var, nm in (("dogal2dk", "S1 natural 2-min"), ("APP120", "S1 APP 120-s (bridge)")):
    g = T7_4[(T7_4.clf == "RF") & (T7_4.scen == "S1-kopru") & (T7_4.variant == var)]; cells = {r.kol: (r.d, r.lo, r.hi, r.kategori, r.AE_kol) for r in g.itertuples() if r.kol in ANAME}
    a8 = R8m[(R8m.senaryo == "S1") & (R8m.torba_turu == ("2dk" if var == "dogal2dk" else "APP120"))]
    for r in a8.itertuples(): cells[ANAME[r.kol]] = (r.d, r.lo, r.hi, r.kategori, r.AE_kol)
    mm = R8main[(R8main.senaryo == "S1") & (R8main.torba_turu == ("2dk" if var == "dogal2dk" else "APP120"))].iloc[0]
    add3(nm, "natural 2-min bags" if var == "dogal2dk" else "APP 120-s draws", int(mm.n), int(mm.n_torba), mm.AE_CC, cells)
# dogs (feat 32)
BTN = {"kayit": "dog × recording (N>=60)", "dogal2": "natural 2-min (N>=60)", "APP120": "APP 120-s"}
for sp in ("K-L1", "K-L2"):
    for bt in ("kayit", "dogal2", "APP120"):
        g = T7_5[(T7_5.scen == "Kopek-ici") & (T7_5.space == sp) & (T7_5.feat.astype(str) == "32") & (T7_5.btype == bt)]
        if not len(g): continue
        cells = {r.kol: (r.d, r.lo, r.hi, r.kategori, r.AE_kol) for r in g.itertuples() if r.kol in ANAME}
        if bt in ("kayit", "dogal2"):
            for r in R8m[(R8m.senaryo == sp) & (R8m.torba_turu == bt)].itertuples(): cells[ANAME[r.kol]] = (r.d, r.lo, r.hi, r.kategori, r.AE_kol)
        add3(f"{sp} within-dog", BTN[bt], int(g.n.iloc[0]), int(g.n_torba.iloc[0]), float(g.AE_CC.iloc[0]), cells)
T3 = pd.DataFrame(rows3)
note3 = ("d = paired AE difference, CC minus arm (d > 0: arm better than CC); 95 % CI = individual-level bootstrap (1000 reps); category from exact sign-flip test (Holm within family) with practical threshold |d| >= 0.01. "
         "Families: R07 source arms (4–5 tests per scenario); R08 DM/DR (3 tests per cell). AE_CC is the bag mean. Rows marked [exploratory] rely on Dunford data. 'N-target' = mean budget of the other individuals. "
         "AE: mean over bags. d: mean over individuals of the paired difference between individual-level bag means; it therefore differs from AE_CC − AE_arm.")
save(T3, "table3_consolidated_results", "Table 3. Consolidated results: CC error and per-arm differences", note3); pd.DataFrame(long3).to_csv(TP / "table3_consolidated_results_long.csv", index=False)

# ================================================================== TABLE 4 — target-labelled arms
SCN4 = {"S1": "S1 (individual-level)", "S2": "S2 (individual-level)", "S3": "S3 (study-level) [exploratory]", "S4-Smit|acik|birey": "S4-Smit-open (individual-level)", "S4-Smit|acik|calisma": "S4-Smit-open (study-level)",
        "S4-Smit|kapali|birey": "S4-Smit-closed (individual-level)", "S4-Smit|kapali|calisma": "S4-Smit-closed (study-level)", "S4-Dunford|acik|calisma": "S4-Dunford-open (study-level) [exploratory]",
        "S4-Dunford|kapali|calisma": "S4-Dunford-closed (study-level) [exploratory]"}
def key4(scen, var, level):
    if scen in ("S1", "S2"): return scen
    if scen == "S3": return "S3"
    return f"{scen}|{var}|{level}"
ARM4 = {"T-ACC": "T-ACC", "PACC-hedef": "PACC-target", "PACC-buzme": "shrinkage", "Yeniden-egitim": "retraining", "Yeniden-egitim-agirlikli": "retraining (weighted)", "Yalniz-hedef-esl": "target-only"}
rows4 = []
def cnt(scen, var, level, m, yont):
    c = C7_6[(C7_6.scen == scen) & (C7_6["var"] == var) & (C7_6.level == level) & (C7_6.m == m) & (C7_6.yontem == yont)]
    return (float(c.hepsi_ge20.iloc[0]) if len(c) else np.nan, float(c.eksik_C.iloc[0]) if len(c) else np.nan)
# rastgele blok: r06_F_tests (m = 300, 1800) + T-ACC (m = 1800, r07_master)
for r in F6[(F6.m.isin([300, 1800])) & (F6.kol_a.isin(ARM4)) & (F6.kol_b == "CC")].itertuples():
    if r.kol_a == "PACC-hedef-g": continue
    k = key4(r.scen, r.var, r.level); lvl = "individual" if r.level == "birey" or r.scen in ("S1", "S2") else "study"
    f20, ec = cnt(r.scen, r.var, r.level, r.m, "rastgele_blok")
    rows4.append(dict(Scenario=SCN4.get(k, k), Level=lvl, Sampling="random block", Arm=ARM4[r.kol_a], m=int(r.m), AE_arm=r.AE_a, AE_CC=r.AE_b, d=r.d, lo=r.lo, hi=r.hi, Category=EN[r.kategori], Holm_p=r.p_holm, family_tests=int(r.aile_n), n=int(r.n),
                      Frac_draws_all_classes_ge20s=f20, Frac_draws_missing_C_column=ec, Source="r06_F_tests.csv"))
for r in M[(M.aile == "hedef-m1800") & M.kol.str.startswith("T-ACC")].itertuples():
    scn = {"S1": ("S1", "30dk_ge50", "birey"), "S2": ("S2", "30dk_ge50", "birey"), "S3 [keşifsel]": ("S3", "-", "calisma"), "S4-Smit-açık": ("S4-Smit", "acik", "birey"), "S4-Smit-kapalı": ("S4-Smit", "kapali", "birey"),
           "S4-Dunford-açık [keşifsel]": ("S4-Dunford", "acik", "calisma"), "S4-Dunford-kapalı [keşifsel]": ("S4-Dunford", "kapali", "calisma")}[r.senaryo]
    k = key4(*scn); f20, ec = cnt(scn[0], scn[1], scn[2], 1800, "rastgele_blok")
    rows4.append(dict(Scenario=SCN4.get(k, k), Level="individual" if scn[2] == "birey" else "study", Sampling="random block", Arm="T-ACC", m=1800, AE_arm=r.AE_kol, AE_CC=r.AE_CC, d=r.d, lo=r.lo, hi=r.hi, Category=EN[r.kategori],
                      Holm_p=r.p_holm, family_tests=int(r.aile_n), n=int(r.n), Frac_draws_all_classes_ge20s=f20, Frac_draws_missing_C_column=ec, Source="r07_master.csv"))
# stratified vs CC-strat (r07_6_tests)
for r in T7_6[(T7_6.m.isin([300, 1800])) & T7_6.karsilastirma.str.endswith("_vs_CC-tab")].itertuples():
    arm = r.karsilastirma.replace("_vs_CC-tab", "").replace("-tab", ""); k = key4(r.scen, r["var"] if False else getattr(r, "var"), r.level)
    f20, ec = cnt(r.scen, getattr(r, "var"), r.level, r.m, "tabakali")
    rows4.append(dict(Scenario=SCN4.get(k, k), Level="individual" if r.level == "birey" else "study", Sampling="stratified", Arm={"T-ACC": "T-ACC", "PACC-hedef": "PACC-target", "Yalniz-hedef": "target-only"}[arm], m=int(r.m), AE_arm=r.AE_a, AE_CC=r.AE_b,
                      d=r.d, lo=r.lo, hi=r.hi, Category=EN[r.kategori], Holm_p=r.p_holm, family_tests=int(r.aile_n), n=int(r.n), Frac_draws_all_classes_ge20s=f20, Frac_draws_missing_C_column=ec, Source="r07_6_tests.csv (CC-stratified reference)"))
T4 = pd.DataFrame(rows4); T4 = T4[T4.m == 1800].sort_values(["Scenario", "Arm", "Sampling"]).reset_index(drop=True)   # R10 C: only m = 1,800; random-block / stratified rows side by side
assert not T4.isna().any().any()
T4["d [95% CI]"] = [f"{r.d:+.4f} [{r.lo:+.4f}; {r.hi:+.4f}]" for r in T4.itertuples()]
note4 = ("d = AE_CC − AE_arm, CC evaluated on the same bags (labelled seconds removed from the bags at individual level). Random-block rows: R06/R07 families; stratified rows: reference is CC on the stratified draw. "
         "'Frac_draws_all_classes_ge20s' = share of label draws containing >= 20 s of every class (r07_6_counts.csv). Retraining and shrinkage were not run with stratified sampling, so those arms have no stratified row (no empty cells). The m curve is in Fig. 5. Dunford rows exploratory.")
save(T4, "table4_target_labelled_arms", "Table 4. Target-labelled arms (m = 1,800 s)", note4)

# ================================================================== SUPPLEMENT
sup = []
def sp(fn, df, desc): df.to_csv(TP / fn, index=False, encoding="utf-8"); sup.append((fn, desc))
sp("supp_S1_R04_bias_variance.csv", rd("r04_decomposition.csv"), "R04 bias–variance decomposition of bag-level error (cells × arm).")
sp("supp_S2_D17_vs_matched32.csv", rd("r06_A2.csv"), "D17 vs matched 32-feature set (S2, harness → collar).")
d_k = rd("r06_D_kural.csv"); sp("supp_S3_delta_rule_kappa.csv", d_k, "δ̂ decision rule and κ-PACC rule (R06 §D) with decision categories.")
sp("supp_S3b_delta_summary.csv", rd("r06_D_delta.csv"), "δ (½‖p − π_source‖₁) median and p90 per bag set.")
sp("supp_S4_dog_S2.csv", T7_5[T7_5.scen == "S2-kopek"], "S2-dog (ABack → ANeck, K-L1, D17) source arms.")
sp("supp_S5_weighted_vs_unweighted_retraining.csv", F6[F6.aile.str.startswith(("F2-RT", "F3-w-u"))], "Weighted and unweighted retraining vs CC and against each other (R06 F2/F3).")
sp("supp_S6_Dunford_A2.csv", rd("r07_3_b_A2.csv"), "Dunford A2 seconds per behaviour: reported, all rows /40, block repeats removed, literal duplicates removed.")
sp("supp_S6b_Dunford_split_unit.csv", rd("r07_3_c_summary.csv"), "Dunford split unit: random 60/40 seconds vs leave-one-cat-out macro-F1 (v1/v2).")
sp("supp_S7_ECE_macroF1.csv", rd("r07_4_cls_summary.csv"), "Macro-F1, accuracy and pooled ECE (RF, HGB, LR).")
sp("supp_S8_DR_h.csv", R8[R8.aile != "birincil"], "DR-h (posterior-mean descriptor) secondary family, R08.")
sp("supp_S9_R08_DR_details.csv", rd("r08_dr_detail.csv"), "R08 DR details: selected number of components, training-bag counts.")
sp("supp_S9b_R08_extrapolation.csv", rd("r08_extrapolation.csv"), "R08 extrapolation diagnostic (evaluation proportions outside training range).")
sp("supp_S9c_R08_dm_boundary.csv", rd("r08_dm_boundary.csv"), "R08 share of bags with a DM boundary solution.")
sp("supp_S10_B1_DM_boundary_classes.csv", pd.read_csv(T / "r09_B1_dm_boundary.csv"), "R09 B1: classes zeroed by DM; share of such bags with true proportion 0; median [IQR] of true proportion.")
sp("supp_S10b_B2_equivalent_inconclusive_RAE.csv", pd.read_csv(T / "r09_B2_equiv_uncertain.csv"), "R09 B2: RAE of arm and CC and signed bias of the rarest class (pp) in EQUIVALENT/INCONCLUSIVE cells.")
sp("supp_S10c_B3_DR_zero_training_bags.csv", pd.read_csv(T / "r09_B3_dr_zero_train.csv"), "R09 B3: share of DR-natural training bags with true class proportion 0.")
sp("supp_S11_analysis_history_v2.csv", rd("r08_analysis_history_v2_2026-11-06.csv"), "Analysis history v2 (all analyses planned/run, with location in paper).")
# ---- R10: S12 (RAE), S13 (DM boundary), S14 (open-set care bias)
from src import r10rae as X10
PR = X10.Pairs(); S12 = X10.s12(PR); sp("supp_S12_rae.csv", S12, "R10 A + R11: RAE (bag mean) of CC and arm, d_RAE = RAE_CC − RAE_arm (individual-level mean, bootstrap CI), AE category and direction-mismatch flag, for every Table 3 cell (rows with note 'source=R11': within-dog R07 arms, computed in R11).")
B1 = pd.read_csv(T / "r09_B1_dm_boundary.csv"); S13 = B1.rename(columns={"senaryo": "cell_scenario", "torba_turu": "cell_bag_type", "sinif": "zeroed_class", "n_torba": "n_bags", "n_dm_sifir": "n_bags_class_zeroed"})
S13["n_bags_true_also_zero"] = (S13.n_bags_class_zeroed * S13.gercek_sifir_orani).round().astype(int); S13 = S13.rename(columns={"gercek_sifir_orani": "share_true_also_zero"})
sp("supp_S13_dm_boundary.csv", S13[["cell_scenario", "cell_bag_type", "zeroed_class", "n_bags", "n_bags_class_zeroed", "n_bags_true_also_zero", "share_true_also_zero"]], "R10 C: classes zeroed by DM per cell (scenario × bag type): number of zeroed bags and share whose true proportion is also 0.")
OB = X10.openset_bias(); sp("supp_S14_openset_care_bias.csv", OB, "R10 B: S4-Smit-open, CC, Care signed bias (pp): cat level (R05), (i) individual mean, (ii) bag mean, 95 % individual bootstrap CI.")
idx = ["# Supplementary table index", "", "| File | Content |", "|---|---|"] + [f"| `{a}` | {b} |" for a, b in sup]
(TP / "supplement_index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")

# ================================================================== FIGURES
INK = "#222222"; GRID = "#dddddd"; C1 = "#2a6fb0"; C2 = "#d9822b"; C3 = "#3a9a8c"; C4 = "#7b5aa6"; C5 = "#777777"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888888", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
MK = {"KAZANIR": ("^", True, "Wins"), "KAYBEDER": ("v", True, "Loses"), "ESDEGER": ("s", False, "Equivalent"), "BELIRSIZ": ("o", False, "Inconclusive")}


def savefig(fig, stem):
    fig.savefig(FP / f"{stem}.png", dpi=300); fig.savefig(FP / f"{stem}.pdf"); plt.close(fig)


# ---- fig2_forest
PAN = []   # (title, DataFrame[label, d, lo, hi, cat, grp])
for sc in M.senaryo.unique():
    if sc in ("S1-tasma", "S1-koşum", "S2-eşlenmiş"): continue
    g = M[(M.senaryo == sc) & (M.aile == "kaynak")][["kol", "d", "lo", "hi", "kategori"]].copy(); g["grp"] = "src"; g = g[g.kol.isin(ANAME)]; g["kol"] = g.kol.map(ANAME)
    lb = LBL[sc]
    if lb in R8MAP:
        a8 = R8m[(R8m.senaryo == R8MAP[lb][0]) & (R8m.torba_turu == R8MAP[lb][1])]
        g = pd.concat([g, pd.DataFrame([dict(kol=ANAME[r.kol], d=r.d, lo=r.lo, hi=r.hi, kategori=r.kategori, grp="r08") for r in a8.itertuples()])], ignore_index=True)
    PAN.append((lb, g))
for sp_ in ("K-L1", "K-L2"):
    rows = []
    for bt, bn in (("kayit", "recording"), ("dogal2", "2-min")):
        gg = T7_5[(T7_5.scen == "Kopek-ici") & (T7_5.space == sp_) & (T7_5.feat.astype(str) == "32") & (T7_5.btype == bt) & T7_5.kol.isin(ANAME)]
        rows += [dict(kol=f"{bn} | {ANAME[r.kol]}", d=r.d, lo=r.lo, hi=r.hi, kategori=r.kategori, grp="src") for r in gg.itertuples()]
        rows += [dict(kol=f"{bn} | {ANAME[r.kol]}", d=r.d, lo=r.lo, hi=r.hi, kategori=r.kategori, grp="r08") for r in R8m[(R8m.senaryo == sp_) & (R8m.torba_turu == bt)].itertuples()]
    PAN.append((f"{sp_} within-dog (n=44 recording; n=35 2-min)", pd.DataFrame(rows)))
nr = (len(PAN) + 1) // 2; fig, axs = plt.subplots(nr, 2, figsize=(12, 3.0 * nr + 1.2)); axs = axs.ravel()
for ax, (ttl, g) in zip(axs, PAN):
    g = g.reset_index(drop=True); ax.axvline(0, color=INK, lw=0.8); ax.axvspan(-0.01, 0.01, color="#eeeeee", zorder=0)
    for i, r in g.iterrows():
        mk, fill, _ = MK[r.kategori]; col = C4 if r.grp == "r08" else C1
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.4); ax.plot(r.d, i, marker=mk, ms=7, color=col, mfc=col if fill else "white", mew=1.4)
    ax.set_yticks(range(len(g))); ax.set_yticklabels(g.kol, fontsize=7.5); ax.invert_yaxis(); ax.set_title(ttl, fontsize=9.5, loc="left"); ax.set_xlabel("d = AE_CC − AE_arm  (d > 0: arm better)", fontsize=8)
for ax in axs[len(PAN):]: ax.axis("off")
h = [plt.Line2D([], [], marker=m, ls="", color=C5, mfc=C5 if f else "white", label=n) for k, (m, f, n) in MK.items()] + [plt.Line2D([], [], color=C1, label="correction arms (PCC, ACC, PACC, EMQ, N-target)"), plt.Line2D([], [], color=C4, label="DM, DR-APP, DR-natural")]
fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=8); fig.suptitle("Scenario × arm: AE_CC − AE_arm [95 % individual bootstrap CI]; grey band |d| < 0.01; [exploratory] = Dunford-based", y=0.995, fontsize=10.5)
fig.tight_layout(rect=(0, 0.04, 1, 0.98)); savefig(fig, "fig2_forest")

# ---- fig3_app_vs_natural
DL = rd("r07_5_delta.csv"); F2d = rd("r07_fig2_data.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
rowsL = []
for var, bn, b8 in (("APP120", "APP 120-s", "APP120"), ("dogal2dk", "natural 2-min", "2dk")):
    for clf in ("RF", "HGB", "LR"):
        r = T7_4[(T7_4.clf == clf) & (T7_4.scen == "S1-kopru") & (T7_4.variant == var) & (T7_4.kol == "PACC")].iloc[0]; rowsL.append((f"{bn} | PACC ({clf})", r.d, r.lo, r.hi, r.kategori, C1))
    for kol, nm in (("DM", "DM"), ("DR-APP-q", "DR-APP"), ("DR-dogal-q", "DR-natural")):
        r = R8m[(R8m.senaryo == "S1") & (R8m.torba_turu == b8) & (R8m.kol == kol)].iloc[0]; rowsL.append((f"{bn} | {nm} (RF)", r.d, r.lo, r.hi, r.kategori, C4))
for i, (lb, d, lo, hi, k, col) in enumerate(rowsL):
    mk, fill, _ = MK[k]; a1.plot([lo, hi], [i, i], color=col, lw=1.5); a1.plot(d, i, marker=mk, ms=7, color=col, mfc=col if fill else "white", mew=1.4)
a1.axvline(0, color=INK, lw=0.8); a1.axvspan(-0.01, 0.01, color="#eeeeee", zorder=0); a1.axhline(5.5, color=GRID, lw=1); a1.set_yticks(range(len(rowsL))); a1.set_yticklabels([r[0] for r in rowsL], fontsize=8); a1.invert_yaxis()
a1.set_xlabel("AE_CC − AE_arm (d > 0: arm better) [95 % CI]"); a1.set_title("Smit S1 (12 cats): APP 120-s vs natural 2-min bags", fontsize=9.5, loc="left")
colsD = {"recording": C1, "natural 2-min": C1, "APP 120-s": C2}; rowsR = []
for sp_ in ("K-L1", "K-L2"):
    for bt, nm in (("kayit", "recording"), ("dogal2", "natural 2-min"), ("APP120", "APP 120-s")):
        t = T7_5[(T7_5.scen == "Kopek-ici") & (T7_5.space == sp_) & (T7_5.feat.astype(str) == "32") & (T7_5.btype == bt) & (T7_5.kol == "PACC")]
        if len(t): rowsR.append((f"Dog {sp_} | {nm} | PACC", t.iloc[0].d, t.iloc[0].lo, t.iloc[0].hi, t.iloc[0].kategori, colsD[nm]))
for i, (lb, d, lo, hi, k, col) in enumerate(rowsR):
    mk, fill, _ = MK[k]; a2.plot([lo, hi], [i, i], color=col, lw=1.5); a2.plot(d, i, marker=mk, ms=7, color=col, mfc=col if fill else "white", mew=1.4)
a2.axvline(0, color=INK, lw=0.8); a2.axvspan(-0.01, 0.01, color="#eeeeee", zorder=0); a2.set_yticks(range(len(rowsR))); a2.set_yticklabels([r[0] for r in rowsR], fontsize=8); a2.invert_yaxis()
a2.set_xlabel("AE_CC − AE_PACC (d > 0: PACC better) [95 % CI]"); a2.set_title("Dogs (n = 44 / 35 / 40), within-dog, 32 features", fontsize=9.5, loc="left")
dtx = "δ medians (½‖p − π_source‖₁): Smit S1 — natural 30-min %.3f, natural 2-min %.3f, APP 120-s %.3f.  Dogs K-L1 (recording / 2-min / APP): %.3f / %.3f / %.3f;  K-L2: %.3f / %.3f / %.3f." % (
    F2d[F2d.torba == "doğal 30 dk"].delta.iloc[0], F2d[F2d.torba == "doğal 2 dk"].delta.iloc[0], F2d[F2d.torba == "APP 120 s"].delta.iloc[0], *[DL[(DL.scen == "Kopek-ici") & (DL.space == sp_) & (DL.feat.astype(str) == "32") & (DL.btype == b)].delta_medyan.iloc[0] for sp_ in ("K-L1", "K-L2") for b in ("kayit", "dogal2", "APP120")])
h = [plt.Line2D([], [], marker=m, ls="", color=C5, mfc=C5 if f else "white", label=n) for k, (m, f, n) in MK.items()] + [plt.Line2D([], [], color=C1, label="PACC"), plt.Line2D([], [], color=C4, label="DM, DR"), plt.Line2D([], [], color=C2, label="APP 120-s (dog panel)")]
fig.legend(handles=h, loc="lower center", ncol=7, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.045)); fig.text(0.5, 0.012, dtx, ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.09, 1, 1)); savefig(fig, "fig3_app_vs_natural")

# ---- fig4_open_set
src5 = pd.read_parquet(P.R05 / "r05_sources_bags.parquet"); v2 = pd.read_parquet(P.R06 / "B_v2_bags.parquet")
CLn = ["Inactive", "Locomotion", "Care"]; fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4)); off = {"CC": -0.18, "PACC": 0.18}; BIAS = {}
for si, (nm, df, sc_, var) in enumerate((("S4-Smit-open", src5, "S4-Smit", "30dk_ge50-acik"), ("S4-Dunford-open [exploratory]", v2, "S4-Dunford", "acik"))):
    g = df[(df.scen == sc_) & (df.variant == var) & (df.m == 0)]
    for arm, col in (("CC", C1), ("PACC", C2)):
        x = g[g.arm == arm]
        for k in range(3):
            b = ((x[f"h{k}"] - x[f"p{k}"]) * 100).groupby(x.ind).mean().values; lo, hi = S6.boot_ci(b); BIAS[(sc_, arm, k)] = float(b.mean())
            xp = k + si * 4 + off[arm]; a1.plot([xp, xp], [lo, hi], color=col, lw=1.6); a1.plot(xp, b.mean(), "o" if arm == "CC" else "s", color=col, ms=6, label=arm if (si == 0 and k == 0) else None)
a1.axhline(0, color=INK, lw=0.8); a1.set_xticks([0, 1, 2, 4, 5, 6]); a1.set_xticklabels([f"{c}\n{('Smit' if i < 3 else 'Dunford')}" for i, c in enumerate(CLn * 2)], fontsize=8); a1.set_ylabel("signed bias (estimate − truth), percentage points"); a1.legend(frameon=False)
a1.set_title("Class-wise signed bias (individual bootstrap CI)", fontsize=9.5, loc="left")
S5 = pickle.load(open(P.R05 / "sources.pkl", "rb")); S6_ = pickle.load(open(P.R06 / "sources_v2.pkl", "rb"))
pos = y1[S5["idx"]]; ps = S5["P4s"][pos == 2].argmax(1); pdn = S6_["P4d"][dun.y1.values == 2].argmax(1); SHARE = {}
for i, (nm, p) in enumerate((("Smit", ps), ("Dunford", pdn))):
    fr = np.bincount(p, minlength=2) / len(p) * 100; SHARE[nm] = (fr[0], len(p))
    a2.bar(i, fr[0], color=C1, width=0.5, label="→ Inactive" if i == 0 else None); a2.bar(i, fr[1], bottom=fr[0], color=C2, width=0.5, label="→ Locomotion" if i == 0 else None)
    a2.text(i, fr[0] / 2, f"{fr[0]:.1f} %", ha="center", color="white", fontsize=9); a2.text(i, fr[0] + fr[1] / 2, f"{fr[1]:.1f} %", ha="center", color=INK, fontsize=9); a2.text(i, 103, f"n = {len(p)} s", ha="center", fontsize=8)
a2.set_xticks([0, 1]); a2.set_xticklabels(["Smit Care", "Dunford Care [expl.]"]); a2.set_ylim(0, 110); a2.set_ylabel("predicted class by the dog model, %"); a2.legend(frameon=False, loc="lower right"); a2.set_title("Care seconds as classified by the dog model", fontsize=9.5, loc="left")
fig.tight_layout(); savefig(fig, "fig4_open_set")

# ---- fig5_label_budget
CT = rd("r07_6_counts.csv"); AR = rd("r07_6_arms.csv"); fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
for (sc_, var, lv, nm, col) in (("S1", "30dk_ge50", "birey", "S1 individual-level", C1), ("S4-Smit", "acik", "birey", "S4-Smit-open individual-level", C2), ("S3", "-", "calisma", "S3 study-level [expl.]", C3)):
    for yo, ls, mk in (("rastgele_blok", "--", "o"), ("tabakali", "-", "s")):
        g = CT[(CT.scen == sc_) & (CT["var"] == var) & (CT.level == lv) & (CT.yontem == yo)].sort_values("m")
        if len(g): a1.plot(g.m, g.hepsi_ge20 * 100, ls=ls, marker=mk, color=col, ms=5, label=f"{nm} – {'random' if yo.startswith('r') else 'stratified'}")
a1.set_xscale("log"); a1.set_xticks([60, 300, 900, 1800]); a1.set_xticklabels(["60", "300", "900", "1,800"]); a1.set_xlabel("label budget m (s)"); a1.set_ylabel("draws with >= 20 s of every class, %"); a1.legend(frameon=False, fontsize=7); a1.set_title("(a) class coverage", loc="left", fontsize=9.5)
SPEC = [("CC", "CC (random)", C5, "o", "-"), ("CC-tab", "CC (stratified)", C5, "o", "--"), ("PACC-hedef", "PACC-target random", C1, "s", "-"), ("PACC-hedef-tab", "PACC-target stratified", C1, "s", "--"),
        ("Yeniden-egitim", "retraining (random)", C2, "^", "-"), ("Yalniz-hedef-esl", "target-only (random)", C3, "D", "-"), ("Yalniz-hedef-tab", "target-only (stratified)", C3, "D", "--")]
for arm, nm, col, mk, ls in SPEC:
    g = AR[(AR.scen == "S4-Smit") & (AR.variant == "acik") & (AR.level == "birey") & (AR.arm == arm)].sort_values("m")
    if len(g): a2.plot(g.m, g.AE, color=col, marker=mk, ls=ls, ms=5, label=nm)
a2.set_xscale("log"); a2.set_xticks([60, 300, 900, 1800]); a2.set_xticklabels(["60", "300", "900", "1,800"]); a2.set_xlabel("label budget m (s)"); a2.set_ylabel("AE (S4-Smit-open, individual-level)"); a2.legend(frameon=False, fontsize=7); a2.set_title("(b) AE", loc="left", fontsize=9.5)
fig.tight_layout(); savefig(fig, "fig5_label_budget")
print("şekiller tamam %.0f s" % (time.time() - t0), flush=True)

# ================================================================== E — paper_numbers.csv
def g1(df, **kw):
    q = df
    for k, v in kw.items(): q = q[q[k] == v]
    assert len(q) == 1, (kw, len(q)); return q.iloc[0]
def dec(s): s = s.replace("%", "").replace("+", "").replace("−", "-").replace(",", "."); return len(s.split(".")[1]) if "." in s else 0
def val(s): return float(s.replace("%", "").replace("+", "").replace("−", "-").replace(",", "."))
NUM = []; COMP = []
def add(id_, tanim, beklenen, hesap, kaynak):
    ok = True; exps = [x.strip() for x in beklenen.split("/")]; assert len(exps) == len(hesap), id_
    for e, h in zip(exps, hesap):
        m_ = round(h, dec(e)) == round(val(e), dec(e)) and abs(round(h, dec(e)) - val(e)) < 1e-9; ok &= bool(m_); COMP.append(dict(kimlik=id_, beklenen=e, hesaplanan_ham=h, hesaplanan_yuvarlanmis=round(h, dec(e)), eslesme=bool(m_)))
    NUM.append(dict(kimlik=id_, tanim=tanim, beklenen=beklenen, hesaplanan=" / ".join(f"{round(h, dec(e))}" for e, h in zip(exps, hesap)), kaynak_dosya=kaynak, eslesme="EVET" if ok else "HAYIR"))
S1c = g1(M, senaryo="S1", kol="CC"); S1n = g1(M, senaryo="S1", kol="N-hedef")
add("N01", "S1 L1 birincil, AE_CC / AE_N-hedef", "0,0260 / 0,1091", [S1c.AE_CC, S1n.AE_kol], "r07_master.csv")
nv = rd("r06_C_naive.csv"); l2 = g1(nv, scen="S1-L2", var="30dk_ge50"); add("N02", "S1 L2, AE_CC / AE_N-hedef", "0,084 / 0,122", [l2.AE_CC, l2.AE_Nh], "r06_C_naive.csv")
add("N03", "S1 L1 d: PCC / ACC / PACC / EMQ", "−0,0152 / −0,0063 / −0,0240 / −0,0440", [g1(M, senaryo="S1", kol=k).d for k in ("PCC", "ACC", "PACC", "EMQ")], "r07_master.csv")
add("N04", "S2 L1 d: ACC / PACC", "−0,0121 / −0,0693", [g1(M, senaryo="S2", kol=k).d for k in ("ACC", "PACC")], "r07_master.csv")
t5 = T7_5[(T7_5.scen == "Kopek-ici") & (T7_5.space == "K-L2") & (T7_5.feat.astype(str) == "32") & (T7_5.btype == "kayit")]
add("N05", "K-L2-32 köpek × kayıt d: ACC / PACC / EMQ", "−0,0312 / −0,0307 / −0,0212", [g1(t5, kol=k).d for k in ("ACC", "PACC", "EMQ")], "r07_5_tests.csv")
add("N06", "S1 birey m = 1.800: her sınıftan ≥ 20 s içeren çekiliş (%), rastgele / tabakalı", "%2,1 / %91,7", [100 * g1(C7_6, scen="S1", var="30dk_ge50", level="birey", m=1800, yontem=y).hepsi_ge20 for y in ("rastgele_blok", "tabakali")], "r07_6_counts.csv")
add("N07", "S1 birey m = 1.800: eksik C sütunu (%), rastgele / tabakalı", "%65,0 / %8,3", [100 * g1(C7_6, scen="S1", var="30dk_ge50", level="birey", m=1800, yontem=y).eksik_C for y in ("rastgele_blok", "tabakali")], "r07_6_counts.csv")
add("N08", "S1 PACC-hedef-tabakalı d (CC-tab'a karşı), m = 60 / 300 / 900", "−0,0257 / −0,0130 / −0,0123", [g1(T7_6, scen="S1", karsilastirma="PACC-hedef-tab_vs_CC-tab", m=m).d for m in (60, 300, 900)], "r07_6_tests.csv")
# R10 B: fig4_open_set shows the individual mean (BIAS = per-individual bag mean); the primary-set (68 bags) value must equal B1(i)
b_i = float(OB.bias_pp.iloc[1]); assert abs(BIAS[("S4-Smit", "CC", 2)] - b_i) < 1e-9, (BIAS[("S4-Smit", "CC", 2)], b_i)
add("N09", "S4-Smit-açık, birincil küme (68 torba): Bakım işaretli yanlılığı CC, birey ortalaması (pp); Bakım saniyelerinin Inaktif'e giden payı (%)", "−12,9 / %78,3", [b_i, SHARE["Smit"][0]], "r05_sources_bags.parquet (R10 B1 i); r05/sources.pkl")
add("N10", "Yalnız hedef − CC, birey, m = 1.800: S4-Smit-açık / S4-Smit-kapalı", "+0,0547 / −0,0288", [g1(M, senaryo=s, kol="Yalniz-hedef-esl (m=1800, rastgele blok)").d for s in ("S4-Smit-açık", "S4-Smit-kapalı")], "r07_master.csv")
add("N11", "S1 APP 120 s PACC − CC: RF / HGB / LR", "+0,0601 / +0,0638 / +0,1526", [g1(T7_4, clf=c, scen="S1-kopru", variant="APP120", kol="PACC").d for c in ("RF", "HGB", "LR")], "r07_4_tests.csv")
add("N12", "S1 doğal 2 dk PACC − CC: RF / HGB / LR", "−0,0185 / −0,0123 / +0,0082", [g1(T7_4, clf=c, scen="S1-kopru", variant="dogal2dk", kol="PACC").d for c in ("RF", "HGB", "LR")], "r07_4_tests.csv")
add("N13", "R08 S1 30 dk d: DM / DR-APP / DR-doğal", "−0,0325 / −0,1269 / −0,0507", [g1(R8m, senaryo="S1", torba_turu="30dk", kol=k).d for k in ("DM", "DR-APP-q", "DR-dogal-q")], "r08_tests.csv")
add("N14", "R08 S1 APP 120 s d: DM / DR-APP", "+0,0477 / +0,0341", [g1(R8m, senaryo="S1", torba_turu="APP120", kol=k).d for k in ("DM", "DR-APP-q")], "r08_tests.csv")
add("N15", "AE_CC: S4-Smit-kapalı / S1-kapalı", "0,0089 / 0,0132", [g1(M, senaryo=s, kol="CC").AE_CC for s in ("S4-Smit-kapalı", "S1-kapalı")], "r07_master.csv")
cs = rd("r07_3_c_summary.csv"); loco = g1(cs, dogrulama="LOCO_birlesik", surum="v2").macroF1
add("N16", "Dunford rastgele − LOCO makro-F1: v1 / v2", "+0,226 / +0,184", [g1(cs, dogrulama="rastgele_60_40", surum=s).macroF1 - loco for s in ("v1", "v2")], "r07_3_c_summary.csv")
j3 = json.load(open(T / "r07_3.json")); add("N17", "Dunford > 42 satırlı saniyelerden blok tekrarı olan / toplam", "394 / 413", [j3["a"]["blok_tekrari"], j3["a"]["saniye_gt42"]], "r07_3.json")
N = pd.DataFrame(NUM); N.to_csv(TP / "paper_numbers.csv", index=False, encoding="utf-8"); pd.DataFrame(COMP).to_csv(TP / "paper_numbers_components.csv", index=False)
print(N.to_string()); print("eşleşen", int((N.eslesme == "EVET").sum()), "/", len(N)); print("TOPLAM %.1f s" % (time.time() - t0))
json.dump(dict(sure_s=time.time() - t0, eslesen=int((N.eslesme == "EVET").sum()), toplam=len(N)), open(TP / "run_info.json", "w"))
