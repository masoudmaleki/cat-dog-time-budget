"""Build Supplementary_Material.xlsx (English) from the public repository tables.
Numbering follows R13 (S1-S20). Run: python3 -I build_supp.py <repo_tables_dir> <out_dir>"""
import sys, re, os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_en import ROWS as HISTORY

T = sys.argv[1]            # .../results/tables
P = os.path.join(T, "paper")
OUT = sys.argv[2]
os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)

def rd(name, paper=True):
    return pd.read_csv(os.path.join(P if paper else T, name))

# ---------------------------------------------------------------- value translation
CAT = {"KAZANIR": "better", "KAYBEDER": "worse", "ESDEGER": "equivalent", "EŞDEĞER": "equivalent",
       "BELIRSIZ": "inconclusive", "BELİRSİZ": "inconclusive", "REFERANS": "reference"}
DIRECT = {
    # scenarios
    "S0 (sızıntılı)": "S0 (leaky)", "S3 [keşifsel]": "S3 [exploratory]", "S4-Smit-açık": "S4-Smit-open",
    "S4-Smit-kapalı": "S4-Smit-closed", "S4-Dunford-açık [keşifsel]": "S4-Dunford-open [exploratory]",
    "S4-Dunford-kapalı [keşifsel]": "S4-Dunford-closed [exploratory]", "R-D [keşifsel]": "R-D [exploratory]",
    "S1-kapalı": "S1-closed", "S1-tasma": "S1-L2 (collar)", "S1-koşum": "S1-L2 (harness)", "S1-kosum": "S1-L2 (harness)",
    "S2-eşlenmiş": "S2-L2 (mapped)", "S2-esl": "S2-L2 (mapped)", "S2-L2 (matched)": "S2-L2 (mapped)",
    "S1-L2": "S1-L2 (collar)", "S4-Smit-kapali": "S4-Smit-closed", "S4-Dunford-kapali": "S4-Dunford-closed [exploratory]",
    "S3": "S3 [exploratory]", "S4-Dunford": "S4-Dunford [exploratory]", "S1-kopru": "S1 bridge (APP vs natural)",
    "Kopek-ici": "within-dog", "S2-kopek": "S2-dog", "S1 2dk": "S1 natural 2-min", "S1-L2 30dk": "S1-L2 (collar) 30-min",
    "S4-Smit-kapali 30dk": "S4-Smit-closed 30-min", "K-L1 dogal2": "K-L1 natural 2-min", "K-L1 kayit": "K-L1 recording",
    "K-L2 dogal2": "K-L2 natural 2-min", "S3 kedi": "S3 [exploratory] cat sample",
    "S4-Dunford-kapali kedi": "S4-Dunford-closed [exploratory] cat sample", "S1-2dk": "S1 natural 2-min",
    "S1-APP120": "S1 APP 120-s",
    # sample types
    "30dk": "30-min", "2dk": "2-min", "APP120": "APP 120-s", "dogal2": "natural 2-min", "dogal2dk": "natural 2-min",
    "kayit": "recording", "kedi": "cat sample", "30dk_ge50": "30-min (coverage ≥ 50%)",
    "30dk_ge50-acik": "30-min, open", "30dk_ge50-kapali": "30-min, closed", "acik": "open", "kapali": "closed",
    "Smit 30 dk (kapsam ≥%50)": "Smit 30-min (coverage ≥ 50%)", "Smit 2 dk (kapsam ≥%50)": "Smit 2-min (coverage ≥ 50%)",
    "köpek × kayıt (N ≥ 60)": "dog recordings (≥ 60 s)", "köpek doğal 2 dk (N ≥ 60)": "dog natural 2-min windows",
    # classes
    "Inaktif": "inactive", "Lokomosyon": "locomotion", "Bakim": "maintenance", "Lying chest": "Lying on chest",
    # arms
    "N-hedef": "N-target", "Yeniden-egitim": "Retraining", "Yeniden-egitim-agirlikli": "Weighted retraining",
    "T-ACC (m=1800, rastgele blok)": "T-ACC (m = 1800, random blocks)",
    "PACC-hedef (m=1800, rastgele blok)": "PACC-target (m = 1800, random blocks)",
    "PACC-buzme (m=1800, rastgele blok)": "PACC-shrinkage (m = 1800, random blocks)",
    "Yeniden-egitim (m=1800, rastgele blok)": "Retraining (m = 1800, random blocks)",
    "Yeniden-egitim-agirlikli (m=1800, rastgele blok)": "Weighted retraining (m = 1800, random blocks)",
    "Yalniz-hedef-esl (m=1800, rastgele blok)": "Target-only, paired (m = 1800, random blocks)",
    "DR-APP-q": "DR-APP", "DR-dogal-q": "DR-natural", "DR-dogal-h": "DR-natural-h", "DR-dogal": "DR-natural",
    # misc
    "calisma": "study", "birey": "individual", "ikincil": "secondary", "birincil": "primary",
    "kesin": "exact", "MC": "Monte Carlo (200,000)", "yari": "half recording", "TOPLAM": "Total",
    "LOCO_birlesik": "leave-one-cat-out (pooled predictions)", "LOCO_kedi": "leave-one-cat-out (per cat)",
    "rastgele_60_40": "random 60/40 split of seconds", "v1": "v1 (with repeated blocks)",
    "v2": "v2 (exact duplicate rows removed)", "h": "posterior histograms (h)", "q": "feature deciles (q)",
    "kedi düzeyi (R05, tüm saniyeler / kedi)": "cat level (all seconds of each cat)",
    "(i) birey ortalaması (birey başına torba ortalaması; R08 §4)": "(i) individual mean (mean over each cat's 30-min windows, then over cats)",
    "(ii) torba ortalaması (68 torba)": "(ii) window mean (68 windows)",
    "'torba düzeyi sınıf vektörü kayıtlı değil'": "window-level class proportions not stored",
    "S2-eşlenmiş L2: torba düzeyi sınıf vektörü kayıtlı değil (yalnız AE)": "S2-L2 (mapped): window-level class proportions not stored (AE only)",
    "source=R11": "computed from stored posteriors (round R11)",
    "Smit 30-min bags (68)": "Smit natural 30-min windows (68)", "Dunford cat bags (N>=60)": "Dunford, one sample per cat (≥ 60 s)",
    "natural 2-min bags": "natural 2-min windows", "APP 120-s draws": "APP 120-s samples", "APP 120-s": "APP 120-s samples",
    "dog × recording (N>=60)": "dog recordings (≥ 60 s)", "natural 2-min (N>=60)": "dog natural 2-min windows",
    "dog × recording (N ≥ 60)": "dog recordings (≥ 60 s)", "dog natural 2-min (N ≥ 60)": "dog natural 2-min windows",
    "r07_master": "r07_master", "r08_tests": "r08_tests",
}
def tr_value(v, col=""):
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s in CAT:
        return CAT[s]
    if s in DIRECT:
        return DIRECT[s]
    if col == "family":  # supp_S5 compound family keys
        for a, b in [("Yeniden-egitim-agirlikli", "Weighted retraining"), ("Yeniden-egitim", "Retraining"),
                     ("calisma", "study"), ("birey", "individual"), ("acik", "open"), ("kapali", "closed"),
                     ("30dk_ge50", "30-min")]:
            s = s.replace(a, b)
        return s
    return s

def translate(df, colmap):
    df = df.rename(columns=colmap)
    for c in df.columns:
        if not is_numeric_dtype(df[c]):
            df[c] = [tr_value(x, c) for x in df[c]]
    return df

STD = {"kategori": "category", "aile_n": "family_size", "p_holm": "p_holm", "pmin": "p_min_attainable",
       "kol": "arm", "n_torba": "n_samples", "AE_kol": "AE_arm", "n": "n_individuals", "lo": "ci_low", "hi": "ci_high",
       "p_yontem": "p_method", "senaryo": "scenario", "scen": "scenario", "torba_turu": "sample_type",
       "btype": "sample_type", "space": "label_space", "uzay": "label_space", "feat": "features", "sinif": "class"}

sheets = []   # (sheet_name, title, notes, source, df)

# S1 ------------------------------------------------------------------
d = rd("supp_S2_D17_vs_matched32.csv")
d = translate(d, {**STD, "est": "estimator", "n_bag": "n_windows", "AE_esl32": "AE_mapped32"})
sheets.append(("S1_D17_vs_matched32", "D17 versus the axis-mapped 32 features in harness-to-collar transfer (S2, Smit, three classes, 68 natural 30-min windows).",
               "d = AE_mapped32 − AE_D17 (individual-level mean; positive: D17 better). Holm over the three estimators.", "supp_S2_D17_vs_matched32.csv", d))
# S2 ------------------------------------------------------------------
h = pd.DataFrame(HISTORY, columns=["round", "hypothesis_or_criterion", "definition", "when_specified", "result", "location_in_paper"])
sheets.append(("S2_Analysis_history", "Analysis history: every hypothesis and criterion tested, when it was specified, its result and where it is reported.",
               "R02–R08 are successive analysis rounds. 'after Rxx results were seen' marks analyses defined after earlier results were available. 'this table only': the result is reported nowhere else in the paper. Effects are differences in AE with 95% individual-bootstrap intervals.", "supp_S11_analysis_history_v2.csv (translated)", h))
# S3 ------------------------------------------------------------------
t3 = rd("table3_consolidated_results_long.csv")
master = rd("r07_master.csv", False); r08 = rd("r08_tests.csv", False); r075 = rd("r07_5_tests.csv", False)
r074 = rd("r07_4_tests.csv", False); r06 = rd("r06_C_naive.csv", False); delta = rd("r07_5_delta.csv", False)
src = []
for _, r in master.iterrows():
    if r.kol in ("PCC", "ACC", "PACC", "EMQ", "N-hedef"):
        src.append(("r07_master.csv", "N-target" if r.kol == "N-hedef" else r.kol, r.d, r.p_holm, r.aile_n, r.AE_CC))
for _, r in r08.iterrows():
    arm = {"DM": "DM", "DR-APP-q": "DR-APP", "DR-dogal-q": "DR-natural"}.get(r.kol)
    if arm: src.append(("r08_tests.csv", arm, r.d, r.p_holm, r.aile_n, r.AE_CC))
for _, r in r075.iterrows():
    if r.scen == "Kopek-ici" and str(r.feat) == "32":
        src.append(("r07_5_tests.csv", "N-target" if r.kol == "N-hedef" else r.kol, r.d, r.p_holm, r.aile_n, r.AE_CC))
for _, r in r074.iterrows():
    if r.clf == "RF" and r.scen == "S1-kopru":
        src.append(("r07_4_tests.csv", r.kol, r.d, r.p_holm, r.aile_n, r.AE_CC))
l2 = r06[r06.scen == "S1-L2"].iloc[0]
src.append(("r06_C_naive.csv (tested alone)", "N-target", -l2.d, l2.p_holm, l2.aile_n, l2.AE_CC))
src = pd.DataFrame(src, columns=["source", "Arm", "d", "p_holm", "family_size", "AE_CC_src"])
rows = []
for _, r in t3.iterrows():
    m = src[(src.Arm == r.Arm) & (np.abs(src.d - r.d) < 1e-6)]
    if len(m) == 0:
        rows.append((np.nan, np.nan, np.nan, "UNMATCHED")); continue
    m = m.iloc[0]
    rows.append((m.p_holm, m.family_size, m.AE_CC_src, m.source))
t3[["p_holm", "family_size", "AE_CC_src", "source"]] = pd.DataFrame(rows, index=t3.index)
# dog N-target rows (present in Table 3 of the paper, absent from the long table)
bmap = {"kayit": "dog × recording (N>=60)", "dogal2": "natural 2-min (N>=60)", "APP120": "APP 120-s"}
add = []
for _, r in r075[(r075.scen == "Kopek-ici") & (r075.feat.astype(str) == "32") & (r075.kol == "N-hedef")].iterrows():
    add.append({"Scenario": f"{r.space} within-dog", "Bag_set": bmap[r.btype], "Arm": "N-target", "AE_arm": r.AE_kol,
                "d": r.d, "lo": r.lo, "hi": r.hi, "category": r.kategori, "n": r.n, "N_bags": r.n_torba,
                "p_holm": r.p_holm, "family_size": r.aile_n, "AE_CC_src": r.AE_CC, "source": "r07_5_tests.csv"})
# DM and DR in the primary samples of S4-Smit-closed, S3 and S4-Dunford-closed (in Table 3, absent from the long table)
r08map = {"S4-Smit-kapali": ("S4-Smit-closed", "Smit 30-min bags (68)"), "S3": ("S3 [exploratory]", "Dunford cat bags (N>=60)"),
          "S4-Dunford-kapali": ("S4-Dunford-closed [exploratory]", "Dunford cat bags (N>=60)")}
for _, r in r08[(r08.aile == "birincil") & r08.senaryo.isin(list(r08map))].iterrows():
    sc, bs = r08map[r.senaryo]
    add.append({"Scenario": sc, "Bag_set": bs, "Arm": {"DM": "DM", "DR-APP-q": "DR-APP", "DR-dogal-q": "DR-natural"}[r.kol],
                "AE_arm": r.AE_kol, "d": r.d, "lo": r.lo, "hi": r.hi, "category": r.kategori, "n": r.n, "N_bags": r.n_torba,
                "p_holm": r.p_holm, "family_size": r.aile_n, "AE_CC_src": r.AE_CC, "source": "r08_tests.csv"})
# S1-L2 (collar) N-target: tested alone in r06 (d there = AE_N-target - AE_CC, so the sign is flipped)
add.append({"Scenario": "S1-L2 (collar)", "Bag_set": "Smit 30-min bags (68)", "Arm": "N-target", "AE_arm": l2.AE_Nh,
            "d": -l2.d, "lo": -l2.hi, "hi": -l2.lo, "category": l2.kategori.replace("KAZANIR", "KAYBEDER"), "n": l2.n_ind,
            "N_bags": l2.n_bag, "p_holm": l2.p_holm, "family_size": l2.aile_n, "AE_CC_src": l2.AE_CC,
            "source": "r06_C_naive.csv (tested alone)"})
t3 = pd.concat([t3, pd.DataFrame(add)], ignore_index=True)
# AE_CC as mean over samples (as in the paper): cats from the source tables; dogs from r07_5_delta
dd = delta[(delta.scen == "Kopek-ici") & (delta.feat.astype(str) == "32")]
def ae_cc_samples(r):
    if "within-dog" in r.Scenario:
        bt = {v: k for k, v in bmap.items()}[r.Bag_set]
        return float(dd[(dd.space == r.Scenario.split()[0]) & (dd.btype == bt)].AE_CC.iloc[0])
    return r.AE_CC_src
t3["AE_CC"] = t3.apply(ae_cc_samples, axis=1)
t3["AE_arm_basis"] = np.where(t3.Scenario.str.contains("within-dog"), "mean over dogs", "mean over samples")
t3["AE_CC_same_basis"] = t3["AE_CC_src"]
t3 = t3[["Scenario", "Bag_set", "Arm", "n", "N_bags", "AE_CC", "AE_CC_same_basis", "AE_arm", "AE_arm_basis", "d", "lo", "hi", "p_holm", "family_size", "category", "source"]]
order = {k: i for i, k in enumerate(dict.fromkeys(t3.Scenario))}
arm_order = {a: i for i, a in enumerate(["PCC", "ACC", "PACC", "EMQ", "N-target", "DM", "DR-APP", "DR-natural"])}
t3 = t3.assign(_o=t3.Scenario.map(order), _a=t3.Arm.map(arm_order)).sort_values(["_o", "Bag_set", "_a"], kind="stable").drop(columns=["_o", "_a"])
t3 = translate(t3, {"Scenario": "scenario", "Bag_set": "sample_set", "Arm": "arm", "n": "n_individuals", "N_bags": "n_samples",
                    "lo": "ci_low", "hi": "ci_high"})
n_unmatched = int((t3.source == "UNMATCHED").sum())
sheets.append(("S3_Table3_long", "All cells of Table 3 and of the protocol comparisons: effects, 95% intervals, Holm-adjusted p-values and decisions.",
               "d = AE_CC − AE_arm, mean over individuals of per-individual differences (positive: arm better than CC); 95% individual-bootstrap interval; exact sign-flip test (dogs: 200,000 Monte Carlo sign assignments), Holm-adjusted within the family given by family_size. AE_CC is the mean over samples, as in Table 3. AE_arm and AE_CC_same_basis are on the basis given in AE_arm_basis (mean over samples for cats; mean over dogs for dogs, as stored). N-target is in the family of the source-calibrated arms except in S1-L2 (collar), where it was tested alone.",
               "table3_consolidated_results_long.csv; r07_master.csv; r08_tests.csv; r07_5_tests.csv; r07_4_tests.csv; r06_C_naive.csv; r07_5_delta.csv", t3))
# S4a -----------------------------------------------------------------
a = r074[r074.clf != "RF"].copy()
a = translate(a, {**STD, "clf": "classifier"})
sheets.append(("S4a_HGB_LR_tests", "Source-calibrated arms with histogram-based gradient boosting (HGB) and logistic regression (LR).",
               "d = AE_CC − AE_arm (positive: arm better). Holm within classifier × scenario × variant. Random forest results are in Table S3.", "r07_4_tests.csv", a))
# S4b -----------------------------------------------------------------
b = translate(rd("supp_S7_ECE_macroF1.csv"), {**STD, "clf": "classifier", "n_birey": "n_individuals", "acc": "accuracy", "variant": "variant"})
sheets.append(("S4b_macroF1_ECE", "Macro-F1, accuracy and pooled expected calibration error (ECE) for the three classifiers.", "", "supp_S7_ECE_macroF1.csv", b))
# S5 ------------------------------------------------------------------
s5 = translate(rd("supp_S13_dm_boundary.csv"), {"cell_scenario": "scenario", "cell_bag_type": "sample_type", "n_bags": "n_samples", "n_bags_class_zeroed": "n_samples_class_zeroed", "n_bags_true_also_zero": "n_samples_true_also_zero"})
sheets.append(("S5_DM_boundary", "Classes set to zero by distribution matching (DM): number of samples and share whose true proportion was also zero.", "", "supp_S13_dm_boundary.csv", s5))
# S6a / S6b -----------------------------------------------------------
s6 = translate(rd("supp_S3_delta_rule_kappa.csv"), {**STD, "set": "sample_set", "kural": "rule", "AE_kural": "AE_rule",
                "AE_alt": "AE_always_corrected", "alt_orani": "share_corrected", "n_ind": "n_individuals"})
sheets.append(("S6a_delta_rule_kappa", "Decision rule 'apply the correction only when the shift visible in CC estimates exceeds τ' (δ̂ rule) and the κ-PACC rule.",
               "τ chosen on S1 windows and frozen. d = AE_CC − AE_rule (positive: rule better). share_corrected: share of samples in which the rule applied the correction.", "supp_S3_delta_rule_kappa.csv", s6))
s6b = translate(rd("supp_S3b_delta_summary.csv"), {"set": "sample_set", "medyan": "median_delta", "p90": "p90_delta"})
_dd = delta[(delta.scen == "Kopek-ici") & (delta.feat.astype(str) == "32")].copy()
_dd = pd.DataFrame({"sample_set": _dd.space + " within-dog " + _dd.btype.map({"kayit": "recording", "dogal2": "natural 2-min", "APP120": "APP 120-s"}),
                    "n": _dd.n_torba.values, "median_delta": _dd.delta_medyan.values, "p90_delta": _dd.delta_p90.values})
s6b = pd.concat([s6b, _dd], ignore_index=True)
sheets.append(("S6b_delta_summary", "Prevalence shift δ = ½‖p − π_source‖₁ by sample set: median and 90th percentile.", "", "supp_S3b_delta_summary.csv; r07_5_delta.csv (dogs, 32 features)", s6b))
# S7a / S7b -----------------------------------------------------------
s7 = translate(rd("supp_S6_Dunford_A2.csv"), {"davranis": "behaviour", "A2": "TableA2_s", "tum_satir_40": "all_rows_div40_s",
               "blok_tekrari_haric_40": "block_repeats_removed_s", "literal_yinelenen_haric_40": "exact_duplicate_rows_removed_s"})
sheets.append(("S7a_Dunford_A2", "Dunford et al. (2024) data: seconds per behaviour as reported in their Table A2 and recomputed from the published file.",
               "Rows / 40 Hz, with all rows, without repeated blocks, and without exactly duplicated rows (the version used here).", "supp_S6_Dunford_A2.csv", s7))
s7b = translate(rd("supp_S6b_Dunford_split_unit.csv"), {"dogrulama": "validation", "surum": "file_version", "n_kosu": "n_runs",
                "acc": "accuracy", "acc_sd": "accuracy_sd", "F1_sd": "macroF1_sd", "F1_min": "macroF1_min", "F1_max": "macroF1_max"})
sheets.append(("S7b_Dunford_split_unit", "Dunford data: effect of the validation unit and of the repeated blocks on macro-F1 (our features and pipeline; not a reproduction of the original analysis).", "", "supp_S6b_Dunford_split_unit.csv", s7b))
# S8 ------------------------------------------------------------------
s8 = translate(rd("supp_S1_R04_bias_variance.csv"), {"scen": "scenario", "size": "window_length_min", "est": "estimator",
               "n_bags": "n_windows", "min_ort": "mean_minutes", "bias_payi": "bias_share"})
sheets.append(("S8_R04_bias_variance", "Seven-class Smit data: bias–variance decomposition of time-budget error by window length and estimator.",
               "ACC and PACC use the confusion matrix estimated on the other cats; ACC-xfit and PACC-xfit use an oracle matrix cross-fitted within the held-out cat (blocks of the opposite type to the window), which is not available in practice. bias_share = bias² / MSE.", "supp_S1_R04_bias_variance.csv", s8))
# S9 ------------------------------------------------------------------
s9 = translate(rd("supp_S4_dog_S2.csv"), STD)
sheets.append(("S9_dog_S2_source_arms", "Dogs, back harness to collar (S2-dog, two classes, D17): source-calibrated arms.",
               "d = AE_CC − AE_arm (positive: arm better). AE values are means over dogs.", "supp_S4_dog_S2.csv", s9))
# S10 -----------------------------------------------------------------
s10 = translate(rd("supp_S5_weighted_vs_unweighted_retraining.csv"), {**STD, "aile": "family", "var": "variant", "kol_a": "arm_a", "kol_b": "arm_b", "n_bag": "n_samples"})
sheets.append(("S10_retraining_weighted", "Retraining on source plus target seconds, unweighted and weighted, against CC and against each other.",
               "m = labelled target seconds. d = AE(arm_b) − AE(arm_a) (positive: arm_a better).", "supp_S5_weighted_vs_unweighted_retraining.csv", s10))
# S11 -----------------------------------------------------------------
s11 = translate(rd("supp_S8_DR_h.csv"), {**STD, "aile": "family", "N_med": "N_median", "PACC_R07": "PACC_category", "PACC_R07_d": "PACC_d"})
sheets.append(("S11_DR_h", "Direct regression with the secondary descriptor (per-class posterior histograms, DR-h).",
               "d = AE_CC − AE_arm (positive: arm better). PACC results for the same cells are given for comparison.", "supp_S8_DR_h.csv", s11))
# S12-S17 -------------------------------------------------------------
s12 = translate(rd("supp_S9_R08_DR_details.csv"), {**STD, "varyant": "variant", "tanimlayici": "descriptor", "a_med": "components_median",
                "a_min": "components_min", "a_maks": "components_max", "n_med": "n_training_samples_median", "n_min": "n_training_samples_min",
                "n_maks": "n_training_samples_max", "kat": "n_outer_folds"})
sheets.append(("S12_R08_DR_details", "Direct regression: selected number of PLS components and number of training samples per outer fold.", "", "supp_S9_R08_DR_details.csv", s12))
s13 = translate(rd("supp_S9b_R08_extrapolation.csv"), {**STD, "varyant": "variant", "dis": "n_outside_training_range", "n": "n_evaluated", "oran": "share_outside_training_range"})
sheets.append(("S13_R08_extrapolation", "Direct regression: share of evaluation samples whose true class proportion lay outside the range of the training samples.", "", "supp_S9b_R08_extrapolation.csv", s13))
s14 = translate(rd("supp_S9c_R08_dm_boundary.csv"), {**STD, "sinir_orani": "share_boundary_solution"})
sheets.append(("S14_R08_DM_boundary_share", "Distribution matching: share of samples with a solution on the boundary of the simplex.", "", "supp_S9c_R08_dm_boundary.csv", s14))
s15 = translate(rd("supp_S10_B1_DM_boundary_classes.csv"), {**STD, "n_dm_sifir": "n_DM_zero", "dm_sifir_orani": "share_DM_zero",
                "gercek_sifir_orani": "share_true_also_zero", "gercek_medyan": "true_median", "gercek_q1": "true_q1", "gercek_q3": "true_q3"})
sheets.append(("S15_DM_zeroed_classes", "Distribution matching: classes set to zero, share of those samples whose true proportion was also zero, and median [IQR] of the true proportion.", "", "supp_S10_B1_DM_boundary_classes.csv", s15))
_s16 = rd("supp_S10b_B2_equivalent_inconclusive_RAE.csv")
def _space(r):
    if isinstance(r.uzay, str): return r.uzay
    for k in ("K-L1", "K-L2"):
        if r.senaryo.startswith(k): return k
    return "L2" if "L2" in r.senaryo else "L1"
_s16["uzay"] = _s16.apply(_space, axis=1)
_names = {"L1": ["inactive", "locomotion", "maintenance"], "L2": ["Lying", "Sitting", "Standing", "Walking", "Trotting", "Grooming", "Eating"],
          "K-L1": ["inactive", "locomotion"], "K-L2": ["Lying on chest", "Sitting", "Standing", "Walking", "Trotting"]}
def _rare(r):
    if isinstance(r.nadir_sinif_ad, str): return r.nadir_sinif_ad
    if pd.notna(r.nadir_sinif): return _names[r.uzay][int(r.nadir_sinif)]
    return np.nan
_s16["nadir_sinif_ad"] = _s16.apply(_rare, axis=1)
_s16 = _s16.drop(columns=["nadir_sinif"])
s16 = translate(_s16, {**STD, "kaynak": "source_table", "RAE_kol": "RAE_arm",
                "nadir_sinif": "rare_class_index", "nadir_gercek_ort_pct": "rare_class_true_mean_pct", "nadir_yanlilik_CC_pp": "rare_class_bias_CC_pp",
                "nadir_yanlilik_kol_pp": "rare_class_bias_arm_pp", "not": "note", "nadir_sinif_ad": "rare_class"})
sheets.append(("S16_equiv_inconclusive_RAE", "Equivalent and inconclusive comparisons: relative error of CC and arm, and signed bias of the rarest class.",
               "Rarest class = class with the smallest mean true proportion. Bias in percentage points.", "supp_S10b_B2_equivalent_inconclusive_RAE.csv", s16))
s17 = translate(rd("supp_S10c_B3_DR_zero_training_bags.csv"), {**STD, "torba_turu": "training_sample_type", "gercek_sifir_orani": "share_true_zero",
                "gercek_ort": "true_mean", "gercek_medyan": "true_median"})
sheets.append(("S17_DR_zero_train_samples", "Direct regression trained on natural windows: share of training samples in which a class was absent.", "", "supp_S10c_B3_DR_zero_training_bags.csv", s17))
# S18-S20 -------------------------------------------------------------
_s18 = rd("supp_S12_rae.csv"); _s18["direction_mismatch"] = _s18["direction_mismatch"].map({1.0: "TRUE", 0.0: "FALSE", True: "TRUE", False: "FALSE"})
s18 = translate(_s18, {"Scenario": "scenario", "Bag_set": "sample_set", "Arm": "arm", "lo": "ci_low_RAE", "hi": "ci_high_RAE", "n_bags": "n_samples", "d_AE_table": "d_AE", "AE_category": "category_AE"})
sheets.append(("S18_RAE_all_cells", "Relative absolute error (RAE) of CC and each arm for every cell of Table S3, with the direction-mismatch flag used in Table 4.",
               "d_RAE = RAE_CC − RAE_arm (mean over individuals; 95% individual-bootstrap interval). direction_mismatch = TRUE when the RAE interval lies above zero but the AE decision is not 'better'.", "supp_S12_rae.csv", s18))
s19 = translate(rd("supp_S14_openset_care_bias.csv"), {"olcu": "measure", "n_birey": "n_individuals", "n_torba": "n_windows", "lo": "ci_low", "hi": "ci_high"})
sheets.append(("S19_openset_care_bias", "Dog-to-cat transfer with maintenance absent from the source (S4-Smit-open): signed bias of the maintenance share under classify-and-count, by averaging level.",
               "Percentage points; 95% individual-bootstrap interval. The paper reports measure (i).", "supp_S14_openset_care_bias.csv", s19))
s20 = translate(r075.copy(), STD)
sheets.append(("S20_dog_tests", "Dogs: all source-calibrated arms and N-target by label set, feature set and sample type (within-dog and S2-dog).",
               "d = AE_CC − AE_arm (positive: arm better). AE values are means over dogs. Holm within label set × features × sample type.", "r07_5_tests.csv", s20))

# ---------------------------------------------------------------- renumber in order of first citation in the manuscript
RENUM = {"S1": "S1", "S2": "S2", "S3": "S3", "S20": "S4", "S9": "S5", "S4a": "S6a", "S4b": "S6b", "S5": "S7", "S14": "S8",
         "S15": "S9", "S17": "S10", "S13": "S11", "S11": "S12", "S12": "S13", "S18": "S14", "S10": "S15", "S19": "S16",
         "S6b": "S17a", "S6a": "S17b", "S7a": "S18a", "S7b": "S18b", "S8": "S19", "S16": "S20"}
def _key(t):
    m = re.match(r"S(\d+)([ab]?)", t); return (int(m.group(1)), m.group(2))
_ref = re.compile(r"(Tables? )(S\d+[ab]?(?:(?:, | and )S\d+[ab]?)*)")
def remap_refs(text):
    if not isinstance(text, str): return text
    def f(m):
        toks = sorted({RENUM[t] for t in re.findall(r"S\d+[ab]?", m.group(2))}, key=_key)
        joined = toks[0] if len(toks) == 1 else (", ".join(toks[:-1]) + " and " + toks[-1])
        return ("Tables " if len(toks) > 1 else "Table ") + joined
    return _ref.sub(f, text)
_new = []
for name, title, notes, source, df in sheets:
    old_id, rest = name.split("_", 1)
    df = df.copy()
    for c in df.columns:
        if not is_numeric_dtype(df[c]): df[c] = [remap_refs(x) for x in df[c]]
    _new.append((RENUM[old_id] + "_" + rest, remap_refs(title), remap_refs(notes), source, df))
sheets = sorted(_new, key=lambda x: _key(x[0].split("_")[0]))
# ---------------------------------------------------------------- write
idx = pd.DataFrame([{"table": "S" + s[0].split("_")[0][1:], "sheet": s[0], "title": s[1], "notes": s[2], "source_file": s[3], "rows": len(s[4])} for s in sheets])
general = pd.DataFrame({"Supplementary Material — Quantification corrections did not reduce time-budget error of reused accelerometer classifiers in cats and dogs": [
    "Conventions unless stated otherwise: AE = mean absolute error of class proportions; d = AE_CC − AE_arm (positive: the arm is better than classify-and-count).",
    "Decisions: better / worse = Holm-adjusted p < 0.05 and |d| ≥ 0.01 in that direction; equivalent = 95% interval within [−0.01, 0.01]; otherwise inconclusive.",
    "Scenarios: S0 random seconds (leaky); S1 new individual; S2 harness → collar; S3 Smit → Dunford; S4 dog → cat (open: target maintenance kept; closed: removed); R-D within Dunford; S1-closed: S1 without maintenance; L2: seven Smit classes; K-L1, K-L2: two and five dog classes. [exploratory] marks results based on the Dunford data.",
    "Sample types: natural 30-min and 2-min windows, cat samples (Dunford), dog recordings, and artificial-prevalence (APP) samples of 120 s.",
    "Code and source tables: https://github.com/masoudmaleki/cat-dog-time-budget"]})
xlsx = os.path.join(OUT, "Supplementary_Material.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    general.to_excel(w, sheet_name="Index", index=False, startrow=0)
    idx.to_excel(w, sheet_name="Index", index=False, startrow=len(general) + 2)
    for name, title, notes, source, df in sheets:
        head = pd.DataFrame({f"Table {name.split('_')[0]}. {title}": [notes] if notes else [""]})
        head.to_excel(w, sheet_name=name, index=False, startrow=0)
        df.to_excel(w, sheet_name=name, index=False, startrow=3)
        df.to_csv(os.path.join(OUT, "csv", name + ".csv"), index=False)
    from openpyxl.styles import Font, Alignment
    for ws in w.book.worksheets:
        for row in ws.iter_rows():
            for c in row:
                if c.row in (1,) or (ws.title != "Index" and c.row == 4) or (ws.title == "Index" and c.row == len(general) + 3):
                    c.font = Font(bold=True)
        for col in ws.columns:
            L = max((len(str(c.value)) if c.value is not None else 0) for c in col[3:]) if ws.max_row > 3 else 10
            ws.column_dimensions[col[0].column_letter].width = min(max(10, L + 2), 60)
        ws.freeze_panes = "A5" if ws.title != "Index" else None
print("unmatched in S3:", n_unmatched)
print("sheets:", len(sheets))
# residual Turkish check
tr_chars = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
tr_words = re.compile(r"\b(kedi|torba|kayit|kayıt|dogal|doğal|acik|açık|kapali|kapalı|birey|calisma|egitim|hedef|sinif|oran|yalniz|kopek|köpek|tasma|kosum|esl|BELIRSIZ|KAZANIR|KAYBEDER|ESDEGER)\b", re.I)
bad = []
for name, _, _, _, df in sheets:
    for c in df.columns:
        if tr_chars.search(str(c)) or tr_words.search(str(c)): bad.append((name, "COLUMN", c))
        if not is_numeric_dtype(df[c]):
            for v in df[c].dropna().astype(str).unique():
                if tr_chars.search(v) or tr_words.search(v): bad.append((name, c, v))
print("residual Turkish:", len(bad))
for x in bad[:40]: print("  ", x)
