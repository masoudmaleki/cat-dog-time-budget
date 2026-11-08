"""consolidate_supp.py - Supplementary Material v5: 23 v4 tables (S1-S20) -> 11 tables (S1-S11).

Reads the v4 tables written by build_supp.py (one CSV per sheet) and, if present, the labelling-strategy
comparison of round R15. No value is recomputed: every number is copied from the v4 tables or from R15.
Writes the workbook, one CSV per table or panel, a LaTeX file of the same content (rounded for print),
and the v4 -> v5 renumbering table.

Run:  python -I consolidate_supp.py <v4_csv_dir> <results_tables_dir> <out_dir>
"""
import sys, os, re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

V4, TAB, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)
TITLE = ("Quantification corrections did not meaningfully reduce time-budget error of reused "
         "accelerometer classifiers in cats and dogs")
AUTHORS = [("Mesut Melek", "Department of Software Engineering, Faculty of Engineering, Gümüşhane University, Gümüşhane, Türkiye"),
           ("Negin Melek", "Department of Electrical and Electronics Engineering, Faculty of Engineering, Gümüşhane University, Gümüşhane, Türkiye")]
CORR = "Negin Melek (negin.melek@gumushane.edu.tr)"
REPO = "https://github.com/masoudmaleki/cat-dog-time-budget"


def v4(name):
    return pd.read_csv(os.path.join(V4, name + ".csv"))


CHECK = []   # (v4 table, rows in v4, rows carried, note)

# ----------------------------------------------------------------- v4 -> v5 table numbers (for text cells)
RENUM = {"S1": "S1", "S2": "S2", "S3": "S3", "S4": "S3", "S5": "S3", "S6a": "S3", "S6b": "S4", "S7": "S5", "S8": "S5",
         "S9": "S5", "S10": "S6", "S11": "S6", "S12": "S3", "S13": "S6", "S14": "S7", "S15": "S3", "S16": "S8",
         "S17a": "S9", "S17b": "S9", "S18a": "S10", "S18b": "S10", "S19": "S11", "S20": "S7"}
PANEL = {"S10": "S6a", "S11": "S6b", "S13": "S6c", "S17a": "S9a", "S17b": "S9b", "S18a": "S10a", "S18b": "S10b"}
_ref = re.compile(r"(Tables? )(S\d+[ab]?(?:(?:, | and )S\d+[ab]?)*)")


def _key(t):
    m = re.match(r"S(\d+)([abc]?)", t)
    return (int(m.group(1)), m.group(2))


def remap(text):
    if not isinstance(text, str):
        return text
    def f(m):
        toks = sorted({RENUM[t] for t in re.findall(r"S\d+[ab]?", m.group(2))}, key=_key)
        j = toks[0] if len(toks) == 1 else ", ".join(toks[:-1]) + " and " + toks[-1]
        return ("Tables " if len(toks) > 1 else "Table ") + j
    return _ref.sub(f, text)


TABLES = []  # dict(id, sheet, title, notes, panels=[(panel_id, panel_title, df, tex_cols)])

# ================================================================= S1
s1 = v4("S1_D17_vs_matched32")
CHECK.append(("S1", len(s1), len(s1), "unchanged"))
TABLES.append(dict(id="S1", sheet="S1_D17_vs_mapped32",
    title="D17 versus the axis-mapped 32 features in harness-to-collar transfer (S2, Smit, three classes, 68 natural 30-min windows).",
    notes="d = AE_mapped32 − AE_D17 (mean over individuals; positive: D17 better). Holm adjustment over the three estimators.",
    panels=[(None, None, s1, None)]))

# ================================================================= S2
s2 = v4("S2_Analysis_history")
for c in s2.columns:
    s2[c] = [remap(x) for x in s2[c]]
CHECK.append(("S2", len(s2), len(s2), "table references renumbered"))
TABLES.append(dict(id="S2", sheet="S2_Analysis_history",
    title="Analysis history: every hypothesis and criterion tested, when it was specified, its result and where it is reported.",
    notes="R02–R08 are successive analysis rounds. 'after Rxx results were seen' marks analyses defined after earlier results were available. "
          "'this table only': the result is reported nowhere else. Effects are differences in AE with 95% individual-bootstrap intervals.",
    panels=[(None, None, s2, None)]))

# ================================================================= S3: all tests against a reference arm
COLS3 = ["group", "classifier", "scenario", "label_set", "features", "sample_set", "level", "m", "arm", "reference",
         "n_individuals", "n_samples", "AE_CC_samples", "AE_reference", "AE_arm", "AE_basis", "d", "ci_low", "ci_high",
         "p", "p_holm", "family_size", "p_method", "category", "source"]
GROUPS = {"A": "Cells of Tables 3 and 6 (random forest)",
          "B": "Dogs: other label and feature sets, and back harness to collar (S2-dog)",
          "C": "Gradient boosting (HGB) and logistic regression (LR)",
          "D": "Direct regression with the secondary descriptor (posterior histograms, DR-h)",
          "E": "Retraining with labelled target seconds, unweighted and weighted",
          "F": "Labelling strategy: stratified minus random, each relative to its own CC on the same seconds"}
rows = []
# --- A: v4 S3
a = v4("S3_Table3_long")
fix_basis = a.scenario.str.contains("within-dog") & a.source.str.startswith("r08")
n_fixed_basis = int(fix_basis.sum())
for i, r in a.iterrows():
    dog = "within-dog" in r.scenario
    basis = "mean over samples" if (not dog or r.source.startswith("r08")) else "mean over dogs (dog means first)"
    rows.append(dict(group="A", classifier="RF", scenario=r.scenario, sample_set=r.sample_set, arm=r.arm, reference="CC",
                     n_individuals=r.n_individuals, n_samples=r.n_samples, AE_CC_samples=r.AE_CC,
                     AE_reference=r.AE_CC_same_basis, AE_arm=r.AE_arm, AE_basis=basis, d=r.d, ci_low=r.ci_low,
                     ci_high=r.ci_high, p_holm=r.p_holm, family_size=r.family_size,
                     p_method="Monte Carlo (200,000)" if dog else "exact", category=r.category, source=r.source))
nA = len(rows)
# --- B: v4 S4 rows not already in A (v4 S5 is a subset of v4 S4)
b4 = v4("S4_dog_tests"); b5 = v4("S5_dog_S2_source_arms")
inA = np.array([((np.abs(a.d - r.d) < 1e-12) & (a.arm == r.arm)).any() for _, r in b4.iterrows()])
s5_in_s4 = sum(int(((np.abs(b4.d - r.d) < 1e-12) & (b4.arm == r.arm) & (b4.scenario == r.scenario)).sum() == 1) for _, r in b5.iterrows())
assert s5_in_s4 == len(b5), "v4 S5 is not contained in v4 S4"
delta = pd.read_csv(os.path.join(TAB, "r07_5_delta.csv"))
BT = {"recording": "kayit", "natural 2-min": "dogal2", "APP 120-s": "APP120"}
SC = {"within-dog": "Kopek-ici", "S2-dog": "S2-kopek"}
SS = {"recording": "dog recordings (≥ 60 s)", "natural 2-min": "dog natural 2-min windows", "APP 120-s": "APP 120-s samples"}
for _, r in b4[~inA].iterrows():
    dd = delta[(delta.scen == SC[r.scenario]) & (delta.space == r.label_space) & (delta.feat.astype(str) == str(r.features)) & (delta.btype == BT[r.sample_type])]
    rows.append(dict(group="B", classifier="RF", scenario=r.scenario, label_set=r.label_space, features=str(r.features),
                     sample_set=SS[r.sample_type], arm=r.arm, reference="CC", n_individuals=r.n_individuals,
                     n_samples=r.n_samples, AE_CC_samples=float(dd.AE_CC.iloc[0]) if len(dd) == 1 else np.nan,
                     AE_reference=r.AE_CC, AE_arm=r.AE_arm, AE_basis="mean over dogs (dog means first)", d=r.d,
                     ci_low=r.ci_low, ci_high=r.ci_high, p=r.p, p_holm=r.p_holm, family_size=r.family_size,
                     p_method=r.p_method, category=r.category, source="r07_5_tests.csv"))
nB = len(rows) - nA
# --- C: v4 S6a
c = v4("S6a_HGB_LR_tests")
for _, r in c.iterrows():
    rows.append(dict(group="C", classifier=r.classifier, scenario=r.scenario, sample_set=r.variant, arm=r.arm, reference="CC",
                     n_individuals=r.n_individuals, n_samples=r.n_samples, AE_CC_samples=r.AE_CC, AE_reference=r.AE_CC,
                     AE_arm=r.AE_arm, AE_basis="mean over samples", d=r.d, ci_low=r.ci_low, ci_high=r.ci_high, p=r.p,
                     p_holm=r.p_holm, family_size=r.family_size, p_method="exact", category=r.category, source="r07_4_tests.csv"))
# --- D: v4 S12
dh = v4("S12_DR_h")
for _, r in dh.iterrows():
    rows.append(dict(group="D", classifier="RF", scenario=r.scenario, sample_set=r.sample_type, arm=r.arm, reference="CC",
                     n_individuals=r.n_individuals, n_samples=r.n_samples, AE_CC_samples=r.AE_CC, AE_reference=r.AE_CC,
                     AE_arm=r.AE_arm, AE_basis="mean over samples", d=r.d, ci_low=r.ci_low, ci_high=r.ci_high, p=r.p,
                     p_holm=r.p_holm, family_size=r.family_size, p_method=r.p_method, category=r.category, source="r08_tests.csv"))
# --- E: v4 S15 (d = AE(arm_b) - AE(arm_a); positive: arm_a better)
e = v4("S15_retraining_weighted")
for _, r in e.iterrows():
    rows.append(dict(group="E", classifier="RF", scenario=r.scenario, sample_set=r.variant, level=r.level, m=r.m,
                     arm=r.arm_a, reference=r.arm_b, n_individuals=r.n_individuals, n_samples=r.n_samples,
                     AE_reference=r.AE_b, AE_arm=r.AE_a, AE_basis="mean over samples", d=r.d, ci_low=r.ci_low,
                     ci_high=r.ci_high, p=r.p, p_holm=r.p_holm, family_size=r.family_size, p_method="exact",
                     category=r.category, source="r06_F_tests.csv"))
# --- F: R15 labelling-strategy comparison (optional)
r15p = os.path.join(TAB, "r15_strategy_vs_own_cc.csv")
nF = 0
if os.path.exists(r15p):
    f = pd.read_csv(r15p)
    need = {"scen", "var", "level", "arm", "m", "n", "n_bag", "AE_CC_rand", "AE_arm_rand", "AE_CC_strat", "AE_arm_strat",
            "d", "lo", "hi", "p", "p_holm", "aile_n", "kategori"}
    assert need <= set(f.columns), f"r15 columns missing: {need - set(f.columns)}"
    ARM = {"T-ACC": "T-ACC", "PACC-hedef": "PACC-target", "Yalniz-hedef": "Target-only"}
    VAR = {"30dk_ge50": "natural 30-min windows", "acik": "open", "kapali": "closed", "-": "Dunford cat samples"}
    SCN = {"S3": "S3 [exploratory]", "S4-Dunford": "S4-Dunford [exploratory]"}
    CATS = {"KAZANIR": "better", "KAYBEDER": "worse", "ESDEGER": "equivalent", "BELIRSIZ": "inconclusive"}
    for _, r in f.iterrows():
        oc = {"acik": "open", "kapali": "closed"}.get(r["var"])
        scn_ = (f"{r.scen}-{oc}" if oc else r.scen)
        scn_ = {"S3": "S3 [exploratory]", "S4-Dunford-open": "S4-Dunford-open [exploratory]", "S4-Dunford-closed": "S4-Dunford-closed [exploratory]"}.get(scn_, scn_)
        ss_ = "Dunford cat sample" if str(r.scen).startswith(("S3", "S4-Dunford")) else "Smit 30-min"
        rows.append(dict(group="F", classifier="RF", scenario=scn_, sample_set=ss_,
                         level={"birey": "individual", "calisma": "study"}.get(r.level, r.level), m=r.m,
                         arm=f"{ARM.get(r.arm, r.arm)}, stratified (gain over own CC)",
                         reference=f"{ARM.get(r.arm, r.arm)}, random blocks (gain over own CC)",
                         n_individuals=r.n, n_samples=r.n_bag, AE_reference=r.AE_arm_rand, AE_arm=r.AE_arm_strat,
                         AE_basis=f"own CC: random {r.AE_CC_rand:.4f}, stratified {r.AE_CC_strat:.4f}",
                         d=r.d, ci_low=r.lo, ci_high=r.hi, p=r.p, p_holm=r.p_holm, family_size=r.aile_n,
                         p_method="exact", category=CATS.get(r.kategori, r.kategori), source="r15_strategy_vs_own_cc.csv"))
        nF += 1
s3 = pd.DataFrame(rows).reindex(columns=COLS3)
CHECK += [("S3", len(a), nA, f"group A; AE basis label corrected in {n_fixed_basis} within-dog DM/DR rows"),
          ("S4", len(b4), int(inA.sum()) + nB, f"{int(inA.sum())} rows already in group A, {nB} added as group B"),
          ("S5", len(b5), len(b5), "all rows contained in v4 S4 (group A or B)"),
          ("S6a", len(c), len(c), "group C"), ("S12", len(dh), len(dh), "group D (PACC comparison columns dropped; PACC rows are in group A)"),
          ("S15", len(e), len(e), "group E")]
TABLES.append(dict(id="S3", sheet="S3_All_tests",
    title="All statistical comparisons of an estimator with a reference arm: effects, 95% intervals, Holm-adjusted p-values and decisions.",
    notes="d = AE_reference − AE_arm, mean over individuals of per-individual differences (positive: the arm is better than the reference); "
          "95% individual-bootstrap interval; exact sign-flip test, or 200,000 random sign assignments where stated; Holm adjustment within "
          "the family whose size is given. AE_CC_samples is the error of CC as a mean over samples, as in Tables 3 and 6 of the paper; "
          "AE_reference and AE_arm are on the basis given in AE_basis. Groups: " + "; ".join(f"{k}, {v}" for k, v in GROUPS.items()) +
          ". In group F, d = [AE_CC − AE_arm]stratified − [AE_CC − AE_arm]random, each term computed on the seconds that remained after that "
          "strategy's labelled seconds were removed (positive: stratified labelling gains more over its own CC).",
    panels=[(None, None, s3, None)]))

# ================================================================= S4
s4 = v4("S6b_macroF1_ECE")
CHECK.append(("S6b", len(s4), len(s4), "unchanged"))
TABLES.append(dict(id="S4", sheet="S4_macroF1_ECE", title="Macro-F1, accuracy and pooled top-label expected calibration error (ECE, 10 bins) of the three classifiers.",
    notes="ECE is pooled over individuals and computed on the highest predicted class probability.", panels=[(None, None, s4, None)]))

# ================================================================= S5: distribution matching
z7, z8, z9 = v4("S7_DM_boundary"), v4("S8_R08_DM_boundary_share"), v4("S9_DM_zeroed_classes")
m79 = z9.merge(z7.rename(columns={"zeroed_class": "class"}), on=["scenario", "sample_type", "class"], suffixes=("", "_s7"), how="outer", indicator=True)
assert (m79._merge == "both").all(), "v4 S7 and S9 do not share the same rows"
assert (m79.n_samples == m79.n_samples_s7).all() and (m79.n_DM_zero == m79.n_samples_class_zeroed).all()
assert np.allclose(m79.share_true_also_zero, m79.share_true_also_zero_s7)
s5 = m79.merge(z8[["scenario", "sample_type", "share_boundary_solution"]], on=["scenario", "sample_type"], how="left")
assert s5.share_boundary_solution.notna().all()
s5 = s5[["scenario", "sample_type", "n_samples", "share_boundary_solution", "class", "n_DM_zero", "share_DM_zero",
         "n_samples_true_also_zero", "share_true_also_zero", "true_median", "true_q1", "true_q3"]].rename(
         columns={"n_samples_true_also_zero": "n_true_also_zero"})
CHECK += [("S7", len(z7), len(z7), "merged into S5 (identical counts verified)"), ("S8", len(z8), len(z8), "merged into S5 as share_boundary_solution"),
          ("S9", len(z9), len(z9), "S5")]
TABLES.append(dict(id="S5", sheet="S5_DM_boundary", title="Distribution matching (DM): solutions on the boundary of the simplex and classes set to zero.",
    notes="share_boundary_solution: share of samples in which DM set at least one class to zero. For each class: number and share of samples in which DM set it to zero, "
          "how many of those samples truly lacked the class, and the median [quartiles] of its true proportion in those samples.",
    panels=[(None, None, s5, None)]))

# ================================================================= S6: direct regression diagnostics (panels)
p6a, p6b, p6c = v4("S10_DR_zero_train_samples"), v4("S11_R08_extrapolation"), v4("S13_R08_DR_details")
CHECK += [("S10", len(p6a), len(p6a), "S6a"), ("S11", len(p6b), len(p6b), "S6b"), ("S13", len(p6c), len(p6c), "S6c")]
TABLES.append(dict(id="S6", sheet="S6_DR_diagnostics", title="Direct compositional regression (DR): diagnostics.", notes="",
    panels=[("S6a", "Training samples of DR-natural: share of samples in which a class was absent, and its mean and median true proportion.", p6a, None),
            ("S6b", "Share of evaluation samples whose true class proportion lay outside the range of the training samples.", p6b, None),
            ("S6c", "Selected number of PLS components and number of training samples per outer fold.", p6c, None)]))

# ================================================================= S7: relative error (v4 S14 + S20)
r14, r20 = v4("S14_RAE_all_cells"), v4("S20_equiv_inconclusive_RAE")
_ok = r14.d_AE_recomputed.notna()
assert (r14.AE_check[_ok] == "OK").all() and np.allclose(r14.d_AE[_ok], r14.d_AE_recomputed[_ok], atol=1e-6)
RARE = ["rare_class", "rare_class_true_mean_pct", "rare_class_bias_CC_pp", "rare_class_bias_arm_pp"]
s7 = r14.drop(columns=["d_AE_recomputed", "AE_check"]).copy()
for cc in RARE:
    s7[cc] = np.nan
s7["rare_class"] = s7["rare_class"].astype(object)
extra = []
n_join = 0
for _, r in r20.iterrows():
    k = s7.index[(s7.arm == r.arm) & (np.abs(s7.d_AE - r.d) < 5e-4) & (np.abs(s7.RAE_CC - r.RAE_CC) < 1e-6)]
    if len(k) == 1:
        assert abs(s7.loc[k[0], "RAE_arm"] - r.RAE_arm) < 1e-9
        for cc in RARE:
            s7.loc[k[0], cc] = r[cc]
        n_join += 1
    else:
        assert len(k) == 0
        extra.append(dict(scenario=r.scenario, sample_set=np.nan, arm=r.arm, category_AE=r.category, d_AE=r.d,
                          n_samples=r.n_samples, n_individuals=r.n_individuals, RAE_CC=r.RAE_CC, RAE_arm=r.RAE_arm,
                          note="target-labelled arm (m = 1,800, random blocks); interval not computed", **{cc: r[cc] for cc in RARE}))
# RAE of Table S3 group-A cells that were added to Table 3 after the RAE round (R15; optional)
r15rae = os.path.join(TAB, "r15_rae_added_cells.csv")
added = pd.DataFrame()
if os.path.exists(r15rae):
    added = pd.read_csv(r15rae)
    A_ = s3[s3.group == "A"]
    for _, r in added.iterrows():
        k = A_[(A_.scenario == r.scenario) & (A_.sample_set == r.sample_set) & (A_.arm == r.arm)]
        assert len(k) == 1 and abs(float(k.d.iloc[0]) - r.d_AE) < 1e-9, ("R15 RAE row does not match Table S3", r.scenario, r.sample_set, r.arm)
        assert not ((s7.arm == r.arm) & (np.abs(s7.d_AE - r.d_AE) < 1e-9) & (s7.scenario == r.scenario)).any(), "R15 RAE row already in v4 S14"
    added = added.reindex(columns=s7.columns)
nA_rae = len(s7) + len(added)
n_rae_ok = int(s7.RAE_CC.notna().sum() + (added.RAE_CC.notna().sum() if len(added) else 0))
n_flag = int((s7.direction_mismatch.astype(str).str.upper() == "TRUE").sum() + ((added.direction_mismatch.astype(str).str.upper() == "TRUE").sum() if len(added) else 0))
s7 = pd.concat([s7, added, pd.DataFrame(extra)], ignore_index=True)
s7 = s7[["scenario", "sample_set", "arm", "category_AE", "d_AE", "n_individuals", "n_samples", "RAE_CC", "RAE_arm", "d_RAE",
         "ci_low_RAE", "ci_high_RAE", "direction_mismatch"] + RARE + ["note"]]
CHECK += [("S14", len(r14), len(r14), "S7 (internal check columns d_AE_recomputed and AE_check verified and dropped)" + (f"; {len(added)} R15 rows added" if len(added) else "")),
          ("S20", len(r20), n_join + len(extra), f"{n_join} rows joined to S7 rows as rare-class columns, {len(extra)} added as rows")]
TABLES.append(dict(id="S7", sheet="S7_RAE", title="Relative absolute error (RAE) of classify-and-count and each estimator, and signed bias of the rarest class.",
    notes=f"The first {nA_rae} rows are comparisons of group A of Table S3; RAE could be computed for {n_rae_ok} of them (window-level class proportions were not stored for the four S2-L2 (mapped) rows), "
          f"and {n_flag} have direction_mismatch = TRUE. d_RAE = RAE_CC − RAE_arm (mean over individuals; 95% individual-bootstrap interval). "
          "direction_mismatch = TRUE when the RAE interval lies above zero but the AE decision is not 'better' (Table 4 of the paper). "
          "Rarest class = class with the smallest mean true proportion; its signed bias is in percentage points and is given for equivalent and inconclusive comparisons. "
          "The last rows are target-labelled arms (m = 1,800, random blocks), for which RAE was computed without an interval." +
          ("" if len(added) == 0 else f" {len(added)} rows were computed in round R15 for cells added to Table 3 after the RAE round."),
    panels=[(None, None, s7, None)]))

# ================================================================= S8
s8 = v4("S16_openset_care_bias")
CHECK.append(("S16", len(s8), len(s8), "unchanged"))
TABLES.append(dict(id="S8", sheet="S8_openset_maintenance_bias",
    title="Dog-to-cat transfer with maintenance absent from the source (S4-Smit-open): signed bias of the maintenance share under classify-and-count, by averaging level.",
    notes="Percentage points; 95% individual-bootstrap interval. The paper reports measure (i).", panels=[(None, None, s8, None)]))

# ================================================================= S9 (panels)
p9a, p9b = v4("S17a_delta_summary"), v4("S17b_delta_rule_kappa")
CHECK += [("S17a", len(p9a), len(p9a), "S9a"), ("S17b", len(p9b), len(p9b), "S9b")]
TABLES.append(dict(id="S9", sheet="S9_prevalence_shift", title="Prevalence shift and shift-dependent decision rules.", notes="",
    panels=[("S9a", "Prevalence shift δ = ½‖p − π_source‖₁ by sample set: median and 90th percentile.", p9a, None),
            ("S9b", "Rule 'apply the correction only when the shift visible in CC estimates exceeds τ' (δ̂ rule) and the κ-PACC rule. "
                    "τ was chosen on S1 windows and frozen; d = AE_CC − AE_rule (positive: rule better); share_corrected: share of samples in which the rule applied the correction.", p9b, None)]))

# ================================================================= S10 (panels; labels of the cleaning rule corrected)
p10a, p10b = v4("S18a_Dunford_A2"), v4("S18b_Dunford_split_unit")
p10b["file_version"] = p10b["file_version"].replace({"v2 (exact duplicate rows removed)": "v2 (repeated blocks removed; used in this study)"})
assert not p10b.file_version.str.contains("exact duplicate").any()
p10a = p10a.rename(columns={"block_repeats_removed_s": "block_repeats_removed_s (used here)",
                            "exact_duplicate_rows_removed_s": "every_duplicate_row_removed_s (comparison only)"})
CHECK += [("S18a", len(p10a), len(p10a), "S10a; note on the version used corrected"), ("S18b", len(p10b), len(p10b), "S10b; version label corrected")]
TABLES.append(dict(id="S10", sheet="S10_Dunford_data", title="Dunford et al. (2024) data: repeated blocks and validation unit.", notes="",
    panels=[("S10a", "Seconds per behaviour as reported in Table A2 of Dunford et al. (2024) and recomputed from the published file (rows / 40 Hz): all rows; "
                     "without the repeated 38–42-row blocks (the version analysed in this study); and, for comparison only, without every exactly duplicated row.", p10a, None),
            ("S10b", "Effect of the validation unit and of the repeated blocks on macro-F1 (our features and pipeline; not a reproduction of the original analysis).", p10b, None)]))

# ================================================================= S11
s11 = v4("S19_R04_bias_variance")
CHECK.append(("S19", len(s11), len(s11), "unchanged"))
TABLES.append(dict(id="S11", sheet="S11_bias_variance", title="Seven-class Smit data: bias–variance decomposition of time-budget error by window length and estimator.",
    notes="ACC and PACC use the confusion matrix estimated on the other cats; ACC-xfit and PACC-xfit use a matrix cross-fitted within the held-out cat "
          "(blocks of the opposite type to the window), which is not available in practice. bias_share = bias² / MSE.",
    panels=[(None, None, s11, None)]))

# ================================================================= checks
chk = pd.DataFrame(CHECK, columns=["v4_table", "rows_v4", "rows_carried", "how"])
chk["v5_table"] = [PANEL.get(t, RENUM[t]) for t in chk.v4_table]
chk = chk[["v4_table", "v5_table", "rows_v4", "rows_carried", "how"]].sort_values("v4_table", key=lambda s: s.map(_key))
lost = int((chk.rows_v4 != chk.rows_carried).sum())
chk.to_csv(os.path.join(OUT, "supp_renumbering_v4_to_v5.csv"), index=False)

# ================================================================= workbook + csv
from openpyxl.styles import Font
general = [
    f"Supplementary Material — {TITLE}",
    "Melek M, Melek N. Code and source tables: " + REPO,
    "Conventions unless stated otherwise: AE = mean absolute error of class proportions; d = AE_reference − AE_arm (positive: the arm is better).",
    "Decisions: better / worse = Holm-adjusted p < 0.05 and |d| ≥ 0.01 in that direction; equivalent = 95% interval within [−0.01, 0.01]; otherwise inconclusive.",
    "Scenarios: S0 random seconds (leaky); S1 new individual; S2 harness → collar; S3 Smit → Dunford; S4 dog → cat (open: target maintenance kept; closed: removed); "
    "R-D within Dunford; S1-closed: S1 without maintenance; L2: seven Smit classes; K-L1, K-L2: two and five dog classes. [exploratory] marks results based on the Dunford data.",
    "Sample types: natural 30-min and 2-min windows, cat samples (Dunford), dog recordings, and artificial-prevalence (APP) samples of 120 s.",
]
idx = []
for t in TABLES:
    for pid, ptitle, df, _ in t["panels"]:
        idx.append(dict(table=pid or t["id"], sheet=t["sheet"], title=(ptitle or t["title"]), rows=len(df)))
idx = pd.DataFrame(idx)
xlsx = os.path.join(OUT, "Supplementary_Material.xlsx")
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    pd.DataFrame({general[0]: general[1:]}).to_excel(w, sheet_name="Index", index=False, startrow=0)
    idx.to_excel(w, sheet_name="Index", index=False, startrow=len(general) + 2)
    for t in TABLES:
        r0 = 0
        head = pd.DataFrame({f"Table {t['id']}. {t['title']}": [t["notes"]]})
        head.to_excel(w, sheet_name=t["sheet"], index=False, startrow=r0); r0 += 3
        for pid, ptitle, df, _ in t["panels"]:
            if pid:
                pd.DataFrame({f"({pid[-1]}) {ptitle}": []}).to_excel(w, sheet_name=t["sheet"], index=False, startrow=r0); r0 += 1
            df.to_excel(w, sheet_name=t["sheet"], index=False, startrow=r0)
            ws = w.sheets[t["sheet"]]
            for cell in ws[r0 + 1]:
                cell.font = Font(bold=True)
            r0 += len(df) + 3
            name = (pid or t["id"]) + "_" + t["sheet"].split("_", 1)[1]
            df.to_csv(os.path.join(OUT, "csv", name + ".csv"), index=False)
    for ws in w.book.worksheets:
        ws["A1"].font = Font(bold=True)
        for col in ws.columns:
            L = max((len(str(c.value)) if c.value is not None else 0) for c in col[3:]) if ws.max_row > 3 else 10
            ws.column_dimensions[col[0].column_letter].width = min(max(10, L + 2), 60)

# ================================================================= LaTeX (rounded for print)
class Raw(str):
    """Already-formatted LaTeX; not escaped."""


def esc(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    if isinstance(s, Raw):
        return str(s)
    s = str(s)
    for a_, b_ in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                   ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a_, b_)
    for a_, b_ in [("δ̂", r"$\hat{\delta}$"), ("δ", r"$\delta$"), ("½", r"$\tfrac{1}{2}$"), ("‖", r"$\Vert$"), ("≥", r"$\geq$"),
                   ("≤", r"$\leq$"), ("→", r"$\rightarrow$"), ("−", r"$-$"), ("×", r"$\times$"), ("²", r"$^{2}$"), ("π", r"$\pi$"),
                   ("κ", r"$\kappa$"), ("τ", r"$\tau$"), ("–", "--"), ("—", "---"), ("≈", r"$\approx$"), ("±", r"$\pm$"),
                   ("Δ", r"$\Delta$"), ("ε", r"$\varepsilon$"), ("λ", r"$\lambda$"), ("₁", r"$_{1}$"), ("’", "'"), ("‘", "`"),
                   ("“", "``"), ("”", "''"), ("…", r"\ldots{}"), ("·", r"$\cdot$"), ("μ", r"$\mu$"), ("σ", r"$\sigma$"),
                   ("θ", r"$\theta$"), ("≠", r"$\neq$"), ("§", r"\S{}")]:
        s = s.replace(a_, b_)
    return s


SIGNED = {"d", "ci_low", "ci_high", "d_RAE", "ci_low_RAE", "ci_high_RAE", "d_AE", "bias_pp", "rare_class_bias_CC_pp", "rare_class_bias_arm_pp"}
PCOL = {"p", "p_holm", "p_min_attainable"}


def num(v, col):
    if isinstance(v, Raw):
        return str(v)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, str):
        return esc(v)
    x = float(v)
    if col in PCOL:
        if x == 0:
            return "0"
        if x < 0.001:
            m_, e_ = f"{x:.1e}".split("e")
            return f"${m_}\\times10^{{{int(e_)}}}$"
        return f"{x:.3f}"
    if col in SIGNED:
        return f"${x:+.3f}$" if col not in ("bias_pp",) and not col.endswith("_pp") else f"${x:+.1f}$"
    if col.startswith("AE") or col in ("ECE_pooled", "mse", "var", "bias2") or col.startswith(("mse_", "var_", "bias2_")):
        return f"{x:.4f}"
    if col.endswith(("_pct", "_s")) or "_s (" in col:
        return f"{x:,.1f}" if abs(x) >= 1000 else f"{x:.1f}" if not col.endswith("_s") and "_s (" not in col else f"{x:,.2f}"
    if x.is_integer() and (col.startswith("n") or col in ("family_size", "m", "rows", "window_length_min", "components_min", "components_max", "kat")):
        return f"{int(x):,}"
    if x.is_integer() and abs(x) >= 10:
        return f"{int(x):,}"
    return f"{x:.3f}"


def plain(s):
    return re.sub(r"\\[a-zA-Z]+|[{}$^]", "", s)


def widths(df, heads, total=25.0, cap=6.0):
    """Column widths (cm) from the longest rendered entry; text columns share what numbers leave."""
    cw = 0.155  # cm per character, scriptsize
    need = []
    for c, h in zip(df.columns, heads):
        vals = [plain(num(v, c)) for v in df[c].head(600)]
        L = max([len(x) for x in vals] + [1])
        H = max(len(w) for w in plain(h).split()) if h.strip() else 1
        need.append((max(L, H) * cw + 0.1, H * cw + 0.1))
    w = [min(n, cap) for n, _ in need]
    s = sum(w) + 0.1 * len(w)
    if s > total:
        # shrink only the wide (text) columns, never below their longest header word
        excess = s - total
        wide = [i for i, x in enumerate(w) if x > 2.0]
        tot_w = sum(w[i] - max(need[i][1], 1.8) for i in wide)
        for i in wide:
            room = w[i] - max(need[i][1], 1.8)
            w[i] -= excess * room / tot_w if tot_w > 0 else 0
    return [round(x, 2) for x in w]


def tex_table(df, heads=None, w=None, size=r"\scriptsize"):
    cols = list(df.columns)
    heads = heads or [esc(c.replace("_", " ")) for c in cols]
    w = w or widths(df, heads)
    spec = "".join(f">{{\\raggedright\\arraybackslash}}p{{{x}cm}}" for x in w)
    L = [r"{" + size, r"\setlength{\tabcolsep}{2.5pt}", r"\begin{longtable}{@{}" + spec + "@{}}", r"\toprule",
         " & ".join(r"\textbf{" + h + "}" for h in heads) + r" \\", r"\midrule", r"\endhead",
         r"\midrule \multicolumn{" + str(len(cols)) + r"}{r}{\textit{continued}} \\ \endfoot", r"\bottomrule", r"\endlastfoot"]
    for r in df.itertuples(index=False):
        L.append(" & ".join(num(v, c) for v, c in zip(r, cols)) + r" \\")
    L += [r"\end{longtable}}"]
    return "\n".join(L)


SHORT = {"Smit natural 30-min windows (68)": "Smit 30-min (68)", "Dunford, one sample per cat (≥ 60 s)": "Dunford cat sample",
         "dog recordings (≥ 60 s)": "dog recordings", "dog natural 2-min windows": "dog 2-min", "APP 120-s samples": "APP 120-s",
         "natural 2-min windows": "Smit 2-min", "30-min (coverage ≥ 50%)": "Smit 30-min", "natural 30-min windows": "Smit 30-min"}


def dci(d_, lo, hi):
    if pd.isna(d_):
        return Raw("")
    return Raw(f"{num(d_, 'd')} [{num(lo, 'd')}, {num(hi, 'd')}]")


# --- S3 print layout
S3P = pd.DataFrame({
    "group": s3.group, "classifier": s3.classifier, "scenario": s3.scenario,
    "labels": [", ".join(str(x) for x in (l, f_) if pd.notna(x) and str(x) != "") for l, f_ in zip(s3.label_set, s3.features)],
    "samples": [SHORT.get(x, x) for x in s3.sample_set],
    "lvl": [", ".join(x for x in ((str(l) if pd.notna(l) else ""), (f"m = {int(m_):,}" if pd.notna(m_) else "")) if x) for l, m_ in zip(s3.level, s3.m)],
    "arm": s3.arm, "reference": s3.reference, "n": s3.n_individuals, "AE_ref": s3.AE_reference, "AE_arm": s3.AE_arm,
    "dci": [dci(*x) for x in zip(s3.d, s3.ci_low, s3.ci_high)],
    "pk": [Raw(f"{num(p_, 'p_holm')} ({int(k)}){'*' if 'Monte' in str(pm) else ''}") if pd.notna(p_) else Raw("") for p_, k, pm in zip(s3.p_holm, s3.family_size, s3.p_method)],
    "category": s3.category})
S3H = ["Grp", "Clf", "Scenario", "Labels, features", "Samples", "Level, $m$", "Arm", "Reference", "$n$", "AE ref.", "AE arm",
       "$d$ [95\\% CI]", "$p_{\\mathrm{Holm}}$ (family)", "Decision"]
S3W = [0.5, 0.6, 2.85, 1.3, 2.1, 1.5, 3.45, 3.0, 0.45, 0.95, 0.95, 3.1, 1.75, 1.45]
# --- S7 print layout
S7P = pd.DataFrame({
    "scenario": s7.scenario, "samples": [SHORT.get(x, x) if isinstance(x, str) else "" for x in s7.sample_set], "arm": s7.arm,
    "ae": [Raw(f"{esc(c_)} ({num(d_, 'd')})") if pd.notna(d_) else Raw(esc(c_)) for c_, d_ in zip(s7.category_AE, s7.d_AE)],
    "n": s7.n_individuals, "RAE_CC": s7.RAE_CC, "RAE_arm": s7.RAE_arm,
    "dci": [dci(*x) for x in zip(s7.d_RAE, s7.ci_low_RAE, s7.ci_high_RAE)],
    "mm": ["yes" if x is True or x == "TRUE" or x == 1 else "" for x in s7.direction_mismatch],
    "rare": s7.rare_class, "true": s7.rare_class_true_mean_pct, "bcc": s7.rare_class_bias_CC_pp, "barm": s7.rare_class_bias_arm_pp,
    "note": s7.note})
S7H = ["Scenario", "Samples", "Arm", "AE decision ($d$)", "$n$", "RAE CC", "RAE arm", "$d_{\\mathrm{RAE}}$ [95\\% CI]", "Mis\\-match",
       "Rarest class", "True \\%", "Bias CC (pp)", "Bias arm (pp)", "Note"]
S7W = [2.5, 2.3, 3.0, 2.8, 0.45, 0.95, 0.95, 3.5, 0.8, 1.5, 0.8, 1.1, 1.1, 2.2]
for col_ in ("true", "bcc", "barm"):
    S7P[col_] = [Raw(f"${x:+.1f}$" if col_ != "true" else f"{x:.1f}") if pd.notna(x) else Raw("") for x in S7P[col_]]

PRINT = {"S3": (S3P, S3H, S3W), "S7": (S7P, S7H, S7W)}
TEXW = {"S2": [0.8, 4.0, 4.3, 3.2, 8.0, 3.6]}

doc = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[utf8]{inputenc}", r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern}",
       r"\usepackage[landscape,margin=1.5cm]{geometry}", r"\usepackage{amsmath}", r"\usepackage{booktabs}", r"\usepackage{longtable}",
       r"\usepackage{array}", r"\usepackage{microtype}", r"\usepackage[hidelinks]{hyperref}", r"\setlength{\parindent}{0pt}",
       r"\setlength{\LTpre}{2pt}", r"\setlength{\LTpost}{8pt}", r"\setlength{\LTleft}{0pt}", r"\pagestyle{plain}",
       r"\begin{document}", r"\begin{center}", r"{\Large\bfseries Supplementary Material}\\[6pt]", r"{\large " + esc(TITLE) + r"}\\[8pt]",
       " \\quad ".join(esc(n) + "$^{" + str(i + 1) + "}$" for i, (n, _) in enumerate(AUTHORS)) + r"\\[4pt]",
       r"{\small " + r"\\ ".join(f"$^{{{i + 1}}}$" + esc(af) for i, (_, af) in enumerate(AUTHORS)) + r"\\[2pt]",
       r"Corresponding author: " + esc(CORR) + r"}", r"\end{center}", r"\medskip", r"{\small"]
for g in general[2:]:
    doc.append(esc(g) + r"\par\smallskip")
doc.append(r"Values are rounded for print. The same tables at full numerical precision are in Supplementary\_Material.xlsx and in the code repository (\url{" + REPO + r"}).\par}")
doc.append(r"\medskip{\small\begin{tabular}{@{}l p{22cm}@{}}\toprule \textbf{Table} & \textbf{Content} \\ \midrule")
for t in TABLES:
    doc.append(t["id"] + " & " + esc(t["title"]) + r" \\")
doc.append(r"\bottomrule\end{tabular}}")
for t in TABLES:
    doc.append(r"\clearpage")
    doc.append(r"\subsection*{Table " + t["id"] + ". " + esc(t["title"]) + "}")
    if t["notes"]:
        extra_ = r" In the print version, * after the family size marks Monte Carlo $p$-values." if t["id"] == "S3" else ""
        doc.append(r"{\footnotesize " + esc(t["notes"]) + extra_ + r"\par}\smallskip")
    for pid, ptitle, df, _ in t["panels"]:
        if pid:
            doc.append(r"\smallskip{\small\textbf{(" + pid[-1] + r")} " + esc(ptitle) + r"\par}")
        if t["id"] in PRINT:
            df_, h_, w_ = PRINT[t["id"]]
            doc.append(tex_table(df_, h_, w_))
        else:
            heads = [esc(c.replace("_", " ")) for c in df.columns]
            doc.append(tex_table(df, heads, TEXW.get(t["id"])))
doc.append(r"\end{document}")
tex = "\n".join(doc)
bad = sorted({ch for ch in tex if ord(ch) > 127 and ch not in "üÜşŞğĞıİçÇöÖé"})
with open(os.path.join(OUT, "Supplementary_Material.tex"), "w", encoding="utf-8") as fh:
    fh.write(tex)

print("tables:", len(TABLES), "| panels:", len(idx))
print("S3 rows: A", nA, "B", nB, "C", len(c), "D", len(dh), "E", len(e), "F", nF, "| total", len(s3))
print("v4 tables carried:", len(chk), "| tables with lost rows:", lost)
print("unescaped non-ASCII in .tex:", bad)
