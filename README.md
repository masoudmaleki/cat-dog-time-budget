# Time-budget error when accelerometer behaviour classifiers are reused

Code and result tables for the study *"Do quantification corrections reduce time-budget error when accelerometer behaviour classifiers are reused?"*

**Question.** When a collar/harness accelerometer behaviour classifier is moved to another individual, sensor placement, study or species, how much does the estimated time budget (behaviour proportions) drift, and which quantification method (CC, PCC, ACC, PACC, EMQ, DM, DR, target-labelled variants) corrects it at the lowest labelling cost?
Primary outcome: time-budget error (AE, RAE, signed per-class bias). Macro-F1 is reported for comparison only.
The unit of splitting is the individual (cat / dog). The only exception is scenario S0, which splits random seconds and is labelled as leaky.

All randomness uses seed `20261105`. Statistics: individual-level paired differences, exact sign-flip test (Monte Carlo for 44-45 dogs), Holm correction within families, individual bootstrap 95 % CI (1,000 resamples).

## Data (not included)

| Dataset | Content | Source |
|---|---|---|
| Smit et al. 2023, *Sensors* 23:7165 | 12 cats, ActiGraph wGT3X-BT, 30 Hz, collar + harness | FigShare, doi:[10.6084/m9.figshare.23605842](https://doi.org/10.6084/m9.figshare.23605842) |
| Dunford et al. 2024, *Ecol Evol* 14:e11380 | 9 cats, Daily Diary, 40 Hz, collar | Dryad, doi:[10.5061/dryad.q2bvq83sx](https://doi.org/10.5061/dryad.q2bvq83sx) |
| Vehkaoja et al. 2022, *Data in Brief* 40:107822 | 45 dogs, ActiGraph GT9X Link, 100 Hz, collar + harness | Mendeley Data, `vxhx934tbn` |

Smit's original R code (feature definitions) is at <https://github.com/MSmit1992/Cat_Accelerometry>; `scripts/00_download.py` fetches it into `third_party/` (git clone fails on Windows because of a `:` in a file name, so a zip fallback is used).

**Download.** `python -I scripts/00_download.py` downloads the Smit data automatically. The Dunford (Dryad) and dog (Mendeley) files may be refused for automated clients (HTTP 401/403); if so, download them by hand into `<data root>/raw/dunford_dryad/` and `<data root>/raw/vehkaoja_mendeley/`. Then run `python -I scripts/00b_convert.py` (raw files to parquet in `<data root>/interim`). Raw files are never modified.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # Python 3.13
export KEDI_DATA_ROOT=/path/to/data                   # Windows (cmd): set KEDI_DATA_ROOT=D:\path\to\data
```

`KEDI_DATA_ROOT` (default `./data`) must contain `raw/`, and receives `interim/` (all derived files, caches). All paths are defined in `config/paths.py`; nothing but code and small result tables is written inside the repository. Run every script as `python -I scripts/<name>.py` from the repository root.

## Running order

Full re-training from the raw data takes about 12-14 h in total (R02 ~82 min, R03 inner CV ~56 min and label-budget run ~6.2 h; the other steps are minutes). Steps marked *no training* only read stored intermediates.

| Step | Scripts | Time |
|---|---|---|
| 1. Data | `00_download.py`, `00b_convert.py`, `01-03_inspect_*.py` | minutes |
| 2. Tests | `python -I tests/test_quant.py`, `test_features.py`, `test_dm.py`, `test_dr.py` | seconds |
| 3. R02 pilot | `04_run_smit_pilot.py --variant a --feat 32` (+ `--variant b --feat 32`, `--variant a --feat 36`), `05_analyze_r02.py` | ~82 min |
| 4. R03 diagnostics | `07_r03_A_sanity.py`, `08_r03_inner.py`, `09_r03_B.py`, `10_r03_D_run.py`, `11_r03_C.py`, `12_r03_E.py`, `13_r03_D_analyze.py` | ~7 h |
| 5. R04 bias-variance | `14_r04_run.py`, `15_r04_figures.py` (no training) | ~20 s |
| 6. R05 study/species shift | `16_r05_features.py`, `17_r05_gateA_counts.py`, `19_r05_sources.py`, `18_r05_ladder.py`, `20_r05_E.py`, `21_r05_F.py`, `22_r05_analyze.py` | hours |
| 7. R06 baselines, gain-shift | `23-36_r06_*.py` | hours |
| 8. R07 robustness, dogs | `37-47_r07_*.py` (`40_r07_4_run.py HGB`, `... LR`; `45_r07_6_strat.py ladder`, `smit`, `dunford`) | hours |
| 9. R08 DM / DR | `48_r08_dog_post.py` (~2 min), `49_r08_gateA.py`, `50_r08_run.py hepsi` (~4 min), `51_r08_analyze.py`, `52_r08_figures.py`, `53_r08_history.py` | minutes |
| 10. Audit, diagnostics | `55_r09_audit.py` (~90 s), `56_r09_diag.py` (~10 s) (no training) | ~2 min |
| 11. Dog RAE inputs | `57_r11_dog_rae.py` (no training; reads the stored R08 posteriors; ~35 s) | ~35 s |
| 12. Paper outputs | `make_paper_outputs.py` (no training) | ~15 s |

Scripts `06_make_r02_report.py` and `54_r08_report.py` only build internal Markdown reports (Turkish) into `results/reports/` and are not needed for the paper outputs. Script docstrings list the files each step reads and writes.
Intermediate results of steps 3-11 (posteriors, bag-level parquet files) go to `<data root>/interim/`. The bag-level tables `results/tables/r0*.csv` that `make_paper_outputs.py` reads are included in this repository, so step 12 can be repeated from the stored intermediates.

### Reproducing the paper tables and figures

```bash
python -I scripts/make_paper_outputs.py      # ~15 s; writes results/tables/paper/ and results/figures/paper/
```

The script also checks 17 headline numbers against the stored tables (`paper_numbers.csv`; it reports `eşleşen 17 / 17`). It needs the intermediates under `<data root>/interim/{r05,r06,r11}` (produced by steps 6-7 and 11) and the feature tables (label counts for Table 1).

## Outputs and their place in the paper

| Output | File(s) in `results/tables/paper/` or `results/figures/paper/` |
|---|---|
| Table 1 - datasets | `table1_datasets.csv/.md` |
| Table 2 - transfer scenarios (S0-S4, K-L1/K-L2) | `table2_transfer_scenarios.csv/.md` |
| Table 3 - consolidated results (d = AE_CC - AE_arm, CI, decision category; AE per arm) | `table3_consolidated_results.csv/.md`, `table3_consolidated_results_long.csv` |
| Table 4 - target-labelled arms, m = 1,800 (random block vs stratified) | `table4_target_labelled_arms.csv/.md` |
| Fig. 2 - forest plot of d by scenario and arm | `fig2_forest.png/.pdf` |
| Fig. 3 - APP vs natural bags (gain and shift delta) | `fig3_app_vs_natural.png/.pdf` |
| Fig. 4 - open set: signed per-class bias, care seconds | `fig4_open_set.png/.pdf` |
| Fig. 5 - label-budget curves | `fig5_label_budget.png/.pdf` |
| Supplementary S1 | `supp_S1_R04_bias_variance.csv` |
| S2 | `supp_S2_D17_vs_matched32.csv` |
| S3, S3b | `supp_S3_delta_rule_kappa.csv`, `supp_S3b_delta_summary.csv` |
| S4 | `supp_S4_dog_S2.csv` |
| S5 | `supp_S5_weighted_vs_unweighted_retraining.csv` |
| S6, S6b | `supp_S6_Dunford_A2.csv`, `supp_S6b_Dunford_split_unit.csv` |
| S7 | `supp_S7_ECE_macroF1.csv` |
| S8 | `supp_S8_DR_h.csv` |
| S9, S9b, S9c | `supp_S9_R08_DR_details.csv`, `supp_S9b_R08_extrapolation.csv`, `supp_S9c_R08_dm_boundary.csv` |
| S10, S10b, S10c | `supp_S10_B1_DM_boundary_classes.csv`, `supp_S10b_B2_equivalent_inconclusive_RAE.csv`, `supp_S10c_B3_DR_zero_training_bags.csv` |
| S11 | `supp_S11_analysis_history_v2.csv` (also `results/tables/r08_analysis_history_v2_2026-11-06.csv`) |
| S12 | `supp_S12_rae.csv` - RAE of CC and arm for every Table 3 cell, d_RAE, direction-mismatch flag (`note = source=R11` marks the within-dog R07-arm rows) |
| S13 | `supp_S13_dm_boundary.csv` |
| S14 | `supp_S14_openset_care_bias.csv` |

`supplement_index.md` lists the supplementary files with one-line descriptions. Labels in the CSV files keep the original (partly Turkish) column and category names (for example `KAZANIR` = wins, `KAYBEDER` = loses, `ESDEGER` = equivalent, `BELIRSIZ` = inconclusive).

## Label spaces

L1: `Inaktif` (lying + sitting + standing / rest), `Lokomosyon` (walk + trot [+ run]), `Bakim` (eat + groom; absent in dogs). L2 (Smit <-> dog only): lying, sitting, standing, walking, trotting. Features: 32 per-second features (Smit definition) and the axis-independent set D17.

## Citation and licence

See `CITATION.cff`. Code: MIT licence (`LICENSE`). The datasets keep their own licences; cite the original publications when using them.

The supplementary workbook (results/supplement/Supplementary_Material.xlsx) is built by scripts/supplement/build_supp.py from the tables in results/tables.
