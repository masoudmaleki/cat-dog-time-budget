r"""01_inspect_smit.py — inspection of the Smit data (items 1-6 and 8; item 7 = the R code, read by hand).
Reads: interim/smit_anno.parquet, smit_accel.parquet (never touches the raw RDATA)
Writes: results/tables/smit_*.csv, smit_summary.json
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P

P.ensure_dirs()
T = P.TABLES
S = {}  # summary

anno = pd.read_parquet(P.INTERIM / "smit_anno.parquet")
BEH = [c for c in anno.columns if c not in ("Timestamp", "Pen", "Cat_id")]


def cat_of(c):
    n = c.split("_", 1)[1]
    for k, v in (("Lying", "Lying"), ("Sitting", "Sitting"), ("Standing", "Standing"), ("Walking", "Walking"),
                 ("Trotting", "Trotting"), ("Running", "Running"), ("Climbing", "Climbing"), ("Jumping", "Jumping"),
                 ("Grooming", "Grooming"), ("Nutrition.Eating", "Eating"), ("Nutrition.Drinking", "Drinking"),
                 ("Littering", "Littering"), ("Scratching", "Scratching"), ("Shake", "Shake"), ("Rubbing", "Rubbing"),
                 ("Playfight", "Playfight"), ("Rolling", "Rolling"), ("Social.Allogrooming", "Allogrooming"),
                 ("Social.Human", "HumanContact"), ("Outofsight", "OutOfSight"), ("Other", "Other"),
                 ("Start", "Start"), ("ActigraphOff", "ActigraphOff")):
        if n.startswith(k):
            return v
    return n


CATMAP = {c: cat_of(c) for c in BEH}
anno["rowsum"] = anno[BEH].sum(axis=1)
anno["label_fine"] = np.where(anno.rowsum == 1, anno[BEH].idxmax(axis=1), np.where(anno.rowsum == 0, "NONE", "MULTI"))
anno["label"] = anno.label_fine.map(lambda x: CATMAP.get(x, x))
anno["sec"] = anno.Timestamp.dt.floor("s")
pd.Series(CATMAP, name="category").rename_axis("column").to_csv(T / "smit_label_map.csv")

# ---------- 1. nesneler
objs = []
for fn, df in (("anno_data.RDATA/longanno", anno.drop(columns=["rowsum", "label_fine", "label", "sec"])),):
    objs.append(dict(nesne=fn, satir=len(df), sutun=df.shape[1], sutunlar=";".join(df.columns), tipler=";".join(map(str, df.dtypes))))

# ---------- accel (only the required columns)
acc_cols = pd.read_parquet(P.INTERIM / "smit_accel.parquet").head(0)  # schema
ACOLS = list(acc_cols.columns)
objs.append(dict(nesne="accel_data.RDATA/longaccl", satir=27778051, sutun=len(ACOLS), sutunlar=";".join(ACOLS),
                 tipler=";".join(map(str, acc_cols.dtypes))))
pd.DataFrame(objs).to_csv(T / "smit_objects.csv", index=False)
S["accel_columns"] = ACOLS
S["n_feature_cols"] = len([c for c in ACOLS if c not in ("Cat_id", "Position", "Timestamp")])

acc = pd.read_parquet(P.INTERIM / "smit_accel.parquet", columns=["Cat_id", "Position", "Timestamp", "X_Mean", "Y_Mean", "Z_Mean", "X_Sum", "VM_Sum", "VM_Mean"])
acc["Cat_id"] = acc.Cat_id.astype("category"); acc["Position"] = acc.Position.astype("category")
acc["sec"] = acc.Timestamp.dt.floor("s")
S["accel_rows"] = len(acc)
S["accel_ts_subsecond"] = int((acc.Timestamp != acc.sec).sum())
S["anno_ts_subsecond"] = int((anno.Timestamp != anno.sec).sum())
# samples-per-second check
smp = acc.sample(2_000_000, random_state=P.SEED)
n_est = (smp.X_Sum / smp.X_Mean).replace([np.inf, -np.inf], np.nan).dropna().round()
S["samples_per_sec_est"] = n_est.value_counts().head(5).to_dict()
S["n_cats"] = int(acc.Cat_id.nunique()); S["cats"] = sorted(acc.Cat_id.astype(str).unique())
S["anno_cats_equal_accel_cats"] = sorted(anno.Cat_id.unique()) == S["cats"]

# ---------- 3. kapsam
S["anno_rows"] = len(anno)
S["anno_rowsum_dist"] = anno.rowsum.value_counts().to_dict()
colsec = anno[BEH].sum().astype(int).rename("sn_anno_sutun")
colsec.to_csv(T / "smit_anno_seconds_by_column.csv")
catsec = anno[BEH].sum().groupby(CATMAP).sum().astype(int)
S["anno_cat_seconds_by_column_sum"] = catsec.to_dict()

# anno -> accel matching (Collar / Harness separately)
accC = acc[acc.Position == "Collar"][["Cat_id", "sec"]].astype({"Cat_id": str})
accH = acc[acc.Position == "Harness"][["Cat_id", "sec"]].astype({"Cat_id": str})
anno["Cat_id"] = anno.Cat_id.astype(str)
anno["has_C"] = anno.set_index(["Cat_id", "sec"]).index.isin(accC.set_index(["Cat_id", "sec"]).index)
anno["has_H"] = anno.set_index(["Cat_id", "sec"]).index.isin(accH.set_index(["Cat_id", "sec"]).index)
S["anno_rows_with_collar"] = int(anno.has_C.sum()); S["anno_rows_with_harness"] = int(anno.has_H.sum())
S["anno_rows_with_both"] = int((anno.has_C & anno.has_H).sum())

rows = []
for pos, col in (("Collar", "has_C"), ("Harness", "has_H")):
    sub = anno[anno[col]]
    for c in BEH:
        rows.append(dict(Position=pos, column=c, category=CATMAP[c], seconds=int(sub[c].sum())))
cov = pd.DataFrame(rows)
cov.to_csv(T / "smit_coverage_position_x_column.csv", index=False)
covc = cov.groupby(["Position", "category"]).seconds.sum().unstack(0)
covc["anno_all"] = catsec
covc.to_csv(T / "smit_coverage_position_x_category.csv")
S["coverage_cat"] = covc.to_dict()
# single-label seconds
single = anno[anno.rowsum == 1]
pd.crosstab(single.label, [single.has_C, single.has_H]).to_csv(T / "smit_single_label_by_match.csv")
# cat x placement
ck = pd.DataFrame({"Collar": anno.groupby("Cat_id").has_C.sum(), "Harness": anno.groupby("Cat_id").has_H.sum(),
                   "anno_rows": anno.groupby("Cat_id").size()})
ck.to_csv(T / "smit_cat_x_position_seconds.csv")
S["cat_x_position"] = ck.to_dict()

REF = {"scored_total": 166754, "OutOfSight": 38395, "Other": 4116, "usable": 124230, "Trotting": 308, "Climbing": 32,
       "Jumping": 239, "Running": 0, "Rolling": 0, "Drinking": 0, "HumanContact": 0}
tot_cols = int(catsec.sum())
data_vals = {"scored_total": int((anno.rowsum >= 1).sum()), "OutOfSight": int(catsec["OutOfSight"]), "Other": int(catsec["Other"]),
             "usable": int(((anno.rowsum == 1) & ~anno.label.isin(["OutOfSight", "Other", "Start", "ActigraphOff"])).sum()),
             **{k: int(catsec[k]) for k in ("Trotting", "Climbing", "Jumping", "Running", "Rolling", "Drinking", "HumanContact")}}
pd.DataFrame({"makale": REF, "veri": data_vals}).assign(fark=lambda d: d.veri - d.makale).to_csv(T / "smit_vs_paper_reference.csv")
S["ref_vs_data"] = {k: (REF[k], data_vals[k]) for k in REF}
S["rowsum_ge1_rows"] = int((anno.rowsum >= 1).sum())
S["sum_of_column_seconds"] = tot_cols
S["max_category_seconds"] = int(catsec.max())

# ---------- 4. unlabelled days
acc_days = acc.groupby([acc.sec.dt.date, "Position"], observed=True).size().unstack(1)
acc_days.to_csv(T / "smit_accel_seconds_by_day.csv")
ab = anno.groupby("Cat_id").sec.agg(["min", "max"])
lab_key = pd.MultiIndex.from_frame(anno[["Cat_id", "sec"]])
acc["Cat_id"] = acc.Cat_id.astype(str)
acc["labelled"] = pd.MultiIndex.from_frame(acc[["Cat_id", "sec"]]).isin(lab_key)
S["accel_rows_unlabelled"] = int((~acc.labelled).sum())
S["accel_rows_labelled"] = int(acc.labelled.sum())
S["accel_unlabelled_dates"] = [str(acc[~acc.labelled].sec.min()), str(acc[~acc.labelled].sec.max())]
S["accel_full_days"] = int((acc_days.sum(axis=1) == 2073600).sum())
S["accel_dates"] = [str(acc.sec.min()), str(acc.sec.max())]
S["anno_window"] = [str(anno.sec.min()), str(anno.sec.max())]
ab.to_csv(T / "smit_anno_window_by_cat.csv")

# ---------- 5. continuity
def seg_stats(ts):
    ts = np.sort(ts.values.astype("datetime64[s]").astype("int64"))
    if len(ts) == 0:
        return {}
    d = np.diff(ts)
    brk = np.where(d > 1)[0]
    starts = np.r_[0, brk + 1]; ends = np.r_[brk, len(ts) - 1]
    lens = ends - starts + 1
    return dict(n=len(ts), first=str(pd.Timestamp(ts[0], unit="s")), last=str(pd.Timestamp(ts[-1], unit="s")),
                gaps_gt1s=int(len(brk)), n_segments=len(lens), longest=int(lens.max()),
                median=float(np.median(lens)), p90=float(np.percentile(lens, 90)), dup=int((d == 0).sum()))


rows = []
for (c, pos), g in acc.groupby(["Cat_id", "Position"], observed=True):
    rows.append(dict(Cat_id=c, Position=pos, set="accel_tum_akis", **seg_stats(g.sec)))
    rows.append(dict(Cat_id=c, Position=pos, set="accel_etiketli_pencere", **seg_stats(g[g.labelled].sec)))
usable_lbl = ~anno.label.isin(["OutOfSight", "Other", "Start", "ActigraphOff", "NONE", "MULTI"])
for pos, col in (("Collar", "has_C"), ("Harness", "has_H")):
    for c, g in anno[anno[col]].groupby("Cat_id"):
        rows.append(dict(Cat_id=c, Position=pos, set="skorlu(rowsum>=1)+accel", **seg_stats(g[g.rowsum >= 1].sec)))
        rows.append(dict(Cat_id=c, Position=pos, set="kullanilabilir(tek etiket, OutOfSight/Other disi)+accel", **seg_stats(g[usable_lbl[g.index]].sec)))
seg = pd.DataFrame(rows)
seg.to_csv(T / "smit_continuity_cat_x_position.csv", index=False)
S["continuity_summary"] = seg.groupby(["set", "Position"]).agg(n=("n", "sum"), gaps=("gaps_gt1s", "sum"), longest_max=("longest", "max"),
                                                                 median_of_medians=("median", "median")).reset_index().to_dict("records")

# windows: consecutive, non-overlapping; from each cat's first annotated second; full windows only
win_rows = []
for pos, col in (("Collar", "has_C"), ("Harness", "has_H")):
    for defn, mask_fn in (("skorlu", lambda g: g.rowsum >= 1), ("kullanilabilir", lambda g: usable_lbl[g.index])):
        for c, g in anno[anno[col]].groupby("Cat_id"):
            t0 = anno[anno.Cat_id == c].sec.min(); t1 = anno[anno.Cat_id == c].sec.max()
            gs = g[mask_fn(g)]
            off = ((gs.sec - t0).dt.total_seconds()).astype(int)
            for wmin in (10, 30, 60):
                W = wmin * 60
                nwin = int(((t1 - t0).total_seconds() + 1) // W)
                cnt = np.bincount((off // W).clip(upper=nwin + 5), minlength=nwin + 6)[:nwin]
                cov_ = cnt / W
                win_rows.append(dict(Position=pos, tanim=defn, Cat_id=c, pencere_dk=wmin, tam_pencere=nwin,
                                     kapsam_ge90=int((cov_ >= 0.9).sum()), kapsam_ge50=int((cov_ >= 0.5).sum()),
                                     medyan_kapsam=float(np.median(cov_)) if nwin else np.nan))
win = pd.DataFrame(win_rows)
win.to_csv(T / "smit_natural_bags_cat_x_position.csv", index=False)
tot = win.groupby(["Position", "tanim", "pencere_dk"])[["tam_pencere", "kapsam_ge90", "kapsam_ge50"]].sum().reset_index()
tot.to_csv(T / "smit_natural_bags_totals.csv", index=False)
S["bags_totals"] = tot.to_dict("records")
S["bags_collar_30_skorlu_per_cat"] = win[(win.Position == "Collar") & (win.pencere_dk == 30) & (win.tanim == "skorlu")][["Cat_id", "tam_pencere", "kapsam_ge90"]].to_dict("records")
S["bags_collar_30_kullanilabilir_per_cat"] = win[(win.Position == "Collar") & (win.pencere_dk == 30) & (win.tanim == "kullanilabilir")][["Cat_id", "tam_pencere", "kapsam_ge90"]].to_dict("records")

# ---------- 6. collar/harness matching
kc = pd.MultiIndex.from_frame(accC[["Cat_id", "sec"]]); kh = pd.MultiIndex.from_frame(accH[["Cat_id", "sec"]])
inter = kc.intersection(kh)
S["match_all_collar_rows"] = len(kc); S["match_all_harness_rows"] = len(kh); S["match_all_both"] = len(inter)
S["match_all_union"] = len(kc.union(kh))
S["match_all_frac_of_union"] = len(inter) / len(kc.union(kh))
bothA = anno[anno.has_C & anno.has_H]
S["match_labelled_collar_only"] = int((anno.has_C & ~anno.has_H).sum())
S["match_labelled_harness_only"] = int((anno.has_H & ~anno.has_C).sum())
S["match_labelled_both"] = len(bothA)
S["match_labelled_frac_of_anno"] = len(bothA) / len(anno)
S["label_agreement_note"] = "Etiketler (Cat_id, saniye) başına tek tabloda; tasma ve koşum aynı etiket satırına bağlanır -> uyum %100 tanım gereği (bağımsız etiket yok)."
# feature agreement: VM_Mean correlation of collar/harness in the same second (informational)
mc = acc[(acc.Position == "Collar") & acc.labelled][["Cat_id", "sec", "VM_Mean"]].merge(
    acc[(acc.Position == "Harness") & acc.labelled][["Cat_id", "sec", "VM_Mean"]], on=["Cat_id", "sec"], suffixes=("_C", "_H"))
S["vm_mean_corr_collar_harness_labelled"] = float(mc.VM_Mean_C.corr(mc.VM_Mean_H))
# second offset: timestamp difference
S["accel_ts_offset_note"] = "Collar/Harness son zaman damgaları 0-41 sn farklı (satır sayıları da farklı)."

# ---------- 8. posture signature
lab = anno[["Cat_id", "sec", "rowsum", "label_fine", "has_C", "has_H"]]
POS = {"Standing": ["Inactive_Standing.Stationary"], "Sitting": ["Inactive_Sitting.Stationary"],
       "Lying": ["Inactive_Lying.Crouch", "Inactive_Lying.Down", "Inactive_Lying.Resting"]}
sig = []
for pos in ("Collar", "Harness"):
    a2 = acc[(acc.Position == pos) & acc.labelled][["Cat_id", "sec", "X_Mean", "Y_Mean", "Z_Mean"]]
    j = a2.merge(lab[lab.rowsum == 1], on=["Cat_id", "sec"])
    for name, cols in POS.items():
        g = j[j.label_fine.isin(cols)]
        cm = g.groupby("Cat_id")[["X_Mean", "Y_Mean", "Z_Mean"]].mean()
        sig.append(dict(dataset="Smit", Position=pos, posture=name, n_sec=len(g), n_cats=g.Cat_id.nunique(),
                        **{f"mean_{k}": g[f"{k}_Mean"].mean() for k in "XYZ"},
                        **{f"sd_{k}": g[f"{k}_Mean"].std() for k in "XYZ"},
                        **{f"catsd_{k}": cm[f"{k}_Mean"].std() for k in "XYZ"}))
pd.DataFrame(sig).to_csv(T / "smit_posture_signature.csv", index=False)
S["posture_signature"] = sig

Path(T / "smit_summary.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in S.items() if k not in ("accel_columns",)}, indent=1, default=str)[:12000])
