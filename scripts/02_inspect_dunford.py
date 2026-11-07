r"""02_inspect_dunford.py — inspection of Dunford 2024.
Reads: interim/dunford_cats.parquet (source: raw/dunford_dryad/Dunford_et_al._Cats_calibrated_data.csv), README.md
Writes: results/tables/dunford_*.csv, dunford_summary.json
Columns: Time (HH:MM:SS, no sub-second part), ID, AccX, AccY, AccZ, Behaviour. No segment column -> chunk = run of consecutive rows with the same ID + Behaviour.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P

df = pd.read_parquet(P.INTERIM / "dunford_cats.parquet")
S = {"rows": len(df), "columns": list(df.columns), "dtypes": df.dtypes.astype(str).to_dict(),
     "head3": df.head(3).to_dict("records"), "n_cats": int(df.ID.nunique()), "cats": sorted(df.ID.unique()),
     "behaviours": sorted(df.Behaviour.unique())}
FS = 40.0
# sampling: rows per second (Time carries no sub-second part)
g = df.groupby(["ID", "Time"], sort=False).size()
S["rows_per_timestamp_second"] = g.value_counts().head(6).to_dict()
S["rows_per_second_median"] = float(g.median())
sec_idx = pd.to_timedelta(df.Time)
S["time_monotonic_within_cat"] = bool(df.groupby("ID").apply(lambda x: pd.to_timedelta(x.Time).is_monotonic_increasing, include_groups=False).all())
d = sec_idx.groupby(df.ID).diff().dt.total_seconds().dropna()
S["time_diff_s_dist"] = d.value_counts().head(6).to_dict()
S["fs_est_hz_from_rows_per_second"] = S["rows_per_second_median"]
# cat x behaviour duration (s) = rows / 40
tab = (df.groupby(["ID", "Behaviour"]).size() / FS).unstack(1).fillna(0).round(2)
tab["TOPLAM"] = tab.sum(axis=1)
tab.loc["TOPLAM"] = tab.sum()
tab.to_csv(P.TABLES / "dunford_cat_x_behaviour_seconds.csv")
S["behaviour_seconds_total"] = tab.loc["TOPLAM"].to_dict()
S["paper_tableA2"] = {"groom": 13.08, "feed": 66.98, "rest": 2491.78, "walk": 533.65, "trot": 86.85, "run": 50.80, "shake": 7.15, "toplam": 3250.28, "parca_gt_2s": 116}
# chunks: ID + consecutive identical Behaviour; 2 s = >80 rows
tsec = pd.to_timedelta(df.Time).dt.total_seconds()
tdiff = tsec.diff()
S["rows_time_decreasing"] = int((tdiff < 0).sum())
key = (df.ID != df.ID.shift()) | (df.Behaviour != df.Behaviour.shift())
run_simple = key.cumsum()
S["runs_simple_total"] = int(run_simple.nunique())
key = key | ~tdiff.isin([0, 1])   # time discontinuity = new chunk
run = key.cumsum()
runs = df.groupby(run).agg(ID=("ID", "first"), Behaviour=("Behaviour", "first"), n=("ID", "size"))
runs["sec"] = runs.n / FS
S["runs_total"] = len(runs); S["runs_gt_2s"] = int((runs.sec > 2).sum())
runs.groupby("Behaviour").sec.agg(["count", "sum", "median", "max"]).round(2).to_csv(P.TABLES / "dunford_runs_by_behaviour.csv")
S["runs_gt_2s_by_behaviour"] = runs[runs.sec > 2].groupby("Behaviour").size().to_dict()
# unit / gravity / axis: per-behaviour axis means and VM
df["VM"] = np.sqrt(df.AccX ** 2 + df.AccY ** 2 + df.AccZ ** 2)
ax = df.groupby("Behaviour")[["AccX", "AccY", "AccZ", "VM"]].agg(["mean", "std"]).round(3)
ax.columns = ["_".join(c) for c in ax.columns]
ax.to_csv(P.TABLES / "dunford_axis_by_behaviour.csv")
rest = df[df.Behaviour.str.lower() == "rest"]
S["rest_axis_mean_g"] = rest[["AccX", "AccY", "AccZ"]].mean().round(4).to_dict()
S["rest_axis_sd_g"] = rest[["AccX", "AccY", "AccZ"]].std().round(4).to_dict()
S["rest_vm_mean_g"] = float(rest.VM.mean()); S["rest_n_cats"] = int(rest.ID.nunique())
rc = rest.groupby("ID")[["AccX", "AccY", "AccZ", "VM"]].mean().round(3)
rc.to_csv(P.TABLES / "dunford_rest_axis_mean_by_cat.csv")
S["acc_range"] = [float(df[["AccX", "AccY", "AccZ"]].min().min()), float(df[["AccX", "AccY", "AccZ"]].max().max())]
S["readme"] = (P.DUNFORD_RAW / "README.md").read_text(encoding="utf-8", errors="replace")
Path(P.TABLES / "dunford_summary.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in S.items() if k != "readme"}, indent=1, default=str))
print(tab.to_string()); print(ax.to_string()); print(rc.to_string())
