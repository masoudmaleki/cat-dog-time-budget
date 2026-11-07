r"""03_inspect_dog.py — inspection of the Vehkaoja 2022 dog data.
Reads: interim/dog_move.parquet, raw/vehkaoja_mendeley/DogInfo.csv, Data_description.txt
Writes: results/tables/dog_*.csv, dog_summary.json
Labels: Behavior_1/2/3, empty = "<undefined>". Note: Data_description.txt lists the x/y names of ANeck swapped (typo); the file's order ANeck_x, ANeck_y, ANeck_z was used.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P

UND = "<undefined>"
df = pd.read_parquet(P.INTERIM / "dog_move.parquet")
S = {"rows": len(df), "columns": list(df.columns), "dtypes": df.dtypes.astype(str).to_dict(), "head3": df.head(3).round(4).to_dict("records"),
     "n_dogs": int(df.DogID.nunique()), "n_tests": int(df.groupby(["DogID", "TestNum"]).ngroups)}
S["data_description"] = (P.DOG_RAW / "Data_description.txt").read_text(encoding="utf-8", errors="replace")
di = pd.read_csv(P.DOG_RAW / "DogInfo.csv")
S["dog_info"] = dict(n=len(di), cols=list(di.columns), weight_min_med_max=[float(di.Weight.min()), float(di.Weight.median()), float(di.Weight.max())],
                     n_breeds=int(di.Breed.nunique()), age_months_min_med_max=[float(di["Age months"].min()), float(di["Age months"].median()), float(di["Age months"].max())],
                     gender_counts=di.Gender.value_counts().to_dict(), dogs_in_info_not_in_data=sorted(set(di.DogID) - set(df.DogID)))
dt = df.groupby(["DogID", "TestNum"]).t_sec.diff().dropna()
S["dt_median_s"] = float(dt.median()); S["fs_hz"] = float(1 / dt.median()); S["dt_dist"] = dt.round(4).value_counts().head(5).to_dict()
FS = S["fs_hz"]
for c in ("Behavior_1", "Behavior_2", "Behavior_3", "Task", "PointEvent"):
    df[c] = df[c].astype("category")
b1, b2, b3 = df.Behavior_1, df.Behavior_2, df.Behavior_3
single = (b2 == UND) & (b3 == UND) & (b1 != UND)
S["total_rows_s"] = len(df) / FS
S["labelled_b1_s"] = float((b1 != UND).sum() / FS); S["undefined_b1_s"] = float((b1 == UND).sum() / FS)
S["rows_with_2_labels_s"] = float(((b2 != UND) & (b3 == UND)).sum() / FS); S["rows_with_3_labels_s"] = float((b3 != UND).sum() / FS)
S["single_label_s_total"] = float(single.sum() / FS)
sl = (df[single].Behavior_1.value_counts() / FS).round(1)
sl.to_csv(P.TABLES / "dog_single_label_seconds.csv", header=["seconds"])
S["single_label_seconds"] = sl.to_dict()
allb = pd.concat([b1.value_counts(), b2.value_counts(), b3.value_counts()], axis=1).fillna(0).astype(int)
allb.columns = ["Behavior_1", "Behavior_2", "Behavior_3"]; (allb / FS).round(1).to_csv(P.TABLES / "dog_all_label_seconds.csv")
S["all_labels"] = sorted(allb.index.astype(str))
S["paper_single"] = {"Galloping": 37, "Lying on chest": 4633, "Sitting": 3895, "Sniffing": 10262, "Standing": 3709, "Trotting": 7174, "Walking": 7503, "toplam_kayit_s": 106110, "tanimsiz_s": 36494}
# posture signature (single-label)
sig = []
for dev in ("ANeck", "ABack"):
    for lab in ("Standing", "Sitting", "Lying chest"):
        g = df[single & (b1 == lab)]
        xyz = g[[f"{dev}_x", f"{dev}_y", f"{dev}_z"]].astype("float64")
        vm = np.sqrt((xyz ** 2).sum(axis=1))
        dm = xyz.groupby(g.DogID.values).mean()
        sig.append(dict(dataset="Dog", Position=dev, posture=lab, n_sec=len(g) / FS, n_dogs=int(g.DogID.nunique()),
                        **{f"mean_{k}": xyz[f"{dev}_{k}"].mean() for k in "xyz"}, **{f"sd_{k}": xyz[f"{dev}_{k}"].std() for k in "xyz"},
                        **{f"dogsd_{k}": dm[f"{dev}_{k}"].std() for k in "xyz"}, vm_mean=vm.mean(), vm_sd=vm.std()))
sig = pd.DataFrame(sig); sig.to_csv(P.TABLES / "dog_posture_signature.csv", index=False)
S["posture_signature"] = sig.round(3).to_dict("records")
S["acc_range_ANeck"] = [float(df[["ANeck_x", "ANeck_y", "ANeck_z"]].min().min()), float(df[["ANeck_x", "ANeck_y", "ANeck_z"]].max().max())]
S["acc_range_ABack"] = [float(df[["ABack_x", "ABack_y", "ABack_z"]].min().min()), float(df[["ABack_x", "ABack_y", "ABack_z"]].max().max())]
Path(P.TABLES / "dog_summary.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in S.items() if k != "data_description"}, indent=1, default=str))
print(sig.round(3).to_string())
