r"""42_r07_5_aback_features.py — R07 §5: dog ABack 1-s features, with exactly the same handling as ANeck (dog section of 16_r05_features.py: resample_poly(3,10), DBA over the recording, 30-sample window = the full second of 100 samples).
Reads: dog_move.parquet (read-only). Writes: <data root>/interim/features/dog_aback_1s_v1.parquet (new file), results/tables/r07_5_aback_summary.json"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pyarrow.parquet as pq
from config import paths as P
from src import features_smit as F

t0 = time.time()
tb = pq.read_table(P.DOG_PARQUET, columns=["DogID", "TestNum", "t_sec", "ABack_x", "ABack_y", "ABack_z"])
dog = tb.column("DogID").to_numpy(); tst = tb.column("TestNum").to_numpy(); tsec = tb.column("t_sec").to_numpy()
A = np.column_stack([tb.column(c).to_numpy() for c in ("ABack_x", "ABack_y", "ABack_z")]).astype(np.float64); del tb
key = dog.astype(np.int64) * 10 + tst.astype(np.int64)
assert (np.argsort(key, kind="stable") == np.arange(len(key))).all()
bounds = np.flatnonzero(np.diff(key)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, len(key)]
out = []; nan_a = int(np.isnan(A).sum())
for a, b in zip(starts, ends):
    n_i = np.rint(tsec[a:b] * 100).astype(np.int64)
    assert n_i[0] == 0 and (np.diff(n_i) == 1).all()
    W = (b - a) // 100; T = W * 100
    df, c, ew = F.signal_features(A[a:a + T], 3, 10, n=30, k=30, partial=True)
    df["DogID"] = int(dog[a]); df["TestNum"] = int(tst[a]); df["sec"] = np.arange(W); df["edge"] = ew
    out.append(df)
DB = pd.concat(out, ignore_index=True); DB.to_parquet(P.FEATURES / "dog_aback_1s_v1.parquet")
json.dump(dict(rows=len(DB), recordings=len(starts), nan_in_signal=nan_a, abs_max=float(np.abs(A).max()), clip_frac_ge7_9=float((np.abs(A) >= 7.9).mean())), open(P.TABLES / "r07_5_aback_summary.json", "w"), indent=1)
print(len(DB), "%.0f s" % (time.time() - t0))
