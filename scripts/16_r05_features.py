r"""16_r05_features.py — R05 A: Dunford (40 Hz) and dog ANeck (100 Hz) resampled to 30 Hz, then the 32 Smit-defined features.
Reads: <data root>/interim/dunford_cats.parquet, dog_move.parquet (read-only). Writes: <data root>/interim/features/{dunford_1s,dunford_1s_native40,dog_aneck_1s}.parquet,
results/tables/r05_features_summary.json
Dunford: chunk = same ID + same label + consecutive rows with Time difference <= 1 s (file order). Seconds whose row count is outside 38-42: the chunk is split there
(the rows of that second are not used). Chunk resampled evenly from 40 Hz: resample_poly(3,4) -> DBA within the chunk -> 30-sample windows, the remainder is dropped.
Dog: each dog x recording is one signal, resample_poly(3,10), DBA over the recording, window = the full second t_sec; a window is kept only if all 100 samples carry the same single label.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, pyarrow.parquet as pq
from config import paths as P
from src import features_smit as F

t0 = time.time(); P.FEATURES.mkdir(parents=True, exist_ok=True)
L1 = ["Inaktif", "Lokomosyon", "Bakim"]
DUN_L1 = {"Rest": 0, "Walk": 1, "Trot": 1, "Run": 1, "Groom": 2, "Feed": 2}
DOG_L1 = {"Lying chest": 0, "Sitting": 0, "Standing": 0, "Walking": 1, "Trotting": 1, "Galloping": 1, "Eating": 2}
summ = {}

# ------------------------------------------------------------------ Dunford
d = pd.read_parquet(P.DUNFORD_CSV)                          # file order is kept (NOT sorted by time; R01)
h, m, s = d.Time.str.split(":", expand=True).astype(int).values.T
d["sec"] = h * 3600 + m * 60 + s
dt = d.sec - d.sec.shift(1)
newc = ~((d.ID == d.ID.shift(1)) & (d.Behaviour == d.Behaviour.shift(1)) & dt.isin([0, 1]))
d["chunk"] = newc.cumsum().values
cnt = d.groupby(["chunk", "sec"]).size().rename("n").reset_index()
summ["dunford_rows_per_second_hist"] = {int(k): int(v) for k, v in cnt.n.value_counts().sort_index().items()}
d = d.merge(cnt, on=["chunk", "sec"], how="left")
d["bad"] = ~d.n.between(38, 42)
summ["dunford"] = dict(rows=len(d), chunks_before_split=int(d.chunk.nunique()), seconds_total=int(len(cnt)),
                       seconds_abnormal=int((~cnt.n.between(38, 42)).sum()), rows_abnormal=int(d.bad.sum()),
                       rows_lt38=int(cnt.n[cnt.n < 38].sum()), seconds_lt38=int((cnt.n < 38).sum()), rows_gt42=int(cnt.n[cnt.n > 42].sum()), seconds_gt42=int((cnt.n > 42).sum()))
# split: the rows of the abnormal second are dropped, the chunk is split into new sub-chunks
d["sub"] = (d.bad != d.bad.shift(1)).cumsum().values
good = d[~d.bad]
rows40, rows30 = [], []; cnts = dict(skew_na=0, kurt_na=0, cor_na=0); lost_tail = 0; n_sub = 0
for sid, g in good.groupby(["chunk", "sub"], sort=False):
    sig = g[["AccX", "AccY", "AccZ"]].values; beh = g.Behaviour.iloc[0]; cat = g.ID.iloc[0]; n_sub += 1
    df30, c, ew = F.signal_features(sig, 3, 4, n=30, k=30, partial=True)
    df40, _, _ = F.signal_features(sig, 1, 1, n=40, k=40, partial=True)
    for k_ in cnts: cnts[k_] += c[k_]
    lost_tail += len(sig) - 40 * (len(sig) // 40) if len(df30) else len(sig)
    if len(df30) == 0: continue
    meta = dict(cat=cat, chunk=int(sid[0]), sub=int(sid[1]), beh=beh, y1=DUN_L1.get(beh, -1))
    a = df30.copy(); a["win"] = np.arange(len(a)); a["edge"] = ew
    for k_, v in meta.items(): a[k_] = v
    rows30.append(a)
    nn = min(len(df30), len(df40)); b = df40.iloc[:nn].copy(); b["win"] = np.arange(nn)
    for k_, v in meta.items(): b[k_] = v
    rows40.append(b)
D30 = pd.concat(rows30, ignore_index=True); D40 = pd.concat(rows40, ignore_index=True)
D30.to_parquet(P.DUNFORD_FEAT); D40.to_parquet(P.DUNFORD_FEAT40)
raw = d[["AccX", "AccY", "AccZ"]].values
summ["dunford"].update(subchunks=n_sub, windows=len(D30), windows_native40=len(D40), tail_rows_dropped=int(lost_tail), edge_windows=int(D30.edge.sum()),
                       rows_in_good=len(good), clip_frac_ge7_9=float((np.abs(raw) >= 7.9).mean()), max_abs_axis=float(np.abs(raw).max()), na_counts=cnts)
print("dunford", summ["dunford"], "%.0f s" % (time.time() - t0), flush=True)

# ------------------------------------------------------------------ Dog (ANeck only)
tb = pq.read_table(P.DOG_PARQUET, columns=["DogID", "TestNum", "t_sec", "ANeck_x", "ANeck_y", "ANeck_z", "Behavior_1", "Behavior_2", "Behavior_3"],
                   read_dictionary=["Behavior_1", "Behavior_2", "Behavior_3"])
dog = tb.column("DogID").to_numpy(); tst = tb.column("TestNum").to_numpy(); tsec = tb.column("t_sec").to_numpy()
A = np.column_stack([tb.column(c).to_numpy() for c in ("ANeck_x", "ANeck_y", "ANeck_z")]).astype(np.float64)
B = {}
for c in ("Behavior_1", "Behavior_2", "Behavior_3"):
    ch = tb.column(c).combine_chunks(); B[c] = (ch.indices.to_numpy(), ch.dictionary.to_pylist())
assert all(v[1][0] == "<undefined>" for v in B.values())
del tb
rec = pd.DataFrame({"d": dog, "t": tst}); key = rec.d.astype(int) * 10 + rec.t.astype(int)
order = np.argsort(key.values, kind="stable"); assert (order == np.arange(len(order))).all(), "kayıtlar bitişik değil"
bounds = np.flatnonzero(np.diff(key.values)) + 1; starts = np.r_[0, bounds]; ends = np.r_[bounds, len(key)]
out = []; dsum = dict(recordings=len(starts), windows_total=0, windows_single=0, windows_multi_or_undefined=0, windows_partial_label=0,
                      clip_n=int((np.abs(A) >= 7.9).sum()), clip_frac_ge7_9=float((np.abs(A) >= 7.9).mean()), max_abs_axis=float(np.abs(A).max()), na_counts=dict(skew_na=0, kurt_na=0, cor_na=0))
for a, b in zip(starts, ends):
    n_i = np.rint(tsec[a:b] * 100).astype(np.int64)
    assert n_i[0] == 0 and (np.diff(n_i) == 1).all(), "kayıt kesintisiz/0'dan başlamıyor"
    W = (b - a) // 100; T = W * 100
    sig = A[a:a + T]
    df, c, ew = F.signal_features(sig, 3, 10, n=30, k=30, partial=True)
    for k_ in dsum["na_counts"]: dsum["na_counts"][k_] += c[k_]
    c1, c2, c3 = (B[k][0][a:a + T].reshape(W, 100) for k in ("Behavior_1", "Behavior_2", "Behavior_3"))
    single = (c1.min(1) == c1.max(1)) & (c1[:, 0] != 0) & (c2.max(1) == 0) & (c3.max(1) == 0)
    dsum["windows_total"] += W; dsum["windows_single"] += int(single.sum()); dsum["windows_multi_or_undefined"] += int((~single).sum())
    names = B["Behavior_1"][1]
    beh = np.array([names[i] if s_ else "" for i, s_ in zip(c1[:, 0], single)])
    df["DogID"] = int(dog[a]); df["TestNum"] = int(tst[a]); df["sec"] = np.arange(W); df["beh"] = beh; df["edge"] = ew
    df["y1"] = [DOG_L1.get(x, -1) if x else -2 for x in beh]
    out.append(df[single])
DG = pd.concat(out, ignore_index=True); DG.to_parquet(P.DOG_FEAT)
dsum["rows_saved"] = len(DG)
summ["dog"] = dsum
json.dump(summ, open(P.TABLES / "r05_features_summary.json", "w"), indent=1, default=float)
print("dog", {k: v for k, v in dsum.items()}, "%.0f s" % (time.time() - t0), flush=True)
