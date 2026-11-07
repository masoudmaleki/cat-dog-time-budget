r"""24_r06_B_diag.py — R06 B: diagnosis of the discarded Dunford seconds (same chunk definition as R05). Reads: dunford_cats.parquet (read-only). Writes: results/tables/r06_B_diag.json, r06_B_seconds.csv"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P

d = pd.read_parquet(P.DUNFORD_CSV)
h, m, s = d.Time.str.split(":", expand=True).astype(int).values.T
d["sec"] = h * 3600 + m * 60 + s
dt = d.sec - d.sec.shift(1)
newc = ~((d.ID == d.ID.shift(1)) & (d.Behaviour == d.Behaviour.shift(1)) & dt.isin([0, 1]))
d["chunk"] = newc.cumsum().values
d["row"] = np.arange(len(d))
g = d.groupby(["chunk", "sec"], sort=False)
S = g.agg(ID=("ID", "first"), beh=("Behaviour", "first"), n=("row", "size"), r0=("row", "min"), r1=("row", "max")).reset_index()
S["pos_in_chunk"] = S.groupby("chunk").cumcount(); S["n_in_chunk"] = S.groupby("chunk").chunk.transform("size")
S["first"] = S.pos_in_chunk == 0; S["last"] = S.pos_in_chunk == S.n_in_chunk - 1
S["prev_n"] = S.groupby("chunk").n.shift(1); S["next_n"] = S.groupby("chunk").n.shift(-1)
S["prev_beh_chunk"] = S.beh.shift(1).where(S.chunk.shift(1) != S.chunk); S["next_beh_chunk"] = S.beh.shift(-1).where(S.chunk.shift(-1) != S.chunk)
# duplicated rows: exactly identical within a second (ID, Behaviour, Time, AccX, AccY, AccZ)
cols = ["ID", "Behaviour", "Time", "AccX", "AccY", "AccZ"]
d["dup"] = d.duplicated(cols, keep="first")
S["n_dup"] = d.groupby(["chunk", "sec"], sort=False).dup.sum().values
S["n_unique"] = S.n - S.n_dup
# block repetition: first half == second half (e.g. 80 = 2x40)
def halves(x):
    a = x[["AccX", "AccY", "AccZ"]].values; n = len(a)
    if n < 2 or n % 2: return False
    return bool(np.array_equal(a[: n // 2], a[n // 2:]))
S["half_repeat"] = g.apply(halves, include_groups=False).values
# in how many different chunks / labels the same (ID, sec) appears
ids = d.groupby(["ID", "sec"]).agg(nchunk=("chunk", "nunique"), nbeh=("Behaviour", "nunique")).reset_index()
S = S.merge(ids, on=["ID", "sec"], how="left")
S["abn"] = ~S.n.between(38, 42)
bins = [0, 19, 37, 42, 60, 90, 10**9]; labels = ["<20", "20-37", "38-42", "43-60", "61-90", ">90"]
S["bin"] = pd.cut(S.n, bins=bins, labels=labels)
out = dict(saniye_toplam=int(len(S)), anormal=int(S.abn.sum()), dagilim_anormal={k: int(v) for k, v in S[S.abn].bin.value_counts().sort_index().items() if v},
           dagilim_tum={k: int(v) for k, v in S.bin.value_counts().sort_index().items()})
gt = S[S.n > 42]
out["gt42"] = dict(n=int(len(gt)), satir=int(gt.n.sum()), yinelenen_satir_toplam=int(gt.n_dup.sum()), saniye_yinelenenli=int((gt.n_dup > 0).sum()),
                   yinelenen_atilinca_38_42=int(((gt.n_unique >= 38) & (gt.n_unique <= 42)).sum()), yarim_tekrar=int(gt.half_repeat.sum()),
                   cok_parca=int((gt.nchunk > 1).sum()), cok_etiket=int((gt.nbeh > 1).sum()), parca_ici_orta=int((~gt["first"] & ~gt["last"]).sum()),
                   parca_ucu=int((gt["first"] | gt["last"]).sum()), tek_saniyelik_parca=int((gt["first"] & gt["last"]).sum()))
for lo, hi, nm in ((43, 60, "43-60"), (61, 90, "61-90"), (91, 10**9, ">90")):
    q = gt[(gt.n >= lo) & (gt.n <= hi)]
    out["gt42_" + nm] = dict(n=int(len(q)), yinelenen=int((q.n_dup > 0).sum()), yarim_tekrar=int(q.half_repeat.sum()), cok_etiket=int((q.nbeh > 1).sum()), cok_parca=int((q.nchunk > 1).sum()),
                            medyan_n=float(q.n.median()), n_unique_medyan=float(q.n_unique.median()))
lt = S[S.n < 38]
out["lt38"] = dict(n=int(len(lt)), parca_ucu=int((lt["first"] | lt["last"]).sum()), ilk=int(lt["first"].sum()), son=int(lt["last"].sum()), orta=int((~lt["first"] & ~lt["last"]).sum()),
                   tek_saniyelik_parca=int((lt["first"] & lt["last"]).sum()), satir=int(lt.n.sum()), kedi_basina={k: int(v) for k, v in lt.ID.value_counts().items()})
out["gt42_kedi_basina"] = {k: int(v) for k, v in gt.ID.value_counts().items()}
# neighbours: row counts of the previous/next second of a >42 second (within the chunk)
out["gt42_komsu_n"] = dict(onceki_38_42=int(gt.prev_n.between(38, 42).sum()), sonraki_38_42=int(gt.next_n.between(38, 42).sum()), onceki_yok=int(gt.prev_n.isna().sum()), sonraki_yok=int(gt.next_n.isna().sum()))
# short seconds: are the neighbours 38-42 and the label the same
S.drop(columns=["r0", "r1"]).to_csv(P.TABLES / "r06_B_seconds.csv", index=False)
json.dump(out, open(P.TABLES / "r06_B_diag.json", "w"), indent=1)
print(json.dumps(out, indent=1))
