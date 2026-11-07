r"""25_r06_B_v2.py — R06 B: Dunford windows with the v2 rule. Reads: dunford_cats.parquet (read-only). Writes: <data root>/interim/features/dunford_v2_1s.parquet, results/tables/r06_B_v2.json, r06_B_v2_counts.csv
v2 (predefined): (1) exactly duplicated rows are dropped; (2) incomplete (<38) seconds at a chunk end are not dropped; (3) the chunk is split at a second with <38 rows in its middle; (4) after duplicates are dropped, the rows
of a single-label second with >42 rows are kept in file order and flagged; a multi-label second is a chunk boundary (rows dropped).
IMPLEMENTATION NOTE (1): individually duplicated rows within a second are legitimate (1,524 of the 2,257 normal 38-42-row seconds contain exactly identical consecutive readings); therefore "exactly duplicated" = the whole
second block repeated verbatim k >= 2 times (n = k*p, 38 <= p <= 42); the first block is kept. Row-level duplication is only counted, as the literal version (r06_B_v2.json: literal_*).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import features_smit as F

DUN_L1 = {"Rest": 0, "Walk": 1, "Trot": 1, "Run": 1, "Groom": 2, "Feed": 2}
d = pd.read_parquet(P.DUNFORD_CSV)
h, m, s = d.Time.str.split(":", expand=True).astype(int).values.T
d["sec"] = h * 3600 + m * 60 + s
dt = d.sec - d.sec.shift(1)
newc = ~((d.ID == d.ID.shift(1)) & (d.Behaviour == d.Behaviour.shift(1)) & dt.isin([0, 1]))
d["chunk"] = newc.cumsum().values
A = d[["AccX", "AccY", "AccZ"]].values
bn = d.groupby(["ID", "sec"]).Behaviour.nunique(); multi = set(bn[bn > 1].index)
cols = ["ID", "Behaviour", "Time", "AccX", "AccY", "AccZ"]
dupflag = d.duplicated(cols, keep="first").values

keep_rows = []; flags = []; sub_ids = []; st = dict(sec=0, normal=0, blokdup=0, blokdup_sonra_normal=0, gt42_flag=0, gt42_multi_atildi=0, lt38_uc=0, lt38_orta_bolme=0, lt38_uc_tek=0, rows_lt38_kept=0,
                                                        literal_normal_to_lt38=0, literal_normal_total=0)
sub = 0
for ck, g in d.groupby("chunk", sort=False):
    secs = [(sc, gg.index.values) for sc, gg in g.groupby("sec", sort=False)]
    ns = len(secs); cur = False
    for pos, (sc, ix) in enumerate(secs):
        n = len(ix); st["sec"] += 1; first = pos == 0; last = pos == ns - 1; flag = False; rows = ix
        if 38 <= n <= 42:
            st["normal"] += 1; st["literal_normal_total"] += 1; st["literal_normal_to_lt38"] += int(n - dupflag[ix].sum() < 38)
        if n > 42:
            a = A[ix]; done = False
            for p_ in range(38, 43):
                if n % p_ == 0 and np.array_equal(a.reshape(n // p_, p_, 3), np.broadcast_to(a[:p_], (n // p_, p_, 3))):
                    rows = ix[:p_]; done = True; st["blokdup"] += 1; st["blokdup_sonra_normal"] += 1; break
            if not done:
                if (g.ID.iloc[0], sc) in multi:
                    st["gt42_multi_atildi"] += 1; sub += 1; cur = False; continue
                flag = True; st["gt42_flag"] += 1
            n2 = len(rows)
            if n2 > 42 and not flag:
                flag = True; st["gt42_flag"] += 1
            if n2 < 38 and not (first or last):
                st["lt38_orta_bolme"] += 1; sub += 1; continue
        elif n < 38:
            if first or last:
                st["lt38_uc"] += 1; st["rows_lt38_kept"] += n; st["lt38_uc_tek"] += int(first and last)
            else:
                st["lt38_orta_bolme"] += 1; sub += 1; continue
        keep_rows.append(rows); flags.append(np.full(len(rows), flag)); sub_ids.append(np.full(len(rows), sub))
    sub += 1
ri = np.concatenate(keep_rows); fl = np.concatenate(flags); sb = np.concatenate(sub_ids)
dd = d.loc[ri, ["ID", "Behaviour", "chunk"]].copy(); dd["flag"] = fl; dd["sub"] = sb; dd["row"] = ri
rows30 = []; nsub = 0
for sid, g in dd.groupby(["chunk", "sub"], sort=False):
    sig = A[g.row.values]; beh = g.Behaviour.iloc[0]; cat = g.ID.iloc[0]
    df30, c, ew = F.signal_features(sig, 3, 4, n=30, k=30, partial=True)
    if len(df30) == 0: continue
    nsub += 1
    fw = np.array([g.flag.values[w * 40:(w + 1) * 40].any() for w in range(len(df30))])
    a = df30.copy(); a["win"] = np.arange(len(a)); a["edge"] = ew; a["flag"] = fw
    for k_, v in dict(cat=cat, chunk=int(sid[0]), sub=int(sid[1]), beh=beh, y1=DUN_L1.get(beh, -1)).items(): a[k_] = v
    rows30.append(a)
D2 = pd.concat(rows30, ignore_index=True); D2.to_parquet(P.DUNFORD_FEAT_V2)
v1 = pd.read_parquet(P.DUNFORD_FEAT); v1 = v1[v1.y1 >= 0]; v2 = D2[D2.y1 >= 0]
cnt = pd.concat([v1.groupby(["cat", "y1"]).size().rename("v1"), v2.groupby(["cat", "y1"]).size().rename("v2")], axis=1).fillna(0).astype(int).reset_index()
cnt["sinif"] = cnt.y1.map({0: "Inaktif", 1: "Lokomosyon", 2: "Bakim"}); cnt.drop(columns="y1").to_csv(P.TABLES / "r06_B_v2_counts.csv", index=False)
X2 = F.d17(v2)
gate = dict(inaktif_VM_Mean_medyan_v2=float(np.median(X2.VM_Mean[v2.y1.values == 0])), n_pencere_v1=len(v1), n_pencere_v2=len(v2), n_flag_pencere=int(v2.flag.sum()), n_edge_v2=int(v2.edge.sum()), nsub=nsub,
            toplam_L1={"v1": v1.groupby("y1").size().to_dict(), "v2": v2.groupby("y1").size().to_dict()},
            na_sayilari=dict(skew=int(v2.VM_Skew.isna().sum()), kurt=int(v2.VM_Kurt.isna().sum())),
            klip_oran_v2=float((np.abs(A[ri]) >= 7.9).mean()), maks_eksen_v2=float(np.abs(A[ri]).max()))
gate["kapi_madde2_gecti_v2"] = bool(0.95 <= gate["inaktif_VM_Mean_medyan_v2"] <= 1.05)
json.dump(dict(st=st, kapi=gate), open(P.TABLES / "r06_B_v2.json", "w"), indent=1, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o))
print(json.dumps(dict(st=st, kapi=gate), indent=1, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o)))
print(cnt.pivot(index="cat", columns="sinif", values=["v1", "v2"]).to_string())
