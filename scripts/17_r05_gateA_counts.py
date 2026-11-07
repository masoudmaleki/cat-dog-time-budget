r"""17_r05_gateA_counts.py — R05 A (gate A items 2-5) and B (data tables).
Reads: Smit table (D.load), features/dunford_1s*.parquet, dog_aneck_1s.parquet, r05_features_summary.json. Writes: results/tables/r05_counts.csv, r05_gateA.json, r05_d17_median_iqr.csv
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, features_smit as F, smit_data as D

T = P.TABLES; out = {}
lab, y1, XC, XH, XC17, XH17 = R.load_smit()
dun, Xd = R.load_dunford(); dun_all = pd.read_parquet(P.DUNFORD_FEAT)
dog_all = pd.read_parquet(P.DOG_FEAT); dog, Xg = R.load_dog()
fs = json.load(open(T / "r05_features_summary.json"))

# ---- item 2: Inaktif VM_Mean median
m = {"Smit_tasma": float(np.median(XC.VM_Mean[y1 == 0])), "Dunford": float(np.median(dun.VM_Mean[dun.y1 == 0])), "Kopek_ANeck": float(np.median(dog.VM_Mean[dog.y1 == 0]))}
out["inaktif_VM_Mean_medyan"] = m; out["kapi_madde2_gecti"] = all(0.95 <= v <= 1.05 for v in m.values())
# ---- item 3: clipping
mx = XC[["X_Min", "X_Max", "Y_Min", "Y_Max", "Z_Min", "Z_Max"]].abs().max(1)
out["kirpilma"] = {"Smit_tasma(saniye düzeyi; ham yok)": dict(frac_saniye_ge7_9=float((mx >= 7.9).mean()), max_abs_eksen=float(mx.max())),
                   "Dunford": dict(frac_ornek_ge7_9=fs["dunford"]["clip_frac_ge7_9"], max_abs_eksen=fs["dunford"]["max_abs_axis"]),
                   "Kopek_ANeck": dict(frac_ornek_ge7_9=fs["dog"]["clip_frac_ge7_9"], max_abs_eksen=fs["dog"]["max_abs_axis"])}
# ---- item 4: sampling sensitivity (Dunford, native 40 Hz vs 30 Hz)
d40 = pd.read_parquet(P.DUNFORD_FEAT40)
k = ["chunk", "sub", "win"]
j = dun_all.merge(d40, on=k, suffixes=("_30", "_40"))
a = F.d17(j[[c + "_30" for c in F.COLS32]].rename(columns=lambda c: c[:-3])); b = F.d17(j[[c + "_40" for c in F.COLS32]].rename(columns=lambda c: c[:-3]))
rel = {}
for c in F.D17_COLS:
    den = b[c].abs(); ok = den > 1e-9
    rel[c] = float(np.median((a[c] - b[c]).abs()[ok] / den[ok]))
out["ornekleme_medyan_goreli_fark"] = rel; out["ornekleme_n_pencere"] = len(j)
# ---- item 5: D17 Inaktif / Lokomosyon median [IQR], three datasets
rows = []
for nm, X, y in (("Smit_tasma", XC17, y1), ("Dunford", Xd, dun.y1.values), ("Kopek_ANeck", Xg, dog.y1.values)):
    for ci, cn in ((0, "Inaktif"), (1, "Lokomosyon")):
        s = X[y == ci]
        for c in F.D17_COLS:
            q = np.percentile(s[c], [25, 50, 75]); rows.append(dict(dataset=nm, L1=cn, feature=c, n=len(s), median=q[1], q25=q[0], q75=q[2]))
pd.DataFrame(rows).to_csv(T / "r05_d17_median_iqr.csv", index=False)
# ---- B: window counts
cnt = []
for c, g in pd.DataFrame({"c": lab.Cat_id, "y": y1}).query("y>=0").groupby("c"):
    for ci in range(3): cnt.append(dict(dataset="Smit", individual=c, L1=R.L1[ci], n=int((g.y == ci).sum())))
for c, g in dun.groupby("cat"):
    for ci in range(3): cnt.append(dict(dataset="Dunford", individual=c, L1=R.L1[ci], n=int((g.y1 == ci).sum())))
for c, g in dog.groupby("DogID"):
    for ci in range(3): cnt.append(dict(dataset="Kopek", individual=int(c), L1=R.L1[ci], n=int((g.y1 == ci).sum())))
cnt = pd.DataFrame(cnt); cnt.to_csv(T / "r05_counts.csv", index=False)
summ = {}
for ds, g in cnt.groupby("dataset"):
    tot = g.groupby("L1").n.sum().to_dict(); per = g.pivot(index="individual", columns="L1", values="n")
    summ[ds] = dict(n_birey=int(g.individual.nunique()), toplam=tot, birey_basina={c: [float(per[c].median()), int(per[c].min()), int(per[c].max())] for c in per.columns},
                    birey_basina_tum_L1_medyan=float(per.sum(1).median()), birey_n_ge1_Bakim=int((per.Bakim > 0).sum()))
out["sayilar"] = summ
# Dunford loss
dd = fs["dunford"]; exp_s = 130011 / 40
n_sh = int((dun_all.beh == "Shake").sum())
out["dunford_kayip"] = dict(beklenen_saniye=3250.28, satir=dd["rows"], anormal_saniye_satir=dd["rows_abnormal"], satir_lt38=dd["rows_lt38"], saniye_lt38=dd["seconds_lt38"], satir_gt42=dd["rows_gt42"],
                            saniye_gt42=dd["seconds_gt42"], iyi_satir=dd["rows_in_good"], pencere=dd["windows"], shake_pencere=n_sh, l1_pencere=int(len(dun)),
                            kayip_anormal_s=dd["rows_abnormal"] / 40, kayip_parca_artigi_s=(dd["rows_in_good"] - 40 * dd["windows"]) / 40, edge_pencere=dd["edge_windows"],
                            edge_oran=dd["edge_windows"] / dd["windows"], sub_parca=dd["subchunks"])
# excluded labels
a = pd.read_parquet(P.INTERIM / "smit_anno.parquet"); beh = [c for c in a.columns if c not in ("Timestamp", "Pen", "Cat_id")]
rs = a[beh].sum(axis=1); fine = a[beh].idxmax(axis=1); sing = rs == 1
ex = fine[sing & ~fine.isin(D.COL_TO_L2.keys())].value_counts()
out["smit_disarida"] = dict(tek_etiket_haric=ex.to_dict(), coklu_etiket_saniye=int((rs > 1).sum()), bos_saniye=int((rs == 0).sum()), l1_kullanilan=int((y1 >= 0).sum()))
out["dunford_disarida"] = dict(Shake_pencere=n_sh, shake_satir=int((pd.read_parquet(P.DUNFORD_CSV, columns=["Behaviour"]).Behaviour == "Shake").sum()))
dogex = dog_all[dog_all.y1 == -1].beh.value_counts()
out["kopek_disarida_tek_etiketli_pencere_s"] = dogex.to_dict()
out["kopek_tek_etiketli_olmayan_pencere"] = fs["dog"]["windows_multi_or_undefined"]
eat = dog[dog.y1 == 2]
out["kopek_eating"] = dict(pencere=int(len(eat)), kopek=int(eat.DogID.nunique()), karar_yeme=bool(len(eat) >= 300 and eat.DogID.nunique() >= 5),
                           kopek_basina=eat.groupby("DogID").size().to_dict())
json.dump(out, open(T / "r05_gateA.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("inaktif_VM_Mean_medyan", "kapi_madde2_gecti", "kirpilma", "sayilar", "dunford_kayip", "kopek_eating")}, indent=0, default=float)[:4500])
print({k: round(v, 3) for k, v in rel.items()})
