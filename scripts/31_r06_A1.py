r"""31_r06_A1.py — R06 A1: recompute the three values of S1 CC AE in R05 (0.0260 / 0.0290 / 0.0292) from the bag sets and r05_bag_level.csv.
Reads: results/tables/r05_bag_level.csv, r05/r05_ladder_bags.parquet, r05/E_ladder_extra.parquet. Writes: results/tables/r06_A1.json
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P

T = P.TABLES
c = pd.read_csv(T / "r05_bag_level.csv", dtype={"torba": str})
c["ae_c"] = (c.p_hat - c.p).abs()
g = c[(c.senaryo == "S1") & (c.kol == "CC")]
keys = ["varyant", "duzey", "m", "tekrar", "hedef_birey", "torba"]
bag = g.groupby(keys, as_index=False).agg(AE=("ae_c", "mean"), N=("N", "first"))
def summ(b):
    return dict(n_torba_satiri=int(len(b)), n_benzersiz_torba=int(b.drop_duplicates(["hedef_birey", "torba"]).shape[0]), n_kedi=int(b.hedef_birey.nunique()), AE=float(b.AE.mean()), ort_N=float(b.N.mean()), toplam_N_benzersiz=int(b.drop_duplicates(["hedef_birey", "torba"]).N.sum()))
a = bag[(bag.varyant == "30dk_ge50") & (bag.m == 0)]
b = bag[(bag.varyant == "30dk_ge50") & (bag.m == 1800)]
mix = bag[bag.varyant == "30dk_ge50"]
out = dict(
    deger_0260=dict(tanim="merdiven: S1 LOCO, 30 dk kapsam >= %50, 68 torba, hicbir saniye cikarilmadi, m=0", **summ(a)),
    deger_0292=dict(tanim="merdiven ek kollari: ayni 68 torba x 20 tekrar; her tekrarda test kedisinin kendi 30 x 60 s = 1.800 s etiketli blogu torbalardan cikarildi (m=1800, duzey birey)", **summ(b)),
    deger_0290=dict(tanim="R05 'D17 bedeli' hesabi: A tablosunda S1|30dk_ge50|CC satirlari m suzgeci olmadan: 68 (m=0) + 1.360 (m=1800) satir karisimi", **summ(mix)),
)
out["karisim_dogrulama"] = float((68 * a.AE.mean() + 1360 * b.AE.mean()) / 1428)
# corrected value of the D17 cost (68 bags, m=0): S1 D17 - S1-32
s32 = c[(c.senaryo == "S1-32") & (c.kol == "CC") & (c.varyant == "30dk_ge50")].assign(ae=lambda d: d.p_hat.sub(d.p).abs()).groupby(["hedef_birey", "torba"]).ae.mean()
s17 = a.set_index(["hedef_birey", "torba"]).AE
out["d17_bedeli_duzeltilmis"] = dict(AE_D17=float(s17.mean()), AE_32=float(s32.mean()), fark=float((s17 - s32.reindex(s17.index)).mean()), n=int(len(s17)))
# check: if the same bag set is counted twice
out["ayni_kume_tutarli"] = bool(abs(a.AE.mean() - 0.026040171029988016) < 1e-6)
json.dump(out, open(T / "r06_A1.json", "w"), indent=1)
print(json.dumps(out, indent=1))
