r"""51_r08_analyze.py — R08 D: F rule (d = AE_CC − AE_arm; individual = cat/dog bag mean; exact sign-flip for n <= 20, 200,000 signed MC draws for dogs; Holm within family; |d| >= 0.01),
primary family = scenario x bag type: {DM, DR-APP, DR-dogal (DR-q)} (3 tests); secondary family: {DR-APP-h, DR-dogal-h} (2 tests). PACC is not in the family (the R07 category is read).
The Q1 criterion (§7.2) and the §7.3 interim-state rule are computed here operationally. Reads: <data root>/interim/r08/run_*.pkl, r07_*.csv. Writes: results/tables/r08_*.csv|json"""
import sys, pickle, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import stats06 as S6, quant as Q

T = P.TABLES; SEED = P.SEED; NMC = 200_000; rng = np.random.default_rng(SEED)
SCEN = ["S1", "S1-L2", "S2", "S4-Smit-kapali", "K-L1", "K-L2", "S3", "S4-Dunford-kapali"]
runs = {s: pickle.load(open(P.R08 / f"run_{s}.pkl", "rb")) for s in SCEN}
bag = pd.concat([runs[s]["bag"] for s in SCEN], ignore_index=True); meta = pd.concat([runs[s]["meta"] for s in SCEN], ignore_index=True)
diag = pd.concat([runs[s]["diag"] for s in SCEN], ignore_index=True); info = pd.concat([runs[s]["info"] for s in SCEN], ignore_index=True)
bag["armk"] = np.where(bag.tanimlayici == "-", bag.kol, bag.kol + "-" + bag.tanimlayici)
bag.drop(columns="armk").to_csv(T / "r08_bag_level.csv", index=False)

# ---- bag-level AE (class mean)
KEY = ["senaryo", "torba_turu", "dis_katman", "hedef_birey", "torba", "N"]
bag["ae"] = (bag.p_hat - bag.p).abs()
ag = bag.groupby(KEY + ["armk"]).ae.mean().rename("AE").reset_index()
W = ag.pivot_table(index=KEY, columns="armk", values="AE").reset_index()
W.to_csv(T / "r08_bag_ae_wide.csv", index=False)

PRIM = ["DM", "DR-APP-q", "DR-dogal-q"]; SECO = ["DR-APP-h", "DR-dogal-h"]


def stat_any(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n <= 20:
        s = S6.stat(d); s["p_yontem"] = "kesin"; return s
    lo, hi = S6.boot_ci(d); obs = abs(d.mean()); cnt = 0; done = 0
    while done < NMC:
        k = min(20_000, NMC - done); sg = rng.choice([-1.0, 1.0], size=(k, n)); cnt += int((np.abs(sg @ d / n) >= obs - 1e-12).sum()); done += k
    return dict(n=n, d=float(d.mean()), lo=lo, hi=hi, p=(1 + cnt) / (NMC + 1), pmin=np.nan, p_yontem="MC")


R7 = {}   # PACC R07 category
m = pd.read_csv(T / "r07_master.csv"); t4 = pd.read_csv(T / "r07_4_tests.csv"); t5 = pd.read_csv(T / "r07_5_tests.csv")


def _t4(sc, var):
    r = t4[(t4.clf == "RF") & (t4.scen == sc) & (t4.variant == var) & (t4.kol == "PACC")].iloc[0]; return r.kategori, r.d


R7[("S1", "30dk")] = _t4("S1", "30dk_ge50"); R7[("S1", "2dk")] = _t4("S1-kopru", "dogal2dk"); R7[("S1", "APP120")] = _t4("S1-kopru", "APP120"); R7[("S2", "30dk")] = _t4("S2", "30dk_ge50")
R7[("S4-Smit-kapali", "30dk")] = _t4("S4-Smit", "30dk_ge50-kapali"); R7[("S3", "kedi")] = _t4("S3", "kedi"); R7[("S4-Dunford-kapali", "kedi")] = _t4("S4-Dunford", "kapali")
r = m[(m.senaryo == "S1-tasma") & (m.kol == "PACC") & (m.uzay == "L2")].iloc[0]; R7[("S1-L2", "30dk")] = (r.kategori, r.d)
for sp in ("K-L1", "K-L2"):
    for bt in ("kayit", "dogal2"):
        r = t5[(t5.scen == "Kopek-ici") & (t5.space == sp) & (t5.feat == "32") & (t5.btype == bt) & (t5.kol == "PACC")].iloc[0]; R7[(sp, bt)] = (r.kategori, r.d)

order = {s: i for i, s in enumerate(SCEN)}
cells = [(sc, bt, gg) for (sc, bt), gg in W.groupby(["senaryo", "torba_turu"], sort=False)]
cells.sort(key=lambda c: (order[c[0]], c[1]))
rows = []
for sc, bt, gg in cells:
    ind = gg.groupby("hedef_birey"); base = ind["CC"].mean()
    for fam, arms in (("birincil", PRIM), ("ikincil", SECO)):
        st = []
        for a in arms:
            if a not in gg or gg[a].isna().all(): continue
            dind = (base - ind[a].mean()).values; s = stat_any(dind)
            st.append(dict(senaryo=sc, torba_turu=bt, kol=a, aile=fam, n_torba=len(gg), AE_CC=float(gg["CC"].mean()), AE_kol=float(gg[a].mean()), N_med=float(gg.N.median()), **s))
        rows += S6.family(st).to_dict("records")
TT = pd.DataFrame(rows)
cat7 = [R7.get((r.senaryo, r.torba_turu), (None, np.nan)) for r in TT.itertuples()]
TT["PACC_R07"] = [c[0] for c in cat7]; TT["PACC_R07_d"] = [c[1] for c in cat7]
TT.to_csv(T / "r08_tests.csv", index=False)

# ---- Q1 criterion (§7.2) and §7.3
def cat_of(sc, bt, arm, fam="birincil"):
    r = TT[(TT.senaryo == sc) & (TT.torba_turu == bt) & (TT.kol == arm) & (TT.aile == fam)]; return r.iloc[0].kategori if len(r) else None


q1 = {}
for v in ("DR-APP-q", "DR-dogal-q"):
    a_ = cat_of("S1", "30dk", v) == "KAZANIR"
    b_ = any(cat_of(sp, "kayit", v) == "KAZANIR" for sp in ("K-L1", "K-L2"))
    loss = [(r.senaryo, r.torba_turu) for r in TT[(TT.kol == v) & (TT.aile == "birincil")].itertuples() if r.kategori == "KAYBEDER"]
    q1[v] = dict(Q1a=bool(a_), Q1b=bool(b_), Q1c=len(loss) == 0, kayip_hucreler=loss, Q1=bool(a_ and b_ and len(loss) == 0))
Q1 = any(x["Q1"] for x in q1.values())
prim = TT[TT.aile == "birincil"]
win_nat = sorted({(r.senaryo, r.torba_turu, r.kol) for r in prim.itertuples() if r.kategori == "KAZANIR" and r.torba_turu != "APP120"})
win_app = sorted({(r.senaryo, r.torba_turu, r.kol) for r in prim.itertuples() if r.kategori == "KAZANIR" and r.torba_turu == "APP120"})
win_sec = sorted({(r.senaryo, r.torba_turu, r.kol) for r in TT[TT.aile == "ikincil"].itertuples() if r.kategori == "KAZANIR" and r.torba_turu != "APP120"})
if Q1: durum = "7.1-ikinci"
elif len(win_nat) > 0: durum = "7.3-ara"
elif len(win_app) > 0: durum = "7.3-yalniz-APP"
else: durum = "7.1-ilk"
json.dump(dict(q1=q1, Q1=bool(Q1), durum=durum, kazanir_dogal=win_nat, kazanir_APP=win_app, kazanir_ikincil_dogal=win_sec,
               kaybeder=[(r.senaryo, r.torba_turu, r.kol, r.aile) for r in TT.itertuples() if r.kategori == "KAYBEDER"]), open(T / "r08_q1.json", "w"), indent=1, default=str)

# ---- ana tablo
main = []
for sc, bt, gg in cells:
    t = TT[(TT.senaryo == sc) & (TT.torba_turu == bt) & (TT.aile == "birincil")]
    row = dict(senaryo=sc, torba_turu=bt, n=int(t.n.iloc[0]), n_torba=len(gg), AE_CC=float(gg["CC"].mean()), PACC_R07=t.PACC_R07.iloc[0], PACC_d=t.PACC_R07_d.iloc[0], aile_n=int(t.aile_n.iloc[0]))
    for a in PRIM:
        r = t[t.kol == a]
        if len(r):
            r = r.iloc[0]; row.update({f"{a}_AE": r.AE_kol, f"{a}_d": r.d, f"{a}_lo": r.lo, f"{a}_hi": r.hi, f"{a}_pholm": r.p_holm, f"{a}_kat": r.kategori})
    main.append(row)
pd.DataFrame(main).to_csv(T / "r08_main.csv", index=False)

# ---- signed bias per class (percentage points; individual mean, individual bootstrap CI)
bb = []
for (sc, bt, arm, cl), gg in bag.groupby(["senaryo", "torba_turu", "armk", "sinif"], sort=False):
    v = ((gg.p_hat - gg.p) * 100).groupby(gg.hedef_birey).mean().values; lo, hi = S6.boot_ci(v)
    bb.append(dict(senaryo=sc, torba_turu=bt, kol=arm, sinif=cl, bias_pp=float(v.mean()), lo=lo, hi=hi, n=len(v)))
pd.DataFrame(bb).to_csv(T / "r08_class_bias.csv", index=False)

# ---- DR details, DM boundary, extrapolation
inf = info.groupby(["senaryo", "varyant", "tanimlayici"]).agg(a_med=("a", "median"), a_min=("a", "min"), a_maks=("a", "max"), n_med=("n_egitim_torba", "median"), n_min=("n_egitim_torba", "min"),
                                                             n_maks=("n_egitim_torba", "max"), kat=("a", "size")).reset_index()
inf.to_csv(T / "r08_dr_detail.csv", index=False)
dmb = meta.groupby(["senaryo", "torba_turu"]).agg(n_torba=("dm_sinir", "size"), sinir_orani=("dm_sinir", "mean")).reset_index(); dmb.to_csv(T / "r08_dm_boundary.csv", index=False)
ex = diag.groupby(["senaryo", "torba_turu", "varyant", "sinif"]).agg(dis=("dis", "sum"), n=("n", "sum")).reset_index(); ex["oran"] = ex.dis / ex.n; ex.to_csv(T / "r08_extrapolation.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
print(TT[["senaryo", "torba_turu", "kol", "aile", "n", "n_torba", "AE_CC", "AE_kol", "d", "lo", "hi", "p_holm", "aile_n", "kategori", "PACC_R07"]].to_string())
print(json.dumps(dict(q1=q1, Q1=bool(Q1), durum=durum), indent=1, default=str)); print(inf.to_string()); print(dmb.to_string())
