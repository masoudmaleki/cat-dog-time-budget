r"""56_r09_diag.py — R09 §B: three diagnostics (no categories; stored outputs only; no model).
B1 DM boundary resolutions: for every class that DM zeroes (p̂ < 1e-6, the R08 boundary definition), the proportion of bags in which the true proportion in the same bag is also 0, and the median [IQR] of the true proportion; per cell (scenario × bag type × class).
B2 EQUIVALENT and UNCERTAIN cells (r07_master.csv + r08_tests.csv primary): RAE of the arm and of CC (bag mean) + signed bias of the rarest class (percentage points, individual mean).
B3 proportion of DR-natural training bags in which the true proportion of a class is 0 (Smit 2 min, Smit 30 min, dog × recording, dog natural 2 min; L1 and L2 separately).
Reads: r08_bag_level.csv, r08_tests.csv, r07_master.csv, r05/r06/r07 bag parquet files, r02_bag_level.csv, Smit/dog tables (labels only). Writes: results/tables/r09_B1_dm_boundary.csv, r09_B2_equiv_uncertain.csv, r09_B3_dr_zero_train.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r07lib as L7, quant as Q, r08scen as SC, r08lib as L8

T = P.TABLES; O5, O6 = P.R05, P.R06
ctx = SC.Ctx()
CLS = {"L1": ["Inaktif", "Lokomosyon", "Bakim"]}
B8 = pd.read_csv(T / "r08_bag_level.csv")
B8["armx"] = B8.kol + np.where(B8.tanimlayici == "-", "", "-" + B8.tanimlayici)
KEY8 = ["senaryo", "torba_turu", "dis_katman", "hedef_birey", "torba"]

# ------------------------------------------------------------------ B1
rows = []
dm = B8[B8.armx == "DM"]
for (sc, bt, k), g in dm.groupby(["senaryo", "torba_turu", "sinif"], sort=False):
    z = g[g.p_hat < 1e-6]
    if len(z) == 0: continue
    q = z.p.quantile([0.25, 0.5, 0.75]).values
    rows.append(dict(senaryo=sc, torba_turu=bt, sinif=k, n_torba=int(g.groupby(KEY8).ngroups), n_dm_sifir=int(len(z)), dm_sifir_orani=len(z) / g.groupby(KEY8).ngroups,
                     gercek_sifir_orani=float((z.p == 0).mean()), gercek_medyan=float(q[1]), gercek_q1=float(q[0]), gercek_q3=float(q[2])))
B1 = pd.DataFrame(rows); B1.to_csv(T / "r09_B1_dm_boundary.csv", index=False)

# ------------------------------------------------------------------ B2
M = pd.read_csv(T / "r07_master.csv"); R8 = pd.read_csv(T / "r08_tests.csv"); R8 = R8[R8.aile == "birincil"]
lad = L7.rd(O5 / "r05_ladder_bags.parquet"); src5 = L7.rd(O5 / "r05_sources_bags.parquet"); v2 = L7.rd(O6 / "B_v2_bags.parquet"); e4 = L7.rd(O6 / "E4_S1kapali_bags.parquet")
DU, SM = L7.load_E()
LD = L7.cat_(L7.rd(O5 / "E_ladder_extra.parquet"), *[L7.rd(p) for p in sorted(O6.glob("E1_ladder_fit_m*.parquet")) if not p.name.endswith("_miss.parquet")]).drop_duplicates(L7.KEYS, keep="last")
SPEC = {"S0 (sızıntılı)": (lad, "S0", "30dk_ge50", None), "S1": (lad, "S1", "30dk_ge50", (LD, "S1", "30dk_ge50", "birey")), "S2": (lad, "S2", "30dk_ge50", (LD, "S2", "30dk_ge50", "birey")),
        "S3 [keşifsel]": (v2, "S3", "kedi", (DU, "S3", "-", "calisma")), "S4-Smit-açık": (src5, "S4-Smit", "30dk_ge50-acik", (SM, "S4-Smit", "acik", "birey")),
        "S4-Smit-kapalı": (src5, "S4-Smit", "30dk_ge50-kapali", (SM, "S4-Smit", "kapali", "birey")), "S4-Dunford-açık [keşifsel]": (v2, "S4-Dunford", "acik", (DU, "S4-Dunford", "acik", "calisma")),
        "S4-Dunford-kapalı [keşifsel]": (v2, "S4-Dunford", "kapali", (DU, "S4-Dunford", "kapali", "calisma")), "R-D [keşifsel]": (v2, "R-D", "kedi", None), "S1-kapalı": (e4, "S1-kapali", "30dk_ge50", None)}
SPEC_N = {"S1": ("S1", "30dk_ge50"), "S2": ("S2", "30dk_ge50"), "S3 [keşifsel]": ("S3", "kedi"), "S4-Smit-açık": ("S4-Smit", "30dk_ge50-acik"), "S4-Smit-kapalı": ("S4-Smit", "30dk_ge50-kapali"),
          "S4-Dunford-açık [keşifsel]": ("S4-Dunford", "acik"), "S4-Dunford-kapalı [keşifsel]": ("S4-Dunford", "kapali"), "R-D [keşifsel]": ("R-D", "kedi")}
NV = pd.read_parquet(O6 / "C_naive_bag.parquet")


def _prop(y, K): return np.bincount(y, minlength=K) / max(len(y), 1)
SMB = {c: _prop(ctx.y1[(ctx.cat == c) & (ctx.y1 >= 0)], 3) for c in ctx.CATS}
DCs = sorted(set(ctx.dcat)); DNB = {c: _prop(ctx.yd[ctx.dcat == c], 3) for c in DCs}


def NHV(scn, ind, Kt):
    """N-target: mean budget of the non-test individuals (same as 32_r06_C_naive.py); with K=2 the care class is dropped and the rest renormalised."""
    if scn in ("S1", "S2", "S4-Smit"): nh = np.mean([SMB[x] for x in ctx.CATS if x != ind], axis=0)
    else: nh = np.mean([DNB[x] for x in DCs if x != ind], axis=0)
    if Kt == 2: nh = nh[:2] / nh[:2].sum()
    return nh[:Kt]


def stats_pair(a, b):
    """a: CC bag df, b: arm bag df (same key: ind, bag, [rep]); columns p0..p2, h0..h2, N, K, RAE. Returns a dict."""
    k = [c for c in ("m", "rep", "ind", "bag") if c in a.columns and c in b.columns]
    j = a.merge(b, on=k, suffixes=("_c", "_a")); K = int(j.K_c.iloc[0])
    p = j[[f"p{i}_c" for i in range(K)]].values; hc = j[[f"h{i}_c" for i in range(K)]].values; ha = j[[f"h{i}_a" for i in range(K)]].values
    rare = int(np.argmin(p.mean(0))) if True else 0
    # rarest class = the class (with true proportion > 0) with the smallest mean true proportion over bags
    bc = pd.DataFrame(dict(ind=j.ind.values, bc=(hc[:, rare] - p[:, rare]) * 100, ba=(ha[:, rare] - p[:, rare]) * 100))
    g = bc.groupby("ind").mean()
    return dict(n_torba=int(len(j)), RAE_CC=float(j.RAE_c.mean()), RAE_kol=float(j.RAE_a.mean()), nadir_sinif=rare, nadir_gercek_ort_pct=float(p[:, rare].mean() * 100),
                nadir_yanlilik_CC_pp=float(g.bc.mean()), nadir_yanlilik_kol_pp=float(g.ba.mean()))


def rae_row(df):   # for data without a RAE column
    return df


NL2 = {(str(r.hedef_birey), str(r.torba)): int(r.N) for r in B8[(B8.senaryo == "S1-L2") & (B8.armx == "CC")].drop_duplicates(KEY8).itertuples()}
out = []
for r in M[M.kategori.isin(["ESDEGER", "BELIRSIZ"])].itertuples():
    base = dict(kaynak="r07_master", senaryo=r.senaryo, kol=r.kol, uzay=r.uzay, AE_CC=r.AE_CC, d=r.d, lo=r.lo, hi=r.hi, kategori=r.kategori, n=r.n, n_torba=r.n_torba)
    try:
        if r.uzay == "L2":
            if r.senaryo == "S2-eşlenmiş": raise KeyError("torba düzeyi sınıf vektörü kayıtlı değil")
            b2 = pd.read_csv(T / "r02_bag_level.csv"); b2 = b2[(b2.varyant == "va_f32") & (b2.bagdef == "30dk_ge50") & (b2.level == "L2") & (b2.basamak == ("S1-tasma" if "tasma" in r.senaryo else "S1-kosum"))]
            def mk(est):
                x = b2[b2.tahminci == est]; w = x.pivot_table(index=["kedi", "torba"], columns="sinif", values=["p", "p_hat"]).dropna(); K = w["p"].shape[1]; o = pd.DataFrame(index=w.index)
                for i in range(K): o[f"p{i}"] = w["p"].iloc[:, i].values; o[f"h{i}"] = w["p_hat"].iloc[:, i].values
                o["K"] = K; o = o.reset_index().rename(columns={"kedi": "ind", "torba": "bag"}); o["N"] = [NL2[(str(a), str(b))] for a, b in zip(o.ind, o.bag)]
                o["RAE"] = [Q.rae(row[[f"h{i}" for i in range(K)]].values.astype(float), row[[f"p{i}" for i in range(K)]].values.astype(float), row["N"]) for _, row in o.iterrows()]
                return o
            st = stats_pair(mk("CC"), mk(r.kol))
        elif r.kol == "N-hedef":
            scn, var = SPEC_N[r.senaryo]; df = {"S1": lad, "S2": lad, "S4-Smit-açık": src5, "S4-Smit-kapalı": src5}.get(r.senaryo, v2)
            g = df[(df.scen == scn) & (df.variant == var) & (df.arm == "CC") & (df.m == 0)].copy(); Kt = int(g.K.iloc[0]); H = []
            for rr in g.itertuples():
                nh = NHV(scn, rr.ind, Kt); pv = np.array([getattr(rr, f"p{i}") for i in range(Kt)]); H.append((nh, Q.rae(nh, pv, rr.N)))
            h = g.copy()
            for i in range(Kt): h[f"h{i}"] = [x[0][i] for x in H]
            h["RAE"] = [x[1] for x in H]; st = stats_pair(g, h)
        elif "m=1800" in r.kol:
            dfT, ts, tv, tl = SPEC[r.senaryo][3]; arm = r.kol.split(" (")[0]
            d0 = dfT[(dfT.scen == ts) & (dfT.variant == tv)]; a = d0[(d0.arm == "CC")].drop_duplicates(["m", "rep", "ind", "bag"]); a = a[a.m == 1800]
            b = d0[(d0.arm == arm) & (d0.level == tl) & (d0.m == 1800)].drop_duplicates(["m", "rep", "ind", "bag"])
            st = stats_pair(a, b)
        else:
            df, scn, var, _ = SPEC[r.senaryo]; g = df[(df.scen == scn) & (df.variant == var) & (df.m == 0)]
            st = stats_pair(g[g.arm == "CC"], g[g.arm == r.kol])
        base.update(st)
    except KeyError as e:
        base["not"] = str(e)
    out.append(base)
# R08
for r in R8[R8.kategori.isin(["ESDEGER", "BELIRSIZ"])].itertuples():
    base = dict(kaynak="r08_tests", senaryo=f"{r.senaryo} {r.torba_turu}", kol=r.kol, uzay="", AE_CC=r.AE_CC, d=r.d, lo=r.lo, hi=r.hi, kategori=r.kategori, n=r.n, n_torba=r.n_torba)
    g = B8[(B8.senaryo == r.senaryo) & (B8.torba_turu == r.torba_turu)]; ka = r.kol
    ca = g[g.armx == "CC"]; aa = g[g.armx == ka]; K = g.sinif.nunique(); cl = list(g.sinif.drop_duplicates())
    def wide(x):
        w = x.pivot_table(index=KEY8, columns="sinif", values=["p", "p_hat"]).reindex(columns=cl, level=1); o = pd.DataFrame(index=w.index)
        for i, c in enumerate(cl): o[f"p{i}"] = w[("p", c)].values; o[f"h{i}"] = w[("p_hat", c)].values
        N = x.drop_duplicates(KEY8).set_index(KEY8).N; o["N"] = N.reindex(o.index).values; o["K"] = len(cl)
        o["RAE"] = [Q.rae(np.array([row[f"h{i}"] for i in range(len(cl))]), np.array([row[f"p{i}"] for i in range(len(cl))]), row["N"]) for _, row in o.iterrows()]
        o = o.reset_index(); o["ind"] = o.hedef_birey; o["bag"] = o.dis_katman.astype(str) + "|" + o.torba.astype(str); return o
    st = stats_pair(wide(ca), wide(aa)); st["nadir_sinif_ad"] = cl[st["nadir_sinif"]]; base.update(st); out.append(base)
B2 = pd.DataFrame(out)
if "nadir_sinif" in B2: B2["nadir_sinif"] = B2.nadir_sinif.astype("Int64")
B2.to_csv(T / "r09_B2_equiv_uncertain.csv", index=False)

# ------------------------------------------------------------------ B3
rows = []


def zero_frac(name, space, bag_rows, y, K, cls):
    for k in range(K):
        pk = np.array([np.mean(y[r] == k) for r in bag_rows])
        rows.append(dict(torba_turu=name, uzay=space, sinif=cls[k], n_torba=len(bag_rows), gercek_sifir_orani=float((pk == 0).mean()), gercek_ort=float(pk.mean()), gercek_medyan=float(np.median(pk))))


nat30 = [ix for (c, w, ix) in ctx.nat30]; nat2 = [ix for (c, w, ix) in ctx.nat2]
zero_frac("Smit 30 dk (kapsam ≥%50)", "L1", nat30, ctx.y1, 3, L8.L1N); zero_frac("Smit 2 dk (kapsam ≥%50)", "L1", nat2, ctx.y1, 3, L8.L1N)
zero_frac("Smit 30 dk (kapsam ≥%50)", "L2", nat30, ctx.y2, 7, ["Lying", "Sitting", "Standing", "Walking", "Trotting", "Grooming", "Eating"])
zero_frac("Smit 2 dk (kapsam ≥%50)", "L2", nat2, ctx.y2, 7, ["Lying", "Sitting", "Standing", "Walking", "Trotting", "Grooming", "Eating"])
for space, y, K, cls in (("K-L1", ctx.dy_L1, 2, L8.L1N[:2]), ("K-L2", ctx.dy_L2, 5, L8.DOGL2)):
    bg = L8.dog_bags(ctx.NK, y)
    zero_frac("köpek × kayıt (N ≥ 60)", space, [rr for d in bg.values() for (_, rr) in d["kayit"]], y, K, cls)
    zero_frac("köpek doğal 2 dk (N ≥ 60)", space, [rr for d in bg.values() for (_, rr) in d["dogal2"]], y, K, cls)
B3 = pd.DataFrame(rows); B3.to_csv(T / "r09_B3_dr_zero_train.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 300, "display.float_format", "{:.4f}".format)
print("tamam", len(B1), len(B2), len(B3))
