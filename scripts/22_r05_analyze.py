r"""22_r05_analyze.py — R05 analysis: summary tables, hypothesis counts, bias²/variance, figures, bag-level CSV.
Reads: r05/r05_ladder_bags.parquet, r05_sources_bags.parquet, E_*.parquet (+ _miss), F_bags.parquet, sources.pkl; results/tables/r05_*.csv/json, r03_E_summary_30dk_ge50.csv.
Writes: results/tables/r05_bag_level.csv, r05_summary_*.csv, r05_hypotheses.json, r05_bv.csv, r05_bias_classes.csv, r05_missing.csv, r05_sensitivity.csv; results/figures/r05_ladder.png, r05_label_budget.png
"Significant gain": the 95% CI of AE_CC - AE_arm lies entirely above 0 (predefined). CI: individual-level bootstrap (1,000), bags paired within individual; repeats are averaged within a bag first.
"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import paths as P
from src import r05lib as R, quant as Q

OUT = P.R05; T = P.TABLES; FG = P.FIGURES; SEED = P.SEED
NAMES = ["Inaktif", "Lokomosyon", "Bakim"]
parts = [OUT / "r05_ladder_bags.parquet", OUT / "r05_sources_bags.parquet", OUT / "F_bags.parquet"] + sorted(OUT.glob("E_*.parquet"))
parts = [p for p in parts if not p.name.endswith("_miss.parquet")]
A = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
A["scen_var"] = A.scen + "|" + A.variant
# ---- bag-level CSV (long)
L = []
for k in range(3):
    s = A[A.K > k]
    L.append(pd.DataFrame(dict(senaryo=s.scen, varyant=s.variant, duzey=s.level, kol=s.arm, m=s.m, tekrar=s.rep, hedef_birey=s.ind, torba=s.bag, N=s.N, sinif=NAMES[k],
                               p=s[f"p{k}"].round(5), p_hat=s[f"h{k}"].round(5))))
pd.concat(L, ignore_index=True).to_csv(T / "r05_bag_level.csv", index=False)
print("torba satırı", len(A), flush=True)


def delta(sub, arm, level=None, col="AE"):
    """AE_CC - AE_arm by m. Returns DataFrame(m, n_bag, n_ind, AE, lo, hi, d, dlo, dhi)."""
    a = sub[(sub.arm == arm) & ((sub.level == level) if level else True)]
    cc = sub[sub.arm == "CC"]; out = []
    if (cc.m > 0).any(): keys = ["m", "rep", "ind", "bag"]; ccx = cc.drop_duplicates(keys)[keys + [col]]
    else: keys = ["ind", "bag"]; ccx = cc.drop_duplicates(keys)[keys + [col]]
    j = a.merge(ccx, on=keys, suffixes=("", "_cc"))
    for m, g in j.groupby("m"):
        g = g.assign(d=g[col + "_cc"] - g[col]).groupby(["ind", "bag"], as_index=False).agg(AE=(col, "mean"), d=("d", "mean"), cc=(col + "_cc", "mean"))
        r = dict(m=int(m), n_bag=len(g), n_ind=g.ind.nunique())
        r["AE"], r["lo"], r["hi"] = R.boot_mean(g.AE.values, g.ind.values); r["d"], r["dlo"], r["dhi"] = R.boot_mean(g.d.values, g.ind.values)
        r["AE_CC"] = float(g.cc.mean()); out.append(r)
    return pd.DataFrame(out)


def pair(sub, a, b, level_a=None, level_b=None, m=None):
    """mean(AE_a - AE_b), paired on (m, rep, ind, bag); positive = b better."""
    keys = ["m", "rep", "ind", "bag"]
    x = sub[(sub.arm == a) & ((sub.level == level_a) if level_a else True)][keys + ["AE"]]; y = sub[(sub.arm == b) & ((sub.level == level_b) if level_b else True)][keys + ["AE"]]
    j = x.merge(y, on=keys, suffixes=("_a", "_b"))
    res = {}
    for mm, g in j.groupby("m"):
        if m is not None and mm not in m: continue
        g = g.assign(d=g.AE_a - g.AE_b).groupby(["ind", "bag"], as_index=False).d.mean(); mu, lo, hi = R.boot_mean(g.d.values, g.ind.values)
        res[int(mm)] = dict(d=mu, lo=lo, hi=hi, n=len(g))
    return res


# ================================================================ source-only and ladder (m = 0)
src_rows = []
for (sc, var), sub in A[(A.m == 0) & (A.level == "-") & ~A.scen.str.contains("APP|kopru")].groupby(["scen", "variant"]):
    for arm in ["CC", "PCC", "ACC", "PACC", "EMQ"]:
        a = sub[sub.arm == arm]
        if a.empty: continue
        g = a.groupby(["ind", "bag"], as_index=False).AE.mean(); r = dict(scen=sc, variant=var, arm=arm, n_bag=len(g), n_ind=g.ind.nunique())
        r["AE"], r["lo"], r["hi"] = R.boot_mean(g.AE.values, g.ind.values)
        g1 = a.groupby(["ind", "bag"], as_index=False).RAE.mean(); r["RAE"], r["RAE_lo"], r["RAE_hi"] = R.boot_mean(g1.RAE.values, g1.ind.values)
        if arm != "CC":
            ccx = sub[sub.arm == "CC"].drop_duplicates(["ind", "bag"])[["ind", "bag", "AE"]]
            j = a.merge(ccx, on=["ind", "bag"], suffixes=("", "_cc")); d = (j.AE_cc - j.AE).values
            r["d"], r["dlo"], r["dhi"] = R.boot_mean(d, j.ind.values); r["sig"] = bool(r["dlo"] > 0)
        src_rows.append(r)
SRC = pd.DataFrame(src_rows); SRC.to_csv(T / "r05_summary_source.csv", index=False)
H = {}

# ---- ladder
def src_val(sc, var, arm, col="AE"):
    r = SRC[(SRC.scen == sc) & (SRC.variant == var) & (SRC.arm == arm)].iloc[0]; return r
H["merdiven_CC"] = {k: (lambda r: [r.AE, r.lo, r.hi, int(r.n_bag)])(src_val(*k, "CC")) for k in
                   [("S0", "30dk_ge50"), ("S1", "30dk_ge50"), ("S2", "30dk_ge50"), ("S4-Smit", "30dk_ge50-acik"), ("R-D", "kedi"), ("S3", "kedi"), ("S4-Dunford", "acik")]}
# S3 - R-D (paired: same cats)
x = A[(A.scen == "S3") & (A.arm == "CC")].drop_duplicates(["ind", "bag"]).set_index("ind").AE; y = A[(A.scen == "R-D") & (A.arm == "CC")].drop_duplicates(["ind", "bag"]).set_index("ind").AE
jj = pd.concat([x, y], axis=1, keys=["s3", "rd"]).dropna(); H["ceza_S3_RD"] = dict(zip(("d", "lo", "hi"), R.boot_mean((jj.s3 - jj.rd).values, jj.index.values)), n=len(jj))
# PACC for the same
for arm in ("PACC", "ACC", "EMQ"):
    x = A[(A.scen == "S3") & (A.arm == arm)].drop_duplicates(["ind", "bag"]).set_index("ind").AE; y = A[(A.scen == "R-D") & (A.arm == arm)].drop_duplicates(["ind", "bag"]).set_index("ind").AE
    jj = pd.concat([x, y], axis=1, keys=["s3", "rd"]).dropna(); H[f"ceza_S3_RD_{arm}"] = dict(zip(("d", "lo", "hi"), R.boot_mean((jj.s3 - jj.rd).values, jj.index.values)))
# D17 cost: S1 (D17) vs S1-32
cl = pd.read_csv(T / "r05_classifier_by_ind.csv")
cs = {}
for sc in ("S1", "S1-32", "S2", "R-D", "K"):
    g = cl[cl.scen == sc]
    cs[sc] = dict(n=len(g), macroF1=R.boot_mean(g.macroF1.values, g.ind.astype(str).values), balacc=R.boot_mean(g.balacc.values, g.ind.astype(str).values))
cs2 = pd.read_csv(T / "r05_classifier_src.csv")
for (sc, var), g in cs2.groupby(["scen", "variant"]):
    cs[f"{sc}|{var}"] = dict(n=len(g), macroF1=R.boot_mean(g.macroF1.values, g.ind.astype(str).values), balacc=R.boot_mean(g.balacc.values, g.ind.astype(str).values))
H["siniflandirici"] = cs
a = cl[cl.scen == "S1"].set_index("ind").macroF1; b = cl[cl.scen == "S1-32"].set_index("ind").macroF1
H["d17_bedeli_macroF1_S1"] = dict(zip(("d", "lo", "hi"), R.boot_mean((a - b).values, a.index.values)))
sub1 = A[(A.scen == "S1") & (A.variant == "30dk_ge50") & (A.arm == "CC")].set_index(["ind", "bag"]).AE; sub2 = A[(A.scen == "S1-32") & (A.variant == "30dk_ge50") & (A.arm == "CC")].set_index(["ind", "bag"]).AE
jj = pd.concat([sub1, sub2], axis=1, keys=["d17", "f32"]).dropna(); H["d17_bedeli_CC_AE_S1"] = dict(zip(("d", "lo", "hi"), R.boot_mean((jj.d17 - jj.f32).values, jj.index.get_level_values(0).values)), AE_d17=float(jj.d17.mean()), AE_32=float(jj.f32.mean()))
# R03 S2-matched L1 (32 features)
r3 = pd.read_csv(T / "r03_E_summary_30dk_ge50.csv"); r3 = r3[r3.scen == "S2-esl"].set_index("est")
H["r03_S2_esl_L1_AE"] = {e: float(r3.loc[e, "AE_L1"]) for e in ("CC", "PCC", "ACC", "PACC", "EMQ") if e in r3.index}

# ================================================================ E: target-labelled arms
E = A[A.m > 0]; EM = []
for (sc, var), sub in E.groupby(["scen", "variant"]):
    full = A[(A.scen == sc) & (A.variant == var)]
    for level in sorted(sub.level.unique()):
        for arm in ["T-ACC", "PACC-hedef", "PACC-buzme", "Yeniden-egitim", "Yalniz-hedef"]:
            if ((sub.arm == arm) & (sub.level == level)).sum() == 0: continue
            dd = delta(full, arm, level); dd.insert(0, "arm", arm); dd.insert(0, "level", level); dd.insert(0, "variant", var); dd.insert(0, "scen", sc); EM.append(dd)
        dd = delta(full, "CC", None) if False else None
EMd = pd.concat(EM, ignore_index=True); EMd["sig"] = EMd.dlo > 0; EMd.to_csv(T / "r05_E_main.csv", index=False)
# CC reference AE (on the remaining seconds) by m
ccref = []
for (sc, var), sub in A[A.arm == "CC"].groupby(["scen", "variant"]):
    for m, g in sub.groupby("m"):
        g2 = g.groupby(["ind", "bag"], as_index=False).AE.mean(); mu, lo, hi = R.boot_mean(g2.AE.values, g2.ind.values); ccref.append(dict(scen=sc, variant=var, m=int(m), AE=mu, lo=lo, hi=hi))
pd.DataFrame(ccref).to_csv(T / "r05_E_ccref.csv", index=False)

# ---- H6: best arm 1-3 at study level, smallest significant m
h6 = {}
for (sc, var), g in EMd[(EMd.level == "calisma") & EMd.arm.isin(["T-ACC", "PACC-hedef", "PACC-buzme"])].groupby(["scen", "variant"]):
    res = {}
    for m, gm in g.groupby("m"):
        b = gm.sort_values("AE").iloc[0]; res[int(m)] = dict(en_iyi=b.arm, AE=b.AE, d=b.d, dlo=b.dlo, dhi=b.dhi, sig=bool(b.dlo > 0), n_ind=int(b.n_ind), n_bag=int(b.n_bag))
    sig_m = [m for m, r in res.items() if r["sig"]]
    h6[f"{sc}|{var}"] = dict(en_kucuk_anlamli_m=(min(sig_m) if sig_m else None), ayrinti=res)
H["H6"] = h6
# ---- H7: S4-Smit, PACC-target individual vs study (positive = individual better)
H["H7"] = {}
for var in ("kapali", "acik"):
    sub = A[(A.scen == "S4-Smit") & (A.variant == var)]
    H["H7"][var] = pair(sub, "PACC-hedef", "PACC-hedef", "calisma", "birey", m=[300, 1800])
# ---- H8: retrain vs best arm (1-3) at the same m; positive = retrain better (AE_best - AE_retrain); repeats 0-4
H["H8"] = {}
for (sc, var, level), g in EMd[EMd.arm.isin(["T-ACC", "PACC-hedef", "PACC-buzme"]) & (EMd.level != "-")].groupby(["scen", "variant", "level"]):
    full = A[(A.scen == sc) & (A.variant == var) & (A.rep < 5)]
    res = {}
    for m in (300, 1800):
        gm = g[g.m == m]
        if gm.empty: continue
        best = gm.sort_values("AE").iloc[0].arm
        pr = pair(full, best, "Yeniden-egitim", level, level, m=[m]).get(m)
        if pr: res[m] = dict(en_iyi_kol=best, **pr)
    H["H8"][f"{sc}|{var}|{level}"] = res
# ladder extra arms: S1/S2 PACC-target and retrain m=1800 (individual) — also present in EMd
H["ladder_extra"] = {}
for sc in ("S1", "S2"):
    sub = A[(A.scen == sc) & (A.variant == "30dk_ge50") & (A.m == 1800)]
    for arm in ("PACC-hedef", "Yeniden-egitim"):
        d = delta(sub, arm, "birey"); H["ladder_extra"][f"{sc}|{arm}"] = d.iloc[0].to_dict() if len(d) else None

# ================================================================ missing-class counts and realised m
mi = []
for f in sorted(OUT.glob("E_*_miss.parquet")):
    d = pd.read_parquet(f)
    for (sc, var, lev, m), g in d.groupby(["scen", "variant", "level", "m"]):
        mi.append(dict(scen=sc, variant=var, level=lev, m=int(m), n_unit=len(g), n_eksik_sinif_birimi=int((g.n_missing > 0).sum()), toplam_eksik_sutun=int(g.n_missing.clip(lower=0).sum()),
                       n_lab_ort=float(g.n_lab.mean()), n_lab_min=int(g.n_lab.min())))
pd.DataFrame(mi).to_csv(T / "r05_missing.csv", index=False)
# target-only skip count: units that have a retrain row but no target-only row
sk = []
for (sc, var, lev), g in A[A.arm.isin(["Yeniden-egitim", "Yalniz-hedef"])].groupby(["scen", "variant", "level"]):
    k = ["m", "rep", "ind"]; rt = g[g.arm == "Yeniden-egitim"][k].drop_duplicates(); yt = g[g.arm == "Yalniz-hedef"][k].drop_duplicates()
    for m, r in rt.groupby("m"): sk.append(dict(scen=sc, variant=var, level=lev, m=int(m), birim=len(r), yalniz_hedef_atlanan=len(r) - len(yt[yt.m == m])))
pd.DataFrame(sk).to_csv(T / "r05_target_only_skips.csv", index=False)

# ================================================================ Care / Locomotion signed bias (CC, PACC, best target-labelled arm; m=1800)
def bias_rows(sub, arm, k):
    g = sub[sub.arm == arm]; b = (g[f"h{k}"] - g[f"p{k}"]) * 100
    g2 = pd.DataFrame(dict(ind=g.ind.values, bag=g.bag.values, b=b.values)).groupby(["ind", "bag"], as_index=False).b.mean(); return R.boot_mean(g2.b.values, g2.ind.values)


SRCV = {("S3", "-"): "kedi", ("S4-Smit", "acik"): "30dk_ge50-acik", ("S4-Smit", "kapali"): "30dk_ge50-kapali", ("S4-Dunford", "acik"): "acik",
        ("S4-Dunford", "kapali"): "kapali", ("S1", "30dk_ge50"): "30dk_ge50", ("S2", "30dk_ge50"): "30dk_ge50"}
bb = []
for (sc, var, lev), g in EMd[(EMd.m == 1800) & EMd.arm.isin(["T-ACC", "PACC-hedef", "PACC-buzme", "Yeniden-egitim", "Yalniz-hedef"])].groupby(["scen", "variant", "level"]):
    best = g.sort_values("AE").iloc[0].arm; full = A[(A.scen == sc) & (A.variant == var)]; srcA = A[(A.scen == sc) & (A.variant == SRCV[(sc, var)]) & (A.m == 0) & (A.level == "-")]
    Kt = int(full.K.iloc[0])
    ccs = full[(full.arm == "CC") & (full.m == 1800)]
    ccs = ccs if len(ccs) else srcA[srcA.arm == "CC"]
    for arm, s_ in (("CC", ccs), ("PACC", srcA[srcA.arm == "PACC"]), (best + " (en iyi, m=1800)", full[(full.arm == best) & (full.m == 1800) & (full.level == lev)])):
        for k, nm in ((1, "Lokomosyon"), (2, "Bakim")):
            if k >= Kt or s_.empty: continue
            mu, lo, hi = bias_rows(s_.assign(arm=arm), arm, k); bb.append(dict(scen=sc, variant=var, level=lev, arm=arm, sinif=nm, bias_pp=mu, lo=lo, hi=hi))
BB = pd.DataFrame(bb); BB.to_csv(T / "r05_bias_classes.csv", index=False)
# CC and PACC for source-only S3/S4/S1/S2 (m=0)
bb2 = []
for (sc, var), sub in A[(A.m == 0) & ~A.scen.str.contains("APP|kopru")].groupby(["scen", "variant"]):
    if sub.K.iloc[0] < 3: continue
    for arm in ("CC", "PACC"):
        for k, nm in ((1, "Lokomosyon"), (2, "Bakim")):
            s_ = sub[sub.arm == arm]
            if not s_.empty: mu, lo, hi = bias_rows(s_, arm, k); bb2.append(dict(scen=sc, variant=var, arm=arm, sinif=nm, bias_pp=mu, lo=lo, hi=hi))
pd.DataFrame(bb2).to_csv(T / "r05_bias_source.csv", index=False)

# ================================================================ open set: distribution of the target's care (Bakim) seconds by the source model
S = pickle.load(open(OUT / "sources.pkl", "rb")); lab, y1, *_ = R.load_smit(); idx = S["idx"].astype(int); dun, _ = R.load_dunford(); yd = dun.y1.values
oc = {}
for nm, y, Pm in (("S4-Smit", y1[idx], S["P4s"]), ("S4-Dunford", yd, S["P4d"])):
    pr = Pm[y == 2].argmax(1); oc[nm] = dict(n_bakim=int((y == 2).sum()), inaktif_pct=float((pr == 0).mean() * 100), lokomosyon_pct=float((pr == 1).mean() * 100))
H["acik_kume_dagilim"] = oc

# ================================================================ H9 APP and bridge
ap = {}
for (sc, var), sub in A[A.scen.str.contains("APP|kopru")].groupby(["scen", "variant"]):
    ccx = sub[sub.arm == "CC"].set_index(["ind", "bag"]).AE
    for arm in ("ACC", "PACC", "EMQ"):
        a = sub[sub.arm == arm].set_index(["ind", "bag"]).AE; jj = pd.concat([ccx, a], axis=1, keys=["cc", "x"]).dropna()
        mu, lo, hi = R.boot_mean((jj.cc - jj.x).values, jj.index.get_level_values(0).values); ap[f"{sc}|{var}|{arm}"] = dict(d=mu, lo=lo, hi=hi, n=len(jj), AE_CC=float(jj.cc.mean()), AE_x=float(jj.x.mean()))
sub = A[(A.scen.str.contains("APP")) & (A.arm.isin(["CC", "PACC-hedef", "PACC"]))]
for sc, g in sub.groupby("scen"):
    ccx = g[g.arm == "CC"].set_index(["ind", "bag"]).AE; a = g[g.arm == "PACC-hedef"].groupby(["ind", "bag"]).AE.mean(); jj = pd.concat([ccx, a], axis=1, keys=["cc", "x"]).dropna()
    mu, lo, hi = R.boot_mean((jj.cc - jj.x).values, jj.index.get_level_values(0).values); ap[f"{sc}|PACC-hedef(300)"] = dict(d=mu, lo=lo, hi=hi, n=len(jj), AE_CC=float(jj.cc.mean()), AE_x=float(jj.x.mean()))
H["APP"] = ap
H["APP_stats"] = json.load(open(T / "r05_F_app_stats.json"))

# ================================================================ bias² / variance (R04 method)
def bv(sub, arm, level=None, m=None):
    g = sub[(sub.arm == arm) & ((sub.level == level) if level else True) & ((sub.m == m) if m is not None else True)]
    rows = []
    for (ind, rep), gg in g.groupby(["ind", "rep"]):
        if len(gg) < 2: continue
        e = (gg[["h0", "h1", "h2"]].values - gg[["p0", "p1", "p2"]].values); e = e[:, ~np.isnan(e).any(0)]
        rows.append(dict(ind=ind, bias2=float((e.mean(0) ** 2).sum()), var=float(e.var(0).sum()), n_bag=len(gg)))
    if not rows: return None
    r = pd.DataFrame(rows).groupby("ind", as_index=False).mean(numeric_only=True); o = dict(n_ind=len(r))
    for c in ("bias2", "var"): o[c], o[c + "_lo"], o[c + "_hi"] = R.boot_mean(r[c].values, r.ind.values)
    o["mse"] = o["bias2"] + o["var"]; return o


BV = []
sub = A[(A.scen == "S4-Smit") & (A.variant == "acik")]
base = A[(A.scen == "S4-Smit") & (A.variant == "30dk_ge50-acik") & (A.m == 0)]
for arm in ("CC", "PACC"):
    o = bv(base, arm);
    if o: BV.append(dict(set="S4-Smit-acik 30dk", arm=arm, **o))
o = bv(sub, "PACC-hedef", "birey", 1800); BV.append(dict(set="S4-Smit-acik 30dk", arm="PACC-hedef(birey,1800)", **o))
o = bv(sub, "PACC-hedef", "calisma", 1800); BV.append(dict(set="S4-Smit-acik 30dk", arm="PACC-hedef(calisma,1800)", **o))
for sc in ("S3-APP", "S4-Dunford-APP"):
    s_ = A[A.scen == sc]
    for arm, lev, m in (("CC", "-", 0), ("PACC", "-", 0), ("PACC-hedef", "calisma", 300)):
        o = bv(s_, arm, lev, m)
        if o: BV.append(dict(set=sc + " (APP 120 s)", arm=arm + (f"({m})" if m else ""), **o))
pd.DataFrame(BV).to_csv(T / "r05_bv.csv", index=False)

# ================================================================ sensitivity: Dunford runs excluded; DBA edge seconds excluded (S3 and S4-Dunford-open only, source-only)
dun_all = pd.read_parquet(P.DUNFORD_FEAT); dun_all = dun_all[dun_all.y1 >= 0].reset_index(drop=True)
sens = []


def ae_set(Pm, y, cats, keepmask, src, Km, Kt):
    out = {}
    for arm in ("CC", "PACC"):
        v = []
        for c in sorted(set(cats)):
            ii = np.where((cats == c) & keepmask)[0]
            if len(ii) < 60: continue
            p = R.truth(y[ii], Kt); ph = R.estimates(Pm[ii], src, Km)[arm]; v.append((c, Q.ae(R.pad3(ph, Kt), p)))
        d = pd.DataFrame(v, columns=["c", "ae"]); out[arm] = (R.boot_mean(d.ae.values, d.c.values), len(d))
    return out


for lbl, mask in (("ana", np.ones(len(dun_all), bool)), ("run_disarida", (dun_all.beh != "Run").values), ("DBA_kenar_disarida", (~dun_all.edge).values)):
    for sc, Pm, src, Km, Kt, ym in (("S3", S["P3"], S["src3"], 3, 3, np.ones(len(dun_all), bool)), ("S4-Dunford-acik", S["P4d"], S["src4"], 2, 3, np.ones(len(dun_all), bool))):
        r = ae_set(Pm, dun_all.y1.values, dun_all.cat.values, mask & ym, src, Km, Kt)
        for arm, (v, n) in r.items(): sens.append(dict(set=sc, durum=lbl, arm=arm, AE=v[0], lo=v[1], hi=v[2], n_kedi=n, n_pencere=int(mask.sum())))
pd.DataFrame(sens).to_csv(T / "r05_sensitivity.csv", index=False)
def _sk(o):
    if isinstance(o, dict): return {(k if isinstance(k, (str, int, float, bool)) or k is None else "|".join(map(str, k))): _sk(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_sk(v) for v in o]
    return o
json.dump(_sk(H), open(T / "r05_hypotheses.json", "w"), indent=1, default=float)

# ================================================================ figures
cols = {"CC": "#444444", "PACC": "#1f77b4", "PACC-hedef": "#2ca02c", "Yeniden-egitim": "#d62728", "T-ACC": "#9467bd", "PACC-buzme": "#8c564b", "Yalniz-hedef": "#ff7f0e"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
smit_steps = [("S0", "30dk_ge50", "S0"), ("S1", "30dk_ge50", "S1"), ("S2", "30dk_ge50", "S2"), ("S4-Smit", "30dk_ge50-acik", "S4-Smit\n(acik)")]
dun_steps = [("R-D", "kedi", "R-D"), ("S3", "kedi", "S3"), ("S4-Dunford", "acik", "S4-Dunford\n(acik)")]
for ax, steps, ttl in ((axes[0], smit_steps, "Smit hedef (30 dk >= %50)"), (axes[1], dun_steps, "Dunford hedef (kedi torbasi; Dunford karisimi)")):
    for arm in ("CC", "PACC"):
        v = [SRC[(SRC.scen == s) & (SRC.variant == v_) & (SRC.arm == arm)].iloc[0] for s, v_, _ in steps]
        ax.errorbar(range(len(steps)), [r.AE for r in v], yerr=[[r.AE - r.lo for r in v], [r.hi - r.AE for r in v]], color=cols[arm], marker="o", ls="-", label=f"{arm} (kaynak)", capsize=2)
    for arm in ("PACC-hedef", "Yeniden-egitim"):
        for lev, ls, mk in (("birey", "--", "s"), ("calisma", ":", "^")):
            xs, ys, lo, hi = [], [], [], []
            for i, (s, v_, _) in enumerate(steps):
                if s in ("R-D", "S0"): continue
                vv = {"S1": "30dk_ge50", "S2": "30dk_ge50", "S4-Smit": "acik", "S3": "-", "S4-Dunford": "acik"}[s]
                d = EMd[(EMd.scen == s) & (EMd.variant == vv) & (EMd.level == lev) & (EMd.arm == arm) & (EMd.m == 1800)]
                if len(d): xs.append(i); ys.append(d.AE.iloc[0]); lo.append(d.AE.iloc[0] - d.lo.iloc[0]); hi.append(d.hi.iloc[0] - d.AE.iloc[0])
            if xs: ax.errorbar(xs, ys, yerr=[lo, hi], color=cols[arm], ls=ls, marker=mk, label=f"{arm} m=1800 ({'birey' if lev == 'birey' else 'calisma'})", capsize=2)
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([s[2] for s in steps]); ax.set_ylabel("AE (L1)"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
axes[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(FG / "r05_ladder.png", dpi=130); plt.close()
panels = [("S3", "-"), ("S4-Smit", "acik"), ("S4-Dunford", "acik"), ("S4-Smit", "kapali")]
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
ccr = pd.DataFrame(ccref)
for ax, (sc, var) in zip(axes.ravel(), panels):
    c0 = ccr[(ccr.scen == sc) & (ccr.variant == var)]
    ax.axhline(c0[c0.m == 0].AE.iloc[0] if (c0.m == 0).any() else c0.AE.mean(), color=cols["CC"], lw=1, label="CC (kaynak)")
    for arm in ("T-ACC", "PACC-hedef", "PACC-buzme", "Yeniden-egitim", "Yalniz-hedef"):
        for lev, ls in (("birey", "--"), ("calisma", "-")):
            d = EMd[(EMd.scen == sc) & (EMd.variant == var) & (EMd.level == lev) & (EMd.arm == arm)].sort_values("m")
            if len(d): ax.plot(d.m, d.AE, color=cols[arm], ls=ls, marker="o", ms=3, label=f"{arm} ({lev})")
    ax.set_xscale("log"); ax.set_xticks(list((60, 300, 900, 1800))); ax.set_xticklabels(["60", "300", "900", "1800"]); ax.set_title(f"{sc} {var if var != '-' else ''}", fontsize=10); ax.set_xlabel("etiketli hedef saniye (m)"); ax.set_ylabel("AE"); ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=6); plt.tight_layout(); plt.savefig(FG / "r05_label_budget.png", dpi=130); plt.close()
print("BİTTİ analiz")
