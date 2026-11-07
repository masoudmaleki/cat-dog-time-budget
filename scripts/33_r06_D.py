r"""33_r06_D.py — R06 D: gain–shift. delta=1/2||p-pi_source||_1, delta_hat=1/2||p_CC-pi_source||_1, kappa=1/2||p_PACC-p_CC||_1, g=AE_CC-AE_arm.
Spearman (individual-cluster bootstrap), decision rule (tau: S1, 6+6 random splits x 50, median frozen).
Reads: r05/r05_ladder_bags.parquet, r05_sources_bags.parquet, F_bags.parquet, r06/B_v2_bags.parquet, Smit table, dog_aneck_1s.parquet.
Writes: results/tables/r06_D_*.csv/json, results/figures/r06_gain_vs_shift.png
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import paths as P
from src import r05lib as R, stats06 as S6

T = P.TABLES; O5 = P.R05; O6 = P.R06; SEED = P.SEED
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); cat = lab.Cat_id.values; ok1 = y1 >= 0; CATS = sorted(set(cat))
dog, _ = R.load_dog(); yg = dog.y1.values[dog.y1.values < 2]


def prop(y, K): return np.bincount(y, minlength=K) / max(len(y), 1)


pi_dog = prop(yg, 2)
lad = pd.read_parquet(O5 / "r05_ladder_bags.parquet"); src5 = pd.read_parquet(O5 / "r05_sources_bags.parquet"); fb = pd.read_parquet(O5 / "F_bags.parquet"); bv2 = pd.read_parquet(O6 / "B_v2_bags.parquet")
SETS = [  # (label, df, (scen, variant), K, type, pi_source(ind))
    ("S1", lad, ("S1", "30dk_ge50"), 3, "dogal30", lambda c: prop(y1[ok1 & (cat != c)], 3)),
    ("S1-2dk", fb, ("S1-kopru", "dogal2dk"), 3, "dogal2", lambda c: prop(y1[ok1 & (cat != c)], 3)),
    ("S1-APP120", fb, ("S1-kopru", "APP120"), 3, "APP", lambda c: prop(y1[ok1 & (cat != c)], 3)),
    ("S2", lad, ("S2", "30dk_ge50"), 3, "dogal30", lambda c: prop(y1[ok1 & (cat != c)], 3)),
    ("S3", bv2, ("S3", "kedi"), 3, "dogal", lambda c: prop(y1[ok1], 3)),
    ("S4-Smit-kapali", src5, ("S4-Smit", "30dk_ge50-kapali"), 2, "dogal30", lambda c: pi_dog),
    ("S4-Dunford-kapali", bv2, ("S4-Dunford", "kapali"), 2, "dogal", lambda c: pi_dog)]
rows = []
for lb, df, (sc, var), K, kind, pif in SETS:
    g = df[(df.scen == sc) & (df.variant == var) & (df.m == 0)]
    W = {a: gg.set_index(["ind", "bag"]) for a, gg in g.groupby("arm")}
    cc = W["CC"]
    for (ind, bag), r in cc.iterrows():
        p = np.array([r[f"p{k}"] for k in range(K)]); hc = np.array([r[f"h{k}"] for k in range(K)])
        rp = W["PACC"].loc[(ind, bag)]; ra = W["ACC"].loc[(ind, bag)]; hp = np.array([rp[f"h{k}"] for k in range(K)]); pi = pif(ind)
        rows.append(dict(set=lb, scen=sc, tur=kind, ind=str(ind), bag=str(bag), N=int(r.N), K=K, delta=0.5 * np.abs(p - pi).sum(), delta_hat=0.5 * np.abs(hc - pi).sum(),
                         kappa=0.5 * np.abs(hp - hc).sum(), AE_CC=float(r.AE), AE_PACC=float(rp.AE), AE_ACC=float(ra.AE)))
B = pd.DataFrame(rows); B["g_PACC"] = B.AE_CC - B.AE_PACC; B["g_ACC"] = B.AE_CC - B.AE_ACC
B.to_parquet(O6 / "D_bag.parquet"); B.to_csv(T / "r06_D_bag.csv", index=False)
# ---- 1. delta summary
d1 = B.groupby("set").delta.agg(n="size", medyan="median", p90=lambda x: x.quantile(0.9)).reset_index()
med = d1.set_index("set").medyan
oran = dict(APP120_over_dogal2dk=float(med["S1-APP120"] / med["S1-2dk"]), APP120_over_dogal30=float(med["S1-APP120"] / med["S1"]))
# ---- 2. Spearman, individual-cluster bootstrap
rng = np.random.default_rng(SEED)


def sp(x, y): return spearmanr(x, y)[0] if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else np.nan


def boot_sp(df, xc, yc, nb=1000):
    inds = df.ind.unique(); groups = {i: df[df.ind == i] for i in inds}; out = []
    for _ in range(nb):
        pick = rng.choice(inds, len(inds)); d = pd.concat([groups[i] for i in pick]); out.append(sp(d[xc].values, d[yc].values))
    out = np.array(out); out = out[~np.isnan(out)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) else (np.nan, np.nan)


cors = []
for lb, g in list(B.groupby("set")) + [("HAVUZ", B)]:
    for xc, yc in (("delta", "delta_hat"), ("delta", "g_PACC"), ("delta_hat", "g_PACC"), ("kappa", "g_PACC"), ("delta_hat", "g_ACC")):
        lo, hi = boot_sp(g, xc, yc, 300 if lb == "HAVUZ" else 1000)
        cors.append(dict(set=lb, x=xc, y=yc, rho=sp(g[xc].values, g[yc].values), lo=lo, hi=hi, n_bag=len(g), n_ind=g.ind.nunique()))
cors = pd.DataFrame(cors); cors.to_csv(T / "r06_D_spearman.csv", index=False)
# ---- 3. decision rule
GRID = np.r_[np.arange(0, 0.6, 0.005), np.inf]


def rule_ae(g, tau, xc, alt): return np.where(g[xc].values > tau, g[alt].values, g.AE_CC.values)


def pick_tau(g, xc, alt):
    ae = np.array([rule_ae(g, t, xc, alt).mean() for t in GRID]); return float(GRID[int(np.argmin(ae))])


S1sets = B[B.set.isin(["S1", "S1-2dk", "S1-APP120"])]
cfg = {"PACC": ("delta_hat", "AE_PACC"), "ACC": ("delta_hat", "AE_ACC"), "kappa-PACC": ("kappa", "AE_PACC")}
taus = {k: [] for k in cfg}; heldout = {k: [] for k in cfg}
for sp_i in range(50):
    r_ = np.random.default_rng(SEED + 600 + sp_i); perm = r_.permutation(CATS); h1, h2 = set(perm[:6]), set(perm[6:])
    A1_, A2_ = S1sets[S1sets.ind.isin(h1)], S1sets[S1sets.ind.isin(h2)]
    for k, (xc, alt) in cfg.items():
        t = pick_tau(A1_, xc, alt); taus[k].append(t)
        heldout[k].append(dict(rule=rule_ae(A2_, t, xc, alt).mean(), cc=A2_.AE_CC.mean(), alt=A2_[alt].mean(), oracle=np.minimum(A2_.AE_CC, A2_[alt]).mean()))
froz = {k: float(np.median(np.array(v))) for k, v in taus.items()}
tau_rows = {k: dict(tau_medyan=froz[k], tau_min=float(np.min(v)), tau_maks=float(np.max(v)), n_sonsuz=int(np.sum(~np.isfinite(v))),
                    heldout_rule=float(np.mean([h["rule"] for h in heldout[k]])), heldout_cc=float(np.mean([h["cc"] for h in heldout[k]])),
                    heldout_alt=float(np.mean([h["alt"] for h in heldout[k]])), heldout_oracle=float(np.mean([h["oracle"] for h in heldout[k]]))) for k, v in taus.items()}
ext = []
for lb in ["S1", "S1-2dk", "S1-APP120", "S2", "S3", "S4-Smit-kapali", "S4-Dunford-kapali"]:
    g = B[B.set == lb]
    for k, (xc, alt) in cfg.items():
        ae_r = rule_ae(g, froz[k], xc, alt); gg = g.assign(ae_r=ae_r, orc=np.minimum(g.AE_CC, g[alt]), dif=g.AE_CC - ae_r)
        s = S6.stat(gg.groupby("ind").dif.mean().values)
        row = dict(set=lb, kural=k, tau=froz[k], AE_kural=float(ae_r.mean()), AE_CC=float(g.AE_CC.mean()), AE_alt=float(g[alt].mean()), AE_oracle=float(gg.orc.mean()),
                   alt_orani=float(np.mean(g[xc].values > froz[k])), d=s["d"], lo=s["lo"], hi=s["hi"], p=s["p"], n_ind=s["n"])
        row["p_holm"] = row["p"]; row["aile_n"] = 1; row["kategori"] = S6.category(row["d"], row["lo"], row["hi"], row["p"]); ext.append(row)
ext = pd.DataFrame(ext); ext.to_csv(T / "r06_D_kural.csv", index=False)
d1.to_csv(T / "r06_D_delta.csv", index=False)
json.dump(dict(oran=oran, tau=tau_rows, froz=froz), open(T / "r06_D.json", "w"), indent=1, default=float)
# ---- 4. plot
fig, ax = plt.subplots(figsize=(7.2, 4.8)); cols = {"S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#2ca02c", "S4-Smit": "#d62728", "S4-Dunford": "#9467bd"}
for lb, g in B.groupby("set"):
    base = "S1" if lb.startswith("S1") else ("S4-Smit" if lb.startswith("S4-Smit") else ("S4-Dunford" if lb.startswith("S4-Dunford") else lb)); app = "APP" in lb
    ax.scatter(g.delta_hat, g.g_PACC, s=14 if app else 18, marker="^" if app else "o", color=cols.get(base, "k"), alpha=0.5, label=lb)
ax.axvline(froz["PACC"] if np.isfinite(froz["PACC"]) else 0.0, color="k", ls="--", lw=1); ax.axhline(0, color="gray", lw=0.6)
ax.set_xlabel("delta_hat = 1/2 ||p_CC - pi_kaynak||_1"); ax.set_ylabel("g_PACC = AE_CC - AE_PACC"); ax.legend(fontsize=6, ncol=2); ax.set_title("Kazanc-kayma (kesikli dikey: dondurulmus tau)")
fig.tight_layout(); fig.savefig(P.FIGURES / "r06_gain_vs_shift.png", dpi=150)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
print(d1); print(oran); print(cors[cors.set.isin(["HAVUZ", "S1", "S2", "S3"])].to_string()); print(json.dumps(tau_rows, indent=0, default=float)); print(ext.to_string())
