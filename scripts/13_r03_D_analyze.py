r"""13_r03_D_analyze.py — R03 D: label-budget curve (H4). Arms: CC, T-ACC, T-ACC-shrinkage, retrain, target-only. Steps: S1-collar, S1-harness; only the T-ACC arms for S2-matched.
Reads: r02/raw_va_f32.pkl, r03/D_<placement>_<cat>.pkl (10), r03/inner_<cat>.pkl (08). Writes: results/tables/r03_D_*.csv, results/figures/r03_label_budget.png
"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import paths as P
from src import evalkit as E, quant as Q, smit_data as D

K = E.K; T = P.TABLES; MS = (60, 300, 900, 1800)
raw = pickle.load(open(P.INTERIM / "r02" / "raw_va_f32.pkl", "rb")); LOCO = {f["cat"]: f for f in raw["loco"]}
ARMS_S1 = ["CC", "T-ACC", "T-ACC-buzme", "Yeniden-egitim", "Yalniz-hedef"]
rows = []; skipped = dict(target_m60=0, target_missing=0, lab_none=0, runs=0)


def hist(pred_hard, m):
    return np.bincount(pred_hard[m], minlength=K) / max(m.sum(), 1)


def run_unit(scen, c, m_, rep, te, rem_mask, lab_pos, Pm, Ctrain, extra):
    """te: all test indices (sorted); rem_mask: unlabelled rows. Returns the evaluate rows (DataFrame)."""
    rem = te[rem_mask]
    pred_all = Pm.argmax(1); pred_rem = pred_all[rem_mask]
    Ct = Q.conf_matrix(E.Y[te][lab_pos], pred_all[lab_pos], K)
    present = np.bincount(E.Y[te][lab_pos], minlength=K) > 0
    lam = m_ / (m_ + 300.0); Cb = Ctrain.copy(); Cb[:, present] = lam * Ct[:, present] + (1 - lam) * Ctrain[:, present]
    def fn(mask):
        cc = np.bincount(pred_rem[mask], minlength=K) / mask.sum()
        o = {"CC": cc, "T-ACC": Q.acc(cc, Ct), "T-ACC-buzme": Q.acc(cc, Cb)}
        for k, arr in extra.items():
            if arr is not None: o[k] = hist(arr, mask)
        return o
    df = E.evaluate([dict(scen=scen, cat=c, te=rem, fn=fn)], "30dk_ge50")
    df["m"] = m_; df["rep"] = rep
    return df


for pos, scen in (("Collar", "S1-tasma"), ("Harness", "S1-kosum")):
    for c in E.CATS:
        d = pickle.load(open(P.INTERIM / "r03" / f"D_{pos}_{c}.pkl", "rb")); f = LOCO[c]; te = f["test_idx"]
        assert np.array_equal(te, d["te"])
        Pm = f["P_cc"] if pos == "Collar" else f["P_hh"]; Ctr = (f["inner_collar"] if pos == "Collar" else f["inner_harness"])["C"]
        for (m_, rep), r in d["runs"].items():
            skipped["runs"] += 1
            if r is None: skipped["lab_none"] += 1; continue
            lab_pos = np.searchsorted(te, r["lab"].astype(int)); rem_mask = np.ones(len(te), bool); rem_mask[lab_pos] = False
            if r["pred_target"] is None: skipped["target_m60"] += 1
            if r["target_missing_classes"]: skipped["target_missing"] += 1
            rows.append(run_unit(scen, c, m_, rep, te, rem_mask, lab_pos, Pm, Ctr,
                                 {"Yeniden-egitim": r["pred_retrain"], "Yalniz-hedef": r["pred_target"]}))
    print("tamam", pos, flush=True)
# S2-matched: only the T-ACC arms (collar target blocks; matched harness model)
for c in E.CATS:
    d = pickle.load(open(P.INTERIM / "r03" / f"D_Collar_{c}.pkl", "rb")); I = pickle.load(open(P.INTERIM / "r03" / f"inner_{c}.pkl", "rb")); te = LOCO[c]["test_idx"]
    ytr = E.Y[I["tr"].astype(int)]; Ctr = Q.conf_matrix(ytr, I["map_oof"].astype(np.float64).argmax(1), K)
    for (m_, rep), r in d["runs"].items():
        if r is None: continue
        lab_pos = np.searchsorted(te, r["lab"].astype(int)); rem_mask = np.ones(len(te), bool); rem_mask[lab_pos] = False
        rows.append(run_unit("S2-esl", c, m_, rep, te, rem_mask, lab_pos, I["map_P_hc"], Ctr, {}))
df = pd.concat(rows, ignore_index=True)
df.drop(columns=["p", "ph"]).to_pickle(P.INTERIM / "r03" / "D_rows.pkl")
# mean over repeats (cat, bag, scen, m, est)
g = df.groupby(["scen", "m", "est", "cat", "bag"], as_index=False).agg(AE=("AE", "mean"), AE_L1=("AE_L1", "mean"), n=("n", "mean"))
g["cid"] = g.cat.map(E.CI)
out = []
for (sc, m_, en), x in g.groupby(["scen", "m", "est"]):
    a = E.boot_ci(x.AE.values, x.cid.values); b = E.boot_ci(x.AE_L1.values, x.cid.values)
    cc = g[(g.scen == sc) & (g.m == m_) & (g.est == "CC")].set_index(["cat", "bag"])
    xx = x.set_index(["cat", "bag"]); j = pd.concat([cc.AE, xx.AE], axis=1, keys=["cc", "x"]).dropna()
    dl = E.boot_ci((j.cc - j.x).values, j.index.get_level_values(0).map(E.CI).values) if en != "CC" else (0.0, 0.0, 0.0)
    j1 = pd.concat([cc.AE_L1, xx.AE_L1], axis=1, keys=["cc", "x"]).dropna()
    dl1 = E.boot_ci((j1.cc - j1.x).values, j1.index.get_level_values(0).map(E.CI).values) if en != "CC" else (0.0, 0.0, 0.0)
    out.append(dict(scen=sc, m=m_, est=en, n_bags=len(x), n_cats=x.cat.nunique(), AE=a[0], AE_lo=a[1], AE_hi=a[2], AE_L1=b[0], AE_L1_lo=b[1], AE_L1_hi=b[2],
                    gain=dl[0], gain_lo=dl[1], gain_hi=dl[2], gain_L1=dl1[0], gain_L1_lo=dl1[1], gain_L1_hi=dl1[2]))
res = pd.DataFrame(out); res.to_csv(T / "r03_D_budget_curve.csv", index=False)
json.dump(skipped, open(T / "r03_D_skips.json", "w"), indent=1)
# ---- bias: Eating/Grooming for the best arm (by AE) at m=1800
bias = []
for sc in ("S1-tasma", "S1-kosum"):
    r1800 = res[(res.scen == sc) & (res.m == 1800)].sort_values("AE")
    best = r1800.iloc[0].est
    for en in ("CC", best):
        x = df[(df.scen == sc) & (df.m == 1800) & (df.est == en)]
        for cn in ("Eating", "Grooming"):
            k = D.L2.index(cn); x2 = x.assign(b=[(r.ph[k] - r.p[k]) * 100 for r in x.itertuples()]).groupby(["cat", "bag"], as_index=False).b.mean()
            m, lo, hi = E.boot_ci(x2.b.values, x2.cat.map(E.CI).values); bias.append(dict(scen=sc, m=1800, est=en, cls=cn, bias_pp=m, lo=lo, hi=hi))
pd.DataFrame(bias).to_csv(T / "r03_D_bias_best.csv", index=False)
# ---- figure
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
cols = {"CC": "#444444", "T-ACC": "#1f77b4", "T-ACC-buzme": "#2ca02c", "Yeniden-egitim": "#d62728", "Yalniz-hedef": "#ff7f0e"}
for ax, sc, ttl in zip(axes, ("S1-tasma", "S1-kosum", "S2-esl"), ("S1-tasma", "S1-koşum", "S2-eşlenmiş (yalnız T-ACC kolları)")):
    for en, col in cols.items():
        x = res[(res.scen == sc) & (res.est == en)].sort_values("m")
        if x.empty: continue
        ax.plot(x.m, x.AE, "-o", color=col, label=en, ms=4); ax.fill_between(x.m, x.AE_lo, x.AE_hi, color=col, alpha=0.12)
    ax.set_xscale("log"); ax.set_xticks(MS); ax.set_xticklabels([str(v) for v in MS]); ax.set_xlabel("etiketli hedef saniye (m)"); ax.set_ylabel("AE (L2)"); ax.set_title(ttl)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(P.FIGURES / "r03_label_budget.png", dpi=130)
pd.set_option("display.width", 220)
for sc in ("S1-tasma", "S1-kosum", "S2-esl"):
    print(sc); print(res[res.scen == sc].pivot(index="est", columns="m", values="AE").round(4).to_string())
print(skipped)
