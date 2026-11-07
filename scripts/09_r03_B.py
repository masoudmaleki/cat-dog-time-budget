r"""09_r03_B.py — R03 B: ACC-oracle (target cat's own C), C distance, condition number, gain–distance Spearman. Steps: S1-collar, S1-harness.
Reads: <data root>/interim/r02/raw_va_f32.pkl (posteriors, training C). Writes: results/tables/r03_B_*.csv/json
"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from config import paths as P
from src import evalkit as E, quant as Q, smit_data as D

K = E.K
res = pickle.load(open(P.INTERIM / "r02" / "raw_va_f32.pkl", "rb"))
T = P.TABLES
units, percat, pair_acc = [], [], {}
for f in res["loco"]:
    c, te = f["cat"], f["test_idx"]; yte = E.Y[te]
    for scen, Pm, inner in (("S1-tasma", f["P_cc"], f["inner_collar"]), ("S1-kosum", f["P_hh"], f["inner_harness"])):
        Ct = Q.conf_matrix(yte, Pm.argmax(1), K); Cs = inner["C"]
        present = np.bincount(yte, minlength=K) > 0
        d = (Cs - Ct)[:, present]
        tv = 0.5 * np.abs(Cs - Ct).sum(0)
        jm = int(np.argmax(np.where(present, tv, -1)))
        for j in np.where(present)[0]:
            for i in range(K):
                pair_acc.setdefault((scen, D.L2[j], D.L2[i]), []).append(abs(Cs[i, j] - Ct[i, j]))
        percat.append(dict(scen=scen, cat=c, fro=float(np.linalg.norm(d)), max_col_tv=float(tv[jm]), max_col_class=D.L2[jm],
                           cond_train=float(np.linalg.cond(Cs)), cond_target=float(np.linalg.cond(Ct)),
                           acc_train=float((Pm.argmax(1) == yte).mean()), n_present=int(present.sum())))
        units.append(dict(scen=scen, cat=c, te=te,
                          fn=(lambda Pm=Pm, inner=inner, Ct=Ct, te=te: (lambda m: {**{k: v for k, v in E.base_estimates(Pm[m], inner).items() if k in ("CC", "ACC")},
                                                                                       "ACC_oracle": Q.acc(Q.cc(Pm[m].argmax(1), K), Ct)}))()))
pc = pd.DataFrame(percat)
out = {}
for bd in ("30dk_ge50", "kedi"):
    df = E.evaluate(units, bd)
    summ = E.summarize(df); summ.to_csv(T / f"r03_B_summary_{bd}.csv", index=False)
    o = {}
    for sc in ("S1-tasma", "S1-kosum"):
        for e in ("ACC", "ACC_oracle"):
            for col in ("AE", "AE_L1"):
                o[f"{sc}|CC-{e}|{col}"] = dict(zip(("diff", "lo", "hi", "n"), E.paired(df, sc, "CC", e, col)))
    out[bd] = o
    if bd == "30dk_ge50":
        # gain and Spearman per cat
        g = df.pivot_table(index=["scen", "cat"], columns="est", values="AE", aggfunc="mean")
        g["gain_ACC"] = g.CC - g.ACC; g["gain_ORACLE"] = g.CC - g.ACC_oracle
        pc = pc.merge(g.reset_index()[["scen", "cat", "gain_ACC", "gain_ORACLE", "CC", "ACC", "ACC_oracle"]], on=["scen", "cat"])
        sp = {}
        for sc in ("S1-tasma", "S1-kosum", "havuz"):
            x = pc if sc == "havuz" else pc[pc.scen == sc]
            for col in ("fro", "max_col_tv", "cond_train"):
                r = spearmanr(x[col], x.gain_ACC); sp[f"{sc}|{col}|gain_ACC"] = [float(r.statistic), float(r.pvalue), len(x)]
        out["spearman_gain_vs_dist"] = sp
        # Eating/Grooming bias: CC, ACC, ACC_oracle
        bi = {}
        for sc in ("S1-tasma", "S1-kosum"):
            for e in ("CC", "ACC", "ACC_oracle"):
                for cn in ("Eating", "Grooming"):
                    bi[f"{sc}|{e}|{cn}"] = E.bias_pp(df, sc, e, D.L2.index(cn))
        out["bias_pp"] = bi
pc.to_csv(T / "r03_B_per_cat.csv", index=False)
pairs = pd.DataFrame([dict(scen=k[0], true_class=k[1], pred_class=k[2], mean_abs_dC=float(np.mean(v)), n_cats=len(v)) for k, v in pair_acc.items()])
pairs = pairs.sort_values("mean_abs_dC", ascending=False); pairs.to_csv(T / "r03_B_pairs.csv", index=False)
json.dump(out, open(T / "r03_B_summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(E.summarize(E.evaluate(units, "30dk_ge50"))[["scen", "est", "n_bags", "AE", "AE_lo", "AE_hi", "AE_L1"]].round(4).to_string())
print(json.dumps(out["30dk_ge50"], indent=0)[:3000]); print(json.dumps(out["spearman_gain_vs_dist"], indent=0))
print(pc.round(3).to_string()); print(pairs.head(12).round(3).to_string())
