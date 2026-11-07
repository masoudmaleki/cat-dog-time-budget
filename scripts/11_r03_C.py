r"""11_r03_C.py — R03 C: calibrate the RF posteriors with the inner cat-grouped CV out-of-fold predictions (isotonic OvR + renormalisation; temperature scaling),
re-run PCC/PACC/EMQ for S1-collar, S1-harness and S2; ECE and Brier before/after.
Reads: r02/raw_va_f32.pkl (outer-model test posteriors), r03/inner_<cat>.pkl (inner-CV OOF posteriors; from 08). Writes: results/tables/r03_C_*.csv/json
Note: the calibrator is learned on the training cats' OOF posteriors; Cp and the EMQ prior are estimated from the calibrated OOF (slightly optimistic for isotonic, which is fitted on this same OOF).
"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from config import paths as P
from src import evalkit as E, quant as Q

K = E.K; T = P.TABLES
raw = pickle.load(open(P.INTERIM / "r02" / "raw_va_f32.pkl", "rb"))
LOCO = {f["cat"]: f for f in raw["loco"]}


def fit_iso(oof, y):
    return [IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(oof[:, k], (y == k).astype(float)) for k in range(K)]


def apply_iso(models, Pm):
    out = np.column_stack([models[k].predict(Pm[:, k]) for k in range(K)])
    s = out.sum(1, keepdims=True); bad = s[:, 0] <= 0
    out[bad] = 1.0 / K; s[bad] = 1.0
    return out / s


def fit_temp(oof, y):
    L = np.log(np.clip(oof, 1e-6, 1.0))
    def nll(t):
        z = L / t; z = z - z.max(1, keepdims=True); lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -lp[np.arange(len(y)), y].mean()
    return minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x


def apply_temp(t, Pm):
    z = np.log(np.clip(Pm, 1e-6, 1.0)) / t; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(Pm, y, nb=15):
    conf = Pm.max(1); ok = (Pm.argmax(1) == y).astype(float); b = np.minimum((conf * nb).astype(int), nb - 1); e = 0.0
    for i in range(nb):
        m = b == i
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def brier(Pm, y):
    oh = np.zeros_like(Pm); oh[np.arange(len(y)), y] = 1; return float(((Pm - oh) ** 2).sum(1).mean())


units, cal_rows, Tvals = [], [], []
for c in E.CATS:
    f = LOCO[c]; I = pickle.load(open(P.INTERIM / "r03" / f"inner_{c}.pkl", "rb"))
    tr = I["tr"].astype(int); ytr = E.Y[tr]; te = f["test_idx"]; yte = E.Y[te]
    for scen, Pm, oof, inner in (("S1-tasma", f["P_cc"], I["raw_collar_oof"], f["inner_collar"]),
                                 ("S1-kosum", f["P_hh"], I["raw_harness_oof"], f["inner_harness"]),
                                 ("S2", f["P_hc"], I["raw_harness_oof"], f["inner_harness"])):
        oof = oof.astype(np.float64)
        iso = fit_iso(oof, ytr); tmp = fit_temp(oof, ytr)
        cal = {"ham": (Pm, oof), "iso": (apply_iso(iso, Pm), apply_iso(iso, oof)), "T": (apply_temp(tmp, Pm), apply_temp(tmp, oof))}
        for kname, (Pc, oofc) in cal.items():
            cal_rows.append(dict(scen=scen, cat=c, kal=kname, ECE=ece(Pc, yte), Brier=brier(Pc, yte), T=tmp if kname == "T" else np.nan))
        ctx = {k: dict(Cp=Q.soft_conf_matrix(ytr, v[1], K), prior_oof=v[1].mean(0)) for k, v in cal.items()}
        def fn(m, Pm=Pm, cal=cal, ctx=ctx, inner=inner):
            out = {"CC": Q.cc(Pm[m].argmax(1), K), "ACC": Q.acc(Q.cc(Pm[m].argmax(1), K), inner["C"])}
            for k, (Pc, _) in cal.items():
                sfx = "" if k == "ham" else "_" + k
                pcc = Q.pcc(Pc[m]); out["PCC" + sfx] = pcc
                out["PACC" + sfx] = Q.pacc(pcc, ctx[k]["Cp"]); out["EMQ" + sfx] = Q.emq(Pc[m], ctx[k]["prior_oof"])[0]
            return out
        units.append(dict(scen=scen, cat=c, te=te, fn=fn))
cal = pd.DataFrame(cal_rows); cal.to_csv(T / "r03_C_calibration_by_cat.csv", index=False)
crow = []
for (sc, k), g in cal.groupby(["scen", "kal"]):
    cid = g.cat.map(E.CI).values; d = dict(scen=sc, kal=k)
    for col in ("ECE", "Brier"):
        m, lo, hi = E.boot_ci(g[col].values, cid); d.update({col: m, col + "_lo": lo, col + "_hi": hi})
    if k == "T": d["T_ort"] = float(g["T"].mean())
    crow.append(d)
pd.DataFrame(crow).to_csv(T / "r03_C_calibration_summary.csv", index=False)
out = {}
for bd in ("30dk_ge50", "kedi"):
    df = E.evaluate(units, bd); summ = E.summarize(df); summ.to_csv(T / f"r03_C_summary_{bd}.csv", index=False)
    o = {}
    for sc in ("S1-tasma", "S1-kosum", "S2"):
        for e in [x for x in df.est.unique() if x != "CC"]:
            for col in ("AE", "AE_L1"):
                o[f"{sc}|CC-{e}|{col}"] = dict(zip(("diff", "lo", "hi", "n"), E.paired(df, sc, "CC", e, col)))
    out[bd] = o
    if bd == "30dk_ge50":
        bi = {}
        for sc in ("S1-tasma", "S1-kosum", "S2"):
            for e in ("CC", "PACC", "PACC_iso", "PACC_T", "EMQ", "EMQ_iso", "EMQ_T"):
                for cn in ("Eating", "Grooming"):
                    bi[f"{sc}|{e}|{cn}"] = E.bias_pp(df, sc, e, E.D.L2.index(cn))
        out["bias_pp"] = bi
json.dump(out, open(T / "r03_C_summary.json", "w"), indent=1, default=float)
pd.set_option("display.width", 220)
print(pd.DataFrame(crow).round(4).to_string())
s = pd.read_csv(T / "r03_C_summary_30dk_ge50.csv"); print(s.pivot(index="est", columns="scen", values="AE").round(4).to_string())
