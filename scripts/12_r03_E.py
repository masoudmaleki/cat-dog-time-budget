r"""12_r03_E.py — R03 E: S2 axis mismatch. S2-raw (R02), S2-matched, S2-invariant; also S1-collar/S1-harness (32 and invariant set).
Reads: r02/raw_va_f32.pkl, r03/inner_<cat>.pkl (08). Writes: results/tables/r03_E_*.csv/json
"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score
from config import paths as P
from src import evalkit as E, quant as Q, smit_data as D, axismap as AM

K = E.K; T = P.TABLES
raw = pickle.load(open(P.INTERIM / "r02" / "raw_va_f32.pkl", "rb")); LOCO = {f["cat"]: f for f in raw["loco"]}


def inner_from_oof(oof, ytr, oofx=None):
    oof = oof.astype(np.float64)
    d = dict(C=Q.conf_matrix(ytr, oof.argmax(1), K), Cp=Q.soft_conf_matrix(ytr, oof, K), prior_oof=oof.mean(0))
    if oofx is not None:
        d["Cx"] = Q.conf_matrix(ytr, oofx.argmax(1), K); d["Cpx"] = Q.soft_conf_matrix(ytr, oofx.astype(np.float64), K)
    return d


def mk_fn(Pm, inner, oracle=False):
    def fn(m):
        e = E.base_estimates(Pm[m], inner)
        if oracle:
            cc = Q.cc(Pm[m].argmax(1), K); e["ACC_oracle"] = Q.acc(cc, inner["Cx"])
        return e
    return fn


units, clf = [], []
for c in E.CATS:
    f = LOCO[c]; I = pickle.load(open(P.INTERIM / "r03" / f"inner_{c}.pkl", "rb"))
    tr = I["tr"].astype(int); ytr = E.Y[tr]; te = f["test_idx"]; yte = E.Y[te]
    ih = f["inner_harness"]; ihx = dict(ih)   # R02: Cx/Cpx var
    S = {
        "S1-tasma": (f["P_cc"], f["inner_collar"], False),
        "S1-kosum": (f["P_hh"], ih, False),
        "S2-ham": (f["P_hc"], ih, True),
        "S2-esl": (I["map_P_hc"], inner_from_oof(I["map_oof"], ytr, I["map_oofx"]), True),
        "S1-kosum-esl": (I["map_P_hh"], inner_from_oof(I["map_oof"], ytr), False),
        "S1-tasma-deg": (I["inv_P_cc"], inner_from_oof(I["inv_collar_oof"], ytr), False),
        "S1-kosum-deg": (I["inv_P_hh"], inner_from_oof(I["inv_harness_oof"], ytr), False),
        "S2-deg": (I["inv_P_hc"], inner_from_oof(I["inv_harness_oof"], ytr, I["inv_harness_oofx"]), True),
    }
    for sc, (Pm, inner, orc) in S.items():
        units.append(dict(scen=sc, cat=c, te=te, fn=mk_fn(Pm, inner, orc)))
        pred = Pm.argmax(1); lbl = np.unique(yte)
        clf.append(dict(scen=sc, cat=c, macroF1=f1_score(yte, pred, labels=lbl, average="macro", zero_division=0),
                        balacc=balanced_accuracy_score(yte, pred), acc=float((pred == yte).mean())))
clf = pd.DataFrame(clf); clf.to_csv(T / "r03_E_classifier_by_cat.csv", index=False)
crow = []
for sc, g in clf.groupby("scen"):
    cid = g.cat.map(E.CI).values; d = dict(scen=sc)
    for col in ("macroF1", "balacc", "acc"):
        m, lo, hi = E.boot_ci(g[col].values, cid); d.update({col: m, col + "_lo": lo, col + "_hi": hi})
    crow.append(d)
pd.DataFrame(crow).to_csv(T / "r03_E_classifier_summary.csv", index=False)
out = {}
for bd in ("30dk_ge50", "kedi"):
    df = E.evaluate(units, bd); summ = E.summarize(df); summ.to_csv(T / f"r03_E_summary_{bd}.csv", index=False)
    o = {}
    for sc in ("S2-ham", "S2-esl", "S2-deg", "S1-tasma", "S1-tasma-deg", "S1-kosum", "S1-kosum-deg"):
        for e in [x for x in df[df.scen == sc].est.unique() if x != "CC"]:
            o[f"{sc}|CC-{e}"] = dict(zip(("diff", "lo", "hi", "n"), E.paired(df, sc, "CC", e)))
    # same estimator difference across scenarios (same bags): how much the mapping reduces the S2 error
    def cross(a, b, est, col="AE"):
        x = df[(df.scen == a) & (df.est == est)].set_index(["cat", "bag"])[col]; y = df[(df.scen == b) & (df.est == est)].set_index(["cat", "bag"])[col]
        j = pd.concat([x, y], axis=1, keys=["a", "b"]).dropna(); return E.boot_ci((j.a - j.b).values, j.index.get_level_values(0).map(E.CI).values) + (len(j),)
    for est in ("CC", "ACC", "PACC", "EMQ", "ACC_oracle"):
        o[f"S2-ham minus S2-esl|{est}"] = dict(zip(("diff", "lo", "hi", "n"), cross("S2-ham", "S2-esl", est)))
        o[f"S2-ham minus S2-deg|{est}"] = dict(zip(("diff", "lo", "hi", "n"), cross("S2-ham", "S2-deg", est)))
    for est in ("CC", "ACC", "PACC", "EMQ"):
        o[f"S1-tasma-deg minus S1-tasma|{est}"] = dict(zip(("diff", "lo", "hi", "n"), cross("S1-tasma-deg", "S1-tasma", est)))
        o[f"S1-kosum-deg minus S1-kosum|{est}"] = dict(zip(("diff", "lo", "hi", "n"), cross("S1-kosum-deg", "S1-kosum", est)))
    out[bd] = o
    if bd == "30dk_ge50":
        bi = {}
        for sc in ("S2-ham", "S2-esl", "S2-deg"):
            for e in ("CC", "ACC", "ACC_oracle"):
                for cn in ("Eating", "Grooming"):
                    bi[f"{sc}|{e}|{cn}"] = E.bias_pp(df, sc, e, D.L2.index(cn))
        out["bias_pp"] = bi
json.dump(out, open(T / "r03_E_summary.json", "w"), indent=1, default=float)
# ---- posture signature (before/after): second-level L2 labels (Lying, Sitting, Standing); mean X,Y,Z (g)
XC, XH = E.features(); XHm = AM.harness_to_collar(XH[D.FEAT32])
rows = []
for k, nm in ((0, "Lying"), (1, "Sitting"), (2, "Standing")):
    m = E.Y == k
    c = XC.loc[m, ["X_Mean", "Y_Mean", "Z_Mean"]].mean().values; h = XH.loc[m, ["X_Mean", "Y_Mean", "Z_Mean"]].mean().values; hm = XHm.loc[m, ["X_Mean", "Y_Mean", "Z_Mean"]].mean().values
    rows.append(dict(durus=nm, n=int(m.sum()), **{f"tasma_{a}": float(v) for a, v in zip("XYZ", c)}, **{f"kosum_ham_{a}": float(v) for a, v in zip("XYZ", h)},
                     **{f"kosum_esl_{a}": float(v) for a, v in zip("XYZ", hm)}, mesafe_ham=float(np.linalg.norm(c - h)), mesafe_esl=float(np.linalg.norm(c - hm)),
                     **{f"fark_ham_{a}": float(abs(c[i] - h[i])) for i, a in enumerate("XYZ")}, **{f"fark_esl_{a}": float(abs(c[i] - hm[i])) for i, a in enumerate("XYZ")}))
pd.DataFrame(rows).to_csv(T / "r03_E_posture_before_after.csv", index=False)
pd.DataFrame(AM.table_rows()).to_csv(T / "r03_E_axis_mapping_table.csv", index=False)
pd.set_option("display.width", 220)
s = pd.read_csv(T / "r03_E_summary_30dk_ge50.csv"); print(s.pivot(index="est", columns="scen", values="AE").round(4).to_string())
print(pd.DataFrame(crow).round(3).to_string()); print(pd.DataFrame(rows).round(3).T.to_string())
