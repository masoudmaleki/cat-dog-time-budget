"""R03 shared evaluation: bags, estimators, cat-level bootstrap. Same definitions as 05_analyze_r02.py in R02.
Reads: smit_anno/accel (via D.load). Writes nothing.
"""
import numpy as np, pandas as pd
from config import paths as P
from src import smit_data as D, quant as Q

K = len(D.L2); SEED = P.SEED
LAB, _XC, _XH = D.load()
Y = LAB.y2.values; CATROW = LAB.Cat_id.values
CATS = sorted(LAB.Cat_id.unique()); CI = {c: i for i, c in enumerate(CATS)}
M13 = np.zeros((3, K))
for j, c in enumerate(D.L2):
    M13[D.L1.index(D.L2_TO_L1[c]), j] = 1
BAGDEFS = {"30dk_ge50": (30, 0.5), "30dk_ge90": (30, 0.9), "10dk_ge90": (10, 0.9), "kedi": None}
WIN = {k: D.make_windows(LAB, *v) for k, v in BAGDEFS.items() if v}
BOOT = np.random.default_rng(SEED).integers(0, len(CATS), size=(1000, len(CATS)))   # same as R02


def features():
    return _XC, _XH


def boot_ci(vals, cat_ids):
    vals = np.asarray(vals, float)
    S = np.bincount(cat_ids, weights=vals, minlength=len(CATS)); N = np.bincount(cat_ids, minlength=len(CATS)).astype(float)
    m = S.sum() / max(N.sum(), 1); n = N[BOOT].sum(1); ok = n > 0
    b = S[BOOT].sum(1)[ok] / n[ok]
    return float(m), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def groups(te, cat, bagdef):
    """te: global row indices. Returns [(cat, bag_no, mask)]."""
    cte = CATROW[te]; out = []
    cl = CATS if cat is None else [cat]
    if BAGDEFS[bagdef] is None:
        for c in cl:
            m = cte == c
            if m.any(): out.append((c, 0, m))
        return out
    win, sel = WIN[bagdef]; wte = win[te]; ss = set(sel)
    for c in cl:
        mc = cte == c
        for w in np.unique(wte[mc]):
            if w >= 0 and (c, int(w)) in ss:
                out.append((c, int(w), mc & (wte == w)))
    return out


def base_estimates(P_, inner):
    """CC, PCC, ACC, PACC, EMQ (R02 definition: EMQ prior = mean of the inner-CV posteriors)."""
    pred = P_.argmax(1); cc = Q.cc(pred, K); pc = Q.pcc(P_)
    return {"CC": cc, "PCC": pc, "ACC": Q.acc(cc, inner["C"]), "PACC": Q.pacc(pc, inner["Cp"]), "EMQ": Q.emq(P_, inner["prior_oof"])[0]}


def evaluate(units, bagdef="30dk_ge50"):
    """units: dict(scen, cat (None=all), te, fn(mask)->{est: p_hat(K)}). Returns the bag rows (DataFrame)."""
    rows = []
    for u in units:
        te = u["te"]; yte = Y[te]
        for c, w, m in groups(te, u["cat"], bagdef):
            n = int(m.sum())
            if n == 0: continue
            p = np.bincount(yte[m], minlength=K) / n
            for en, ph in u["fn"](m).items():
                p1, h1 = M13 @ p, M13 @ ph
                rows.append(dict(scen=u["scen"], est=en, cat=c, bag=w, n=n, AE=Q.ae(ph, p), RAE=Q.rae(ph, p, n),
                                 AE_L1=Q.ae(h1, p1), RAE_L1=Q.rae(h1, p1, n), p=p, ph=ph))
    df = pd.DataFrame(rows); df["cid"] = df.cat.map(CI)
    return df


def summarize(df, cols=("AE", "AE_L1", "RAE")):
    out = []
    for (sc, en), g in df.groupby(["scen", "est"]):
        d = dict(scen=sc, est=en, n_bags=len(g))
        for c in cols:
            m, lo, hi = boot_ci(g[c].values, g.cid.values); d.update({c: m, c + "_lo": lo, c + "_hi": hi})
        out.append(d)
    return pd.DataFrame(out)


def paired(df, scen, a, b, col="AE"):
    """mean(col_a − col_b) over paired bags, cat bootstrap CI. (a=CC, b=X) -> positive = X better."""
    x = df[df.scen == scen]
    xa = x[x.est == a].set_index(["cat", "bag"])[col]; xb = x[x.est == b].set_index(["cat", "bag"])[col]
    j = pd.concat([xa, xb], axis=1, keys=["a", "b"]).dropna()
    cid = j.index.get_level_values(0).map(CI).values
    return boot_ci((j.a - j.b).values, cid) + (len(j),)


def bias_pp(df, scen, est, cls_idx):
    g = df[(df.scen == scen) & (df.est == est)]
    v = np.array([(r.ph[cls_idx] - r.p[cls_idx]) * 100 for r in g.itertuples()])
    return boot_ci(v, g.cid.values)
