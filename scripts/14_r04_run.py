r"""14_r04_run.py — R04: cross-fitted oracle (ACC-xfit, PACC-xfit), bag-length sweep, bias²/variance decomposition,
analytic prediction of the ACC variance, crossover criterion.
Reads: r02/raw_va_f32.pkl (S1-collar/S1-harness posteriors, training C), r03/inner_<cat>.pkl (S2-matched posteriors and inner-CV OOF). No new RF training.
Writes: <data root>/interim/r04/bags.pkl (bag-level intermediate), results/tables/r04_*.csv/json, results/figures/r04_*.png
Bags: 30-min blocks from the cat's first labelled second (block b: even = A, odd = B). Lengths 5/10/30/60 min: R01 windows (coverage >= 50%);
5/10/30-min bags lie inside a single block, 60-min bags span two adjacent blocks (A+B). 'yari' (half) = all seconds of the A blocks (or of the B blocks).
ACC-xfit: C from the blocks of the OPPOSITE type to the bag's block; for 60-min (A+B) bags, C from all blocks outside the bag's two blocks (deviation, noted in the report); for 'yari', C from the opposite half.
"""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from config import paths as P
from src import evalkit as E, quant as Q, smit_data as D

K = E.K; Y = E.Y; LAB = E.LAB; T = P.TABLES
OUT = P.INTERIM / "r04"; OUT.mkdir(parents=True, exist_ok=True)
raw = pickle.load(open(P.INTERIM / "r02" / "raw_va_f32.pkl", "rb")); LOCO = {f["cat"]: f for f in raw["loco"]}
SEC = LAB.sec.values.astype("datetime64[s]").astype(np.int64)
CATROW = E.CATROW
BLK = np.zeros(len(LAB), int)
for c in E.CATS:
    m = np.where(CATROW == c)[0]; BLK[m] = (SEC[m] - SEC[m].min()) // 1800
WIN = {s: D.make_windows(LAB, s, 0.5) for s in (5, 10, 30, 60)}
SIZES = ["5", "10", "30", "60", "yari"]
EST = ["CC", "PCC", "ACC", "PACC", "EMQ", "ACC-xfit", "PACC-xfit"]
t0 = time.time()


def inner_from_oof(oof, ytr):
    oof = oof.astype(np.float64)
    return dict(C=Q.conf_matrix(ytr, oof.argmax(1), K), Cp=Q.soft_conf_matrix(ytr, oof, K), prior_oof=oof.mean(0))


def make_bags(c, te):
    """[(size, bag_id, rows (global idx, sorted by time), xfit_rows)]"""
    mine = te[np.argsort(SEC[te], kind="stable")]; out = []
    blk = BLK[mine]
    for s in (5, 10, 30, 60):
        win, sel = WIN[s]
        for (cc_, w) in sel:
            if cc_ != c: continue
            mk = win[mine] == w; rows = mine[mk]
            if len(rows) == 0: continue
            bset = np.unique(BLK[rows])
            if s <= 30:
                assert len(bset) == 1
                xr = mine[(blk % 2) != (bset[0] % 2)]
            else:
                xr = mine[~np.isin(blk, bset)]
            out.append((str(s), int(w), rows, xr))
    for h, nm in ((0, 0), (1, 1)):
        rows = mine[blk % 2 == h]; xr = mine[blk % 2 != h]
        if len(rows) and len(xr): out.append(("yari", nm, rows, xr))
    return out


def analytic(C, q, rows_pred, n, ncls):
    """C^-1 (diag(q) - q q^T)/N_eff C^-T; trace = total class variance. rows_pred: predicted labels within the bag, sorted by time."""
    h = rows_pred
    rs, ws = [], []
    for k in range(ncls):
        x = (h == k).astype(float)
        if 0 < q[k] < 1 and len(x) > 2 and x.std() > 0 and x[:-1].std() > 0 and x[1:].std() > 0:
            rs.append(np.corrcoef(x[:-1], x[1:])[0, 1]); ws.append(q[k] * (1 - q[k]))
    rho = float(np.clip(np.average(rs, weights=ws), 0.0, 0.99)) if rs else 0.0
    neff = n * (1 - rho) / (1 + rho)
    cond = np.linalg.cond(C); use_pinv = bool(cond > 1e12)
    Ci = np.linalg.pinv(C) if use_pinv else np.linalg.inv(C)
    S = (np.diag(q) - np.outer(q, q)) / neff
    return float(np.trace(Ci @ S @ Ci.T)), rho, neff, use_pinv, float(cond)


rows = []
for c in E.CATS:
    f = LOCO[c]; I = pickle.load(open(P.INTERIM / "r03" / f"inner_{c}.pkl", "rb"))
    te = f["test_idx"]; ytr = Y[I["tr"].astype(int)]
    SC = {"S1-tasma": (f["P_cc"], f["inner_collar"]), "S1-kosum": (f["P_hh"], f["inner_harness"]),
          "S2-esl": (I["map_P_hc"], inner_from_oof(I["map_oof"], ytr))}
    pos = {int(g): i for i, g in enumerate(te)}   # global -> row within te
    bags = make_bags(c, te)
    for scen, (Pm, inner) in SC.items():
        pred_all = Pm.argmax(1)
        for size, bid, rws, xr in bags:
            ix = np.array([pos[int(g)] for g in rws]); xix = np.array([pos[int(g)] for g in xr])
            Pb = Pm[ix]; pb = pred_all[ix]; n = len(ix)
            p = np.bincount(Y[rws], minlength=K) / n
            cc = Q.cc(pb, K); pc = Q.pcc(Pb)
            Cx = Q.conf_matrix(Y[xr], pred_all[xix], K); Cpx = Q.soft_conf_matrix(Y[xr], Pm[xix], K)
            est = {"CC": cc, "PCC": pc, "ACC": Q.acc(cc, inner["C"]), "PACC": Q.pacc(pc, inner["Cp"]),
                   "EMQ": Q.emq(Pb, inner["prior_oof"])[0], "ACC-xfit": Q.acc(cc, Cx), "PACC-xfit": Q.pacc(pc, Cpx)}
            v_acc = analytic(inner["C"], cc, pb, n, K); v_x = analytic(Cx, cc, pb, n, K)
            for en, ph in est.items():
                e = ph - p
                rows.append(dict(scen=scen, size=size, est=en, cat=c, bag=bid, n=n, e=e, AE=float(np.abs(e).mean()), SE=float((e ** 2).sum()),
                                 vpred=v_acc[0] if en == "ACC" else (v_x[0] if en == "ACC-xfit" else np.nan),
                                 rho=v_acc[1], neff=v_acc[2], pinv=v_acc[3] if en == "ACC" else (v_x[3] if en == "ACC-xfit" else False)))
    print("tamam", c, "%.0f s" % (time.time() - t0), flush=True)
df = pd.DataFrame(rows); df["cid"] = df.cat.map(E.CI)
df.to_pickle(OUT / "bags.pkl")

# ---------------- 2. decomposition (within cat; equal-weight mean over cats; ddof=0 -> MSE = bias² + variance)
cell = []
for (sc, sz, en, c), g in df.groupby(["scen", "size", "est", "cat"]):
    if len(g) < 2: continue
    E_ = np.vstack(g.e.values)
    b2 = float((E_.mean(0) ** 2).sum()); v = float(E_.var(0, ddof=0).sum())
    cell.append(dict(scen=sc, size=sz, est=en, cat=c, cid=E.CI[c], n_bags=len(g), bias2=b2, var=v, mse=b2 + v, AE=float(g.AE.mean()), n_sec=float(g.n.mean())))
cell = pd.DataFrame(cell); cell.to_csv(T / "r04_decomp_by_cat.csv", index=False)
dec = []
for (sc, sz, en), g in cell.groupby(["scen", "size", "est"]):
    d = dict(scen=sc, size=sz, est=en, n_cats=len(g), n_bags=int(g.n_bags.sum()), min_ort=float(g.n_sec.mean() / 60))
    for col in ("bias2", "var", "mse", "AE"):
        m, lo, hi = E.boot_ci(g[col].values, g.cid.values); d.update({col: m, col + "_lo": lo, col + "_hi": hi})
    d["bias_payi"] = d["bias2"] / d["mse"]
    dec.append(d)
dec = pd.DataFrame(dec); dec["sz_o"] = dec["size"].map({s: i for i, s in enumerate(SIZES)}); dec["est_o"] = dec.est.map({e: i for i, e in enumerate(EST)})
dec = dec.sort_values(["scen", "sz_o", "est_o"]).drop(columns=["sz_o", "est_o"]); dec.to_csv(T / "r04_decomposition.csv", index=False)

# ---------------- 1/4. paired AE differences (CC − X; positive = X better) and the rate at which ACC-xfit beats CC
pv = df.pivot_table(index=["scen", "size", "cat", "bag"], columns="est", values="AE")
pv_se = df.pivot_table(index=["scen", "size", "cat", "bag"], columns="est", values="SE")
pw = []
for (sc, sz), g in pv.groupby(level=[0, 1]):
    cid = g.index.get_level_values(2).map(E.CI).values
    for x in ("ACC", "ACC-xfit", "PACC", "PACC-xfit", "PCC", "EMQ"):
        d = dict(scen=sc, size=sz, est=x, n_bags=len(g))
        m, lo, hi = E.boot_ci((g["CC"] - g[x]).values, cid); d.update(dAE=m, dAE_lo=lo, dAE_hi=hi)
        s2 = pv_se.loc[g.index]; m, lo, hi = E.boot_ci((s2["CC"] - s2[x]).values, cid); d.update(dSE=m, dSE_lo=lo, dSE_hi=hi)
        d["p_win_AE"] = float((g["CC"] > g[x]).mean())
        pw.append(d)
pw = pd.DataFrame(pw); pw.to_csv(T / "r04_paired_vs_CC.csv", index=False)

# ---------------- 3. analytic vs observed variance (ACC and ACC-xfit), cell = scenario × length × cat (>=2 bags); observed: ddof=1
vp = []
for en in ("ACC", "ACC-xfit"):
    for (sc, sz, c), g in df[df.est == en].groupby(["scen", "size", "cat"]):
        if len(g) < 3: continue
        E_ = np.vstack(g.e.values)
        vp.append(dict(est=en, scen=sc, size=sz, cat=c, n_bags=len(g), obs=float(E_.var(0, ddof=1).sum()), pred=float(g.vpred.mean()), pinv_frac=float(g.pinv.mean()),
                       rho=float(g.rho.mean()), neff_ort=float(g.neff.mean()), n_ort=float(g.n.mean())))
vp = pd.DataFrame(vp); vp["oran"] = vp.obs / vp.pred; vp.to_csv(T / "r04_var_pred_cells.csv", index=False)
vs = []
for (en, sc, sz), g in vp.groupby(["est", "scen", "size"]):
    r = spearmanr(g.obs, g.pred) if len(g) >= 3 else None
    vs.append(dict(est=en, scen=sc, size=sz, n_cells=len(g), obs_ort=g.obs.mean(), pred_ort=g.pred.mean(), oran_ortalama=g.obs.mean() / g.pred.mean(),
                   oran_geo=float(np.exp(np.log(g.oran.clip(lower=1e-12)).mean())), spearman=float(r.statistic) if r is not None else np.nan,
                   p=float(r.pvalue) if r is not None else np.nan, pinv_frac=g.pinv_frac.mean(), rho_ort=g.rho.mean()))
for en, g in vp.groupby("est"):
    r = spearmanr(g.obs, g.pred); vs.append(dict(est=en, scen="hepsi", size="hepsi", n_cells=len(g), obs_ort=g.obs.mean(), pred_ort=g.pred.mean(), oran_ortalama=g.obs.mean() / g.pred.mean(),
                                                 oran_geo=float(np.exp(np.log(g.oran.clip(lower=1e-12)).mean())), spearman=float(r.statistic), p=float(r.pvalue),
                                                 pinv_frac=g.pinv_frac.mean(), rho_ort=g.rho.mean()))
vs = pd.DataFrame(vs); vs.to_csv(T / "r04_var_pred_summary.csv", index=False)

# ---------------- 4. crossover criterion: ACC wins <=> bias²_CC(cat, scenario, length) > variance_ACC(analytic, bag)
crit = []
b2cc = cell[cell.est == "CC"].set_index(["scen", "size", "cat"]).bias2
for en in ("ACC", "ACC-xfit"):
    g = df[df.est == en].copy()
    cc_ = df[df.est == "CC"].set_index(["scen", "size", "cat", "bag"])
    g = g.set_index(["scen", "size", "cat", "bag"])
    g["AE_CC"] = cc_.AE; g["SE_CC"] = cc_.SE
    g = g.reset_index(); g["b2cc"] = [b2cc.get((r.scen, r.size, r.cat), np.nan) for r in g.itertuples()]
    g = g.dropna(subset=["b2cc"])
    g["pred_win"] = g.b2cc > g.vpred; g["win_AE"] = g.AE_CC > g.AE; g["win_SE"] = g.SE_CC > g.SE
    for key, gg in list(g.groupby(["scen", "size"])) + [(("hepsi", "hepsi"), g)]:
        crit.append(dict(est=en, scen=key[0], size=key[1], n_bags=len(gg), p_win_AE=gg.win_AE.mean(), isabet_AE=float((gg.pred_win == gg.win_AE).mean()),
                         taban_AE=float(max(gg.win_AE.mean(), 1 - gg.win_AE.mean())), p_win_SE=gg.win_SE.mean(), isabet_SE=float((gg.pred_win == gg.win_SE).mean()),
                         pred_win_orani=float(gg.pred_win.mean())))
pd.DataFrame(crit).to_csv(T / "r04_criterion.csv", index=False)
json.dump(dict(sure_s=time.time() - t0, n_bag_satiri=len(df), n_bag=int(df[(df.est == "CC") & (df.scen == "S1-tasma")].shape[0])), open(T / "r04_run_info.json", "w"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(dec[["scen", "size", "est", "n_cats", "n_bags", "bias2", "var", "mse", "AE"]].round(5).to_string())
print(vs.round(3).to_string()); print(pd.DataFrame(crit).round(3).to_string())
