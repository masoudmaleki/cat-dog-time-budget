r"""05_analyze_r02.py — R02 analysis: estimators from the posteriors, bag metrics, cat-level bootstrap, H1-H3, oracle.
Reads: <data root>/interim/r02/raw_<tag>.pkl (output of 04), smit_anno/accel (for the label table)
Writes: results/tables/r02_*.csv, r02_hypotheses.json
Primary: tag va_f32, 30-min bags with coverage>=50%, L2. Sensitivity: other bags, vb_f32 (7000 cap), va_f36 (36 features).
"""
import sys, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, balanced_accuracy_score
from config import paths as P
from src import smit_data as D, quant as Q

K = len(D.L2); SEED = P.SEED
T = P.TABLES
lab, _, _ = D.load()   # features are not needed but load is the only entry point; memory is small
y_all = lab.y2.values; cats_row = lab.Cat_id.values
CATS = sorted(lab.Cat_id.unique()); CI = {c: i for i, c in enumerate(CATS)}
M13 = np.zeros((3, K));
for j, c in enumerate(D.L2): M13[D.L1.index(D.L2_TO_L1[c]), j] = 1
BAGDEFS = {"30dk_ge50": (30, 0.5), "30dk_ge90": (30, 0.9), "10dk_ge90": (10, 0.9), "kedi": None}
WIN = {}
for name, d in BAGDEFS.items():
    if d: WIN[name] = D.make_windows(lab, *d)
BOOT = np.random.default_rng(SEED).integers(0, len(CATS), size=(1000, len(CATS)))   # common cat resampling


def boot_ci(vals, cat_ids, stat="mean"):
    """vals: bag-level values, cat_ids: 0..11. Cat-level bootstrap: ratio sum/n. Returns (mean, lo, hi)."""
    vals = np.asarray(vals, float); S = np.bincount(cat_ids, weights=vals, minlength=len(CATS)); N = np.bincount(cat_ids, minlength=len(CATS)).astype(float)
    m = S.sum() / max(N.sum(), 1)
    n = N[BOOT].sum(1); ok = n > 0
    b = S[BOOT].sum(1)[ok] / n[ok]
    return m, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_dist(vals, cat_ids):
    S = np.bincount(cat_ids, weights=np.asarray(vals, float), minlength=len(CATS)); N = np.bincount(cat_ids, minlength=len(CATS)).astype(float)
    n = N[BOOT].sum(1); return S[BOOT].sum(1) / np.where(n > 0, n, np.nan)


def estimates(P_, inner, scen):
    out = {}
    pred = P_.argmax(1)
    cc = Q.cc(pred, K); pc = Q.pcc(P_)
    out["CC"] = cc; out["PCC"] = pc
    out["ACC"] = Q.acc(cc, inner["C"]); out["PACC"] = Q.pacc(pc, inner["Cp"])
    out["EMQ"] = Q.emq(P_, inner["prior_oof"])[0]
    out["EMQ_prtrain"] = Q.emq(P_, inner["prior_train"])[0]
    if scen == "S2":
        out["ACC_oracle"] = Q.acc(cc, inner["Cx"]); out["PACC_oracle"] = Q.pacc(pc, inner["Cpx"])
    return out


def units(res):
    """Scenario units: (scen, cat_or_None, test_idx, P, inner)."""
    U = []
    if res["s0"] is not None:
        U.append(("S0", None, res["s0"]["test_idx"], res["s0"]["P_cc"], res["s0"]["inner_collar"]))
    for f in res["loco"]:
        U.append(("S1-tasma", f["cat"], f["test_idx"], f["P_cc"], f["inner_collar"]))
        U.append(("S1-kosum", f["cat"], f["test_idx"], f["P_hh"], f["inner_harness"]))
        U.append(("S2", f["cat"], f["test_idx"], f["P_hc"], f["inner_harness"]))
    return U


def analyze(tag):
    res = pickle.load(open(P.INTERIM / "r02" / f"raw_{tag}.pkl", "rb"))
    bag_rows, cls_rows, clf_rows = [], [], []
    for scen, cat, te, P_, inner in units(res):
        yte = y_all[te]; cte = cats_row[te]; pred = P_.argmax(1)
        # classifier metrics per cat
        for c in (CATS if cat is None else [cat]):
            m = cte == c
            if m.sum() == 0: continue
            yy, pp = yte[m], pred[m]
            lbl = np.unique(yy)
            clf_rows.append(dict(tag=tag, scen=scen, cat=c, n=int(m.sum()), macroF1=f1_score(yy, pp, labels=lbl, average="macro", zero_division=0),
                                 balacc=balanced_accuracy_score(yy, pp),
                                 macroF1_L1=f1_score(M13.argmax(0)[yy], M13.argmax(0)[pp], average="macro", zero_division=0),
                                 balacc_L1=balanced_accuracy_score(M13.argmax(0)[yy], M13.argmax(0)[pp])))
        for bd, d in BAGDEFS.items():
            groups = []   # (cat, bag_no, mask)
            if d is None:
                for c in (CATS if cat is None else [cat]):
                    m = cte == c
                    if m.any(): groups.append((c, 0, m))
            else:
                win, sel = WIN[bd]; wte = win[te]; selset = set(sel)
                for c in (CATS if cat is None else [cat]):
                    mc = cte == c
                    for w in np.unique(wte[mc]):
                        if w >= 0 and (c, int(w)) in selset:
                            groups.append((c, int(w), mc & (wte == w)))
            for c, w, m in groups:
                n = int(m.sum()); p = np.bincount(yte[m], minlength=K) / n
                est = estimates(P_[m], inner, scen)
                for en, ph in est.items():
                    for lvl, (pp_, hh_) in {"L2": (p, ph), "L1": (M13 @ p, M13 @ ph)}.items():
                        bag_rows.append(dict(tag=tag, bagdef=bd, level=lvl, scen=scen, est=en, cat=c, bag=w, n=n,
                                             AE=Q.ae(hh_, pp_), RAE=Q.rae(hh_, pp_, n)))
                        names = D.L2 if lvl == "L2" else D.L1
                        for k, nm in enumerate(names):
                            cls_rows.append((tag, bd, lvl, scen, en, c, w, nm, float(pp_[k]), float(hh_[k])))
    bag = pd.DataFrame(bag_rows)
    cls = pd.DataFrame(cls_rows, columns=["tag", "bagdef", "level", "scen", "est", "cat", "bag", "class", "p", "p_hat"])
    clf = pd.DataFrame(clf_rows)
    return bag, cls, clf, res


def main():
    tags = [t for t in ("va_f32", "vb_f32", "va_f36") if (P.INTERIM / "r02" / f"raw_{t}.pkl").exists()]
    BAG, CLS, CLF, RES = [], [], [], {}
    for t in tags:
        b, c, f, r = analyze(t); BAG.append(b); CLS.append(c); CLF.append(f); RES[t] = r
        print("analiz tamam", t, len(b))
    bag = pd.concat(BAG); cls = pd.concat(CLS); clf = pd.concat(CLF)
    bag["cid"] = bag.cat.map(CI)
    # ---- bag-level CSV (requested format)
    out = cls.rename(columns={"scen": "basamak", "tag": "varyant", "est": "tahminci", "cat": "kedi", "bag": "torba", "class": "sinif"})
    out[["p", "p_hat"]] = out[["p", "p_hat"]].round(5)
    out.to_csv(T / "r02_bag_level.csv", index=False)
    clf.to_csv(T / "r02_classifier_metrics_by_cat.csv", index=False)
    # ---- summary table
    rows = []
    for (tag, bd, lvl, scen, est), g in bag.groupby(["tag", "bagdef", "level", "scen", "est"]):
        a = boot_ci(g.AE, g.cid.values); r = boot_ci(g.RAE, g.cid.values)
        rows.append(dict(tag=tag, bagdef=bd, level=lvl, scen=scen, est=est, n_bags=len(g), n_cats=g.cat.nunique(), AE=a[0], AE_lo=a[1], AE_hi=a[2], RAE=r[0], RAE_lo=r[1], RAE_hi=r[2]))
    summ = pd.DataFrame(rows); summ.to_csv(T / "r02_summary_all.csv", index=False)
    # ---- true budget variability (S1-collar bags, via the CC row)
    v = cls[(cls.tag == "va_f32") & (cls.scen == "S1-tasma") & (cls.est == "CC")]
    var = v.groupby(["bagdef", "level", "class"]).p.agg(n_bags="count", mean="mean", sd="std", min="min", max="max").reset_index()
    var.to_csv(T / "r02_true_budget_variability.csv", index=False)
    # ---- signed bias per class
    cls["bias_pp"] = (cls.p_hat - cls.p) * 100
    cls["cid"] = cls.cat.map(CI)
    brow = []
    for (tag, bd, lvl, scen, est, k), g in cls.groupby(["tag", "bagdef", "level", "scen", "est", "class"]):
        m, lo, hi = boot_ci(g.bias_pp, g.cid.values); brow.append(dict(tag=tag, bagdef=bd, level=lvl, scen=scen, est=est, cls=k, n=len(g), bias_pp=m, lo=lo, hi=hi, true_mean_pp=g.p.mean() * 100))
    bias = pd.DataFrame(brow); bias.to_csv(T / "r02_class_bias.csv", index=False)
    # ---- classifier metric summary
    crow = []
    for (tag, scen), g in clf.groupby(["tag", "scen"]):
        cid = g.cat.map(CI).values
        d = dict(tag=tag, scen=scen, n_cats=len(g))
        for col in ("macroF1", "balacc", "macroF1_L1", "balacc_L1"):
            m, lo, hi = boot_ci(g[col].values, cid); d.update({col: m, col + "_lo": lo, col + "_hi": hi})
        crow.append(d)
    clfs = pd.DataFrame(crow); clfs.to_csv(T / "r02_classifier_metrics_summary.csv", index=False)
    # ---- hypotheses
    H = {}
    def pairs(tag, bd, lvl, scen, a, b):
        x = bag[(bag.tag == tag) & (bag.bagdef == bd) & (bag.level == lvl) & (bag.scen == scen)]
        xa = x[x.est == a].set_index(["cat", "bag"]).AE; xb = x[x.est == b].set_index(["cat", "bag"]).AE
        j = pd.concat([xa, xb], axis=1, keys=["a", "b"]).dropna()
        return j
    for tag in tags:
        for bd in BAGDEFS:
            for lvl in ("L2", "L1"):
                key = f"{tag}|{bd}|{lvl}"
                h1 = {}
                for e in ("ACC", "PACC", "EMQ", "PCC"):
                    j = pairs(tag, bd, lvl, "S1-tasma", "CC", e)
                    dlt = (j.a - j.b).values; cid = j.index.get_level_values(0).map(CI).values
                    h1[e] = dict(zip(("diff_CC_minus_X", "lo", "hi"), boot_ci(dlt, cid)), n_bags=len(j))
                h2 = {}
                for e in ("ACC", "PACC", "EMQ"):
                    j1 = pairs(tag, bd, lvl, "S1-tasma", "CC", e); j2 = pairs(tag, bd, lvl, "S2", "CC", e)
                    d1 = (j1.a - j1.b); d2 = (j2.a - j2.b)
                    idx = d1.index.intersection(d2.index); cid = idx.get_level_values(0).map(CI).values
                    dd = (d2.loc[idx] - d1.loc[idx]).values
                    h2[e] = dict(S1=boot_ci(d1.loc[idx].values, cid), S2=boot_ci(d2.loc[idx].values, cid), S2_minus_S1=boot_ci(dd, cid), n_bags=len(idx))
                # oracle recovery
                x_s2 = bag[(bag.tag == tag) & (bag.bagdef == bd) & (bag.level == lvl) & (bag.scen == "S2")].set_index(["cat", "bag", "est"]).AE
                x_s1 = bag[(bag.tag == tag) & (bag.bagdef == bd) & (bag.level == lvl) & (bag.scen == "S1-tasma") & (bag.est == "ACC")].set_index(["cat", "bag"]).AE
                a2 = x_s2.xs("ACC", level="est"); orc = x_s2.xs("ACC_oracle", level="est")
                idx = a2.index.intersection(orc.index).intersection(x_s1.index); cid = idx.get_level_values(0).map(CI).values
                A2, O, A1 = a2.loc[idx].values, orc.loc[idx].values, x_s1.loc[idx].values
                def rec(bi):
                    cc_ = cid; s2 = np.bincount(cc_, weights=A2, minlength=12); so = np.bincount(cc_, weights=O, minlength=12); s1 = np.bincount(cc_, weights=A1, minlength=12); nn = np.bincount(cc_, minlength=12)
                    n = nn[bi].sum(); m2, mo, m1 = s2[bi].sum() / n, so[bi].sum() / n, s1[bi].sum() / n
                    return m2, mo, m1
                m2, mo, m1 = rec(np.arange(12)); full = (m2 - mo) / (m2 - m1) * 100 if m2 - m1 > 1e-9 else np.nan
                bs = []
                for b in BOOT:
                    a, o, c1 = rec(b)
                    bs.append((a - o) / (a - c1) * 100 if a - c1 > 1e-9 else np.nan)
                bs = np.array(bs)
                H[key] = dict(H1=h1, H2=h2, oracle=dict(AE_ACC_S2=m2, AE_oracle_S2=mo, AE_ACC_S1tasma=m1, loss=m2 - m1, recovery_pct=full,
                                                        recovery_lo=float(np.nanpercentile(bs, 2.5)) if np.isfinite(bs).any() else None,
                                                        recovery_hi=float(np.nanpercentile(bs, 97.5)) if np.isfinite(bs).any() else None,
                                                        frac_boot_valid=float(np.isfinite(bs).mean())))
    # H3: cat-level ranking
    h3 = []
    kb = bag[(bag.bagdef == "kedi") & (bag.level == "L2")]
    cl = clf.set_index(["tag", "scen", "cat"]).macroF1
    for tag in tags:
        for est in ("CC", "PCC", "ACC", "PACC", "EMQ"):
            allx, ally = [], []
            for scen in ("S1-tasma", "S1-kosum", "S2"):
                g = kb[(kb.tag == tag) & (kb.scen == scen) & (kb.est == est)].sort_values("cat")
                f1 = np.array([cl[(tag, scen, c)] for c in g.cat]); ae_ = g.AE.values
                r = spearmanr(f1, ae_)
                h3.append(dict(tag=tag, est=est, scen=scen, n=len(g), rho=float(r.statistic), p=float(r.pvalue)))
                allx += list(f1); ally += list(ae_)
            r = spearmanr(allx, ally); h3.append(dict(tag=tag, est=est, scen="havuz(3x12)", n=len(allx), rho=float(r.statistic), p=float(r.pvalue)))
    pd.DataFrame(h3).to_csv(T / "r02_h3_spearman.csv", index=False)
    json.dump(H, open(T / "r02_hypotheses.json", "w"), indent=1, default=float)
    # ---- helper: data summary
    tot = lab.y2[lab.y2 >= 0].map(dict(enumerate(D.L2))).value_counts().rename("saniye").to_frame()
    tot["L1"] = [D.L2_TO_L1[c] for c in tot.index]
    tot.to_csv(T / "r02_class_seconds.csv")
    ex = dict(toplam_etiket_satiri=len(lab), kullanilabilir_R01=int(lab.usable.sum()), L2_saniye=int((lab.y2 >= 0).sum()),
              kullanilabilir_ama_L2_disi=int((lab.usable & (lab.y2 < 0)).sum()), kullanilabilir_degil=int((~lab.usable).sum()))
    json.dump(ex, open(T / "r02_excluded_seconds.json", "w"), indent=1)
    bn = []
    for bd, w in WIN.items():
        pass
    nb = {bd: len(w[1]) for bd, w in WIN.items()}; json.dump(nb, open(T / "r02_bag_counts.json", "w"))
    # run times
    json.dump({t: RES[t]["times"] for t in tags}, open(T / "r02_runtimes.json", "w"), indent=1, default=float)
    print("BİTTİ", nb)


if __name__ == "__main__":
    main()
