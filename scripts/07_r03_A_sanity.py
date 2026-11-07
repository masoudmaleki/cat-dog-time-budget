r"""07_r03_A_sanity.py — R03 A: in S0, estimate C from random-second inner CV (StratifiedKFold(4)) and compare ACC/PACC/EMQ with CC.
Reads: <data root>/interim/r02/raw_va_f32.pkl (S0 test posteriors; same model as R02), smit_* parquet.
Writes: results/tables/r03_A_*.csv, r03_A_summary.json
"""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from config import paths as P
from src import evalkit as E, quant as Q, smit_data as D

K, SEED = E.K, P.SEED
res = pickle.load(open(P.INTERIM / "r02" / "raw_va_f32.pkl", "rb"))
s0 = res["s0"]; te = s0["test_idx"]; P_te = s0["P_cc"]
idx_all = np.where(E.Y >= 0)[0]
perm = np.random.default_rng(SEED).permutation(idx_all); ntr = int(0.7 * len(perm))
tr = np.sort(perm[:ntr]); assert len(tr) == s0["n_train"] and not set(tr) & set(te)
XC = E.features()[0][D.FEAT32].values
t0 = time.time()
oof = np.zeros((len(tr), K))
for a, b in StratifiedKFold(4, shuffle=True, random_state=SEED).split(tr, E.Y[tr]):
    m = RandomForestClassifier(300, n_jobs=-1, random_state=SEED + 1).fit(XC[tr[a]], E.Y[tr[a]])
    p = m.predict_proba(XC[tr[b]]); full = np.zeros((len(b), K)); full[:, m.classes_] = p; oof[b] = full
y = E.Y[tr]
rnd = dict(C=Q.conf_matrix(y, oof.argmax(1), K), Cp=Q.soft_conf_matrix(y, oof, K), prior_oof=oof.mean(0))
grp = s0["inner_collar"]
print("iç CV süresi", round(time.time() - t0), "s; OOF doğruluk", round(float((oof.argmax(1) == y).mean()), 4), "kedi-gruplu OOF doğruluk", round(grp["oof_acc"], 4))
print("test doğruluk", round(float((P_te.argmax(1) == E.Y[te]).mean()), 4))


def fn(m):
    e = E.base_estimates(P_te[m], rnd)
    g = E.base_estimates(P_te[m], grp)
    return {"CC": e["CC"], "PCC": e["PCC"], "ACC_rastgeleCV": e["ACC"], "PACC_rastgeleCV": e["PACC"], "EMQ_rastgeleCV": e["EMQ"],
            "ACC_kediCV(R02)": g["ACC"], "PACC_kediCV(R02)": g["PACC"], "EMQ_kediCV(R02)": g["EMQ"]}


out = {}
for bd in ("30dk_ge50", "kedi"):
    df = E.evaluate([dict(scen="S0", cat=None, te=te, fn=fn)], bd)
    summ = E.summarize(df); summ.to_csv(P.TABLES / f"r03_A_summary_{bd}.csv", index=False)
    print(bd, "\n", summ[["est", "n_bags", "AE", "AE_lo", "AE_hi", "AE_L1"]].round(4).to_string())
    d = {}
    for e in ("ACC_rastgeleCV", "PACC_rastgeleCV", "EMQ_rastgeleCV", "ACC_kediCV(R02)"):
        d[e] = dict(zip(("CC_minus_X", "lo", "hi", "n"), E.paired(df, "S0", "CC", e)))
        d[e + "_L1"] = dict(zip(("CC_minus_X", "lo", "hi", "n"), E.paired(df, "S0", "CC", e, "AE_L1")))
    out[bd] = d
    print(json.dumps(d, indent=1))
json.dump(dict(rnd_C_diag=np.diag(rnd["C"]).tolist(), grp_C_diag=np.diag(grp["C"]).tolist(), oof_acc_rand=float((oof.argmax(1) == y).mean()),
               oof_acc_grp=grp["oof_acc"], test_acc=float((P_te.argmax(1) == E.Y[te]).mean()), paired=out),
          open(P.TABLES / "r03_A_summary.json", "w"), indent=1, default=float)
# ---- diagnostic checks (code correctness)
Ct = Q.conf_matrix(E.Y[te], P_te.argmax(1), K)
diag = {}
diag["C_fro_rastgele_vs_test"] = float(np.linalg.norm(rnd["C"] - Ct)); diag["C_fro_kediCV_vs_test"] = float(np.linalg.norm(grp["C"] - Ct))
diag["C_diag_rastgele"] = np.diag(rnd["C"]).round(3).tolist(); diag["C_diag_test"] = np.diag(Ct).round(3).tolist(); diag["C_diag_kediCV"] = np.diag(grp["C"]).round(3).tolist()
diag["C_kolon_toplam_min_max"] = [float(rnd["C"].sum(0).min()), float(rnd["C"].sum(0).max())]
# (1) pool level (all S0 test seconds as one bag): algebraic identity ACC(C_test) == true proportion
p_true = np.bincount(E.Y[te], minlength=K) / len(te)
cc_ = Q.cc(P_te.argmax(1), K)
pool = {"CC": cc_, "ACC_rastgeleCV": Q.acc(cc_, rnd["C"]), "ACC_C_test(özdeşlik)": Q.acc(cc_, Ct), "ACC_kediCV": Q.acc(cc_, grp["C"]),
        "PACC_rastgeleCV": Q.pacc(Q.pcc(P_te), rnd["Cp"]), "EMQ_rastgeleCV": Q.emq(P_te, rnd["prior_oof"])[0]}
diag["havuz_AE"] = {k: Q.ae(v, p_true) for k, v in pool.items()}
# (2) within-bag identity: estimating C from the bag's own labels -> ACC must recover the bag proportion exactly
ae_self = []
for c, w, m in E.groups(te, None, "30dk_ge50"):
    yb, pb = E.Y[te][m], P_te[m].argmax(1); Cb = Q.conf_matrix(yb, pb, K)
    ae_self.append(Q.ae(Q.acc(Q.cc(pb, K), Cb), np.bincount(yb, minlength=K) / len(yb)))
diag["torba_ici_ozdeslik_AE_max"] = float(np.max(ae_self)); diag["torba_ici_ozdeslik_AE_ort"] = float(np.mean(ae_self))
# (3) within-bag error: bag dependence of C. ‖.‖_F between the per-bag true C and the global C_test
fro = []
for c, w, m in E.groups(te, None, "30dk_ge50"):
    yb, pb = E.Y[te][m], P_te[m].argmax(1); Cb = Q.conf_matrix(yb, pb, K); fro.append(np.linalg.norm(Cb - Ct))
diag["torba_C_vs_kuresel_C_test_fro_ort"] = float(np.mean(fro))
# (4) clustering of consecutive-second errors within a bag: lag-1 s autocorrelation of the error probability (misclassification) in the same bag
sec = E.LAB.sec.values[te]; o = np.argsort(sec); err = (P_te.argmax(1) != E.Y[te])[o].astype(float)
cat_o = E.CATROW[te][o]; same = cat_o[1:] == cat_o[:-1]
diag["hata_otokorelasyon_lag1"] = float(np.corrcoef(err[1:][same], err[:-1][same])[0, 1])
print(json.dumps(diag, indent=1, ensure_ascii=True))
json.dump(diag, open(P.TABLES / "r03_A_diagnostics.json", "w"), indent=1, default=float)

# ---- (5) time-clustering test: fake bags of the same size as the real ones but drawn at random from the cat's test seconds (time structure destroyed)
rng = np.random.default_rng(SEED)
rows = []
for rep in range(20):
    for c, w, m in E.groups(te, None, "30dk_ge50"):
        idx_c = np.where(E.CATROW[te] == c)[0]
        pick = rng.choice(idx_c, int(m.sum()), replace=False)
        pb = P_te[pick]; yb = E.Y[te][pick]; p = np.bincount(yb, minlength=K) / len(pick)
        est = E.base_estimates(pb, rnd)
        rows.append(dict(rep=rep, cat=c, bag=w, **{f"AE_{k}": Q.ae(v, p) for k, v in est.items()}))
pseudo = pd.DataFrame(rows)
cid = pseudo.cat.map(E.CI).values
ps = {}
for k in ("CC", "ACC", "PACC", "EMQ"):
    ps[k] = E.boot_ci(pseudo[f"AE_{k}"].values, cid)
ps["CC_minus_ACC"] = E.boot_ci((pseudo.AE_CC - pseudo.AE_ACC).values, cid)
ps["CC_minus_PACC"] = E.boot_ci((pseudo.AE_CC - pseudo.AE_PACC).values, cid)
ps["CC_minus_EMQ"] = E.boot_ci((pseudo.AE_CC - pseudo.AE_EMQ).values, cid)
print("rastgele (zamansal yapısı bozulmuş) sahte torbalar, 20 tekrar:", json.dumps(ps, indent=1))
json.dump(ps, open(P.TABLES / "r03_A_pseudo_bags.json", "w"), indent=1, default=float)

# ---- (6) iid check: random bags from the whole S0 test set (cats mixed), at real bag sizes. C fixed -> ACC/PACC/EMQ should beat CC
rng = np.random.default_rng(SEED + 7)
sizes = [int(m.sum()) for _, _, m in E.groups(te, None, "30dk_ge50")]
rows = []
for rep in range(20):
    for n_ in sizes:
        pick = rng.choice(len(te), n_, replace=False)
        pb = P_te[pick]; p = np.bincount(E.Y[te][pick], minlength=K) / n_
        est = E.base_estimates(pb, rnd)
        rows.append({f"AE_{k}": Q.ae(v, p) for k, v in est.items()})
iid = pd.DataFrame(rows)
res_iid = {k: [float(iid[f"AE_{k}"].mean()), float(iid[f"AE_{k}"].sem())] for k in ("CC", "ACC", "PACC", "EMQ")}
res_iid["n_torba"] = len(iid)
print("iid torbalar (kediler karışık):", json.dumps(res_iid))
json.dump(res_iid, open(P.TABLES / "r03_A_iid_bags.json", "w"), indent=1)
