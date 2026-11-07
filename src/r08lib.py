r"""R08 shared: data/bag builders, APP draws, per-fold CC/PACC/DM/DR runner. Reads: R05–R07 intermediates. Writes nothing (the scripts write).
Row indices 'global' = row number of the relevant domain table (Smit: lab; dog: NK; Dunford: v2 table filtered to y1>=0).
"""
import pickle
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, quant as Q, quant_dm as DM, quant_dr as DRm, smit_data as D, features_smit as F, r07clf as C07

SEED = P.SEED
L1N = ["Inaktif", "Lokomosyon", "Bakim"]; DOGL2 = ["Lying chest", "Sitting", "Standing", "Walking", "Trotting"]
ARMS = [("CC", "-"), ("PACC", "-"), ("DM", "-"), ("DR-APP", "q"), ("DR-dogal", "q"), ("DR-APP", "h"), ("DR-dogal", "h")]


# ------------------------------------------------------------------ dog
def dog_tables():
    NK = pd.read_parquet(P.DOG_FEAT).reset_index(drop=True)
    X32 = NK[F.COLS32].values.astype(np.float64); X17 = F.d17(NK).values.astype(np.float64)
    beh = NK.beh.values; dogid = NK.DogID.values
    y_L2 = np.array([DOGL2.index(b) if b in DOGL2 else -1 for b in beh]); y_L1 = np.where(NK.y1.values == 2, -1, NK.y1.values)
    return NK, X32, X17, y_L2, y_L1, dogid


def dog_bags(NK, y, dogs=None):
    """R07 §5 bags (same as bags_for in 43_r07_5_dog.py): {dog: {'kayit': [(id, rows)], 'dogal2': [...]}}, N >= 60."""
    use = np.where(y >= 0)[0]; dogid = NK.DogID.values; rec_ = NK.TestNum.values; sec = NK.sec.values
    out = {}
    for d in (sorted(set(dogid[use])) if dogs is None else dogs):
        r = use[dogid[use] == d]; out[d] = dict(kayit=[], dogal2=[])
        for t in np.unique(rec_[r]):
            rr = r[rec_[r] == t]
            if len(rr) >= 60: out[d]["kayit"].append((f"{t}", rr))
            w = sec[rr] // 120
            for k in np.unique(w):
                q = rr[w == k]
                if len(q) >= 60: out[d]["dogal2"].append((f"{t}_{k}", q))
    return out


# ------------------------------------------------------------------ APP draws (training bags)
def app_train(y, K, rows_by_ind, code, nbag=100, size=120, maxtry=1000):
    """nbag APP bags per individual (Dirichlet(1,..,1) proportions; a draw exceeding the individual's class seconds is rejected; at most 1,000 attempts; without replacement within the bag).
    Seed: SeedSequence([SEED, 800, code, individual order]) — separate from the evaluation APP draws (600/700 sequences).
    Returns (bags {individual: [(rows, p)]}, stats DataFrame)."""
    bags = {}; st = []
    for i, ind in enumerate(sorted(rows_by_ind)):
        rows = rows_by_ind[ind]; cl = {k: rows[y[rows] == k] for k in range(K)}
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 800, code, i])); out = []; tries = 0
        for b in range(nbag):
            for _ in range(maxtry):
                tries += 1; p = rng.dirichlet(np.ones(K)); n = C07.largest_remainder(p, size)
                if all(n[k] <= len(cl[k]) for k in range(K)):
                    out.append((np.concatenate([rng.choice(cl[k], n[k], replace=False) for k in range(K)]), n / size)); break
        bags[ind] = out
        pr = np.array([b[1] for b in out]) if out else np.zeros((0, K))
        st.append(dict(kod=code, birey=str(ind), kabul=len(out), deneme=tries, kabul_orani=len(out) / tries, **{f"p{k}_ort": (pr[:, k].mean() if len(pr) else np.nan) for k in range(K)},
                       **{f"p{k}_sd": (pr[:, k].std() if len(pr) else np.nan) for k in range(K)}))
    return bags, pd.DataFrame(st)


# ------------------------------------------------------------------ descriptors
def desc_q(X, bags):
    return np.array([DRm.descr_q(X, rows) for rows in bags])


def desc_h(Pg, bags):
    return np.array([DRm.descr_h(Pg, rows) for rows in bags])


def truth(y, rows, K):
    return np.bincount(y[rows], minlength=K) / len(rows)


# ------------------------------------------------------------------ fold runner
def run_fold(f, log=None):
    """f: dict
      scen, fold, K, cls (class names), btype order
      Xs (global source feature array), tr (source training rows), ys (global source labels), inds (global source individuals), oof (len(tr), K) inner-CV posteriors
      nat: [(individual, rows)], app: [(individual, rows, p)]
      Xt, Pt (global target features, outer-model posteriors), yt (global target labels)
      bags: {btype: [(individual, bag_id, rows)]}
      C, Cp (source confusion matrices; for CC/PACC)
    Returns (row list [dict], meta list, diagnostics list)."""
    K = f["K"]; tr = f["tr"]; ys = f["ys"]; oof = f["oof"].astype(np.float64)
    prior_tr = np.bincount(ys[tr], minlength=K) / len(tr)
    H, present = DM.class_cond_hists(oof, ys[tr], K)
    Poof = np.full((len(f["Xs"]), K), np.nan); Poof[tr] = oof
    rows_out = []; meta = []; diag = []
    # evaluation bags: reference and DM
    allb = [(bt, ind, bid, rows) for bt, lst in f["bags"].items() for (ind, bid, rows) in lst]
    est = {}
    for j, (bt, ind, bid, rows) in enumerate(allb):
        Pb = f["Pt"][rows]; n = len(rows); cc = Q.cc(Pb.argmax(1), K); pc = Pb.mean(0)
        dm, bound, Lval = DM.dm(Pb, H, present, [np.full(K, 1.0 / K), prior_tr, cc])
        est[j] = {("CC", "-"): cc, ("PACC", "-"): Q.pacc(pc, f["Cp"]), ("DM", "-"): dm}
        meta.append(dict(senaryo=f["scen"], torba_turu=bt, dis_katman=f["fold"], hedef_birey=str(ind), torba=str(bid), N=n, dm_sinir=bool(bound), dm_L=Lval))
    # DR: descriptors
    Dq_e = desc_q(f["Xt"], [b[3] for b in allb]); Dh_e = desc_h(f["Pt"], [b[3] for b in allb])
    info = {}
    for var, bl in (("DR-APP", [(i, r, p) for (i, r, p) in f["app"]]), ("DR-dogal", [(i, r, truth(ys, r, K)) for (i, r) in f["nat"]])):
        if len(bl) < 10:
            info[var] = None; continue
        grp = np.array([str(b[0]) for b in bl]); pb = np.array([b[2] for b in bl]); Nb = np.array([len(b[1]) for b in bl]); rr = [b[1] for b in bl]
        for tdesc, De, Xb in (("q", Dq_e, desc_q(f["Xs"], rr)), ("h", Dh_e, desc_h(Poof, rr))):
            mdl = DRm.DR().fit(Xb, pb, Nb, grp)
            ph = mdl.predict(De)
            for j in range(len(allb)): est[j][(var, tdesc)] = ph[j]
            info[(var, tdesc)] = dict(a=mdl.a, n_bag=len(bl), amax=mdl.amax_eff)
        # extrapolation diagnostic (per class): proportion range [min, max] in the training bags
        lo, hi = pb.min(0), pb.max(0)
        for bt in f["bags"]:
            ix = [j for j, b in enumerate(allb) if b[0] == bt]
            ptrue = np.array([truth(f["yt"], allb[j][3], K) for j in ix])
            for k in range(K): diag.append(dict(senaryo=f["scen"], dis_katman=f["fold"], torba_turu=bt, varyant=var, sinif=f["cls"][k], dis=int(((ptrue[:, k] < lo[k]) | (ptrue[:, k] > hi[k])).sum()), n=len(ix),
                                                  egitim_min=float(lo[k]), egitim_maks=float(hi[k])))
    for j, (bt, ind, bid, rows) in enumerate(allb):
        p = truth(f["yt"], rows, K); n = len(rows)
        for (arm, tdesc), ph in est[j].items():
            ainfo = info.get((arm, tdesc)) if arm.startswith("DR") else None
            for k in range(K):
                rows_out.append(dict(senaryo=f["scen"], torba_turu=bt, kol=arm, tanimlayici=tdesc, dis_katman=f["fold"], hedef_birey=str(ind), torba=str(bid), N=n, sinif=f["cls"][k],
                                     p=float(p[k]), p_hat=float(ph[k]), a=(ainfo["a"] if ainfo else np.nan)))
    fi_ = [dict(senaryo=f["scen"], dis_katman=f["fold"], varyant=v[0], tanimlayici=v[1], a=i["a"], amax=i["amax"], n_egitim_torba=i["n_bag"]) for v, i in info.items() if i is not None and isinstance(v, tuple)]
    return rows_out, meta, diag, fi_
