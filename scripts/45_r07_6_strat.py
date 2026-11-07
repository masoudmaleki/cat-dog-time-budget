r"""45_r07_6_strat.py — R07 §6: stratified target labelling. Arms (no retraining): CC-strat (same bags), T-ACC-strat, PACC-target-strat; extra: Target-only-strat (S4-Smit-open individual, m=300/1800, 5 repeats, RF 200).
m in {60,300,900,1800}, 20 repeats. Individual: S1, S2, S4-Smit (open/closed); study: S3, S4-Dunford (open/closed), S4-Smit (open/closed). At the individual level the selected seconds are removed from the bags; at the study level they are not.
Source models and posteriors come from R05/R06 (no new model, only the extra retrain arm). Folds as in R06 (SEED+1, SEED+2 permutations).
Reads: r05/ladder_{S1,S2}_*.pkl, r05/sources.pkl, r06/sources_v2.pkl, Smit table, features/*. Writes: <data root>/interim/r07/T_<name>.parquet, _miss.parquet
Argument: ladder | smit | dunford"""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, r06lib as L, r07strat as ST

part = sys.argv[1]; SEED = P.SEED; O5 = P.R05; O6 = P.R06; OUT = P.INTERIM / "r07"; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
LOG = open(P.LOGS / f"r07_6_{part}.log", "a", encoding="utf-8")
MS = (60, 300, 900, 1800); REPS = 20; REPS_FIT = 5; FIT_MS = (300, 1800)
KEEP = {"CC": "CC-tab", "T-ACC": "T-ACC-tab", "PACC-hedef": "PACC-hedef-tab", "Yalniz-hedef-esl": "Yalniz-hedef-tab"}


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def finish(rec, miss, name):
    pd.DataFrame(CNT).to_parquet(OUT / f"T_{name}_cnt.parquet"); CNT.clear()
    w = rec.wide(); w = w[w.arm.isin(KEEP)].copy(); w["arm"] = w.arm.map(KEEP)
    w.to_parquet(OUT / f"T_{name}.parquet")
    pd.DataFrame(miss, columns=["scen", "variant", "level", "m", "rep", "unit", "n_lab", "n_missing", "n_cls"]).to_parquet(OUT / f"T_{name}_miss.parquet")


CNT = []


def dr(runs, K, m, rng, y, scen, var, level, rep, unit):
    """stratified draw + record of the per-class second counts (E3 comparison)"""
    lr, cap = ST.draw_strat(runs, K, m, rng); cn = np.bincount(y[lr], minlength=K)[:K] if len(lr) else np.zeros(K, int)
    CNT.append(dict(scen=scen, var=var, level=level, m=m, rep=rep, unit=unit, n_lab=int(len(lr)), K=K, **{f"c{k}": int(cn[k]) for k in range(K)}))
    return lr


def one(rec, miss, scen, var, level, m, rep, c, bags, rem, y, Pm, pred, src, labr, Kt, hard=None):
    r2 = R.Rec(); L.eval_unit(r2, miss, scen, var, level, m, rep, c, bags, rem, y, Pm, pred, src, labr, Kt, hard); rec.rows += r2.rows


lab, y1, XC, XH, XC17, XH17 = R.load_smit(); idx_all = np.where(y1 >= 0)[0]; cat_full = lab.Cat_id.values; CATS = sorted(set(cat_full))
XS, XHs = XC17.values, XH17.values; SEC = lab.sec.values.astype("datetime64[s]").astype(np.int64)

if part == "ladder":
    BAGS30 = R.smit_bags(lab, y1, 30, 0.5); rec = R.Rec(); miss = []
    for ci, c in enumerate(CATS):
        for scen, trX in (("S1", XS), ("S2", XHs)):
            d = pickle.load(open(O5 / f"ladder_{scen}_{c}.pkl", "rb")); te = d["te"].astype(int); Pm = d["P"].astype(np.float64); src = d["src"]
            pred = Pm.argmax(1); pos = {int(g): i for i, g in enumerate(te)}; yte = y1[te]; ts = SEC[te]; srt = np.argsort(ts, kind="stable")
            bags = [(w, np.array([pos[int(g)] for g in ix if int(g) in pos])) for cc, w, ix in BAGS30 if cc == c]
            runs = ST.make_runs(srt, ts[srt], yte[srt], 3)
            for m in MS:
                for rep in range(REPS):
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, 800 + ci, m, rep, 7 if scen == "S1" else 8]))
                    labr = dr(runs, 3, m, rng, yte, scen, "30dk_ge50", "birey", rep, c)
                    rem = np.ones(len(te), bool); rem[labr] = False
                    one(rec, miss, scen, "30dk_ge50", "birey", m, rep, c, bags, rem, yte, Pm, pred, src, labr, 3)
        log("ladder", c, "%.0f s" % (time.time() - t0))
    finish(rec, miss, "ladder")

if part == "smit":
    dog, Xg = R.load_dog(); keepd = dog.y1.values < 2; Xg = Xg.values[keepd]; yg = dog.y1.values[keepd]
    S = pickle.load(open(O5 / "sources.pkl", "rb"))
    STR_S = [list(a) for a in np.array_split(np.random.default_rng(SEED + 1).permutation(CATS), 4)]
    for var in ("acik", "kapali"):
        Kt = 2 if var == "kapali" else 3; acik = var == "acik"; vcode = 12 + (var == "acik")
        tg = L.SmitTarget(lab, y1, XS, SEC, CATS, idx_all, (y1 < 2) if var == "kapali" else None, S["P4s"]); Pm, src = tg.P, S["src4"]; pred = Pm.argmax(1)
        runs_c = {c: ST.make_runs(tg.sorted_c[c], tg.t[tg.sorted_c[c]], tg.y[tg.sorted_c[c]], Kt) for c in CATS}
        rec = R.Rec(); miss = []; wcache = {}
        for ci, c in enumerate(CATS):
            si = [i for i, s in enumerate(STR_S) if c in s][0]; pool_cats = [x for x in CATS if x not in STR_S[si]]; bg = tg.bags.get(c, [])
            for m in MS:
                for rep in range(REPS):
                    labI = dr(runs_c[c], Kt, m, np.random.default_rng(np.random.SeedSequence([SEED, 900 + ci, m, rep, vcode])), tg.y, "S4-Smit", var, "birey", rep, c)
                    rem = np.ones(len(tg.y), bool); rem[labI] = False
                    key = (si, m, rep)
                    if key not in wcache: wcache[key] = dr(ST.merge_runs([runs_c[x] for x in pool_cats], Kt), Kt, m, np.random.default_rng(np.random.SeedSequence([SEED, 950 + si, m, rep, vcode])), tg.y, "S4-Smit", var, "calisma", rep, f"s{si}")
                    labW = wcache[key]; hardI = None
                    if acik and rep < REPS_FIT and m in FIT_MS:
                        hardI = L.fit_arms(Xg, yg, tg.X, tg.y, labI, tg.rows_c[c], SEED + 7 * ci + 10 * rep + m % 89, weighted=False, unweighted=False, target_only=True)
                    one(rec, miss, "S4-Smit", var, "birey", m, rep, c, bg, rem, tg.y, Pm, pred, src, labI, Kt, hardI)
                    one(rec, miss, "S4-Smit", var, "calisma", m, rep, c, bg, None, tg.y, Pm, pred, src, labW, Kt)
            log("smit", var, c, "%.0f s" % (time.time() - t0))
        finish(rec, miss, f"S4-Smit_{var}")

if part == "dunford":
    dun, Xd = R.load_dunford(v2=True); Xd = Xd.values; yd = dun.y1.values; dcat = dun.cat.values; DCATS = sorted(set(dcat))
    dcid = pd.factorize(dun.chunk.astype(str) + "_" + dun["sub"].astype(str))[0]
    S = pickle.load(open(O6 / "sources_v2.pkl", "rb"))
    STR_D = [list(a) for a in np.array_split(np.random.default_rng(SEED + 2).permutation(DCATS), 3)]
    for scen, var, Pm0, src, Kt, vcode in (("S3", "-", S["P3"], S["src3"], 3, 21), ("S4-Dunford", "kapali", S["P4d"], S["src4"], 2, 22), ("S4-Dunford", "acik", S["P4d"], S["src4"], 3, 23)):
        tg = L.DunTarget(yd, dcat, dcid, Xd, DCATS, var == "kapali"); Pm = Pm0[tg.rows]; pred = Pm.argmax(1)
        bags = {c: [(0, tg.rows_c[c])] for c in DCATS if len(tg.rows_c[c]) >= 60}
        runs_c = {}
        for c in DCATS:
            lst = []
            for k, rows in tg.chunks_c[c].items(): lst.append(ST.make_runs(rows, np.arange(len(rows)), tg.y[rows], Kt))
            runs_c[c] = ST.merge_runs(lst, Kt)
        rec = R.Rec(); miss = []
        for si, strat in enumerate(STR_D):
            pool_cats = [c for c in DCATS if c not in strat]; ev = [c for c in strat if c in bags]; pr = ST.merge_runs([runs_c[x] for x in pool_cats], Kt)
            for m in MS:
                for rep in range(REPS):
                    labr = dr(pr, Kt, m, np.random.default_rng(np.random.SeedSequence([SEED, 1000 + si, m, rep, vcode])), tg.y, scen, var, "calisma", rep, f"k{si}")
                    for c in ev: one(rec, miss, scen, var, "calisma", m, rep, c, bags[c], None, tg.y, Pm, pred, src, labr, Kt)
            log("dunford", scen, var, "katman", si, "%.0f s" % (time.time() - t0))
        finish(rec, miss, f"{scen}_{var}")
log("BITTI", part, "%.0f s" % (time.time() - t0))
