r"""27_r06_E_dunford.py — R06 E1/E2/E3 Dunford v2: S3, S4-Dunford (closed/open), study level, m in {60,300,900,1800}.
Arms: CC (ref), T-ACC, PACC-target, PACC-shrinkage, PACC-target-g (20 repeats); Retrain (unweighted), Retrain-weighted, Target-only-matched (5 repeats; RF 200). New draws (v2 data).
Reads: features/dunford_v2_1s.parquet, dog_aneck_1s.parquet, Smit table, r06/sources_v2.pkl. Writes: r06/E1_<scen>_<var>.parquet, _miss.parquet, results/tables/r06_strata.json
"""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, r06lib as L

FIT_MS = (300, 1800)   # time limit: model training (retrain, target-only) only at these m; PACC arms at all m
SEED = P.SEED; OUT = P.R06; T = P.TABLES; t0 = time.time()
LOG = open(P.LOGS / "r06_E_dunford.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


lab, y1, XC, XH, XC17, XH17 = R.load_smit(); idx_all = np.where(y1 >= 0)[0]; XS = XC17.values
dun, Xd = R.load_dunford(v2=True); Xd = Xd.values; yd = dun.y1.values; dcat = dun.cat.values; DCATS = sorted(set(dcat))
dcid = pd.factorize(dun.chunk.astype(str) + "_" + dun["sub"].astype(str))[0]
dog, Xg = R.load_dog(); keepd = dog.y1.values < 2; Xg = Xg.values[keepd]; yg = dog.y1.values[keepd]
S = pickle.load(open(OUT / "sources_v2.pkl", "rb"))
STR_D = [list(a) for a in np.array_split(np.random.default_rng(SEED + 2).permutation(DCATS), 3)]
json.dump(dict(dunford=STR_D), open(T / "r06_strata.json", "w"))


def run_dunford(scen, var, tgt, Pm, src, Xsrc, ysrc, Kt, vcode):
    f = OUT / f"E1_{scen}_{var}.parquet"
    if f.exists(): log("atla", f.name); return
    rec = R.Rec(); miss = []; pred = Pm.argmax(1)
    bags = {c: [(0, tgt.rows_c[c])] for c in DCATS if len(tgt.rows_c[c]) >= 60}
    for c in bags:
        n = len(bags[c][0][1]); p = R.truth(tgt.y[bags[c][0][1]], Kt)
        rec.add(scen, var, "calisma", "CC", 0, 0, c, 0, n, p, R.pad3(np.bincount(pred[bags[c][0][1]], minlength=Pm.shape[1]) / n, Kt))
    for si, strat in enumerate(STR_D):
        pool_cats = [c for c in DCATS if c not in strat]; ev = [c for c in strat if c in bags]
        for m in L.MS:
            for rep in range(L.REPS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 300 + si, m, rep, vcode]))
                labr = tgt.draw(pool_cats, m, rng)
                if len(labr) == 0: miss.append((scen, var, "calisma", m, rep, f"katman{si}", 0, -1, -1)); continue
                fo = None
                if rep < L.REPS_RT and ev and m in FIT_MS:
                    rows_pred = np.concatenate([tgt.rows_c[c] for c in ev])
                    fo = L.fit_arms(Xsrc, ysrc, tgt.X, tgt.y, labr, rows_pred, SEED + 1000 * si + 10 * rep + m % 97)
                for c in ev:
                    L.eval_unit(rec, miss, scen, var, "calisma", m, rep, c, bags[c], None, tgt.y, Pm, pred, src, labr, Kt, fo)
        log(scen, var, "katman", si, "%.0f s" % (time.time() - t0))
    rec.wide().to_parquet(f); pd.DataFrame(miss, columns=["scen", "variant", "level", "m", "rep", "unit", "n_lab", "n_missing", "n_cls"]).to_parquet(OUT / f"E1_{scen}_{var}_miss.parquet")


tgt3 = L.DunTarget(yd, dcat, dcid, Xd, DCATS, False)
run_dunford("S3", "-", tgt3, S["P3"][tgt3.rows], S["src3"], XS[idx_all], y1[idx_all], 3, 1)
for var in ("kapali", "acik"):
    tg = L.DunTarget(yd, dcat, dcid, Xd, DCATS, var == "kapali"); Kt = 2 if var == "kapali" else 3
    run_dunford("S4-Dunford", var, tg, S["P4d"][tg.rows], S["src4"], Xg, yg, Kt, 2 + (var == "acik"))
log("BITTI E dunford", "%.0f s" % (time.time() - t0))
