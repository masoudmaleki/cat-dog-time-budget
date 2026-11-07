r"""28_r06_E_smit.py — R06 E1/E2/E3 S4-Smit (dog -> Smit collar): open (weighted retrain + matched target-only) and closed (matched target-only only); individual and study level.
R05 draws exactly (same seeds). The unweighted retrain is taken from R05; it is reproduced within 1e-6 in one cell (if it cannot be reproduced, all cells are re-run).
Reads: Smit table, dog_aneck_1s.parquet, r05/sources.pkl, r05/E_S4-Smit_acik.parquet. Writes: r06/E1_S4-Smit_<var>.parquet, _miss.parquet, r06/E1_repro.json
Argument: list of m values (default 60,300,900,1800).
"""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, r06lib as L

FIT_MS = (300, 1800)   # time limit: model training only at these m
SEED = P.SEED; OUT = P.R06; O5 = P.R05; T = P.TABLES; t0 = time.time()
MS = tuple(int(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else L.MS
LOG = open(P.LOGS / "r06_E_smit.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


lab, y1, XC, XH, XC17, XH17 = R.load_smit(); idx_all = np.where(y1 >= 0)[0]; XS = XC17.values; cat_full = lab.Cat_id.values; CATS = sorted(set(cat_full))
SEC = lab.sec.values.astype("datetime64[s]").astype(np.int64)
dog, Xg = R.load_dog(); keepd = dog.y1.values < 2; Xg = Xg.values[keepd]; yg = dog.y1.values[keepd]
S = pickle.load(open(O5 / "sources.pkl", "rb"))
STR_S = [list(a) for a in np.array_split(np.random.default_rng(SEED + 1).permutation(CATS), 4)]
R5 = pd.read_parquet(O5 / "E_S4-Smit_acik.parquet")


def repro_check():
    """Single cell: open set, Cho, m=300, rep=0, individual; reproduce the unweighted retrain and compare the bag AE with R05."""
    f = OUT / "E1_repro.json"
    if f.exists(): return json.load(open(f))["ok"]
    var, c, m, rep, ci = "acik", "Cho", 300, 0, 0; vcode = 3
    tg = L.SmitTarget(lab, y1, XS, SEC, CATS, idx_all, None, S["P4s"]); Pm = tg.P; pred = Pm.argmax(1)
    labI = tg.draw([c], m, np.random.default_rng(np.random.SeedSequence([SEED, 200 + ci, m, rep, vcode])))
    rem = np.ones(len(tg.y), bool); rem[labI] = False
    fo = L.fit_arms(Xg, yg, tg.X, tg.y, labI, tg.rows_c[c], SEED + 7 * ci + 10 * rep + m % 89, weighted=False, unweighted=True, target_only=False)
    rec = R.Rec(); miss = []
    L.eval_unit(rec, miss, "S4-Smit", var, "birey", m, rep, c, tg.bags[c], rem, tg.y, Pm, pred, S["src4"], labI, 3, fo)
    new = rec.wide(); new = new[new.arm == "Yeniden-egitim"].sort_values("bag").AE.values
    old = R5[(R5.arm == "Yeniden-egitim") & (R5.level == "birey") & (R5.m == m) & (R5.rep == rep) & (R5.ind == c)].sort_values("bag").AE.values
    dif = float(np.abs(new - old).max()) if len(new) == len(old) and len(new) else float("nan")
    ok = bool(dif <= 1e-6); json.dump(dict(ok=ok, max_abs_diff=dif, n_bag=int(len(new))), open(f, "w")); log("repro", ok, dif, len(new)); return ok


REPRO = repro_check()


def run_smit(var):
    f = OUT / f"E1_S4-Smit_{var}.parquet"
    if f.exists(): log("atla", f.name); return
    Kt = 2 if var == "kapali" else 3; vcode = 2 + (var == "acik"); acik = var == "acik"
    tg = L.SmitTarget(lab, y1, XS, SEC, CATS, idx_all, (y1 < 2) if var == "kapali" else None, S["P4s"]); Pm, src = tg.P, S["src4"]; pred = Pm.argmax(1)
    rec = R.Rec(); miss = []; wcache = {}
    for ci, c in enumerate(CATS):
        si = [i for i, s in enumerate(STR_S) if c in s][0]; pool_cats = [x for x in CATS if x not in STR_S[si]]
        for m in MS:
            for rep in range(L.REPS):
                labI = tg.draw([c], m, np.random.default_rng(np.random.SeedSequence([SEED, 200 + ci, m, rep, vcode])))
                rem = np.ones(len(tg.y), bool); rem[labI] = False
                key = (si, m, rep)
                if key not in wcache: wcache[key] = tg.draw(pool_cats, m, np.random.default_rng(np.random.SeedSequence([SEED, 100 + si, m, rep, vcode])))
                labW = wcache[key]; bg = tg.bags.get(c, []); hardI = hardW = None
                if rep < L.REPS_RT and m in FIT_MS:
                    hardI = L.fit_arms(Xg, yg, tg.X, tg.y, labI, tg.rows_c[c], SEED + 7 * ci + 10 * rep + m % 89, weighted=acik, unweighted=(acik and not REPRO), target_only=True)
                    kw = ("fw", si, m, rep)
                    if kw not in wcache:
                        rows_pred = np.concatenate([tg.rows_c[x] for x in STR_S[si]])
                        wcache[kw] = L.fit_arms(Xg, yg, tg.X, tg.y, labW, rows_pred, SEED + 31 * si + 10 * rep + m % 83, weighted=acik, unweighted=(acik and not REPRO), target_only=True)
                    hardW = wcache[kw]
                L.eval_unit(rec, miss, "S4-Smit", var, "birey", m, rep, c, bg, rem, tg.y, Pm, pred, src, labI, Kt, hardI)
                r2 = R.Rec(); L.eval_unit(r2, miss, "S4-Smit", var, "calisma", m, rep, c, bg, rem, tg.y, Pm, pred, src, labW, Kt, hardW)
                rec.rows += [r for r in r2.rows if r[3] != "CC"]
        log("S4-Smit", var, c, "%.0f s" % (time.time() - t0))
    rec.wide().to_parquet(f); pd.DataFrame(miss, columns=["scen", "variant", "level", "m", "rep", "unit", "n_lab", "n_missing", "n_cls"]).to_parquet(OUT / f"E1_S4-Smit_{var}_miss.parquet")


for var in ("acik", "kapali"): run_smit(var)
log("BITTI E smit", "%.0f s" % (time.time() - t0))
