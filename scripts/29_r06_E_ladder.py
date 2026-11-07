r"""29_r06_E_ladder.py — R06 E1/E3 ladder: S1 and S2 individual level, m in {1800, 300}. Argument 1: 'e3' (only PACC-target/-g, no model) or 'fit' (weighted retrain; 5 repeats). Argument 2: list of m values.
R05 draws exactly (seed [SEED,400+ci,m,rep,7|8]); in R05 the retrain existed only for m=1800, rep 0-2: that unweighted arm is taken from R05, and the unweighted arm is also run in the other cells.
Reads: Smit table, r05/ladder_{S1,S2}_<cat>.pkl. Writes: r06/E1_ladder_<phase>_m<m>.parquet
"""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, r06lib as L

SEED = P.SEED; OUT = P.R06; O5 = P.R05; t0 = time.time()
phase = sys.argv[1]; MS = tuple(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 else (1800, 300)
LOG = open(P.LOGS / "r06_E_ladder.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


lab, y1, XC, XH, XC17, XH17 = R.load_smit(); idx_all = np.where(y1 >= 0)[0]; cat_full = lab.Cat_id.values; CATS = sorted(set(cat_full))
XS, XHs = XC17.values, XH17.values; SEC = lab.sec.values.astype("datetime64[s]").astype(np.int64)
BAGS30 = R.smit_bags(lab, y1, 30, 0.5)
for m in MS:
    f = OUT / f"E1_ladder_{phase}_m{m}.parquet"
    if f.exists(): log("atla", f.name); continue
    rec = R.Rec(); miss = []
    for ci, c in enumerate(CATS):
        for scen, trX in (("S1", XS), ("S2", XHs)):
            d = pickle.load(open(O5 / f"ladder_{scen}_{c}.pkl", "rb")); te = d["te"].astype(int); Pm = d["P"].astype(np.float64); src = d["src"]
            pred = Pm.argmax(1); pos = {int(g): i for i, g in enumerate(te)}; yte = y1[te]; ts = SEC[te]; srt = np.argsort(ts, kind="stable")
            bags = [(w, np.array([pos[int(g)] for g in ix if int(g) in pos])) for cc, w, ix in BAGS30 if cc == c]
            tr = idx_all[cat_full[idx_all] != c]; Xte = XS[te]
            for rep in range(L.REPS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 400 + ci, m, rep, 7 if scen == "S1" else 8]))
                s = L.pick_blocks(ts[srt], m // 60, rng, [])
                labr = np.concatenate([srt[a:a + 60] for a in s]) if s else np.array([], int)
                rem = np.ones(len(te), bool); rem[labr] = False; hard = None
                if phase == "fit" and rep < L.REPS_RT and len(labr):
                    hard = L.fit_arms(trX[tr], y1[tr], Xte, yte, labr, np.arange(len(te)), SEED + 5 * ci + rep, weighted=True, unweighted=not (m == 1800 and rep < 3), target_only=False)
                L.eval_unit(rec, miss, scen, "30dk_ge50", "birey", m, rep, c, bags, rem, yte, Pm, pred, src, labr, 3, hard)
        log(phase, m, c, "%.0f s" % (time.time() - t0))
    rec.wide().to_parquet(f); pd.DataFrame(miss, columns=["scen", "variant", "level", "m", "rep", "unit", "n_lab", "n_missing", "n_cls"]).to_parquet(OUT / f"E1_ladder_{phase}_m{m}_miss.parquet")
log("BITTI ladder", phase)
