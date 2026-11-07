r"""50_r08_run.py <scenario|hepsi> — R08 B/C: CC, PACC (from stored posteriors), DM, DR-APP/DR-dogal (q and h descriptors). No new RF.
Reads: R05–R07 posteriors (r05, r06, r02, r03), r08/dog_*.pkl. Writes: <data root>/interim/r08/run_<scenario>.pkl (bag rows, meta, diagnostics, a); no per-fold checkpoint (scenarios are small)."""
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r08scen as S, r08lib as L

OUT = P.R08; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(P.LOGS / "r08_run.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


ctx = S.Ctx(); want = S.SCENS if sys.argv[1] == "hepsi" else sys.argv[1:]
for scen in want:
    t0 = time.time(); R_, M_, D_, I_ = [], [], [], []
    for f in S.folds(ctx, scen):
        r, m, d, i = L.run_fold(f); R_ += r; M_ += m; D_ += d; I_ += i
        log(scen, f["fold"], "satır", len(r), "%.0f s" % (time.time() - t0))
    pickle.dump(dict(bag=pd.DataFrame(R_), meta=pd.DataFrame(M_), diag=pd.DataFrame(D_), info=pd.DataFrame(I_)), open(OUT / f"run_{scen}.pkl", "wb"))
    log("BITTI", scen, "%.0f s" % (time.time() - t0))
a, b = ctx.app(1)[1], None
for code in (1, 2, 3, 4):
    ctx.app(code)[1].assign(kod=code).to_csv(P.TABLES / f"r08_app_train_stats_kod{code}.csv", index=False) if code in ctx._app else None
