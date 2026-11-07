r"""23_r06_A2.py — R06 A2: S2 (Smit harness 11 cats -> collar) L1, matched-32 features (R03-E mapping table: src/axismap.py). New model; the stored R05 S2 posteriors are used for D17.
Settings as in R05 (500 trees, inner CV GroupKFold(4) 300 trees, seed SEED+ci). Reads: Smit table. Writes: <data root>/interim/r06/A2_S2esl_<cat>.pkl
"""
import sys, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import paths as P
from src import r05lib as R, axismap as AM

SEED = P.SEED; OUT = P.R06; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); cat = lab.Cat_id.values; CATS = sorted(set(cat)); idx = np.where(y1 >= 0)[0]
XHm = AM.harness_to_collar(XH).values.astype(np.float64); XCv = XC.values
for ci, c in enumerate(CATS):
    f = OUT / f"A2_S2esl_{c}.pkl"
    if f.exists(): continue
    te = idx[cat[idx] == c]; tr = idx[cat[idx] != c]
    src = R.source_fit(XHm[tr], y1[tr], cat[tr], 3, SEED + ci)
    Pm = src["model"].predict_proba(XCv[te]); del src["model"]
    pickle.dump(dict(te=te.astype(np.int32), P=Pm.astype(np.float32), src=src), open(f, "wb"))
    print(c, "%.0f s" % (time.time() - t0), flush=True)
print("BITTI A2", flush=True)
