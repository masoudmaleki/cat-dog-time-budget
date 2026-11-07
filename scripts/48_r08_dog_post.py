r"""48_r08_dog_post.py — R08: reproduces the outer/inner-CV posteriors of dog K-L2-32 and K-L1-32 with EXACTLY the same settings and seeds as R07 §5
(R07 stored only AE; the posteriors were not stored). GroupKFold(5) (group = dog), source_fit (RF 500, inner CV GroupKFold(4) 300 trees), seed SEED + 10*ci + fold (K-L2: ci=0, K-L1: ci=2).
Reads: features/dog_aneck_1s.parquet. Writes: <data root>/interim/r08/dog_<K-L1|K-L2>_f<k>.pkl (te, P, tr, src)."""
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from config import paths as P
from src import r05lib as R, features_smit as F
from src import r08lib as L

OUT = P.R08; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
NK, X32, X17, y_L2, y_L1, dogid = L.dog_tables()
for ci_, (space, y, K) in ((0, ("K-L2", y_L2, 5)), (2, ("K-L1", y_L1, 2))):
    use = np.where(y >= 0)[0]
    for fi, (tri, tei) in enumerate(GroupKFold(5).split(use, groups=dogid[use])):
        f = OUT / f"dog_{space}_f{fi}.pkl"
        if f.exists(): continue
        tr = use[tri]; te = use[tei]
        src = R.source_fit(X32[tr], y[tr], dogid[tr], K, P.SEED + 10 * ci_ + fi); model = src.pop("model")
        Pm = R.pad_proba(model, X32[te], K)
        pickle.dump(dict(te=te.astype(np.int32), tr=tr.astype(np.int32), P=Pm, src=src), open(f, "wb"))
        print(space, "katman", fi, "%.0f s" % (time.time() - t0), flush=True)
print("BITTI", "%.0f s" % (time.time() - t0))
