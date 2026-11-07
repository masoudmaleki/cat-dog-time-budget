r"""30_r06_E4.py — R06 E4: S1-closed reference. Smit collar LOCO; care (Bakim) seconds removed from training and test; 2 classes; D17; CC, ACC, PACC (+PCC, EMQ stored).
Bags identical to S4-Smit-closed (R.smit_bags mask y1<2). Reads: Smit table. Writes: r06/E4_S1kapali_bags.parquet
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import paths as P
from src import r05lib as R

SEED = P.SEED; OUT = P.R06; t0 = time.time()
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); cat = lab.Cat_id.values; CATS = sorted(set(cat)); XS = XC17.values
idx = np.where(y1 >= 0)[0]; idx2 = idx[y1[idx] < 2]
bags = R.smit_bags(lab, y1, 30, 0.5, mask=(y1 < 2))
rec = R.Rec()
for ci, c in enumerate(CATS):
    te = idx2[cat[idx2] == c]; tr = idx2[cat[idx2] != c]
    src = R.source_fit(XS[tr], y1[tr], cat[tr], 2, SEED + ci); Pm = src.pop("model").predict_proba(XS[te])
    for cc, w, ix in bags:
        if cc != c: continue
        ii = np.searchsorted(te, ix); p = R.truth(y1[ix], 2)
        assert (te[ii] == ix).all()
        for arm, ph in R.estimates(Pm[ii], src, 2).items(): rec.add("S1-kapali", "30dk_ge50", "-", arm, 0, 0, c, w, len(ii), p, ph)
    print(c, "%.0f s" % (time.time() - t0), flush=True)
rec.wide().to_parquet(OUT / "E4_S1kapali_bags.parquet"); print("BITTI E4", flush=True)
