"""Prepares the Smit data for R02: labelled rows only, L2/L1 labels, bag windows.
Reads: interim/smit_anno.parquet, smit_accel.parquet (only the rows inside the label window). Writes nothing.
"""
import numpy as np, pandas as pd, pyarrow.dataset as ds
from config import paths as P

L2 = ["Lying", "Sitting", "Standing", "Walking", "Trotting", "Grooming", "Eating"]
L1 = ["Inaktif", "Lokomosyon", "Bakim"]
L2_TO_L1 = {"Lying": "Inaktif", "Sitting": "Inaktif", "Standing": "Inaktif", "Walking": "Lokomosyon",
            "Trotting": "Lokomosyon", "Grooming": "Bakim", "Eating": "Bakim"}
L1_IDX = np.array([L1.index(L2_TO_L1[c]) for c in L2])  # L2 index -> L1 index

COL_TO_L2 = {"Inactive_Lying.Crouch": "Lying", "Inactive_Lying.Down": "Lying", "Inactive_Lying.Resting": "Lying",
             "Inactive_Sitting.Stationary": "Sitting", "Inactive_Sitting.Up": "Sitting", "Inactive_Sitting.Down": "Sitting",
             "Inactive_Standing.Stationary": "Standing", "Inactive_Standing.Up": "Standing",
             "Active_Walking": "Walking", "Active_Trotting": "Trotting",
             "Maintenance_Grooming": "Grooming", "Maintenance_Scratching": "Grooming",
             "Maintenance_Nutrition.Eating": "Eating"}
NON_USABLE = {"Other_Outofsight", "Other_Other", "Other_Start", "Other_ActigraphOff"}

AXIS = [f"{a}_{s}" for a in "XYZ" for s in ("Mean", "Min", "Max", "Sum", "sd", "Skew", "Kurt")]
VMF = [f"VM_{s}" for s in ("Mean", "Min", "Max", "Sum", "sd", "Skew", "Kurt")]
FEAT32 = AXIS + VMF + ["ODBA30_Mean", "Cor_XY", "Cor_XZ", "Cor_YZ"]
FEAT36 = AXIS + VMF + [f"ODBA{w}_Mean" for w in (15, 30, 45, 60, 90)] + ["Cor_XY", "Cor_XZ", "Cor_YZ"]


def load():
    """Returns: lab (one row per second: Cat_id, sec, usable, y2 (L2 index or -1), ...), XC, XH (feature DataFrames, same order as lab)"""
    a = pd.read_parquet(P.INTERIM / "smit_anno.parquet")
    beh = [c for c in a.columns if c not in ("Timestamp", "Pen", "Cat_id")]
    a["rowsum"] = a[beh].sum(axis=1)
    fine = a[beh].idxmax(axis=1)
    single = a.rowsum == 1
    a["usable"] = single & ~fine.isin(NON_USABLE)           # R01 definition
    a["l2"] = np.where(single, fine.map(COL_TO_L2), None)
    a["y2"] = a.l2.map({c: i for i, c in enumerate(L2)}).fillna(-1).astype(int)
    a["sec"] = a.Timestamp.dt.floor("s")
    a["Cat_id"] = a.Cat_id.astype(str)
    lab = a[["Cat_id", "sec", "usable", "y2"]].copy()
    tmin, tmax = a.sec.min(), a.sec.max()
    dset = ds.dataset(P.INTERIM / "smit_accel.parquet")
    cols = ["Cat_id", "Position", "Timestamp"] + FEAT36
    filt = (ds.field("Timestamp") >= tmin.to_pydatetime()) & (ds.field("Timestamp") <= tmax.to_pydatetime())
    acc = dset.to_table(columns=cols, filter=filt).to_pandas()
    acc["sec"] = acc.Timestamp.dt.floor("s")
    out = {}
    for pos in ("Collar", "Harness"):
        x = acc[acc.Position == pos].drop(columns=["Position", "Timestamp"])
        x["Cat_id"] = x.Cat_id.astype(str)
        m = lab[["Cat_id", "sec"]].merge(x, on=["Cat_id", "sec"], how="left")
        assert len(m) == len(lab) and m[FEAT36].notna().all().all(), "eşleşme/NaN sorunu"
        out[pos] = m[FEAT36].astype("float32").reset_index(drop=True)
    return lab.reset_index(drop=True), out["Collar"], out["Harness"]


def make_windows(lab, minutes, min_cov):
    """As in R01: from each cat's first labelled second, non-overlapping, full windows. Coverage = usable / window seconds.
    Returns: win with the same length as lab (window number, -1 = outside a window) and the list of selected (Cat_id, win)."""
    W = minutes * 60
    win = np.full(len(lab), -1, dtype=int)
    sel = []
    for c, g in lab.groupby("Cat_id"):
        t0, t1 = g.sec.min(), g.sec.max()
        nwin = int(((t1 - t0).total_seconds() + 1) // W)
        off = ((g.sec - t0).dt.total_seconds()).astype(int).values
        w = off // W
        ok = w < nwin
        win[g.index[ok]] = w[ok]
        cnt = np.bincount(w[g.usable.values & ok], minlength=nwin)
        for k in range(nwin):
            if cnt[k] / W >= min_cov:
                sel.append((c, k))
    return win, sel
