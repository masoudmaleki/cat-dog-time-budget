"""R07 shared: R06 bag-level data + paired-difference / F-rule helpers (same logic as R06 35_r06_F_tests.py)."""
import numpy as np, pandas as pd
from config import paths as P
from src import stats06 as S6

O5 = P.R05; O6 = P.R06; T = P.TABLES
KEYS = ["scen", "variant", "level", "arm", "m", "rep", "ind", "bag"]


def rd(p): return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def cat_(*xs):
    xs = [x for x in xs if len(x)]
    return pd.concat(xs, ignore_index=True) if xs else pd.DataFrame(columns=KEYS + ["AE"])


def load_E():
    DU = cat_(*[rd(O6 / f"E1_{s}_{v}.parquet") for s, v in (("S3", "-"), ("S4-Dunford", "kapali"), ("S4-Dunford", "acik"))])
    SM = cat_(*[rd(O5 / f"E_S4-Smit_{v}.parquet") for v in ("acik", "kapali")], *[rd(O6 / f"E1_S4-Smit_{v}.parquet") for v in ("acik", "kapali")]).drop_duplicates(KEYS, keep="last")
    return DU, SM


def pairs(df, scen, var, level, a, b, ms):
    """d = AE_b - AE_a (positive: a better); per m (individual-mean Series, n_bag, AE_a, AE_b). CC takes no level filter."""
    d = df[(df.scen == scen) & (df.variant == var)]
    def sel(arm): return d[d.arm == arm] if arm == "CC" else d[(d.arm == arm) & (d.level == level)]
    A_ = sel(a).drop_duplicates(["m", "rep", "ind", "bag"]); B_ = sel(b).drop_duplicates(["m", "rep", "ind", "bag"])
    j = A_[["m", "rep", "ind", "bag", "AE"]].merge(B_[["m", "rep", "ind", "bag", "AE"]], on=["m", "rep", "ind", "bag"], suffixes=("_a", "_b"))
    j["d"] = j.AE_b - j.AE_a; out = {}
    for mm, g in j.groupby("m"):
        if mm in ms: out[int(mm)] = (g.groupby("ind").d.mean(), len(g), float(g.AE_a.mean()), float(g.AE_b.mean()))
    return out


def family(name, items):
    rows = []
    for lab_, (dind, nb, aa, ab) in items:
        rows.append(dict(**lab_, n_bag=nb, AE_a=aa, AE_b=ab, **S6.stat(dind.values)))
    df = S6.family(rows)
    if len(df): df.insert(0, "aile", name)
    return df
