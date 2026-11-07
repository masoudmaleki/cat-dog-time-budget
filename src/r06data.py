"""R06 data collector: bag-level parquet files (R05 + R06) in a single table. Primary set: Smit 68 bags (30dk_ge50); Dunford (v2) cat bag N >= 60."""
import numpy as np, pandas as pd
from config import paths as P

O5 = P.R05; O6 = P.R06


def rd(p): return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def load_r05():
    parts = [O5 / "r05_ladder_bags.parquet", O5 / "r05_sources_bags.parquet", O5 / "F_bags.parquet", O5 / "E_ladder_extra.parquet"] + sorted(p for p in O5.glob("E_*.parquet") if not p.name.endswith("_miss.parquet") and p.name != "E_ladder_extra.parquet")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def long_probs(A):
    """Not needed to recompute AE; helper for the class vectors: (p, h) arrays."""
    K = A.K.values; p = A[["p0", "p1", "p2"]].values; h = A[["h0", "h1", "h2"]].values
    return p, h, K
