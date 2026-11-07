"""R06 F: decision rule. Individual-level paired differences; exact sign-flip (2^n), Holm within family, |d| >= 0.01 threshold, CI = individual bootstrap (1,000)."""
import itertools
import numpy as np, pandas as pd
from config import paths as P

SEED = P.SEED; THR = 0.01
_SIGNS = {}


def signs(n):
    if n not in _SIGNS: _SIGNS[n] = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    return _SIGNS[n]


def signflip_p(d):
    d = np.asarray(d, float); n = len(d)
    if n == 0: return np.nan
    obs = abs(d.mean()); perm = np.abs(signs(n) @ d / n)
    return float(np.mean(perm >= obs - 1e-12))


def boot_ci(d, nboot=1000):
    d = np.asarray(d, float); n = len(d)
    B = np.random.default_rng(SEED).integers(0, n, size=(nboot, n))
    b = d[B].mean(1); return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def stat(dind):
    """dind: per-individual mean difference (positive = the compared arm is better). Returns dict(n, d, lo, hi, p)."""
    dind = np.asarray(dind, float); dind = dind[~np.isnan(dind)]
    lo, hi = boot_ci(dind) if len(dind) else (np.nan, np.nan)
    return dict(n=int(len(dind)), d=float(dind.mean()) if len(dind) else np.nan, lo=lo, hi=hi, p=signflip_p(dind), pmin=(2.0 / 2 ** len(dind)) if len(dind) else np.nan)


def holm(ps):
    ps = np.asarray(ps, float); k = len(ps); order = np.argsort(ps); adj = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (k - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def category(d, lo, hi, ph):
    if ph < 0.05 and d >= THR: return "KAZANIR"
    if ph < 0.05 and d <= -THR: return "KAYBEDER"
    if lo >= -THR and hi <= THR: return "ESDEGER"
    return "BELIRSIZ"


def family(rows):
    """rows: list of dict(stat output + labels). Applies Holm and adds the category; returns a DataFrame, 'aile_n' = number of tests."""
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["p_holm"] = holm(df.p.values); df["aile_n"] = len(df)
    df["kategori"] = [category(r.d, r.lo, r.hi, r.p_holm) for r in df.itertuples()]
    return df


def paired_ind(a, b, keys=("m", "rep", "ind", "bag"), col="AE"):
    """a, b: bag-level DataFrames (same key); per-individual mean of (a - b). Positive = b better. Returns a Series (index = ind)."""
    k = list(keys)
    j = a[k + [col]].merge(b[k + [col]], on=k, suffixes=("_a", "_b"))
    j["d"] = j[col + "_a"] - j[col + "_b"]
    return j.groupby("ind").d.mean()
