r"""Python counterpart of Smit's 1-s feature definitions (R05, gate A).
Source: R01 §4.7 + the R code (third_party/Cat_Accelerometry_zip/.../FeatureEngineering.R).
- sd: R sd (ddof=1). Skew: moments::skewness = m3/m2^1.5 (scipy.stats.skew bias=True). Kurt: moments::kurtosis = m4/m2^2, NOT excess
  (scipy.stats.kurtosis fisher=False, bias=True).
- ODBA30: static component zoo::rollmean(k=30, align="center"). zoo source (R/rollmean.R, rollmean.zoo): ix <- floor((1+k)/2):ceiling(n-k/2);
  rval[j] = mean(x[j..j+k-1]) is assigned to ix[1]. For k=30 the first valid (1-based) index is 15 => for the i-th sample the window is x[i-14 .. i+15] (14 left, 15 right).
  In Smit na.pad=TRUE: the file start (14 samples) and end (15 samples) are NA. Here `partial=True` gives the partial-window mean at the edges (R05 primary).
- Cor: within-second Pearson; if an axis is constant R gives NA -> 0 (the count is returned). Skew/Kurt on a constant window give NaN -> 0 (same as Smit Prep.datasets.R; the count is returned).
"""
import numpy as np, pandas as pd
from scipy.signal import resample_poly

STATS = ("Mean", "Min", "Max", "Sum", "sd", "Skew", "Kurt")
COLS32 = [f"{a}_{s}" for a in ("X", "Y", "Z") for s in STATS] + [f"VM_{s}" for s in STATS] + ["ODBA30_Mean", "Cor_XY", "Cor_XZ", "Cor_YZ"]
D17_COLS = ([f"VM_{s}" for s in ("Mean", "Sum", "Min", "Max", "sd", "Skew", "Kurt")] + ["ODBA30_Mean"] +
            [f"sd_s{i}" for i in (1, 2, 3)] + [f"Kurt_s{i}" for i in (1, 2, 3)] + [f"absCor_s{i}" for i in (1, 2, 3)])


def rollmean_center(x, k=30, partial=True):
    """zoo::rollmean(x, k, align='center') (over the whole signal). Returns (rm, edge_mask). Edge: file start/end, k-1 samples in total."""
    x = np.asarray(x, dtype=np.float64); n = len(x)
    left = (1 + k) // 2 - 1            # zoo: first valid 1-based index floor((1+k)/2) -> 0-based left
    right = k - 1 - left
    rm = np.full(n, np.nan); edge = np.zeros(n, bool)
    if n >= k:
        rm[left:n - right] = np.convolve(x, np.ones(k) / k, mode="valid")
    edge[:left] = True; edge[n - right:] = True
    if partial:
        for i in list(range(0, min(left, n))) + list(range(max(n - right, left), n)):
            lo, hi = max(0, i - left), min(n, i + right + 1)
            rm[i] = x[lo:hi].mean()
    return rm, edge


def _moments(w):
    """w: (W, n). Returns mean, sd(ddof=1), skew, kurt, constant_mask."""
    n = w.shape[1]; mu = w.mean(1); d = w - mu[:, None]
    m2 = (d ** 2).mean(1); m3 = (d ** 3).mean(1); m4 = (d ** 4).mean(1)
    const = w.max(1) == w.min(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = m3 / m2 ** 1.5; ku = m4 / m2 ** 2
    sk[const] = np.nan; ku[const] = np.nan
    sd = np.sqrt((d ** 2).sum(1) / (n - 1)) if n > 1 else np.zeros(len(w))
    return mu, sd, sk, ku, const


def window_features(sig, rm, n=30):
    """sig (T,3) X,Y,Z; rm (T,3) static component (rollmean). floor(T/n) windows from T (the remainder is dropped). Returns (DataFrame COLS32, counter dict)."""
    T = (len(sig) // n) * n; W = T // n
    S = np.asarray(sig[:T], np.float64).reshape(W, n, 3); R = np.asarray(rm[:T], np.float64).reshape(W, n, 3)
    ch = {"X": S[:, :, 0], "Y": S[:, :, 1], "Z": S[:, :, 2]}; ch["VM"] = np.sqrt((S ** 2).sum(2))
    out = {}; cnt = {"skew_na": 0, "kurt_na": 0, "cor_na": 0}
    for a, w in ch.items():
        mu, sd, sk, ku, const = _moments(w)
        cnt["skew_na"] += int(np.isnan(sk).sum()); cnt["kurt_na"] += int(np.isnan(ku).sum())
        out[f"{a}_Mean"] = mu; out[f"{a}_Min"] = w.min(1); out[f"{a}_Max"] = w.max(1); out[f"{a}_Sum"] = w.sum(1)
        out[f"{a}_sd"] = sd; out[f"{a}_Skew"] = np.nan_to_num(sk, nan=0.0); out[f"{a}_Kurt"] = np.nan_to_num(ku, nan=0.0)
    out["ODBA30_Mean"] = np.abs(S - R).sum(2).mean(1)
    for (i, j), nm in (((0, 1), "Cor_XY"), ((0, 2), "Cor_XZ"), ((1, 2), "Cor_YZ")):
        a, b = S[:, :, i], S[:, :, j]; da = a - a.mean(1, keepdims=True); db = b - b.mean(1, keepdims=True)
        den = np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
        bad = (a.max(1) == a.min(1)) | (b.max(1) == b.min(1)) | (den == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = (da * db).sum(1) / den
        c[bad] = 0.0; cnt["cor_na"] += int(bad.sum()); out[nm] = c
    return pd.DataFrame(out)[COLS32], cnt


def signal_features(sig, up=1, down=1, n=30, k=30, partial=True):
    """One uninterrupted chunk: (optional resample_poly) -> DBA (whole chunk) -> windows of n samples.
    Returns (DataFrame, counter, edge_window_mask: windows whose DBA was computed with a partial window)."""
    s = np.asarray(sig, np.float64)
    if (up, down) != (1, 1):
        s = resample_poly(s, up, down, axis=0, padtype="line")
    rm = np.empty_like(s); edge = np.zeros(len(s), bool)
    for c in range(3):
        rm[:, c], e = rollmean_center(s[:, c], k, partial)
    df, cnt = window_features(s, rm, n)
    W = len(df); ew = edge_windows(len(s), n, k)[:W] if W else np.zeros(0, bool)
    return df, cnt, ew


def edge_windows(T, n=30, k=30):
    left = (1 + k) // 2 - 1; right = k - 1 - left
    W = T // n; m = np.zeros(T, bool); m[:left] = True; m[T - right:] = True
    return m[:W * n].reshape(W, n).any(1) if W else np.zeros(0, bool)


def d17(df):
    """The axis-independent 17-feature set (D17). df: DataFrame containing the COLS32 columns."""
    o = pd.DataFrame(index=df.index)
    for s in ("Mean", "Sum", "Min", "Max", "sd", "Skew", "Kurt"): o[f"VM_{s}"] = df[f"VM_{s}"].values
    o["ODBA30_Mean"] = df["ODBA30_Mean"].values
    for nm, cols, fn in (("sd", [f"{a}_sd" for a in "XYZ"], lambda v: v), ("Kurt", [f"{a}_Kurt" for a in "XYZ"], lambda v: v),
                         ("absCor", ["Cor_XY", "Cor_XZ", "Cor_YZ"], np.abs)):
        v = -np.sort(-fn(df[cols].values.astype(np.float64)), axis=1)
        for i in range(3): o[f"{nm}_s{i + 1}"] = v[:, i]
    return o[D17_COLS]
