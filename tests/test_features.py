r"""Gate A item 1: features agree with a hand-written numpy version of the R definition within 1e-9; Kurt/Skew on normal noise; rollmean centring; D17 invariance.
Run: python -I tests/test_features.py
"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import stats
from src import features_smit as F

rng = np.random.default_rng(20261105)
res = {}


def naive_rollmean_zoo(x, k):
    """Literal translation of the index rule of zoo::rollmean.zoo (1-based R indices)."""
    n = len(x); out = np.full(n, np.nan)
    ix = list(range(math.floor((1 + k) / 2), math.ceil(n - k / 2) + 1))      # R: floor((1+k)/2):ceiling(n-k/2)
    rval = [sum(x[j:j + k]) / k for j in range(n - k + 1)]                    # R: cumsum(y)/k, rval[j] = mean(x[j..j+k-1])
    assert len(ix) == len(rval)
    for i1, v in zip(ix, rval): out[i1 - 1] = v
    return out


# --- 1) rollmean centring (k = 30, even): window x[i-14 .. i+15]
x = rng.normal(size=200)
ref = naive_rollmean_zoo(x, 30)
rm, edge = F.rollmean_center(x, 30, partial=False)
assert np.array_equal(np.isnan(ref), np.isnan(rm)), "NaN deseni zoo ile farklı"
assert np.nanmax(np.abs(ref - rm)) < 1e-12
valid = np.where(~np.isnan(rm))[0]
assert valid[0] == 14 and valid[-1] == 200 - 16, (valid[0], valid[-1])        # 0-based: first 14, last n-16 (15 samples on the right)
i = 60; assert abs(rm[i] - x[i - 14:i + 16].mean()) < 1e-12                  # window i-14 .. i+15
res["rollmean_k30_pencere"] = "x[i-14..i+15]"
rmp, edge = F.rollmean_center(x, 30, partial=True)
assert not np.isnan(rmp).any() and edge.sum() == 29
assert abs(rmp[3] - x[0:3 + 16].mean()) < 1e-12 and abs(rmp[-2] - x[200 - 2 - 14:].mean()) < 1e-12
# odd k (zoo uses the same formula): k=5 -> x[i-2..i+2]
r5, _ = F.rollmean_center(x, 5, partial=False); assert abs(r5[10] - x[8:13].mean()) < 1e-12 and np.array_equal(np.isnan(r5), np.isnan(naive_rollmean_zoo(x, 5)))

# --- 2) agreement of the features with the hand-written version (3 windows, non-constant signal)
n = 30; W = 7
sig = rng.normal(0, 0.6, size=(W * n, 3)) + np.array([0.1, -0.7, 0.4])
sig[2 * n:3 * n, 1] = 0.25                 # constant axis -> Skew/Kurt/Cor rules
rm = np.column_stack([F.rollmean_center(sig[:, c], 30, True)[0] for c in range(3)])
df, cnt = F.window_features(sig, rm, n)


def naive(w, r):
    o = {}
    ax = {"X": w[:, 0], "Y": w[:, 1], "Z": w[:, 2], "VM": np.sqrt((w ** 2).sum(1))}
    for a, v in ax.items():
        m = v.mean(); d = v - m; m2 = (d ** 2).mean(); m3 = (d ** 3).mean(); m4 = (d ** 4).mean()
        o[f"{a}_Mean"] = m; o[f"{a}_Sum"] = v.sum(); o[f"{a}_Min"] = v.min(); o[f"{a}_Max"] = v.max()
        o[f"{a}_sd"] = math.sqrt((d ** 2).sum() / (len(v) - 1))
        o[f"{a}_Skew"] = 0.0 if v.max() == v.min() else m3 / m2 ** 1.5
        o[f"{a}_Kurt"] = 0.0 if v.max() == v.min() else m4 / m2 ** 2
    o["ODBA30_Mean"] = np.abs(w - r).sum(1).mean()
    for (i, j), nm in (((0, 1), "Cor_XY"), ((0, 2), "Cor_XZ"), ((1, 2), "Cor_YZ")):
        a, b = w[:, i], w[:, j]
        o[nm] = 0.0 if (a.max() == a.min() or b.max() == b.min()) else float(np.corrcoef(a, b)[0, 1])
    return o


worst = 0.0
for k in range(W):
    o = naive(sig[k * n:(k + 1) * n], rm[k * n:(k + 1) * n])
    for c, v in o.items():
        worst = max(worst, abs(df.loc[k, c] - v))
assert worst < 1e-9, worst
res["oznitelik_maks_mutlak_fark"] = worst
assert cnt["cor_na"] == 2 and cnt["skew_na"] == 1 and cnt["kurt_na"] == 1, cnt   # constant Y window: Cor_XY, Cor_YZ; Y_Skew, Y_Kurt
# scipy counterparts
w = sig[:n, 0]
assert abs(df.loc[0, "X_Skew"] - stats.skew(w, bias=True)) < 1e-12 and abs(df.loc[0, "X_Kurt"] - stats.kurtosis(w, fisher=False, bias=True)) < 1e-12
assert abs(df.loc[0, "X_sd"] - w.std(ddof=1)) < 1e-12

# --- 3) normal noise with 10^6 samples: Kurt 3 ± 0.02, Skew 0 ± 0.02
z = rng.normal(size=(1_000_000, 3)); rz = np.zeros_like(z)
d6, _ = F.window_features(z, rz, 1_000_000)
res["normal_X_Kurt"] = float(d6.loc[0, "X_Kurt"]); res["normal_X_Skew"] = float(d6.loc[0, "X_Skew"])
assert abs(d6.loc[0, "X_Kurt"] - 3) < 0.02 and abs(d6.loc[0, "X_Skew"]) < 0.02, (d6.loc[0, "X_Kurt"], d6.loc[0, "X_Skew"])

# --- 4) D17 axis permutation and sign invariance
base = F.d17(df)
perm = sig[:, [2, 0, 1]] * np.array([-1, 1, -1])
rm2 = np.column_stack([F.rollmean_center(perm[:, c], 30, True)[0] for c in range(3)])
d2, _ = F.window_features(perm, rm2, n)
dd = np.abs(F.d17(d2).values - base.values).max()
assert dd < 1e-9, dd
res["d17_permutasyon_isaret_maks_fark"] = float(dd)
assert list(base.columns) == F.D17_COLS and len(F.D17_COLS) == 17

# --- 5) resample_poly: chunk length and number of windows (Dunford up=3 down=4: 40 Hz -> 30 Hz)
s40 = rng.normal(size=(40 * 12, 3)); d30, c, ew = F.signal_features(s40, 3, 4)
assert len(d30) == 12 and ew.sum() == 2, (len(d30), ew.sum())    # 12 s, DBA with partial windows: first and last window
print("TÜM TESTLER GEÇTİ", res)
