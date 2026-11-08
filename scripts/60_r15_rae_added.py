r"""60_r15_rae_added.py — R15 A2: RAE for the 16 Table 3 cells added after R10/R11 (no model is trained; only stored bag-level outputs are read).
Same method as src/r10rae.py (R10/R11): epsilon = 1/(2N) smoothing (rae_vec), individual-level d_RAE = mean over individuals of (RAE_CC - RAE_arm),
bootstrap CI and seed from src.stats06 (S6.boot_ci via r10rae.boot_ind). CC and arm are taken on the same bags.
Gate (as R11): d_AE recomputed from the bag-level values must equal d in results/supplement/csv/S3_Table3_long.csv (|diff| < 1e-9); otherwise the cell is not written.
Reads: S3_Table3_long.csv, r08_bag_level.csv, r02_bag_level.csv, interim/r11/dog_rae_bags.parquet, Smit/dog label tables (read-only, via src.r08scen.Ctx)
Writes: results/tables/r15_rae_added_cells.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from config import paths as P
from src import r10rae as X, smit_data as D

T = P.TABLES
L3 = pd.read_csv(Path(P.TABLES).parent / "supplement" / "csv" / "S3_Table3_long.csv")   # v4 table (the v4 copy is results/supplement/v4/csv after step B5)
DUN = "Dunford, one sample per cat (≥ 60 s)"; SMIT = "Smit natural 30-min windows (68)"
DOGSETS = ["APP 120-s samples", "dog recordings (≥ 60 s)", "dog natural 2-min windows"]
CELLS = [("S3 [exploratory]", DUN, a) for a in ("DM", "DR-APP", "DR-natural")] + [("S4-Smit-closed", SMIT, a) for a in ("DM", "DR-APP", "DR-natural")] \
    + [("S4-Dunford-closed [exploratory]", DUN, a) for a in ("DM", "DR-APP", "DR-natural")] + [("S1-L2 (collar)", SMIT, "N-target")] \
    + [(f"{k} within-dog", s, "N-target") for k in ("K-L1", "K-L2") for s in DOGSETS]
assert len(CELLS) == 16
pr = X.Pairs(); c = pr.ctx
BT = {"APP 120-s samples": "APP120", "dog recordings (≥ 60 s)": "kayit", "dog natural 2-min windows": "dogal2"}
dog11 = pd.read_parquet(P.INTERIM / "r11" / "dog_rae_bags.parquet")


def prop(y, K): return np.bincount(y, minlength=K) / max(len(y), 1)


def nt_dog(space, btype):
    """N-target for within-dog: mean budget of the other dogs (43_r07_5_dog.py: budg over usable seconds, dogs equal weight)."""
    y, K = (c.dy_L1, 2) if space == "K-L1" else (c.dy_L2, 5); use = np.where(y >= 0)[0]; ud = sorted(set(c.dogid[use]))
    budg = {d: prop(y[use[c.dogid[use] == d]], K) for d in ud}
    g = dog11[(dog11.space == space) & (dog11.btype == btype) & (dog11.arm == "CC")]
    p = g[[f"p{i}" for i in range(K)]].values; N = g.N.values; h = np.array([np.mean([budg[x] for x in ud if x != d], axis=0) for d in g.dog])
    return pd.DataFrame(dict(ind=g.dog.astype(str).values, AE_c=g.AE.values, AE_a=np.abs(h - p).mean(1), RAE_c=g.RAE.values, RAE_a=X.rae_vec(h, p, N)))


def nt_l2():
    """S1-L2 collar N-target: as 32_r06_C_naive.py (7 L2 classes, mean budget of the other cats), N from the stored R08 CC bags."""
    r2 = pr.r02; r2 = r2[(r2.varyant == "va_f32") & (r2.bagdef == "30dk_ge50") & (r2.level == "L2") & (r2.basamak == "S1-tasma") & (r2.tahminci == "CC")]
    K2 = len(D.L2); b2 = {k: np.bincount(c.y2[(c.cat == k) & (c.y2 >= 0)], minlength=K2) / ((c.cat == k) & (c.y2 >= 0)).sum() for k in c.CATS}; rows = []
    for (ind, bag), g in r2.groupby(["kedi", "torba"]):
        g = g.set_index("sinif").reindex(D.L2); pv = g.p.values[None]; hv = g.p_hat.values[None]; N = np.array([pr.NL2[(str(ind), str(bag))]])
        nh = np.mean([b2[x] for x in c.CATS if x != ind], axis=0)[None]
        rows.append(dict(ind=str(ind), AE_c=np.abs(hv - pv).mean(), AE_a=np.abs(nh - pv).mean(), RAE_c=X.rae_vec(hv, pv, N)[0], RAE_a=X.rae_vec(nh, pv, N)[0]))
    return pd.DataFrame(rows)


out = []; fails = []
for sc, bs, arm in CELLS:
    lr = L3[(L3.scenario == sc) & (L3.sample_set == bs) & (L3.arm == arm)]
    assert len(lr) == 1, (sc, bs, arm, len(lr)); lr = lr.iloc[0]
    if arm == "N-target" and sc.endswith("within-dog"): j, note = nt_dog(sc.split(" ")[0], BT[bs]), ""
    elif arm == "N-target": j, note = nt_l2(), ""
    else: j, note = pr.get(sc, bs, arm)
    if j is None or len(j) == 0: fails.append((sc, bs, arm, note or "empty pairing")); continue
    dA = (j.AE_c - j.AE_a).groupby(j.ind).mean(); dR = (j.RAE_c - j.RAE_a).groupby(j.ind).mean(); lo, hi = X.boot_ind(dR.values)
    err = abs(dA.mean() - lr.d)
    if not err < 1e-9: fails.append((sc, bs, arm, f"AE gate failed: recomputed d_AE={dA.mean():.9f} vs S3 d={lr.d:.9f}")); continue
    out.append(dict(scenario=sc, sample_set=bs, arm=arm, category_AE=lr.category, d_AE=lr.d, n_samples=len(j), n_individuals=len(dR), RAE_CC=j.RAE_c.mean(), RAE_arm=j.RAE_a.mean(),
                    d_RAE=dR.mean(), ci_low_RAE=lo, ci_high_RAE=hi, direction_mismatch=bool(lo > 0 and lr.category != "better"), note="computed in round R15"))
R = pd.DataFrame(out); R.to_csv(T / "r15_rae_added_cells.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format)
print(R.drop(columns=["note"]).to_string()); print("computed", len(R), "/ 16; not computable:", fails)
