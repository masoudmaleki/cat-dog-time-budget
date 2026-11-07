r"""57_r11_dog_rae.py — R11 §A: within-dog K-L1/K-L2 (32 features) × R07 arms (PCC, ACC, PACC, EMQ) bag-level p, p̂ (for RAE).
NO model is trained: reads the stored R08 posteriors (dog_<K-L1|K-L2>_f<k>.pkl); the arms are the same as in R07 §5 (R.estimates), the APP 120-s draws use the same seed as app_bags in R07 43_r07_5_dog.py.
Gate: in every cell d (= AE_CC - AE_arm, dog mean) agrees with r07_5_tests.csv within 1e-4.
Reads: features/dog_aneck_1s.parquet, interim/r08/dog_*.pkl, results/tables/r07_5_tests.csv. Writes: interim/r11/dog_rae_bags.parquet, results/tables/r11_gate.csv"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, quant as Q, r07clf as C, r08lib as L

SEED = P.SEED; OUT = P.INTERIM / "r11"; OUT.mkdir(parents=True, exist_ok=True)
NK, X32, X17, y_L2, y_L1, dogid = L.dog_tables(); rec_ = NK.TestNum.values
ARMS = ["CC", "PCC", "ACC", "PACC", "EMQ"]


def prop(y, K): return np.bincount(y, minlength=K) / max(len(y), 1)


def app_bags(y, K, rows, di, tag):  # same as 43_r07_5_dog.py
    cl = {k: rows[y[rows] == k] for k in range(K)}
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 700, tag, di])); out = []
    for b in range(50):
        ok = False
        for _ in range(1000):
            p = rng.dirichlet(np.ones(K)); n = C.largest_remainder(p, 120)
            if all(n[k] <= len(cl[k]) for k in range(K)): ok = True; break
        if ok: out.append((b, np.concatenate([rng.choice(cl[k], n[k], replace=False) for k in range(K)])))
    return out


rows = []
for ci_, space, y, K in ((0, "K-L2", y_L2, 5), (2, "K-L1", y_L1, 2)):
    bg = L.dog_bags(NK, y); use = np.where(y >= 0)[0]; N = len(y)
    for fi in range(5):
        d = pickle.load(open(P.R08 / f"dog_{space}_f{fi}.pkl", "rb")); te = d["te"].astype(int); src = d["src"]
        Pm = np.zeros((N, K)); Pm[te] = d["P"]
        for dg in sorted(set(dogid[te])):
            units = [("kayit", bid, rr) for bid, rr in bg[dg]["kayit"]] + [("dogal2", bid, rr) for bid, rr in bg[dg]["dogal2"]]
            units += [("APP120", str(b), rr) for b, rr in app_bags(y, K, use[dogid[use] == dg], K, int(dg) + 1000 * ci_)]
            for bt, bid, rr in units:
                n = len(rr); p = prop(y[rr], K); es = R.estimates(Pm[rr], src, K)
                for arm in ARMS:
                    ph = es[arm]; pp = np.full(5, np.nan); hh = np.full(5, np.nan); pp[:K] = p; hh[:K] = ph
                    rows.append((space, bt, arm, int(dg), str(bid), n, K, *pp, *hh, Q.ae(ph, p), Q.rae(ph, p, n)))
B = pd.DataFrame(rows, columns=["space", "btype", "arm", "dog", "bag", "N", "K"] + [f"p{i}" for i in range(5)] + [f"h{i}" for i in range(5)] + ["AE", "RAE"])
B.to_parquet(OUT / "dog_rae_bags.parquet")
T7 = pd.read_csv(P.TABLES / "r07_5_tests.csv"); T7 = T7[(T7.scen == "Kopek-ici") & (T7.feat.astype(str) == "32")]; g = []
for r in T7.itertuples():
    if r.kol not in ARMS or r.kol == "CC": continue
    x = B[(B.space == r.space) & (B.btype == r.btype)]; a = x[x.arm == "CC"].set_index(["dog", "bag"]).AE; b = x[x.arm == r.kol].set_index(["dog", "bag"]).AE
    dd = (a - b).groupby(level=0).mean(); d = float(dd.mean())
    g.append(dict(space=r.space, btype=r.btype, kol=r.kol, n_dog=len(dd), n_bags=len(a), d_new=d, d_r07=r.d, fark=abs(d - r.d), gecti=int(abs(d - r.d) <= 1e-4)))
G = pd.DataFrame(g); G.to_csv(P.TABLES / "r11_gate.csv", index=False); print(G.to_string()); print("GECTI", int(G.gecti.sum()), "/", len(G))
