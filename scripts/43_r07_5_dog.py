r"""43_r07_5_dog.py — R07 §5: within-study repeat in dogs (n = 45 dogs), ANeck, single-label seconds only.
Configurations: label K-L2 (5 classes) / K-L1 (2 classes), features 32 (primary) / D17 (sensitivity); S2-dog: ABack model -> ANeck, K-L1, D17.
Split GroupKFold(5) (group = dog), inner CV GroupKFold(4), RF 500 (inner 300), seed SEED + fold (R05 source_fit).
Bags: (i) dog x recording (N >= 60); (ii) natural 2 min (coverage >= 50%); (iii) APP 120 s, 50 bags per dog (Dirichlet, R05-F protocol).
Arms: N-target, CC, PCC, ACC, PACC, EMQ. delta = 1/2 ||p - pi_source||_1 (pi_source = class proportions of the training dogs).
Reads: features/dog_aneck_1s.parquet, dog_aback_1s_v1.parquet. Writes: <data root>/interim/r07/dog_bags.parquet, results/tables/r07_5_*.csv"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from config import paths as P
from src import r05lib as R, quant as Q, features_smit as F, r07clf as C

SEED = P.SEED; OUT = P.INTERIM / "r07"; OUT.mkdir(parents=True, exist_ok=True); T = P.TABLES; t0 = time.time()
LOG = open(P.LOGS / "r07_5_dog.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


NK = pd.read_parquet(P.DOG_FEAT)
NB = pd.read_parquet(P.FEATURES / "dog_aback_1s_v1.parquet")
NK = NK.reset_index(drop=True)
M = NK[["DogID", "TestNum", "sec", "beh", "y1"]].merge(NB[["DogID", "TestNum", "sec"] + F.COLS32], on=["DogID", "TestNum", "sec"], how="left", validate="1:1")
assert not M[F.COLS32].isna().any().any()
X_neck32 = NK[F.COLS32].values.astype(np.float64); X_back32 = M[F.COLS32].values.astype(np.float64)
X_neck17 = F.d17(NK).values.astype(np.float64); X_back17 = F.d17(M).values.astype(np.float64)
L2N = ["Lying chest", "Sitting", "Standing", "Walking", "Trotting"]
beh = NK.beh.values; dogid = NK.DogID.values; rec_ = NK.TestNum.values; sec = NK.sec.values
# labels
y_L2 = np.array([L2N.index(b) if b in L2N else -1 for b in beh]); y_L1 = np.where(NK.y1.values == 2, -1, NK.y1.values)   # Eating (1 window) excluded
DOGS = sorted(set(dogid[(y_L1 >= 0) | (y_L2 >= 0)]))


def prop(y, K): return np.bincount(y, minlength=K) / max(len(y), 1)


def bags_for(y, K, units_idx):
    """units_idx: usable (y>=0) row indices (all dogs). Returns dict dog -> {'kayit': [(id, rows)], 'dogal2': [...]}."""
    out = {}
    for d in DOGS:
        r = units_idx[dogid[units_idx] == d]; out[d] = dict(kayit=[], dogal2=[])
        for t in np.unique(rec_[r]):
            rr = r[rec_[r] == t]
            if len(rr) >= 60: out[d]["kayit"].append((int(t), rr))
            w = sec[rr] // 120
            for k in np.unique(w):
                q = rr[w == k]
                if len(q) >= 60: out[d]["dogal2"].append((f"{t}_{k}", q))
    return out


def app_bags(y, K, rows, di, tag):
    cl = {k: rows[y[rows] == k] for k in range(K)}
    if any(len(v) == 0 for v in cl.values()):    # class missing -> the Dirichlet draw is rejected in that class; draw only over the available classes
        pass
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 700, tag, di])); out = []; tries = 0; acc = 0
    for b in range(50):
        ok = False
        for _ in range(1000):
            tries += 1; p = rng.dirichlet(np.ones(K)); n = C.largest_remainder(p, 120)
            if all(n[k] <= len(cl[k]) for k in range(K)): ok = True; break
        if ok: acc += 1; out.append((b, np.concatenate([rng.choice(cl[k], n[k], replace=False) for k in range(K)])))
    return out, tries, acc


rows_out = []; app_stats = []; skips = []
CONFIGS = [("Kopek-ici", "K-L2", "32", y_L2, 5, X_neck32, X_neck32), ("Kopek-ici", "K-L2", "D17", y_L2, 5, X_neck17, X_neck17),
           ("Kopek-ici", "K-L1", "32", y_L1, 2, X_neck32, X_neck32), ("Kopek-ici", "K-L1", "D17", y_L1, 2, X_neck17, X_neck17),
           ("S2-kopek", "K-L1", "D17", y_L1, 2, X_back17, X_neck17)]
for ci_, (scen, space, feat, y, K, XS_, XT_) in enumerate(CONFIGS):
    use = np.where(y >= 0)[0]; ud = np.array(sorted(set(dogid[use]))); bagset = bags_for(y, K, use)
    # dog budgets (N-target)
    budg = {d: prop(y[use[dogid[use] == d]], K) for d in ud}
    folds = list(GroupKFold(5).split(use, groups=dogid[use]))
    for fi, (tri, tei) in enumerate(folds):
        tr = use[tri]; te = use[tei]
        src = R.source_fit(XS_[tr], y[tr], dogid[tr], K, SEED + 10 * ci_ + fi); model = src.pop("model")
        Pm = np.zeros((len(y), K)); Pm[te] = R.pad_proba(model, XT_[te], K); pred = Pm.argmax(1)
        pi = prop(y[tr], K)
        for d in sorted(set(dogid[te])):
            nh = np.mean([budg[x] for x in ud if x != d], axis=0)
            units = [("kayit", f"{t}", rr) for t, rr in bagset[d]["kayit"]] + [("dogal2", f"{t}", rr) for t, rr in bagset[d]["dogal2"]]
            ab, tries, acc = app_bags(y, K, use[dogid[use] == d], K, d + 1000 * ci_)
            app_stats.append(dict(scen=scen, space=space, feat=feat, dog=int(d), kabul=acc, deneme=tries, kabul_orani=acc / tries))
            units += [("APP120", str(b), rr) for b, rr in ab]
            for btype, bid, rr in units:
                n = len(rr); p = prop(y[rr], K); delta = 0.5 * np.abs(p - pi).sum()
                es = R.estimates(Pm[rr], src, K); es["N-hedef"] = nh
                for arm, ph in es.items():
                    rows_out.append((scen, space, feat, btype, arm, int(d), bid, n, Q.ae(ph, p), delta))
        log(scen, space, feat, "katman", fi, "%.0f s" % (time.time() - t0))
B = pd.DataFrame(rows_out, columns=["scen", "space", "feat", "btype", "arm", "dog", "bag", "N", "AE", "delta"])
B.to_parquet(OUT / "dog_bags.parquet"); pd.DataFrame(app_stats).to_csv(T / "r07_5_app_acceptance.csv", index=False)
json.dump(dict(n_dog={f"{c[0]}|{c[1]}|{c[2]}": int(B[(B.scen == c[0]) & (B.space == c[1]) & (B.feat == c[2])].dog.nunique()) for c in CONFIGS}, sure_s=time.time() - t0), open(T / "r07_5_info.json", "w"), indent=1)
log("BITTI", "%.0f s" % (time.time() - t0))
