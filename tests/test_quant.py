"""Unit test: synthetic 3 classes, known prior shift. Run: python -I tests/test_quant.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from src import quant as Q

SEED = 20261105
K = 3
MEANS = np.array([[0, 0], [1.2, 0.0], [0.6, 1.1]]) * 1.8   # with scale 1.0 the RF calibration residual is ~0.02-0.04 (report R02 section 7)


def sample(n_per_class, rng):
    X = np.vstack([rng.normal(MEANS[k], 1.0, size=(n, 2)) for k, n in enumerate(n_per_class)])
    y = np.concatenate([np.full(n, k) for k, n in enumerate(n_per_class)])
    return X, y


def main():
    rng = np.random.default_rng(SEED)
    Xtr, ytr = sample([4000, 4000, 4000], rng)                  # training prior: 1/3,1/3,1/3
    rf = RandomForestClassifier(300, min_samples_leaf=20, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    oof_P = cross_val_predict(RandomForestClassifier(300, min_samples_leaf=20, random_state=SEED, n_jobs=-1),
                              Xtr, ytr, cv=StratifiedKFold(4, shuffle=True, random_state=SEED), method="predict_proba")
    oof_pred = oof_P.argmax(1)
    C = Q.conf_matrix(ytr, oof_pred, K); Cp = Q.soft_conf_matrix(ytr, oof_P, K)
    prior = np.bincount(ytr, minlength=K) / len(ytr)
    true_prev = np.array([0.7, 0.2, 0.1])                         # known prior shift
    errs = {m: [] for m in ("CC", "ACC", "PCC", "PACC", "EMQ")}
    for rep in range(20):
        r = np.random.default_rng(SEED + rep)
        Xte, yte = sample(np.round(true_prev * 3000).astype(int), r)
        P = rf.predict_proba(Xte)
        est = {"CC": Q.cc(P.argmax(1), K), "PCC": Q.pcc(P)}
        est["ACC"] = Q.acc(est["CC"], C); est["PACC"] = Q.pacc(est["PCC"], Cp)
        est["EMQ"] = Q.emq(P, prior)[0]
        for m, v in est.items():
            errs[m].append(np.abs(v - true_prev).max())
    mean = {m: float(np.mean(v)) for m, v in errs.items()}
    print("maks |p_hat-p| ortalaması (20 tekrar):", {m: round(v, 4) for m, v in mean.items()})
    ok = True
    for m in ("ACC", "PACC", "EMQ"):
        c1 = mean[m] < mean["CC"]; c2 = mean[m] <= 0.02
        print(f"{m}: CC'den iyi={c1}, <=0.02={c2}")
        ok &= c1 and c2
    # simplex checks
    assert abs(Q.acc(np.array([.5, .3, .2]), np.eye(3)).sum() - 1) < 1e-9
    assert np.allclose(Q.emq(np.full((10, 3), 1 / 3), np.ones(3) / 3)[0], 1 / 3)
    assert Q.rae(np.array([.5, .5]), np.array([.5, .5]), 100) == 0
    print("TEST", "GEÇTİ" if ok else "KALDI")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
