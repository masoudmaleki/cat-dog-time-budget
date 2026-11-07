"""Gate A.1 (Task 08): DM unit test. Synthetic 3 classes (generator from test_quant.py), known prior shift, bag N = 10,000.
Run: python -I tests/test_dm.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from src import quant as Q, quant_dm as DM
from tests.test_quant import sample, SEED, K


def main():
    rng = np.random.default_rng(SEED)
    Xtr, ytr = sample([4000, 4000, 4000], rng)
    mk = lambda: RandomForestClassifier(300, min_samples_leaf=20, random_state=SEED, n_jobs=-1)
    rf = mk().fit(Xtr, ytr)
    oof = cross_val_predict(mk(), Xtr, ytr, cv=StratifiedKFold(4, shuffle=True, random_state=SEED), method="predict_proba")
    H, present = DM.class_cond_hists(oof, ytr, K); prior = np.bincount(ytr, minlength=K) / len(ytr)
    true = np.array([0.7, 0.2, 0.1]); e_dm, e_cc, mx = [], [], []
    for rep in range(10):
        r = np.random.default_rng(SEED + rep)
        Xte, yte = sample(np.round(true * 10000).astype(int), r)
        P = rf.predict_proba(Xte); cc = Q.cc(P.argmax(1), K)
        ph, bound, L = DM.dm(P, H, present, [np.full(K, 1 / K), prior, cc])
        assert abs(ph.sum() - 1) < 1e-9 and (ph >= 0).all()
        e_dm.append(Q.ae(ph, true)); e_cc.append(Q.ae(cc, true)); mx.append(np.abs(ph - true).max())
    print("AE DM %.4f  AE CC %.4f  maks |p_hat-p| DM (10 tekrar, en kötü) %.4f" % (np.mean(e_dm), np.mean(e_cc), max(mx)))
    ok = max(mx) <= 0.02 and np.mean(e_dm) < np.mean(e_cc)
    # K=2 equivalence: with two-column posteriors the same solution as HDy (first column only)
    P2 = np.c_[P[:, 0], 1 - P[:, 0]]; H2, pr2 = DM.class_cond_hists(np.c_[oof[:, 0], 1 - oof[:, 0]], (ytr == 0).astype(int) * 0 + (ytr != 0).astype(int) * 1, 2)
    p2, _, _ = DM.dm(P2, H2, pr2, [np.full(2, .5)])
    grid = np.linspace(0, 1, 2001); t = DM.hist_norm(P2[:, 0])
    Lg = [DM.hd(t, g * H2[0, 0] + (1 - g) * H2[0, 1]) for g in grid]
    print("K=2: DM %.4f, ızgara HDy %.4f" % (p2[0], grid[int(np.argmin(Lg))]))
    ok &= abs(p2[0] - grid[int(np.argmin(Lg))]) < 2e-3
    print("TEST", "GEÇTİ" if ok else "KALDI"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
