"""Gate A.2 (Task 08): DR unit tests. Run: python -I tests/test_dr.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import ortho_group
from src import quant_dr as DR, quant as Q

SEED = 20261105
from src.quant_dr import PLS_TOL
ok = True


def chk(name, cond, info=""):
    global ok
    ok &= bool(cond); print(("GEÇTİ " if cond else "KALDI ") + name, info)


rng = np.random.default_rng(SEED)
# 1. ilr round trip
for K in (2, 3, 5, 7):
    x = rng.dirichlet(np.ones(K), size=200); err = np.abs(DR.ilr_inv(DR.ilr(x)) - x).max()
    chk(f"ilr gidiş-dönüş K={K}", err < 1e-12, f"maks hata {err:.2e}")
# 2. zero replacement
for K in (2, 3, 5, 7):
    N = 120; p = np.zeros(K); p[0] = 1.0 if K == 2 else 0.6; p[1] = 1 - p[0] if K == 2 else 0.4
    q = DR.zero_replace(p, N); chk(f"sıfır değiştirme K={K}", abs(q.sum() - 1) < 1e-12 and (q > 0).all(), f"toplam-1 {q.sum() - 1:.1e}, min {q.min():.4f}")
# 3. basis invariance (PLSR proportions)
K = 5; n = 200; d = 30
Xb = rng.normal(size=(n, d)); W = rng.normal(size=(d, K)); pb = np.exp(Xb @ W * 0.3); pb /= pb.sum(1, keepdims=True)
Z = DR.ilr(pb); Qm = ortho_group.rvs(K - 1, random_state=SEED); Z2 = Z @ Qm
Xs = (Xb - Xb.mean(0)) / Xb.std(0, ddof=1)
from sklearn.cross_decomposition import PLSRegression
for a in (1, 3, 6):
    m1 = PLSRegression(a, scale=False, tol=PLS_TOL, max_iter=20000).fit(Xs, Z); m2 = PLSRegression(a, scale=False, tol=PLS_TOL, max_iter=20000).fit(Xs, Z2)
    p1 = DR.ilr_inv(m1.predict(Xs)); p2 = DR.ilr_inv(m2.predict(Xs) @ Qm.T)
    chk(f"taban değişmezliği a={a}", np.abs(p1 - p2).max() < 1e-8, f"maks fark {np.abs(p1 - p2).max():.2e}")
# 3b. prediction with a components from a single fit == separate fit
Xt = rng.normal(size=(50, d)); pr = DR.pls_preds_by_a(Xs, Z, (Xt - Xb.mean(0)) / Xb.std(0, ddof=1), 8)
for a in (1, 4, 8):
    m = PLSRegression(a, scale=False, tol=PLS_TOL, max_iter=20000).fit(Xs, Z); e = np.abs(m.predict((Xt - Xb.mean(0)) / Xb.std(0, ddof=1)) - pr[a - 1]).max()
    chk(f"kesik bileşen tahmini a={a}", e < 1e-8, f"maks fark {e:.2e}")
# 4. synthetic bags
K, F, Nb = 3, 4, 300
MU = np.array([[0, 0, 0, 0], [1.5, 0, 1.0, 0], [0, 1.5, 0, 1.0]], float)


def make(nbags, rs):
    D, P_, G = [], [], []
    for b in range(nbags):
        p = rs.dirichlet(np.ones(K)); n = np.floor(p * Nb).astype(int); n[np.argmax(p)] += Nb - n.sum()
        X = np.vstack([rs.normal(MU[k], 1.0, size=(n[k], F)) for k in range(K)])
        D.append(DR.descr_q(X, np.arange(len(X)))); P_.append(n / Nb); G.append(b % 25)
    return np.array(D), np.array(P_), np.array(G)


rs = np.random.default_rng(SEED + 1)
Dtr, Ptr, Gtr = make(500, rs); Dte, Pte, _ = make(200, rs)
mdl = DR.DR().fit(Dtr, Ptr, np.full(len(Ptr), Nb), Gtr)
ae_dr = np.abs(mdl.predict(Dte) - Pte).mean(); ae_base = np.abs(Ptr.mean(0) - Pte).mean()
chk("sentetik: DR AE < taban AE / 2", ae_dr < ae_base / 2, f"DR {ae_dr:.4f}, taban {ae_base:.4f}, a={mdl.a}")
print("TEST", "GEÇTİ" if ok else "KALDI"); sys.exit(0 if ok else 1)
