"""R06 shared: the draw/evaluation logic of the R05 E script (same seeds) + weighted retrain and matched target-only.
Source logic: scripts/20_r05_E.py (copy; the R05 draws are reproduced exactly)."""
import numpy as np, pandas as pd
from src import r05lib as R, quant as Q

MS = (60, 300, 900, 1800); REPS = 20; REPS_RT = 5


def pick_blocks(t, need, rng, taken):
    n = len(t)
    starts = np.where(t[59:] - t[:-59] == 59)[0] if n >= 60 else np.array([], int)
    out = []
    for s in rng.permutation(starts):
        if len(out) == need: break
        if all(abs(s - q) >= 60 for q in taken + out): out.append(int(s))
    return out


class SmitTarget:
    def __init__(self, lab, y1, XS, SEC, cats, idx_all, mask=None, P_=None):
        self.cats = cats
        sel = np.ones(len(idx_all), bool) if mask is None else mask[idx_all]
        self.full = idx_all[sel]; self.pos = {int(g): i for i, g in enumerate(self.full)}
        cat_full = lab.Cat_id.values
        self.y = y1[self.full]; self.ind = cat_full[self.full]; self.t = SEC[self.full]; self.X = XS[self.full]
        self.P = None if P_ is None else P_[sel]
        self.rows_c = {c: np.where(self.ind == c)[0] for c in cats}
        self.sorted_c = {c: r[np.argsort(self.t[r], kind="stable")] for c, r in self.rows_c.items()}
        self.bags = {}
        for c, w, ix in R.smit_bags(lab, y1, 30, 0.5, mask=mask):
            self.bags.setdefault(c, []).append((w, np.array([self.pos[int(g)] for g in ix])))

    def draw(self, cats, m, rng):
        need = m // 60; taken = {c: [] for c in cats}; out = []; cand = list(cats)
        while len(out) < need and cand:
            c = cand[rng.integers(len(cand))]; rs = self.sorted_c[c]; t = self.t[rs]
            s = pick_blocks(t, 1, rng, taken[c])
            if not s: cand.remove(c); continue
            taken[c].append(s[0]); out.append(rs[s[0]:s[0] + 60])
        return np.concatenate(out) if out else np.array([], int)


class DunTarget:
    def __init__(self, yd, dcat, dcid, Xd, DCATS, drop_bakim):
        keep = (yd < 2) if drop_bakim else np.ones(len(yd), bool)
        self.rows = np.where(keep)[0]; self.y = yd[self.rows]; self.ind = dcat[self.rows]; self.cid = dcid[self.rows]; self.X = Xd[self.rows]
        self.rows_c = {c: np.where(self.ind == c)[0] for c in DCATS}
        self.chunks_c = {c: {k: np.where(self.cid == k)[0] for k in np.unique(self.cid[self.rows_c[c]])} for c in DCATS}

    def draw(self, cats, m, rng):
        pool = [v for c in cats for v in self.chunks_c[c].values()]; tot = 0; out = []
        for k in rng.permutation(len(pool)):
            if tot + len(pool[k]) <= m: out.append(pool[k]); tot += len(pool[k])
            if tot == m: break
        return np.concatenate(out) if out else np.array([], int)


def eval_unit(rec, miss, scen, var, level, m, rep, c, bags, rem, y, Pm, pred, src, labrows, Kt, hard=None, with_ref=True):
    """Same arms as R05 (CC, T-ACC, PACC-target, PACC-shrinkage) + hard: {arm: hard prediction array}."""
    Km = Pm.shape[1]; yl = y[labrows]; okl = yl < Km; nlab = len(labrows)
    Ct, pres = R.target_C(yl[okl], pred[labrows][okl], src["C"], False)
    Cpt, _ = R.target_C(yl[okl], Pm[labrows][okl], src["Cp"], True)
    lam = nlab / (nlab + 300.0); Cb = R.shrink(Cpt, src["Cp"], pres, lam)
    miss.append((scen, var, level, m, rep, c, nlab, int((~pres).sum()), int(len(pres))))
    for w, rows in bags:
        rr = rows if rem is None else rows[rem[rows]]
        if len(rr) == 0: continue
        p = R.truth(y[rr], Kt); n = len(rr)
        q = np.bincount(pred[rr], minlength=Km) / n; pc = Pm[rr].mean(0)
        rec.add(scen, var, level, "CC", m, rep, c, w, n, p, R.pad3(q, Kt))
        rec.add(scen, var, level, "T-ACC", m, rep, c, w, n, p, R.pad3(Q.acc(q, Ct), Kt))
        rec.add(scen, var, level, "PACC-hedef", m, rep, c, w, n, p, R.pad3(Q.pacc(pc, Cpt), Kt))
        rec.add(scen, var, level, "PACC-buzme", m, rep, c, w, n, p, R.pad3(Q.pacc(pc, Cb), Kt))
        # E3: PACC-target-g: fall back to CC if any class has < 20 s
        cnts = np.bincount(yl[okl], minlength=Km)
        if (cnts < 20).any(): rec.add(scen, var, level, "PACC-hedef-g", m, rep, c, w, n, p, R.pad3(q, Kt))
        else: rec.add(scen, var, level, "PACC-hedef-g", m, rep, c, w, n, p, R.pad3(Q.pacc(pc, Cpt), Kt))
        if hard:
            for arm, hp in hard.items():
                if hp is not None: rec.add(scen, var, level, arm, m, rep, c, w, n, p, R.pad3(np.bincount(hp[rr], minlength=Kt) / n, Kt))


def fit_arms(Xsrc, ysrc, X, y, labrows, rows_pred, seed, weighted=True, unweighted=True, target_only=True):
    """Unweighted retrain (RT-u), weighted retrain (RT-w; source 1, target N_source/m) and matched target-only (YH-esl; missing class = 0, single class = everything is that class).
    Returns {arm: hard prediction array (X space; only rows_pred filled)}."""
    out = {}; Xl, yl = X[labrows], y[labrows]; Xa = np.vstack([Xsrc, Xl]); ya = np.concatenate([ysrc, yl])

    def pr(m_):
        h = np.full(len(X), -1, int); h[rows_pred] = m_.classes_[m_.predict_proba(X[rows_pred]).argmax(1)]; return h
    if unweighted: out["Yeniden-egitim"] = pr(R.rf(200, seed).fit(Xa, ya))
    if weighted:
        w = np.concatenate([np.ones(len(ysrc)), np.full(len(yl), len(ysrc) / len(yl))])
        out["Yeniden-egitim-agirlikli"] = pr(R.rf(200, seed).fit(Xa, ya, sample_weight=w))
    if target_only: out["Yalniz-hedef-esl"] = pr(R.rf(200, seed).fit(Xl, yl))
    return out
