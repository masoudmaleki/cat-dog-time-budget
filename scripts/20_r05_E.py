r"""20_r05_E.py — R05 E: target-labelled arms. m in {60,300,900,1800} target seconds; arms: CC (ref), T-ACC, PACC-target, PACC-shrinkage (20 repeats), retrain and target-only (RF 200 trees, 5 repeats).
Level 'calisma' (study): labels come from the target study's non-test individuals (Dunford 3 folds x 3 cats; Smit 4 folds x 3 cats, seed 20261105). Level 'birey' (individual; S4-Smit only): from the test cat's own 60-s blocks.
At the individual level the labelled seconds are removed from the evaluation bags; the study-level arms are evaluated on the same remaining seconds (H7 pairing).
Ladder extra arms (S1, S2; individual; m=1800): PACC-target (20 repeats), retrain (3 repeats).
Reads: Smit table, features/*, r05/sources.pkl, r05/ladder_*.pkl. Writes: r05/E_*.parquet (bag level), results/tables/r05_E_missing.csv
Labelled-block rule: in Smit, consecutive (uninterrupted seconds), non-overlapping 60-s blocks; in Dunford, whole chunks (chunk, sub) in random order, total <= m.
"""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, quant as Q

SEED = P.SEED; OUT = P.R05; T = P.TABLES; MS = (60, 300, 900, 1800); REPS = 20; REPS_RT = 5
LOG = open(P.LOGS / "r05_E.log", "a", encoding="utf-8"); t0 = time.time()


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


lab, y1, XC, XH, XC17, XH17 = R.load_smit()
idx_all = np.where(y1 >= 0)[0]; cat_full = lab.Cat_id.values; CATS = sorted(set(cat_full))
SEC = lab.sec.values.astype("datetime64[s]").astype(np.int64)
XS, XHs = XC17.values, XH17.values
dun, Xd = R.load_dunford(); Xd = Xd.values; yd = dun.y1.values; dcat = dun.cat.values; DCATS = sorted(set(dcat))
dcid = pd.factorize(dun.chunk.astype(str) + "_" + dun["sub"].astype(str))[0]
dog, Xg = R.load_dog(); keepd = dog.y1.values < 2; Xg = Xg.values[keepd]; yg = dog.y1.values[keepd]
S = pickle.load(open(OUT / "sources.pkl", "rb"))
BAGS30 = R.smit_bags(lab, y1, 30, 0.5)
STR_S = [list(a) for a in np.array_split(np.random.default_rng(SEED + 1).permutation(CATS), 4)]
STR_D = [list(a) for a in np.array_split(np.random.default_rng(SEED + 2).permutation(DCATS), 3)]
json.dump(dict(smit=STR_S, dunford=STR_D), open(T / "r05_strata.json", "w"))


def pick_blocks(rows_sorted_by_t, t, need, rng, taken):
    """'need' non-overlapping uninterrupted 60-s blocks from one cat. Returns (rows, list of starts)."""
    n = len(t)
    starts = np.where(t[59:] - t[:-59] == 59)[0] if n >= 60 else np.array([], int)
    out = []
    for s in rng.permutation(starts):
        if len(out) == need: break
        if all(abs(s - q) >= 60 for q in taken + out): out.append(int(s))
    return out


class SmitTarget:
    def __init__(self, mask=None, P_=None):
        sel = np.ones(len(idx_all), bool) if mask is None else mask[idx_all]
        self.full = idx_all[sel]; self.pos = {int(g): i for i, g in enumerate(self.full)}
        self.y = y1[self.full]; self.ind = cat_full[self.full]; self.t = SEC[self.full]; self.X = XS[self.full]
        self.P = None if P_ is None else P_[sel]
        self.rows_c = {c: np.where(self.ind == c)[0] for c in CATS}
        self.sorted_c = {c: r[np.argsort(self.t[r], kind="stable")] for c, r in self.rows_c.items()}
        self.bags = {}
        for c, w, ix in R.smit_bags(lab, y1, 30, 0.5, mask=mask):
            self.bags.setdefault(c, []).append((w, np.array([self.pos[int(g)] for g in ix])))

    def draw(self, cats, m, rng):
        need = m // 60; taken = {c: [] for c in cats}; out = []; cand = list(cats)
        while len(out) < need and cand:
            c = cand[rng.integers(len(cand))]; rs = self.sorted_c[c]; t = self.t[rs]
            s = pick_blocks(rs, t, 1, rng, taken[c])
            if not s: cand.remove(c); continue
            taken[c].append(s[0]); out.append(rs[s[0]:s[0] + 60])
        return np.concatenate(out) if out else np.array([], int)


class DunTarget:
    def __init__(self, drop_bakim):
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


def eval_unit(rec, miss, scen, var, level, m, rep, c, bags, rem, y, Pm, pred, src, labrows, Kt, hard=None, tag=""):
    """bags: [(w, rows)] in the target row space; rem: remaining mask (None = all); hard: {arm: hard prediction array (target row space)}."""
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
        if m == 0 or tag == "ref":
            rec.add(scen, var, level, "CC", m, rep, c, w, n, p, R.pad3(q, Kt)); continue
        rec.add(scen, var, level, "CC", m, rep, c, w, n, p, R.pad3(q, Kt))
        rec.add(scen, var, level, "T-ACC", m, rep, c, w, n, p, R.pad3(Q.acc(q, Ct), Kt))
        rec.add(scen, var, level, "PACC-hedef", m, rep, c, w, n, p, R.pad3(Q.pacc(pc, Cpt), Kt))
        rec.add(scen, var, level, "PACC-buzme", m, rep, c, w, n, p, R.pad3(Q.pacc(pc, Cb), Kt))
        if hard:
            for arm, hp in hard.items():
                if hp is not None: rec.add(scen, var, level, arm, m, rep, c, w, n, p, R.pad3(np.bincount(hp[rr], minlength=Kt) / n, Kt))


def fit_arms(Xsrc, ysrc, X, y, labrows, rows_pred, seed):
    """Retrain and target-only (RF 200). Returns {arm: hard prediction array (same space as X; only rows_pred filled)}; None if target-only was skipped."""
    out = {}
    Xl, yl = X[labrows], y[labrows]
    m1 = R.rf(200, seed).fit(np.vstack([Xsrc, Xl]), np.concatenate([ysrc, yl]))
    h = np.full(len(X), -1, int); h[rows_pred] = m1.classes_[m1.predict_proba(X[rows_pred]).argmax(1)]; out["Yeniden-egitim"] = h
    if len(np.unique(yl)) >= 2:
        m2 = R.rf(200, seed).fit(Xl, yl); h2 = np.full(len(X), -1, int); h2[rows_pred] = m2.classes_[m2.predict_proba(X[rows_pred]).argmax(1)]; out["Yalniz-hedef"] = (h2, set(m2.classes_.tolist()))
    else:
        out["Yalniz-hedef"] = None
    return out


def target_only_ok(arm_out, y, rows_c_rem):
    """Return the prediction if every class present in the target cat (in the remaining seconds) is also in the labelled set, otherwise None (skip)."""
    a = arm_out.get("Yalniz-hedef")
    if a is None: return None
    h, cls = a
    return h if set(np.unique(y[rows_c_rem]).tolist()) <= cls else None


# ======================================================================= S3 and S4-Dunford (study level only)
def run_dunford(scen, var, tgt, Pm, src, Xsrc, ysrc, Kt, vcode):
    f = OUT / f"E_{scen}_{var}.parquet"
    if f.exists(): log("atla", f.name); return
    rec = R.Rec(); miss = []; pred = Pm.argmax(1)
    bags = {c: [(0, tgt.rows_c[c])] for c in DCATS if len(tgt.rows_c[c]) >= 60}
    for c in bags:                                       # CC reference (no labels removed)
        n = len(bags[c][0][1]); p = R.truth(tgt.y[bags[c][0][1]], Kt)
        rec.add(scen, var, "calisma", "CC", 0, 0, c, 0, n, p, R.pad3(np.bincount(pred[bags[c][0][1]], minlength=Pm.shape[1]) / n, Kt))
    for si, strat in enumerate(STR_D):
        pool_cats = [c for c in DCATS if c not in strat]; ev = [c for c in strat if c in bags]
        for m in MS:
            for rep in range(REPS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 300 + si, m, rep, vcode]))
                labr = tgt.draw(pool_cats, m, rng)
                if len(labr) == 0: miss.append((scen, var, "calisma", m, rep, f"katman{si}", 0, -1, -1)); continue
                hard = None
                if rep < REPS_RT and ev:
                    rows_pred = np.concatenate([tgt.rows_c[c] for c in ev])
                    fo = fit_arms(Xsrc, ysrc, tgt.X, tgt.y, labr, rows_pred, SEED + 1000 * si + 10 * rep + m % 97)
                for c in ev:
                    hard = None
                    if rep < REPS_RT:
                        hard = {"Yeniden-egitim": fo["Yeniden-egitim"], "Yalniz-hedef": target_only_ok(fo, tgt.y, tgt.rows_c[c])}
                    eval_unit(rec, miss, scen, var, "calisma", m, rep, c, bags[c], None, tgt.y, Pm, pred, src, labr, Kt, hard)
        log(scen, var, "katman", si, "%.0f s" % (time.time() - t0))
    rec.wide().to_parquet(f); pd.DataFrame(miss, columns=["scen", "variant", "level", "m", "rep", "unit", "n_lab", "n_missing", "n_cls"]).to_parquet(OUT / f"E_{scen}_{var}_miss.parquet")


tgt3 = DunTarget(False)
run_dunford("S3", "-", tgt3, S["P3"][tgt3.rows], S["src3"], XS[idx_all], y1[idx_all], 3, 1)
for var in ("kapali", "acik"):
    tg = DunTarget(var == "kapali"); Kt = 2 if var == "kapali" else 3
    run_dunford("S4-Dunford", var, tg, S["P4d"][tg.rows], S["src4"], Xg, yg, Kt, 2 + (var == "acik"))


# ======================================================================= S4-Smit (calisma + birey)
def run_smit_s4(var):
    f = OUT / f"E_S4-Smit_{var}.parquet"
    if f.exists(): log("atla", f.name); return
    Kt = 2 if var == "kapali" else 3; vcode = 2 + (var == "acik")
    tg = SmitTarget(mask=(y1 < 2) if var == "kapali" else None, P_=S["P4s"]); Pm, src = tg.P, S["src4"]; pred = Pm.argmax(1)
    rec = R.Rec(); miss = []; wcache = {}
    for ci, c in enumerate(CATS):
        si = [i for i, s in enumerate(STR_S) if c in s][0]; pool_cats = [x for x in CATS if x not in STR_S[si]]
        for m in MS:
            for rep in range(REPS):
                rngI = np.random.default_rng(np.random.SeedSequence([SEED, 200 + ci, m, rep, vcode]))
                labI = tg.draw([c], m, rngI)
                rem = np.ones(len(tg.y), bool); rem[labI] = False
                key = (si, m, rep)
                if key not in wcache:
                    wcache[key] = tg.draw(pool_cats, m, np.random.default_rng(np.random.SeedSequence([SEED, 100 + si, m, rep, vcode])))
                labW = wcache[key]
                bg = tg.bags.get(c, [])
                hardI = hardW = None
                if rep < REPS_RT:
                    fo = fit_arms(Xg, yg, tg.X, tg.y, labI, tg.rows_c[c], SEED + 7 * ci + 10 * rep + m % 89)
                    hardI = {"Yeniden-egitim": fo["Yeniden-egitim"], "Yalniz-hedef": target_only_ok(fo, tg.y, tg.rows_c[c][rem[tg.rows_c[c]]])}
                    kw = ("fw", si, m, rep)
                    if kw not in wcache:
                        ev = [x for x in STR_S[si]]; rows_pred = np.concatenate([tg.rows_c[x] for x in ev])
                        wcache[kw] = fit_arms(Xg, yg, tg.X, tg.y, labW, rows_pred, SEED + 31 * si + 10 * rep + m % 83)
                    fw = wcache[kw]
                    hardW = {"Yeniden-egitim": fw["Yeniden-egitim"], "Yalniz-hedef": target_only_ok(fw, tg.y, tg.rows_c[c][rem[tg.rows_c[c]]])}
                eval_unit(rec, miss, "S4-Smit", var, "birey", m, rep, c, bg, rem, tg.y, Pm, pred, src, labI, Kt, hardI)
                # study level: on the same remaining seconds; the CC reference is the same as the individual-level row (no duplicates)
                r2 = R.Rec()
                eval_unit(r2, miss, "S4-Smit", var, "calisma", m, rep, c, bg, rem, tg.y, Pm, pred, src, labW, Kt, hardW)
                rec.rows += [r for r in r2.rows if r[3] != "CC"]
        log("S4-Smit", var, c, "%.0f s" % (time.time() - t0))
    rec.wide().to_parquet(f); pd.DataFrame(miss, columns=["scen", "variant", "level", "m", "rep", "unit", "n_lab", "n_missing", "n_cls"]).to_parquet(OUT / f"E_S4-Smit_{var}_miss.parquet")


for var in ("kapali", "acik"): run_smit_s4(var)


# ======================================================================= Ladder extra arms (S1, S2; individual; m=1800)
def run_ladder_extra():
    f = OUT / "E_ladder_extra.parquet"
    if f.exists(): log("atla", f.name); return
    rec = R.Rec(); miss = []; m = 1800
    tg = SmitTarget(); full_pos = tg.pos
    for ci, c in enumerate(CATS):
        for scen, trX in (("S1", XS), ("S2", XHs)):
            d = pickle.load(open(OUT / f"ladder_{scen}_{c}.pkl", "rb")); te = d["te"].astype(int); Pm = d["P"].astype(np.float64); src = d["src"]
            pred = Pm.argmax(1); pos = {int(g): i for i, g in enumerate(te)}
            yte = y1[te]; ts = SEC[te]; srt = np.argsort(ts, kind="stable")
            bags = [(w, np.array([pos[int(g)] for g in ix if int(g) in pos])) for cc, w, ix in BAGS30 if cc == c]
            tr = idx_all[cat_full[idx_all] != c]
            for rep in range(REPS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 400 + ci, m, rep, 7 if scen == "S1" else 8]))
                s = pick_blocks(srt, ts[srt], m // 60, rng, [])
                labr = np.concatenate([srt[a:a + 60] for a in s]) if s else np.array([], int)
                rem = np.ones(len(te), bool); rem[labr] = False
                hard = None
                if rep < 3:
                    m1 = R.rf(200, SEED + 5 * ci + rep).fit(np.vstack([trX[tr], (XS if scen == "S1" else XS)[te][labr]]), np.concatenate([y1[tr], yte[labr]]))
                    h = np.full(len(te), -1, int); h[:] = m1.classes_[m1.predict_proba(XS[te]).argmax(1)]; hard = {"Yeniden-egitim": h}
                eval_unit(rec, miss, scen, "30dk_ge50", "birey", m, rep, c, bags, rem, yte, Pm, pred, src, labr, 3, hard)
        log("ladder-extra", c, "%.0f s" % (time.time() - t0))
    rec.wide().to_parquet(f); pd.DataFrame(miss, columns=["scen", "variant", "level", "m", "rep", "unit", "n_lab", "n_missing", "n_cls"]).to_parquet(OUT / "E_ladder_extra_miss.parquet")


run_ladder_extra()
log("BİTTİ E", "%.0f s" % (time.time() - t0))
