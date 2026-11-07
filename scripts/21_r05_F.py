r"""21_r05_F.py — R05 F: APP (artificial prior shift) and bridge check. APP is secondary only.
Dunford APP (S3 and S4-Dunford-closed): 50 bags x 120 s per test cat; proportions Dirichlet(1,..,1) (as many components as classes); a draw exceeding the cat's window count in a class is rejected
(at most 1,000 attempts); sampling without replacement within the bag. Arms: CC, ACC, PACC, EMQ (source) + study-level PACC-target (m=300; repeat for bag b = b mod 20, same label draws as the E script).
Bridge (Smit S1, D17): with the same protocol, 50 APP bags x 120 s per test cat and natural 2-min bags (R02 window rule, coverage >= 50%).
Reads: r05/sources.pkl, r05/ladder_S1_<cat>.pkl, Smit table, features/dunford_1s.parquet. Writes: r05/F_bags.parquet, results/tables/r05_F_app_stats.json
"""
import sys, json, pickle, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, quant as Q, smit_data as D

SEED = P.SEED; OUT = P.R05; T = P.TABLES; t0 = time.time()
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); cat = lab.Cat_id.values; CATS = sorted(set(cat))
dun, Xd = R.load_dunford(); yd = dun.y1.values; dcat = dun.cat.values; DCATS = sorted(set(dcat))
dcid = pd.factorize(dun.chunk.astype(str) + "_" + dun["sub"].astype(str))[0]
S = pickle.load(open(OUT / "sources.pkl", "rb"))
STR_D = [list(a) for a in np.array_split(np.random.default_rng(SEED + 2).permutation(DCATS), 3)]
rec = R.Rec(); stats = {}


def largest_remainder(p, size):
    x = p * size; n = np.floor(x).astype(int); r = size - n.sum()
    if r > 0: n[np.argsort(-(x - n))[:r]] += 1
    return n


def app_draw(cls_rows, K, nbag, size, rng, maxtry=1000):
    """cls_rows: {k: row array}. Returns (bag list [(rows, p_true) | None], total_attempts, accepted)."""
    out = []; tries = 0; acc = 0
    for b in range(nbag):
        ok = False
        for _ in range(maxtry):
            tries += 1; p = rng.dirichlet(np.ones(K)); n = largest_remainder(p, size)
            if all(n[k] <= len(cls_rows[k]) for k in range(K)): ok = True; break
        if ok:
            acc += 1; out.append((np.concatenate([rng.choice(cls_rows[k], n[k], replace=False) for k in range(K)]), n / size))
        else:
            out.append(None)
    return out, tries, acc


def chunk_draw(keep_mask, cats, m, rng):
    ch = {}
    for c in cats:
        rows = np.where((dcat == c) & keep_mask)[0]
        for k in np.unique(dcid[rows]): ch[(c, k)] = rows[dcid[rows] == k]
    pool = list(ch.values()); tot = 0; out = []
    for k in rng.permutation(len(pool)):
        if tot + len(pool[k]) <= m: out.append(pool[k]); tot += len(pool[k])
        if tot == m: break
    return np.concatenate(out) if out else np.array([], int)


# ------------------------------------------------------------------ Dunford APP
for scen, var, vcode, Kt in (("S3", "-", 1, 3), ("S4-Dunford", "kapali", 2, 2)):
    keep = np.ones(len(yd), bool) if scen == "S3" else (yd < 2)
    Pm = S["P3"] if scen == "S3" else S["P4d"]; src = S["src3"] if scen == "S3" else S["src4"]; Km = Pm.shape[1]
    pred = Pm.argmax(1); st = {}
    for c in DCATS:
        rows = np.where((dcat == c) & keep)[0]
        if len(rows) == 0: continue
        cls = {k: rows[yd[rows] == k] for k in range(Kt)}
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 500, vcode, DCATS.index(c)]))
        bags, tries, acc = app_draw(cls, Kt, 50, 120, rng)
        pr = np.array([b[1] for b in bags if b is not None])
        st[c] = dict(kabul=acc, deneme=tries, kabul_orani=acc / tries, atlanan_torba=50 - acc, n_pencere=len(rows),
                     p_ort=(pr.mean(0).round(3).tolist() if len(pr) else None), p_sd=(pr.std(0).round(3).tolist() if len(pr) else None))
        si = [i for i, s in enumerate(STR_D) if c in s][0]; pool = [x for x in DCATS if x not in STR_D[si]]
        wl = {}
        for rep in range(20):
            rng2 = np.random.default_rng(np.random.SeedSequence([SEED, 300 + si, 300, rep, vcode]))
            wl[rep] = chunk_draw(keep, pool, 300, rng2)
        for b, bg in enumerate(bags):
            if bg is None: continue
            rr, p = bg; q = np.bincount(pred[rr], minlength=Km) / len(rr); pc = Pm[rr].mean(0)
            for arm, ph in R.estimates(Pm[rr], src, Km).items():
                if arm == "PCC": continue
                rec.add(scen + "-APP", var, "-", arm, 0, 0, c, b, 120, p, R.pad3(ph, Kt))
            labr = wl[b % 20]
            if len(labr):
                yl = yd[labr]; okl = yl < Km
                Cpt, pres = R.target_C(yl[okl], Pm[labr][okl], src["Cp"], True)
                rec.add(scen + "-APP", var, "calisma", "PACC-hedef", 300, b % 20, c, b, 120, p, R.pad3(Q.pacc(pc, Cpt), Kt))
    stats[scen + "|" + var] = st
    log_ = {k: round(v["kabul_orani"], 4) for k, v in st.items()}; print(scen, "kabul orani", log_, flush=True)

# ------------------------------------------------------------------ Bridge: Smit S1
win2, sel2 = D.make_windows(lab, 2, 0.5); ss = set(sel2); st = {}
for ci, c in enumerate(CATS):
    d = pickle.load(open(OUT / f"ladder_S1_{c}.pkl", "rb")); te = d["te"].astype(int); Pm = d["P"].astype(np.float64); src = d["src"]
    pos = {int(g): i for i, g in enumerate(te)}; yte = y1[te]
    cls = {k: np.where(yte == k)[0] for k in range(3)}
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 600, ci]))
    bags, tries, acc = app_draw(cls, 3, 50, 120, rng)
    pr = np.array([b[1] for b in bags if b is not None])
    st[c] = dict(kabul=acc, deneme=tries, kabul_orani=acc / tries, p_ort=pr.mean(0).round(3).tolist(), p_sd=pr.std(0).round(3).tolist())
    for b, bg in enumerate(bags):
        if bg is None: continue
        rr, p = bg
        for arm, ph in R.estimates(Pm[rr], src, 3).items():
            if arm != "PCC": rec.add("S1-kopru", "APP120", "-", arm, 0, 0, c, b, 120, p, ph)
    ws = [w for (cc, w) in sel2 if cc == c]
    for w in ws:
        ix = np.where((cat == c) & (win2 == w) & (y1 >= 0))[0]; ii = np.array([pos[int(g)] for g in ix if int(g) in pos])
        if len(ii) == 0: continue
        p = R.truth(yte[ii], 3)
        for arm, ph in R.estimates(Pm[ii], src, 3).items():
            if arm != "PCC": rec.add("S1-kopru", "dogal2dk", "-", arm, 0, 0, c, w, len(ii), p, ph)
stats["S1-kopru"] = st
json.dump(stats, open(T / "r05_F_app_stats.json", "w"), indent=1, default=float)
rec.wide().to_parquet(OUT / "F_bags.parquet")
print("BİTTİ F", "%.0f s" % (time.time() - t0), flush=True)
