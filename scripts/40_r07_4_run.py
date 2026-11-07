r"""40_r07_4_run.py — R07 §4: HGB / LR with S1, S2, S3, S4-Smit (open/closed), S4-Dunford (open/closed), S1 APP120 + natural 2-min bridge. L1, D17.
Usage: python -I scripts/40_r07_4_run.py HGB|LR. Dunford = v2 (R06). Bags as in R05/R06. APP seed as in R05 ([SEED,600,ci]).
Reads: Smit table, features/dunford_v2_1s.parquet, dog_aneck_1s.parquet. Writes: <data root>/interim/r07/bags_<clf>.parquet, results/tables/r07_4_cls_<clf>.csv"""
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from config import paths as P
from src import r05lib as R, r07clf as C, smit_data as D

kind = sys.argv[1]; SEED = P.SEED; OUT = P.INTERIM / "r07"; OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
LOG = open(P.LOGS / f"r07_4_{kind}.log", "a", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


lab, y1, XC, XH, XC17, XH17 = R.load_smit(); cat = lab.Cat_id.values; CATS = sorted(set(cat)); idx = np.where(y1 >= 0)[0]
XS, XHs = XC17.values, XH17.values; BAGS30 = R.smit_bags(lab, y1, 30, 0.5)
dun, Xd = R.load_dunford(v2=True); Xd = Xd.values; yd = dun.y1.values; dcat = dun.cat.values; DC = sorted(set(dcat))
dog, Xg = R.load_dog(); keep = dog.y1.values < 2; Xg = Xg.values[keep]; yg = dog.y1.values[keep]; gid = dog.DogID.values[keep]
rec = R.Rec(); cls_rows = []; ece_store = {}


def est(P_, src, K): return R.estimates(P_, src, K)


def clf_row(scen, var, ind, yt, P_):
    pred = P_.argmax(1); cls_rows.append(dict(clf=kind, scen=scen, variant=var, ind=ind, n=len(yt), macroF1=f1_score(yt, pred, labels=np.unique(yt), average="macro", zero_division=0), acc=float((pred == yt).mean())))
    ece_store.setdefault((scen, var), []).append((yt, P_))


# ---------------- S1, S2 (LOCO) + APP bridge
win2, sel2 = D.make_windows(lab, 2, 0.5)
for ci, c in enumerate(CATS):
    te = idx[cat[idx] == c]; tr = idx[cat[idx] != c]; pos = {int(g): i for i, g in enumerate(te)}
    for scen, XT in (("S1", XS), ("S2", XHs)):
        f = OUT / f"{kind}_{scen}_{c}.pkl"
        if f.exists(): d = pickle.load(open(f, "rb"))
        else:
            src = C.source_fit(kind, XT[tr], y1[tr], cat[tr], 3); Pm = C.pad(src.pop("model"), XS[te], 3)
            d = dict(P=Pm, src=src); pickle.dump(d, open(f, "wb"))
        Pm, src = d["P"], d["src"]; clf_row(scen, "-", c, y1[te], Pm)
        for cc, w, ix in BAGS30:
            if cc != c: continue
            ii = np.array([pos[int(g)] for g in ix]); p = R.truth(y1[ix], 3)
            for arm, ph in est(Pm[ii], src, 3).items(): rec.add(scen, "30dk_ge50", "-", arm, 0, 0, c, w, len(ii), p, ph)
        if scen == "S1":
            yte = y1[te]; cl = {k: np.where(yte == k)[0] for k in range(3)}
            bags = C.app_draw(cl, 3, 50, 120, np.random.default_rng(np.random.SeedSequence([SEED, 600, ci])))
            for b, bg in enumerate(bags):
                if bg is None: continue
                rr, p = bg
                for arm, ph in est(Pm[rr], src, 3).items(): rec.add("S1-kopru", "APP120", "-", arm, 0, 0, c, b, 120, p, ph)
            for w in [w for (cc, w) in sel2 if cc == c]:
                ix = np.where((cat == c) & (win2 == w) & (y1 >= 0))[0]; ii = np.array([pos[int(g)] for g in ix if int(g) in pos])
                if len(ii) == 0: continue
                p = R.truth(yte[ii], 3)
                for arm, ph in est(Pm[ii], src, 3).items(): rec.add("S1-kopru", "dogal2dk", "-", arm, 0, 0, c, w, len(ii), p, ph)
    log("LOCO", c, "%.0f s" % (time.time() - t0))

# ---------------- S3: Smit collar -> Dunford (v2)
f = OUT / f"{kind}_src3.pkl"
if f.exists(): d3 = pickle.load(open(f, "rb"))
else:
    src3 = C.source_fit(kind, XS[idx], y1[idx], cat[idx], 3); d3 = dict(P=C.pad(src3.pop("model"), Xd, 3), src=src3); pickle.dump(d3, open(f, "wb"))
for c in DC:
    ii = np.where(dcat == c)[0]; clf_row("S3", "-", c, yd[ii], d3["P"][ii])
    if len(ii) < 60: continue
    p = R.truth(yd[ii], 3)
    for arm, ph in est(d3["P"][ii], d3["src"], 3).items(): rec.add("S3", "kedi", "-", arm, 0, 0, c, 0, len(ii), p, ph)
log("S3", "%.0f s" % (time.time() - t0))

# ---------------- S4: dog ANeck (K=2) -> Smit / Dunford
f = OUT / f"{kind}_src4.pkl"
if f.exists(): d4 = pickle.load(open(f, "rb"))
else:
    s4 = C.source_fit(kind, Xg, yg, gid, 2); m4 = s4.pop("model"); d4 = dict(Ps=C.pad(m4, XS[idx], 2), Pd=C.pad(m4, Xd, 2), src=s4); pickle.dump(d4, open(f, "wb"))
pos_idx = {int(g): i for i, g in enumerate(idx)}
for var in ("kapali", "acik"):
    Kt = 2 if var == "kapali" else 3
    for c in DC:
        ii = np.where((dcat == c) & ((yd < 2) if var == "kapali" else True))[0]
        if len(ii) == 0: continue
        clf_row("S4-Dunford", var, c, yd[ii], np.hstack([d4["Pd"][ii], np.zeros((len(ii), 1))]) if var == "acik" else d4["Pd"][ii])
        if len(ii) < 60: continue
        p = R.truth(yd[ii], Kt)
        for arm, ph in est(d4["Pd"][ii], d4["src"], 2).items(): rec.add("S4-Dunford", var, "-", arm, 0, 0, c, 0, len(ii), p, R.pad3(ph, Kt))
    bags = R.smit_bags(lab, y1, 30, 0.5, mask=(y1 < 2) if var == "kapali" else None)
    for c, w, ix in bags:
        ii = np.array([pos_idx[int(g)] for g in ix]); p = R.truth(y1[ix], Kt)
        for arm, ph in est(d4["Ps"][ii], d4["src"], 2).items(): rec.add("S4-Smit", "30dk_ge50-" + var, "-", arm, 0, 0, c, w, len(ii), p, R.pad3(ph, Kt))
    for c in CATS:
        ix = np.where((cat == c) & (y1 >= 0) & ((y1 < 2) if var == "kapali" else True))[0]; ii = np.array([pos_idx[int(g)] for g in ix])
        clf_row("S4-Smit", var, c, y1[ix], np.hstack([d4["Ps"][ii], np.zeros((len(ii), 1))]) if var == "acik" else d4["Ps"][ii])
log("S4", "%.0f s" % (time.time() - t0))
rec.wide().to_parquet(OUT / f"bags_{kind}.parquet")
cd = pd.DataFrame(cls_rows)
ece = {f"{s}|{v}": C.ece_top(np.vstack([p for _, p in L]), np.concatenate([y for y, _ in L])) for (s, v), L in ece_store.items()}
cd["ECE_pooled"] = [ece[f"{r.scen}|{r.variant}"] for r in cd.itertuples()]
cd.to_csv(P.TABLES / f"r07_4_cls_{kind}.csv", index=False); log("BITTI", kind, "%.0f s" % (time.time() - t0))
