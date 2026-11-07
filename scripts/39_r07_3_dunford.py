r"""39_r07_3_dunford.py — R07 §3: Dunford data quality and validation leakage.
a) Block repetition in seconds with >42 rows: are timestamp + signal exactly identical. b) Comparison with Table A2 (rows/40; with/without duplicates).
c) 7 classes (including shake), 1-s windows (resampled to 30 Hz, R05 primary), 32 features, RF 500 trees: (i) v1 random 60%/40% split of seconds x20, (ii) v2 the same x20, (iii) v2 LOCO.
This is not a reproduction; it only shows the size of the effect of the split unit and of the duplicates.
Reads: dunford_cats.parquet, features/dunford_1s.parquet (v1), dunford_v2_1s.parquet (v2). Writes: results/tables/r07_3_*.csv/json"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from config import paths as P
from src import r05lib as R, features_smit as F

T = P.TABLES; SEED = P.SEED; t0 = time.time()
A2 = dict(Groom=13.08, Feed=66.98, Rest=2491.78, Walk=533.65, Trot=86.85, Run=50.80, Shake=7.15)
out = {}

# ---------------- a) and b)
d = pd.read_parquet(P.DUNFORD_CSV)
h, m, s = d.Time.str.split(":", expand=True).astype(int).values.T
d["sec"] = h * 3600 + m * 60 + s
dt = d.sec - d.sec.shift(1)
d["chunk"] = (~((d.ID == d.ID.shift(1)) & (d.Behaviour == d.Behaviour.shift(1)) & dt.isin([0, 1]))).cumsum().values
A = d[["AccX", "AccY", "AccZ"]].values
g = d.groupby(["chunk", "sec"], sort=False)
cnt = g.size(); big = cnt[cnt > 42]
n_big = len(big); n_blk = n_time_same = n_sig_same = n_rows_removed = 0; keep = np.ones(len(d), bool)
for (ck, sc), n in big.items():
    ix = g.indices[(ck, sc)]; a = A[ix]; times = d.Time.values[ix]; ok = False
    for p_ in range(38, 43):
        if n % p_ == 0 and np.array_equal(a.reshape(n // p_, p_, 3), np.broadcast_to(a[:p_], (n // p_, p_, 3))):
            ok = True; n_blk += 1; n_time_same += int(len(set(times)) == 1); n_sig_same += 1; keep[ix[p_:]] = False; break
out["a"] = dict(saniye_gt42=int(n_big), blok_tekrari=int(n_blk), blok_tekrari_zaman_damgasi_ayni=int(n_time_same), blok_tekrari_sinyal_birebir=int(n_sig_same), blok_olmayan=int(n_big - n_blk),
                kaldirilan_satir=int((~keep).sum()))
lit = d.duplicated(["ID", "Behaviour", "Time", "AccX", "AccY", "AccZ"], keep="first").values
rows = []
for b_, a_ in A2.items():
    x = d.Behaviour == b_
    rows.append(dict(davranis=b_, A2=a_, tum_satir_40=x.sum() / 40, blok_tekrari_haric_40=(x & keep).sum() / 40, literal_yinelenen_haric_40=(x & ~lit).sum() / 40))
B = pd.DataFrame(rows); B.loc[len(B)] = dict(davranis="TOPLAM", A2=3250.28, tum_satir_40=len(d) / 40, blok_tekrari_haric_40=keep.sum() / 40, literal_yinelenen_haric_40=(~lit).sum() / 40)
B.to_csv(T / "r07_3_b_A2.csv", index=False); print(B.to_string()); print(out["a"])


# ---------------- c)
def load(v2):
    df = pd.read_parquet(P.DUNFORD_FEAT_V2 if v2 else P.DUNFORD_FEAT).reset_index(drop=True)
    X = df[F.COLS32].values.astype(np.float64); return df, X


def metr(y, p):
    lb = np.unique(y); return float((y == p).mean()), float(f1_score(y, p, labels=lb, average="macro", zero_division=0))


res = []; dupinfo = []
for ver, v2 in (("v1", False), ("v2", True)):
    df, X = load(v2); y = pd.factorize(df.beh, sort=True)[0]; n = len(y)
    key = None
    if not v2:
        # duplicated window pairs: same cat and same 32-feature vector (9 decimals)
        kk = pd.util.hash_pandas_object(pd.DataFrame(np.round(X, 9)).assign(c=df.cat.values), index=False).values
        gid = pd.factorize(kk)[0]; sz = np.bincount(gid); dupg = np.where(sz > 1)[0]
        out["v1_yinelenen_pencere_grubu"] = int(len(dupg)); out["v1_yinelenen_pencere_sayisi"] = int(sz[dupg].sum())
    for rep in range(20):
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 7, int(v2), rep])); perm = rng.permutation(n); ntr = int(round(0.6 * n)); tr, te = perm[:ntr], perm[ntr:]
        mdl = R.rf(500, SEED + rep).fit(X[tr], y[tr]); p = mdl.predict(X[te]); acc, f1 = metr(y[te], p)
        res.append(dict(dogrulama="rastgele_60_40", surum=ver, rep=rep, n_pencere=n, acc=acc, macroF1=f1))
        if not v2:
            intr = np.zeros(n, bool); intr[tr] = True
            ng = len(dupg); split = 0; pairs_split = 0; pairs_tot = 0
            for gi in dupg:
                mem = np.where(gid == gi)[0]; t_ = intr[mem]; k_ = len(mem); pairs_tot += k_ * (k_ - 1) // 2
                a_ = int(t_.sum()); pairs_split += a_ * (k_ - a_); split += int(0 < a_ < k_)
            dupinfo.append(dict(rep=rep, grup=ng, grup_ayri_dusen=split, cift_toplam=pairs_tot, cift_ayri=pairs_split))
    if v2:
        cats = sorted(set(df.cat)); P_all = np.zeros(n, int); accs = []
        for ci, c in enumerate(cats):
            te = np.where(df.cat.values == c)[0]; tr = np.where(df.cat.values != c)[0]
            mdl = R.rf(500, SEED + ci).fit(X[tr], y[tr]); P_all[te] = mdl.predict(X[te])
            a_, f_ = metr(y[te], P_all[te]); res.append(dict(dogrulama="LOCO_kedi", surum="v2", rep=ci, n_pencere=len(te), acc=a_, macroF1=f_, kedi=c))
        a_, f_ = metr(y, P_all); res.append(dict(dogrulama="LOCO_birlesik", surum="v2", rep=0, n_pencere=n, acc=a_, macroF1=f_))
    print(ver, "%.0f s" % (time.time() - t0), flush=True)
R_ = pd.DataFrame(res); R_.to_csv(T / "r07_3_c_runs.csv", index=False)
pd.DataFrame(dupinfo).to_csv(T / "r07_3_c_dup_split.csv", index=False)
summ = R_.groupby(["dogrulama", "surum"]).agg(n_kosu=("acc", "size"), acc=("acc", "mean"), acc_sd=("acc", "std"), macroF1=("macroF1", "mean"), F1_sd=("macroF1", "std"), F1_min=("macroF1", "min"), F1_max=("macroF1", "max")).reset_index()
summ.to_csv(T / "r07_3_c_summary.csv", index=False); print(summ.to_string())
di = pd.DataFrame(dupinfo); out["v1_ayri_dusen_grup_orani_ort"] = float((di.grup_ayri_dusen / di.grup).mean()); out["v1_ayri_dusen_cift_orani_ort"] = float((di.cift_ayri / di.cift_toplam).mean())
out["v1_ayri_dusen_cift_orani_aralik"] = [float((di.cift_ayri / di.cift_toplam).min()), float((di.cift_ayri / di.cift_toplam).max())]
json.dump(out, open(T / "r07_3.json", "w"), indent=1); print(json.dumps(out, indent=1)); print("sure %.0f s" % (time.time() - t0))
