r"""44_r07_5_analyze.py — R07 §5: dog results, F rule. d = AE_CC − AE_arm (d>0: arm better), individual = dog (bag mean), CI = dog bootstrap (1,000).
With n = 45 dogs the 2^n permutations cannot be enumerated exactly: the sign-flip p is Monte Carlo (200,000 random sign draws, p = (1 + #{|T*| >= |T|}) / (B + 1)); the deviation is stated in the report.
Family = scenario x bag type x label space x features; Holm over 5 tests (PCC, ACC, PACC, EMQ, N-target). H9: APP120 − natural 2-min gain contrast (PACC, ACC; family of 2).
Reads: <data root>/interim/r07/dog_bags.parquet. Writes: results/tables/r07_5_tests.csv, r07_5_delta.csv, r07_5_h9.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import stats06 as S6

SEED = P.SEED; T = P.TABLES; NMC = 200_000
B = pd.read_parquet(P.INTERIM / "r07" / "dog_bags.parquet")
rng = np.random.default_rng(SEED)


def stat_mc(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    lo, hi = S6.boot_ci(d)
    if n <= 20: p = S6.signflip_p(d); how = "kesin"
    else:
        obs = abs(d.mean()); cnt = 0; done = 0
        while done < NMC:
            k = min(20_000, NMC - done); sg = rng.choice([-1.0, 1.0], size=(k, n)); cnt += int((np.abs(sg @ d / n) >= obs - 1e-12).sum()); done += k
        p = (1 + cnt) / (NMC + 1); how = "MC"
    return dict(n=n, d=float(d.mean()), lo=lo, hi=hi, p=float(p), p_yontem=how)


rows = []; dl = []; h9 = []
ARMS = ["PCC", "ACC", "PACC", "EMQ", "N-hedef"]
for (scen, space, feat), g in B.groupby(["scen", "space", "feat"]):
    ind = {}
    for bt, gb in g.groupby("btype"):
        dd = gb.drop_duplicates(["bag", "dog"]).groupby("bag").size()
        d_ = gb.groupby("arm").AE.mean()
        W = gb.groupby(["dog", "arm"]).AE.mean().unstack()           # mean AE per dog
        ind[bt] = W
        dl.append(dict(scen=scen, space=space, feat=feat, btype=bt, n_dog=int(W.shape[0]), n_torba=int(gb[gb.arm == "CC"].shape[0]), AE_CC=float(gb[gb.arm == "CC"].AE.mean()),
                       delta_medyan=float(gb[gb.arm == "CC"].delta.median()), delta_p90=float(gb[gb.arm == "CC"].delta.quantile(0.9)), N_medyan=float(gb[gb.arm == "CC"].N.median())))
        st = []
        for a in ARMS:
            dv = (W["CC"] - W[a]).dropna()
            s = stat_mc(dv.values); nb = int(gb[gb.arm == a].shape[0])
            st.append(dict(scen=scen, space=space, feat=feat, btype=bt, kol=a, n_torba=nb, AE_CC=float(W["CC"][dv.index].mean()), AE_kol=float(W[a][dv.index].mean()), **s))
        df = S6.family(st); rows += df.to_dict("records")
    if "APP120" in ind and "dogal2" in ind:
        for a in ("PACC", "ACC"):
            ga = (ind["APP120"]["CC"] - ind["APP120"][a]); gn = (ind["dogal2"]["CC"] - ind["dogal2"][a]); j = pd.concat([ga, gn], axis=1, keys=["app", "nat"]).dropna()
            s = stat_mc((j.app - j.nat).values); h9.append(dict(scen=scen, space=space, feat=feat, kol=a, d_APP=float(j.app.mean()), d_dogal2=float(j.nat.mean()), **s))
TT = pd.DataFrame(rows); TT.to_csv(T / "r07_5_tests.csv", index=False)
DL = pd.DataFrame(dl); DL.to_csv(T / "r07_5_delta.csv", index=False)
H9 = pd.DataFrame(h9)
out = []
for (sc, sp, ft), g in H9.groupby(["scen", "space", "feat"]):
    f = S6.family(g.to_dict("records")); out.append(f)
H9 = pd.concat(out); H9.to_csv(T / "r07_5_h9.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.4f}".format)
print(DL.to_string()); print(TT[["scen", "space", "feat", "btype", "kol", "n", "n_torba", "AE_CC", "AE_kol", "d", "lo", "hi", "p_holm", "kategori"]].to_string()); print(H9.to_string())
