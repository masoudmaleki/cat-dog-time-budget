r"""49_r08_gateA.py — R08 gate A item 3: reproduction from the stored posteriors (within 1e-4).
S1 L1 primary set: CC AE 0.0260; PACC − CC −0.0240 (d = AE_CC − AE_PACC, individual mean). S1-L2 (32): CC AE 0.0837. K-L2-32 dog x recording CC AE: r07_5_tests.csv.
In addition: the S1 APP120 evaluation draws are identical to F_bags (R05). Reads: r05/ladder_S1_*.pkl, r02/raw_va_f32.pkl, r08/dog_*.pkl, results/tables/r07_5_tests.csv, r05/F_bags.parquet. Writes: results/tables/r08_gateA.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import quant as Q, r08scen as S, r08lib as L

ctx = S.Ctx(); res = {}; rows = []
for scen in ("S1", "S1-L2", "K-L2", "K-L1"):
    for f in S.folds(ctx, scen):
        K = f["K"]
        for bt, lst in f["bags"].items():
            for ind, bid, rr in lst:
                Pb = f["Pt"][rr]; p = L.truth(f["yt"], rr, K); cc = Q.cc(Pb.argmax(1), K); pa = Q.pacc(Pb.mean(0), f["Cp"])
                rows.append((scen, bt, str(ind), str(bid), len(rr), Q.ae(cc, p), Q.ae(pa, p)))
B = pd.DataFrame(rows, columns=["scen", "bt", "ind", "bag", "N", "AE_CC", "AE_PACC"])
g = B[(B.scen == "S1") & (B.bt == "30dk")]; di = (g.AE_CC - g.AE_PACC).groupby(g.ind).mean()
res["S1_30dk"] = dict(n_bag=len(g), AE_CC_torba_ort=float(g.AE_CC.mean()), AE_CC_birey_ort=float(g.groupby("ind").AE_CC.mean().mean()), d_PACC=float(di.mean()), hedef_CC=0.0260, hedef_d=-0.0240)
g = B[(B.scen == "S1-L2") & (B.bt == "30dk")]
res["S1L2_30dk"] = dict(n_bag=len(g), AE_CC_torba_ort=float(g.AE_CC.mean()), AE_CC_birey_ort=float(g.groupby("ind").AE_CC.mean().mean()), hedef_CC=0.0837)
T7 = pd.read_csv(P.TABLES / "r07_5_tests.csv")
for sp in ("K-L2", "K-L1"):
    for bt in ("kayit", "dogal2"):
        g = B[(B.scen == sp) & (B.bt == bt)]; t = T7[(T7.scen == "Kopek-ici") & (T7.space == sp) & (T7.feat == "32") & (T7.btype == bt)].iloc[0]
        res[f"{sp}_{bt}"] = dict(n_bag=len(g), n_dog=int(g.ind.nunique()), AE_CC_birey_ort=float(g.groupby("ind").AE_CC.mean().mean()), AE_CC_torba_ort=float(g.AE_CC.mean()), r07_AE_CC=float(t.AE_CC), r07_n_torba=int(t.n_torba))
# APP120 draw check (same p as R05 F_bags)
FB = pd.read_parquet(P.R05 / "F_bags.parquet"); FB = FB[(FB.scen == "S1-kopru") & (FB.variant == "APP120") & (FB.arm == "CC")]
ctx2 = [f for f in S.folds(ctx, "S1")]; mine = []
for f in ctx2:
    for ind, bid, rr in f["bags"]["APP120"]: mine.append((ind, int(bid), *L.truth(f["yt"], rr, 3)))
M = pd.DataFrame(mine, columns=["ind", "bag", "q0", "q1", "q2"]); J = M.merge(FB[["ind", "bag", "p0", "p1", "p2"]].assign(bag=lambda d: d.bag.astype(int)), on=["ind", "bag"], how="outer", indicator=True)
res["APP120_kontrol"] = dict(benim=len(M), R05=len(FB), eslesen=int((J._merge == "both").sum()), maks_fark=float(np.abs(J[["q0", "q1", "q2"]].values - J[["p0", "p1", "p2"]].values)[J._merge == "both"].max()))
json.dump(res, open(P.TABLES / "r08_gateA.json", "w"), indent=1)
print(json.dumps(res, indent=1))
