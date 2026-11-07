r"""36_r06_E3_counts.py — R06 E3: per-class second counts in the label draws (same seeds as R05/R06); missing C column (0 s) and <20 s proportions. No model.
Reads: Smit table, dog_aneck_1s.parquet, features/dunford_v2_1s.parquet, r05/ladder_{S1,S2}_<cat>.pkl (te only). Writes: results/tables/r06_E3_counts.csv
Classes are counted in the source class space (S3/S1/S2: 3 classes; S4: dog source 2 classes; care seconds in the open set are in the separate 'Bakim_etiket' column).
"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P
from src import r05lib as R, r06lib as L

SEED = P.SEED; O5 = P.R05; T = P.TABLES
lab, y1, XC, XH, XC17, XH17 = R.load_smit(); idx_all = np.where(y1 >= 0)[0]; XS = XC17.values; cat_full = lab.Cat_id.values; CATS = sorted(set(cat_full))
SEC = lab.sec.values.astype("datetime64[s]").astype(np.int64)
dun, Xd = R.load_dunford(v2=True); Xd = Xd.values; yd = dun.y1.values; dcat = dun.cat.values; DCATS = sorted(set(dcat)); dcid = pd.factorize(dun.chunk.astype(str) + "_" + dun["sub"].astype(str))[0]
STR_S = [list(a) for a in np.array_split(np.random.default_rng(SEED + 1).permutation(CATS), 4)]
STR_D = [list(a) for a in np.array_split(np.random.default_rng(SEED + 2).permutation(DCATS), 3)]
rows = []   # unit level: scen, var, level, m, unit, c0..c2


def add(scen, var, level, m, unit, y, Km):
    cnt = np.bincount(y[y < Km], minlength=Km); rows.append(dict(scen=scen, var=var, level=level, m=m, unit=unit, n_lab=int(len(y)), bakim_etiket=int((y == 2).sum()) if Km == 2 else 0, **{f"c{k}": int(cnt[k]) for k in range(Km)}, Km=Km))


# Dunford
for scen, var, drop, Km, vcode in (("S3", "-", False, 3, 1), ("S4-Dunford", "kapali", True, 2, 2), ("S4-Dunford", "acik", False, 2, 3)):
    tg = L.DunTarget(yd, dcat, dcid, Xd, DCATS, drop)
    for si, strat in enumerate(STR_D):
        pool = [c for c in DCATS if c not in strat]
        for m in L.MS:
            for rep in range(L.REPS):
                labr = tg.draw(pool, m, np.random.default_rng(np.random.SeedSequence([SEED, 300 + si, m, rep, vcode])))
                add(scen, var, "calisma", m, f"k{si}r{rep}", tg.y[labr], Km)
# Smit S4
seen = set()
for var, mask, vcode in (("kapali", (y1 < 2), 2), ("acik", None, 3)):
    tg = L.SmitTarget(lab, y1, XS, SEC, CATS, idx_all, mask, None)
    for ci, c in enumerate(CATS):
        si = [i for i, s in enumerate(STR_S) if c in s][0]; pool = [x for x in CATS if x not in STR_S[si]]
        for m in L.MS:
            for rep in range(L.REPS):
                labI = tg.draw([c], m, np.random.default_rng(np.random.SeedSequence([SEED, 200 + ci, m, rep, vcode])))
                add("S4-Smit", var, "birey", m, f"{c}r{rep}", tg.y[labI], 2)
                if (var, si, m, rep) not in seen:   # study-level draw once per (si, m, rep)
                    seen.add((var, si, m, rep))
                    labW = tg.draw(pool, m, np.random.default_rng(np.random.SeedSequence([SEED, 100 + si, m, rep, vcode])))
                    add("S4-Smit", var, "calisma", m, f"s{si}r{rep}", tg.y[labW], 2)
# ladder
for ci, c in enumerate(CATS):
    for scen in ("S1", "S2"):
        d = pickle.load(open(O5 / f"ladder_{scen}_{c}.pkl", "rb")); te = d["te"].astype(int); yte = y1[te]; ts = SEC[te]; srt = np.argsort(ts, kind="stable")
        for m in (300, 1800):
            for rep in range(L.REPS):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 400 + ci, m, rep, 7 if scen == "S1" else 8]))
                s = L.pick_blocks(ts[srt], m // 60, rng, [])
                labr = np.concatenate([srt[a:a + 60] for a in s]) if s else np.array([], int)
                add(scen, "30dk_ge50", "birey", m, f"{c}r{rep}", yte[labr], 3)
U = pd.DataFrame(rows).fillna(0); U.to_parquet(P.R06 / "E3_units.parquet")
out = []
for (scen, var, level, m), g in U.groupby(["scen", "var", "level", "m"]):
    Km = int(g.Km.iloc[0]); cm = g[[f"c{k}" for k in range(Km)]].values
    for k in range(Km):
        out.append(dict(scen=scen, var=var, level=level, m=m, sinif=["Inaktif", "Lokomosyon", "Bakim"][k], n_cekilis=len(g), medyan_s=float(np.median(cm[:, k])), p10_s=float(np.percentile(cm[:, k], 10)),
                        eksik_C_orani=float((cm[:, k] == 0).mean()), lt20_orani=float((cm[:, k] < 20).mean())))
    out.append(dict(scen=scen, var=var, level=level, m=m, sinif="HEPSI>=20", n_cekilis=len(g), medyan_s=np.nan, p10_s=np.nan, eksik_C_orani=float((cm.min(1) == 0).mean()), lt20_orani=float((cm.min(1) < 20).mean()),
                    hepsi_ge20_orani=float((cm.min(1) >= 20).mean()), n_lab_ort=float(g.n_lab.mean())))
O = pd.DataFrame(out); O.to_csv(T / "r06_E3_counts.csv", index=False)
pd.set_option("display.width", 250, "display.max_rows", 300, "display.float_format", "{:.3f}".format)
print(O[O.sinif == "HEPSI>=20"].to_string())
