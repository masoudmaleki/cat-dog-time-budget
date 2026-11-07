r"""37_r07_1.py — R07 §1: target-only (matched) − CC and target-only − weighted retrain; m = 300, 1,800; family = scenario x level (Holm over m).
Reads: r05/E_S4-Smit_*.parquet, r06/E1_*.parquet (R06 draws). Writes: results/tables/r07_1_yh.csv"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src import r07lib as L
DU, SM = L.load_E(); MS = (300, 1800); out = []
cells = [(DU, "S3", "-", "calisma"), (DU, "S4-Dunford", "acik", "calisma"), (DU, "S4-Dunford", "kapali", "calisma")] + \
        [(SM, "S4-Smit", v, lv) for v in ("acik", "kapali") for lv in ("birey", "calisma")]
for df, sc, var, lv in cells:
    for b, tag in (("CC", "YH-CC"), ("Yeniden-egitim-agirlikli", "YH-RTw")):
        pr = L.pairs(df, sc, var, lv, "Yalniz-hedef-esl", b, MS)
        if not pr: continue
        r = L.family(f"R07-1|{sc}|{var}|{lv}|{tag}", [(dict(scen=sc, var=var, level=lv, karsilastirma=tag, m=m), pr[m]) for m in MS if m in pr]); out.append(r)
T = pd.concat(out); T.to_csv(L.T / "r07_1_yh.csv", index=False)
pd.set_option("display.width", 250, "display.float_format", "{:.4f}".format)
print(T[["scen", "var", "level", "karsilastirma", "m", "n", "AE_a", "AE_b", "d", "lo", "hi", "p_holm", "aile_n", "kategori"]].to_string())
