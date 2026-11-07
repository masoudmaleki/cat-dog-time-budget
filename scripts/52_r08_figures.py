r"""52_r08_figures.py — R08 E: fig1_main_v2.png = same settings as fig1_main.png (fig1 block of 47_r07_7_figures.py), + DM, DR-APP, DR-natural rows (R08, primary set).
Reads: results/tables/r07_master.csv, r08_tests.csv. Writes: results/figures/fig1_main_v2.png (does not touch fig1_main.png)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import paths as P

T = P.TABLES; FG = P.FIGURES
INK = "#222222"; GRID = "#dddddd"; C1 = "#2a6fb0"; C2 = "#d9822b"; C3 = "#3a9a8c"; C4 = "#7b5aa6"; C5 = "#777777"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888888", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
MK = {"KAZANIR": ("^", True), "KAYBEDER": ("v", True), "ESDEGER": ("s", False), "BELIRSIZ": ("o", False)}
M = pd.read_csv(T / "r07_master.csv"); M = M[(M.uzay == "L1") & (M.kategori != "REFERANS")]
R8 = pd.read_csv(T / "r08_tests.csv"); R8 = R8[R8.aile == "birincil"]
MAP = {"S1": ("S1", "30dk"), "S2": ("S2", "30dk"), "S3 [keşifsel]": ("S3", "kedi"), "S4-Smit-kapalı": ("S4-Smit-kapali", "30dk"), "S4-Dunford-kapalı [keşifsel]": ("S4-Dunford-kapali", "kedi")}
NAME = {"DM": "DM (R08)", "DR-APP-q": "DR-APP (R08)", "DR-dogal-q": "DR-doğal (R08)"}
sc = [s for s in M.senaryo.unique()]
fig, axs = plt.subplots(5, 2, figsize=(12, 15)); axs = axs.ravel()
for ax, s in zip(axs, sc):
    g = M[M.senaryo == s][["kol", "aile", "d", "lo", "hi", "kategori"]].copy()
    if s in MAP:
        r8 = R8[(R8.senaryo == MAP[s][0]) & (R8.torba_turu == MAP[s][1])]
        add = pd.DataFrame([dict(kol=NAME[r.kol], aile="R08", d=r.d, lo=r.lo, hi=r.hi, kategori=r.kategori) for r in r8.itertuples()])
        g = pd.concat([g, add], ignore_index=True)
    g = g.reset_index(drop=True); ax.axvline(0, color=INK, lw=0.8); ax.axvspan(-0.01, 0.01, color="#eeeeee", zorder=0)
    for i, r in g.iterrows():
        mk, fill = MK[r.kategori]; col = C1 if r.aile == "kaynak" else (C4 if r.aile == "R08" else C2)
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.4); ax.plot(r.d, i, marker=mk, ms=7, color=col, mfc=col if fill else "white", mew=1.4)
    ax.set_yticks(range(len(g))); ax.set_yticklabels([k.replace(" (m=1800, rastgele blok)", " m1800") for k in g.kol], fontsize=7.5); ax.invert_yaxis(); ax.set_title(s, fontsize=9.5, loc="left")
    ax.set_xlabel("d = AE_CC − AE_kol  (d>0: kol daha iyi)", fontsize=8)
for ax in axs[len(sc):]: ax.axis("off")
h = [plt.Line2D([], [], marker=m, ls="", color=C5, mfc=C5 if f else "white", label=k) for k, (m, f) in MK.items()] + [plt.Line2D([], [], color=C1, label="kaynak kolları"), plt.Line2D([], [], color=C2, label="hedef etiketli (m=1.800, rastgele blok)"),
                                                                                                                      plt.Line2D([], [], color=C4, label="DM, DR (R08; birincil torba kümesi)")]
fig.legend(handles=h, loc="lower center", ncol=7, frameon=False); fig.suptitle("Senaryo × kol: CC − kol AE farkı (RF, L1, D17); gri bant |d| < 0,01", y=0.995, fontsize=11)
fig.tight_layout(rect=(0, 0.03, 1, 0.98)); fig.savefig(FG / "fig1_main_v2.png"); plt.close(fig); print("tamam")
