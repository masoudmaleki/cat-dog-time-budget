"""fig2_forest_v5.py - Figure 2 of the paper, redrawn to fit one A4 page (16 cm wide, about 19 cm high).

Reads only stored test tables (no model training, no intermediate files):
  results/tables/r07_master.csv, r08_tests.csv, r07_5_tests.csv
Writes results/figures/paper/fig2_forest.pdf and .png (overwrites the R09 version).
Run from the repository root:  python -I scripts/fig2_forest_v5.py results/tables results/figures/paper
"""
import sys, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T, FP = sys.argv[1], sys.argv[2]
os.makedirs(FP, exist_ok=True)
M = pd.read_csv(os.path.join(T, "r07_master.csv"))
R8 = pd.read_csv(os.path.join(T, "r08_tests.csv")); R8 = R8[R8.aile == "birincil"]
T75 = pd.read_csv(os.path.join(T, "r07_5_tests.csv"))

INK = "#222222"; GRID = "#dddddd"; C1 = "#2a6fb0"; C2 = "#d9822b"
MK = {"KAZANIR": ("^", True, "Better"), "KAYBEDER": ("v", True, "Worse"), "ESDEGER": ("s", False, "Equivalent"), "BELIRSIZ": ("o", False, "Inconclusive")}
ARMS = ["N-target", "PCC", "ACC", "PACC", "EMQ", "DM", "DR-APP", "DR-natural"]
SRC = {"N-hedef": "N-target", "PCC": "PCC", "ACC": "ACC", "PACC": "PACC", "EMQ": "EMQ"}
R8A = {"DM": "DM", "DR-APP-q": "DR-APP", "DR-dogal-q": "DR-natural"}
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7, "axes.edgecolor": "#888888", "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "xtick.labelsize": 6, "ytick.labelsize": 6.5, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False})

# (title, r07_master scenario, r08 scenario or None, r08 sample type)
CATS = [("S0 random seconds (leaky), n = 12", "S0 (sızıntılı)", None, None),
        ("S1 new individual, n = 12", "S1", "S1", "30dk"),
        ("S2 harness → collar, n = 12", "S2", "S2", "30dk"),
        ("S4-Smit dog → cat, open, n = 12", "S4-Smit-açık", None, None),
        ("S4-Smit dog → cat, closed, n = 12", "S4-Smit-kapalı", "S4-Smit-kapali", "30dk"),
        ("S1-closed (reference), n = 12", "S1-kapalı", None, None)]
DUN = [("S3 cross-study [expl.], n = 8", "S3 [keşifsel]", "S3", "kedi"),
       ("S4-Dunford open [expl.], n = 8", "S4-Dunford-açık [keşifsel]", None, None),
       ("S4-Dunford closed [expl.], n = 7", "S4-Dunford-kapalı [keşifsel]", "S4-Dunford-kapali", "kedi"),
       ("Within Dunford (reference) [expl.], n = 8", "R-D [keşifsel]", None, None)]


def cat_panel(sc, r8s, r8t):
    rows = {}
    for r in M[(M.senaryo == sc) & (M.aile == "kaynak") & M.kol.isin(list(SRC))].itertuples():
        rows[SRC[r.kol]] = [(r.d, r.lo, r.hi, r.kategori, C1)]
    if r8s:
        for r in R8[(R8.senaryo == r8s) & (R8.torba_turu == r8t) & R8.kol.isin(list(R8A))].itertuples():
            rows[R8A[r.kol]] = [(r.d, r.lo, r.hi, r.kategori, C1)]
    return rows


def dog_panel(space):
    rows = {}
    for bt, col in (("kayit", C1), ("dogal2", C2)):
        for r in T75[(T75.scen == "Kopek-ici") & (T75.space == space) & (T75.feat.astype(str) == "32") & (T75.btype == bt) & T75.kol.isin(list(SRC))].itertuples():
            rows.setdefault(SRC[r.kol], []).append((r.d, r.lo, r.hi, r.kategori, col))
        for r in R8[(R8.senaryo == space) & (R8.torba_turu == bt) & R8.kol.isin(list(R8A))].itertuples():
            rows.setdefault(R8A[r.kol], []).append((r.d, r.lo, r.hi, r.kategori, col))
    return rows


PANELS = [(t, cat_panel(s, a, b)) for t, s, a, b in CATS]
PANELS += [("K-L1 dogs, two classes, n = 44 / 35", dog_panel("K-L1")), ("K-L2 dogs, five classes, n = 44 / 35", dog_panel("K-L2"))]
PANELS += [(t, cat_panel(s, a, b)) for t, s, a, b in DUN]

fig, axs = plt.subplots(4, 3, figsize=(16 / 2.54, 19 / 2.54), sharey=True)
for k, (ax, (ttl, rows)) in enumerate(zip(axs.ravel(), PANELS)):
    ax.axvline(0, color=INK, lw=0.6); ax.axvspan(-0.01, 0.01, color="#eeeeee", zorder=0)
    for i, arm in enumerate(ARMS):
        pts = rows.get(arm, [])
        offs = [0.0] if len(pts) == 1 else [-0.17, 0.17]
        for (d, lo, hi, cat, col), o in zip(pts, offs):
            mk, fill, _ = MK[cat]
            ax.plot([lo, hi], [i + o, i + o], color=col, lw=0.9)
            ax.plot(d, i + o, marker=mk, ms=3.6, color=col, mfc=col if fill else "white", mew=0.8)
    ax.set_yticks(range(len(ARMS))); ax.set_yticklabels(ARMS); ax.set_ylim(len(ARMS) - 0.5, -0.5)
    ax.set_title(ttl, loc="left", pad=3)
    ax.tick_params(axis="both", length=2, pad=1.5)
    ax.locator_params(axis="x", nbins=4)
    if k >= 9:
        ax.set_xlabel("$d$ = AE$_{CC}$ − AE$_{arm}$", labelpad=1)
h = [plt.Line2D([], [], marker=m, ls="", color="#555555", mfc="#555555" if f else "white", ms=4, label=n) for m, f, n in MK.values()]
h += [plt.Line2D([], [], color=C1, lw=1.2, label="cats: primary samples; dogs: recordings"), plt.Line2D([], [], color=C2, lw=1.2, label="dogs: natural 2-min windows")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0), handlelength=1.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.045, 1, 1), h_pad=0.9, w_pad=0.6)
fig.savefig(os.path.join(FP, "fig2_forest.pdf")); fig.savefig(os.path.join(FP, "fig2_forest.png"), dpi=300)
w, hgt = fig.get_size_inches()
print(f"fig2_forest: {w * 2.54:.1f} x {hgt * 2.54:.1f} cm, panels: {len(PANELS)}")
