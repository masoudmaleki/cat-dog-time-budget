r"""15_r04_figures.py — R04 figures. Reads: results/tables/r04_paired_vs_CC.csv, r04_var_pred_cells.csv, r04_decomposition.csv. Writes: results/figures/r04_var_pred.png, r04_crossover.png"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import paths as P

T, F = P.TABLES, P.FIGURES
pw = pd.read_csv(T / "r04_paired_vs_CC.csv"); vp = pd.read_csv(T / "r04_var_pred_cells.csv"); dec = pd.read_csv(T / "r04_decomposition.csv")
half_min = float(dec[dec["size"] == "yari"].min_ort.mean())
XM = {"5": 5.0, "10": 10.0, "30": 30.0, "60": 60.0, "yari": half_min}
col = {"S1-tasma": "#1f77b4", "S1-kosum": "#d95f02", "S2-esl": "#2ca02c"}
name = {"S1-tasma": "S1-tasma", "S1-kosum": "S1-koşum", "S2-esl": "S2-eşlenmiş"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, est, ttl in zip(axes, ("ACC-xfit", "ACC"), ("CC − ACC-xfit (sızıntısız hedef C)", "CC − ACC (eğitim kedileri C)")):
    for sc in col:
        x = pw[(pw.scen == sc) & (pw.est == est)].copy(); x["m"] = x["size"].map(XM); x = x.sort_values("m")
        ax.plot(x.m, x.dAE, "-o", color=col[sc], label=name[sc], ms=4)
        ax.fill_between(x.m, x.dAE_lo, x.dAE_hi, color=col[sc], alpha=0.12)
    ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log"); ax.set_xticks(list(XM.values()))
    ax.set_xticklabels(["5", "10", "30", "60", "yarı\n(≈%d)" % round(half_min)]); ax.set_xlabel("torba boyu (dk)"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.3)
axes[0].set_ylabel("AE_CC − AE_X  (pozitif = düzeltme kazanır)"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(F / "r04_crossover.png", dpi=130); plt.close()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
mk = {"5": "o", "10": "s", "30": "^", "60": "D"}
for ax, est in zip(axes, ("ACC", "ACC-xfit")):
    x = vp[vp.est == est]
    for sc in col:
        for sz, m in mk.items():
            g = x[(x.scen == sc) & (x["size"].astype(str) == sz)]
            ax.scatter(g.pred, g.obs, c=col[sc], marker=m, s=28, alpha=0.8, label=f"{name[sc]} {sz} dk" if est == "ACC" and sz == "30" else None)
    lim = [min(x.pred.min(), x.obs.min()) * 0.7, max(x.pred.max(), x.obs.max()) * 1.4]
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("analitik varyans (iz, N_eff ile)"); ax.set_ylabel("gözlenen varyans (kedi içi, ddof=1)"); ax.set_title(est + " (işaret: 5○ 10□ 30△ 60◇ dk)", fontsize=10); ax.grid(alpha=0.3, which="both")
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(F / "r04_var_pred.png", dpi=130); plt.close()
print("ok", half_min)
