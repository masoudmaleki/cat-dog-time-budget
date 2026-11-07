r"""47_r07_7_figures.py — R07 §7 figures (fixed settings; no new model).
fig1_main: scenario x arm, d = AE_CC − AE_arm [CI], category marked (RF, L1). fig2_app_vs_natural: Smit and dog, CC − PACC gain and median delta for natural / APP.
fig3_open_set: S4-Smit-open and S4-Dunford-open signed bias per class (CC, pp) + distribution of the care seconds over the predicted classes.
fig4_label_budget: (a) proportion of draws with >=20 s from every class, (b) AE (S4-Smit-open individual): CC, PACC-target (random/stratified), retrain, target-only.
Reads: results/tables/r07_master.csv, r07_5_*.csv, r06_D_delta.csv, r07_6_*.csv, r05/r06 bag parquet files, sources pkl. Writes: results/figures/fig*.png"""
import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import paths as P
from src import r05lib as R, stats06 as S6

T = P.TABLES; FG = P.FIGURES; FG.mkdir(parents=True, exist_ok=True)
INK = "#222222"; GRID = "#dddddd"; C1 = "#2a6fb0"; C2 = "#d9822b"; C3 = "#3a9a8c"; C4 = "#7b5aa6"; C5 = "#777777"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888888", "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})
MK = {"KAZANIR": ("^", True), "KAYBEDER": ("v", True), "ESDEGER": ("s", False), "BELIRSIZ": ("o", False)}

# ---------------- fig1
M = pd.read_csv(T / "r07_master.csv"); M = M[(M.uzay == "L1") & (M.kategori != "REFERANS")]
sc = [s for s in M.senaryo.unique()]
fig, axs = plt.subplots(5, 2, figsize=(12, 15)); axs = axs.ravel()
for ax, s in zip(axs, sc):
    g = M[M.senaryo == s].reset_index(drop=True); ax.axvline(0, color=INK, lw=0.8)
    ax.axvspan(-0.01, 0.01, color="#eeeeee", zorder=0)
    for i, r in g.iterrows():
        mk, fill = MK[r.kategori]; col = C1 if r.aile == "kaynak" else C2
        ax.plot([r.lo, r.hi], [i, i], color=col, lw=1.4); ax.plot(r.d, i, marker=mk, ms=7, color=col, mfc=col if fill else "white", mew=1.4)
    ax.set_yticks(range(len(g))); ax.set_yticklabels([k.replace(" (m=1800, rastgele blok)", " m1800") for k in g.kol], fontsize=7.5); ax.invert_yaxis(); ax.set_title(s, fontsize=9.5, loc="left")
    ax.set_xlabel("d = AE_CC − AE_kol  (d>0: kol daha iyi)", fontsize=8)
for ax in axs[len(sc):]: ax.axis("off")
h = [plt.Line2D([], [], marker=m, ls="", color=C5, mfc=C5 if f else "white", label=k) for k, (m, f) in MK.items()] + [plt.Line2D([], [], color=C1, label="kaynak kolları"), plt.Line2D([], [], color=C2, label="hedef etiketli (m=1.800, rastgele blok)")]
fig.legend(handles=h, loc="lower center", ncol=6, frameon=False); fig.suptitle("Senaryo × kol: CC − kol AE farkı (RF, L1, D17); gri bant |d| < 0,01", y=0.995, fontsize=11)
fig.tight_layout(rect=(0, 0.03, 1, 0.98)); fig.savefig(FG / "fig1_main.png"); plt.close(fig)

# ---------------- fig2
rows = []
FB = pd.read_parquet(P.R05 / "F_bags.parquet"); FB = FB[(FB.scen == "S1-kopru") & (FB.m == 0)]
dd = pd.read_csv(T / "r06_D_delta.csv") if (T / "r06_D_delta.csv").exists() else None
D6 = pd.read_csv(T / "r06_D_bag.csv")
for var, nm in (("dogal2dk", "doğal 2 dk"), ("APP120", "APP 120 s")):
    g = FB[FB.variant == var]; cc = g[g.arm == "CC"][["ind", "bag", "AE"]]; pa = g[g.arm == "PACC"][["ind", "bag", "AE"]]; j = cc.merge(pa, on=["ind", "bag"], suffixes=("_c", "_p")); j["d"] = j.AE_c - j.AE_p
    dind = j.groupby("ind").d.mean().values; lo, hi = S6.boot_ci(dind); st = "S1-2dk" if var == "dogal2dk" else "S1-APP120"
    rows.append(dict(tur="Smit (12 kedi, L1)", torba=nm, d=dind.mean(), lo=lo, hi=hi, delta=float(D6[D6.set == st].delta.median())))
j30 = pd.read_csv(T / "r07_master.csv"); r = j30[(j30.senaryo == "S1") & (j30.kol == "PACC") & (j30.uzay == "L1")].iloc[0]
rows.append(dict(tur="Smit (12 kedi, L1)", torba="doğal 30 dk", d=r.d, lo=r.lo, hi=r.hi, delta=float(D6[D6.set == "S1"].delta.median())))
DT = pd.read_csv(T / "r07_5_tests.csv"); DL = pd.read_csv(T / "r07_5_delta.csv")
for sp, nmx in (("K-L1", "K-L1"), ("K-L2", "K-L2")):
    for bt, nm in (("kayit", "köpek×kayıt"), ("dogal2", "doğal 2 dk"), ("APP120", "APP 120 s")):
        t = DT[(DT.scen == "Kopek-ici") & (DT.space == sp) & (DT.feat == "32") & (DT.btype == bt) & (DT.kol == "PACC")]
        dl = DL[(DL.scen == "Kopek-ici") & (DL.space == sp) & (DL.feat == "32") & (DL.btype == bt)]
        if len(t) and len(dl): t = t.iloc[0]; rows.append(dict(tur=f"Köpek (n=44, {nmx})", torba=nm, d=t.d, lo=t.lo, hi=t.hi, delta=float(dl.delta_medyan.iloc[0])))
F2 = pd.DataFrame(rows); F2.to_csv(T / "r07_fig2_data.csv", index=False)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.5, 1]}, sharey=True)
lab = [f"{r.tur} | {r.torba}" for r in F2.itertuples()]; y = np.arange(len(F2))
cols = {"doğal 30 dk": C1, "doğal 2 dk": C1, "köpek×kayıt": C1, "APP 120 s": C2}
for i, r in enumerate(F2.itertuples()):
    a1.plot([r.lo, r.hi], [i, i], color=cols[r.torba], lw=1.5); a1.plot(r.d, i, "o", color=cols[r.torba], ms=7, mfc=cols[r.torba] if r.torba == "APP 120 s" else "white", mew=1.5)
    a2.barh(i, r.delta, color=cols[r.torba], height=0.55)
a1.axvline(0, color=INK, lw=0.8); a1.set_yticks(y); a1.set_yticklabels(lab, fontsize=8); a1.invert_yaxis(); a1.set_xlabel("CC − PACC AE kazancı (d>0: PACC daha iyi) [birey bootstrap %95 GA]")
a2.set_xlabel("δ medyanı (½‖p − π_kaynak‖₁)"); fig.suptitle("Doğal torbalar (mavi, boş) ve APP (turuncu, dolu): PACC kazancı ve önsel kayması", fontsize=10.5); fig.tight_layout(); fig.savefig(FG / "fig2_app_vs_natural.png"); plt.close(fig)

# ---------------- fig3
src5 = pd.read_parquet(P.R05 / "r05_sources_bags.parquet"); v2 = pd.read_parquet(P.R06 / "B_v2_bags.parquet")
CL = ["Inaktif", "Lokomosyon", "Bakım"]; fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4)); off = {"CC": -0.18, "PACC": 0.18}
for si, (nm, df, sc_, var) in enumerate((("S4-Smit-açık", src5, "S4-Smit", "30dk_ge50-acik"), ("S4-Dunford-açık", v2, "S4-Dunford", "acik"))):
    g = df[(df.scen == sc_) & (df.variant == var) & (df.m == 0)]
    for arm, col in (("CC", C1), ("PACC", C2)):
        x = g[g.arm == arm]
        for k in range(3):
            b = ((x[f"h{k}"] - x[f"p{k}"]) * 100).groupby(x.ind).mean().values; lo, hi = S6.boot_ci(b)
            xp = k + si * 4 + off[arm]; a1.plot([xp, xp], [lo, hi], color=col, lw=1.6); a1.plot(xp, b.mean(), "o" if arm == "CC" else "s", color=col, ms=6, label=arm if (si == 0 and k == 0) else None)
a1.axhline(0, color=INK, lw=0.8); a1.set_xticks([0, 1, 2, 4, 5, 6]); a1.set_xticklabels([f"{c}\n{('Smit' if i < 3 else 'Dunford')}" for i, c in enumerate(CL * 2)], fontsize=8); a1.set_ylabel("işaretli yanlılık (tahmin − gerçek), yüzde puan"); a1.legend(frameon=False); a1.set_title("Sınıf bazında işaretli yanlılık (birey bootstrap GA)", fontsize=9.5, loc="left")
lab, y1, *_ = R.load_smit(); S5 = pickle.load(open(P.R05 / "sources.pkl", "rb")); S6_ = pickle.load(open(P.R06 / "sources_v2.pkl", "rb")); dun, _ = R.load_dunford(v2=True)
pos = y1[S5["idx"]]; ps = S5["P4s"][pos == 2].argmax(1); pdn = S6_["P4d"][dun.y1.values == 2].argmax(1)
for i, (nm, p) in enumerate((("Smit", ps), ("Dunford", pdn))):
    fr = np.bincount(p, minlength=2) / len(p) * 100; a2.bar(i, fr[0], color=C1, width=0.5, label="→ Inaktif" if i == 0 else None); a2.bar(i, fr[1], bottom=fr[0], color=C2, width=0.5, label="→ Lokomosyon" if i == 0 else None)
    a2.text(i, fr[0] / 2, f"%{fr[0]:.1f}", ha="center", color="white", fontsize=9); a2.text(i, fr[0] + fr[1] / 2, f"%{fr[1]:.1f}", ha="center", color=INK, fontsize=9); a2.text(i, 103, f"n={len(p)} s", ha="center", fontsize=8)
a2.set_xticks([0, 1]); a2.set_xticklabels(["Smit Bakım", "Dunford Bakım"]); a2.set_ylim(0, 110); a2.set_ylabel("tahmin edilen sınıf, %"); a2.legend(frameon=False, loc="lower right"); a2.set_title("Bakım saniyelerinin köpek modeli tahmini", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig(FG / "fig3_open_set.png"); plt.close(fig)

# ---------------- fig4
CT = pd.read_csv(T / "r07_6_counts.csv"); AR = pd.read_csv(T / "r07_6_arms.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
for (sc_, var, lv, nm, col) in (("S1", "30dk_ge50", "birey", "S1 birey", C1), ("S4-Smit", "acik", "birey", "S4-Smit-açık birey", C2), ("S3", "-", "calisma", "S3 çalışma", C3)):
    for yo, ls, mk in (("rastgele_blok", "--", "o"), ("tabakali", "-", "s")):
        g = CT[(CT.scen == sc_) & (CT["var"] == var) & (CT.level == lv) & (CT.yontem == yo)].sort_values("m")
        if len(g): a1.plot(g.m, g.hepsi_ge20 * 100, ls=ls, marker=mk, color=col, ms=5, label=f"{nm} – {'rastgele' if yo.startswith('r') else 'tabakalı'}")
a1.set_xscale("log"); a1.set_xticks([60, 300, 900, 1800]); a1.set_xticklabels(["60", "300", "900", "1.800"]); a1.set_xlabel("etiket bütçesi m (s)"); a1.set_ylabel("her sınıftan ≥ 20 s içeren çekiliş, %"); a1.legend(frameon=False, fontsize=7); a1.set_title("(a) sınıf kapsamı", loc="left", fontsize=9.5)
SPEC = [("CC", "CC (rastgele)", C5, "o", "-"), ("CC-tab", "CC (tabakalı)", C5, "o", "--"), ("PACC-hedef", "PACC-hedef rastgele", C1, "s", "-"), ("PACC-hedef-tab", "PACC-hedef tabakalı", C1, "s", "--"),
        ("Yeniden-egitim", "yeniden eğitim (rastgele)", C2, "^", "-"), ("Yalniz-hedef-esl", "yalnız hedef (rastgele)", C3, "D", "-"), ("Yalniz-hedef-tab", "yalnız hedef (tabakalı)", C3, "D", "--")]
for arm, nm, col, mk, ls in SPEC:
    g = AR[(AR.scen == "S4-Smit") & (AR.variant == "acik") & (AR.level == "birey") & (AR.arm == arm)] if arm != "CC" else AR[(AR.scen == "S4-Smit") & (AR.variant == "acik") & (AR.arm == "CC") & (AR.level == "birey")]
    g = g.sort_values("m")
    if len(g): a2.plot(g.m, g.AE, color=col, marker=mk, ls=ls, ms=5, label=nm)
a2.set_xscale("log"); a2.set_xticks([60, 300, 900, 1800]); a2.set_xticklabels(["60", "300", "900", "1.800"]); a2.set_xlabel("etiket bütçesi m (s)"); a2.set_ylabel("AE (S4-Smit-açık, birey düzeyi)"); a2.legend(frameon=False, fontsize=7); a2.set_title("(b) AE", loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig(FG / "fig4_label_budget.png"); plt.close(fig)
print(F2.to_string()); print("tamam")
