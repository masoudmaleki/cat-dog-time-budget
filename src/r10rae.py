r"""r10rae.py — R10 §A (RAE table, S12) and §B (open-set care bias) computations. No model; stored bag-level outputs only.
Reads: results/tables/{r07_master,r07_4_tests,r07_5_tests,r08_bag_level,r08_tests,r02_bag_level,r05_bias_classes}.csv, paper/table3_consolidated_results_long.csv,
      <data root>/interim/{r05,r06}/*_bags.parquet (RF bags; RAE column stored), R08 context (only for the N-target budgets, as in B2).
"""
import numpy as np, pandas as pd
from config import paths as P
from src import r07lib as L7, quant as Q, r08scen as SC, stats06 as S6

T = P.TABLES; O5, O6 = P.R05, P.R06
KEY8 = ["senaryo", "torba_turu", "dis_katman", "hedef_birey", "torba"]
ARM_KOL = {"PCC": "PCC", "ACC": "ACC", "PACC": "PACC", "EMQ": "EMQ", "N-target": "N-hedef", "DM": "DM", "DR-APP": "DR-APP-q", "DR-natural": "DR-dogal-q"}
R08ARM = {"DM": "DM", "DR-APP": "DR-APP-q", "DR-natural": "DR-dogal-q"}
R08SPEC = {"S1": ("S1", "30dk"), "S1-L2 (collar)": ("S1-L2", "30dk"), "S2": ("S2", "30dk"), "S4-Smit-closed": ("S4-Smit-kapali", "30dk"), "S3 [exploratory]": ("S3", "kedi"),
           "S4-Dunford-closed [exploratory]": ("S4-Dunford-kapali", "kedi"), "S1 natural 2-min": ("S1", "2dk"), "S1 APP 120-s (bridge)": ("S1", "APP120")}
DOGBT = {"dog × recording (N>=60)": "kayit", "natural 2-min (N>=60)": "dogal2"}
DOGBT11 = dict(DOGBT, **{"APP 120-s": "APP120"})   # R11: per-bag p/p-hat for the R07 arms (interim/r11/dog_rae_bags.parquet, made by scripts/57_r11_dog_rae.py)


def rae_vec(h, p, N):
    eps = (1.0 / (2.0 * np.asarray(N, float)))[:, None]
    ps = p + eps; ps = ps / ps.sum(1, keepdims=True); hs = h + eps; hs = hs / hs.sum(1, keepdims=True)
    return (np.abs(hs - ps) / ps).mean(1)


def _wide_from_long(x, cl):
    w = x.pivot_table(index=KEY8, columns="sinif", values=["p", "p_hat"]).reindex(columns=cl, level=1)
    N = x.drop_duplicates(KEY8).set_index(KEY8).N.reindex(w.index).values
    p = np.column_stack([w[("p", c)].values for c in cl]); h = np.column_stack([w[("p_hat", c)].values for c in cl])
    o = pd.DataFrame(dict(AE=np.abs(h - p).mean(1), RAE=rae_vec(h, p, N)), index=w.index)
    return o


class Pairs:
    def __init__(self):
        self.M = pd.read_csv(T / "r07_master.csv"); self.B8 = pd.read_csv(T / "r08_bag_level.csv")
        self.B8["armx"] = self.B8.kol + np.where(self.B8.tanimlayici == "-", "", "-" + self.B8.tanimlayici)
        self.lad = L7.rd(O5 / "r05_ladder_bags.parquet"); self.src5 = L7.rd(O5 / "r05_sources_bags.parquet"); self.v2 = L7.rd(O6 / "B_v2_bags.parquet"); self.e4 = L7.rd(O6 / "E4_S1kapali_bags.parquet")
        self.F = L7.rd(O5 / "F_bags.parquet"); self.r02 = pd.read_csv(T / "r02_bag_level.csv")
        self.ctx = SC.Ctx(); self._c8 = {}
        NK = self.B8[(self.B8.senaryo == "S1-L2") & (self.B8.armx == "CC")].drop_duplicates(KEY8)
        self.NL2 = {(str(r.hedef_birey), str(r.torba)): int(r.N) for r in NK.itertuples()}
        c = self.ctx
        def prop(y, K): return np.bincount(y, minlength=K) / max(len(y), 1)
        self.SMB = {k: prop(c.y1[(c.cat == k) & (c.y1 >= 0)], 3) for k in c.CATS}; self.DCs = sorted(set(c.dcat)); self.DNB = {k: prop(c.yd[c.dcat == k], 3) for k in self.DCs}

    # ---- R08 (B8) -----------------------------------------------------------------
    def r08(self, scn, bt, kol):
        g = self.B8[(self.B8.senaryo == scn) & (self.B8.torba_turu == bt)]
        if (scn, bt) not in self._c8:
            cl = list(g.sinif.drop_duplicates()); self._c8[(scn, bt)] = {a: _wide_from_long(x, cl) for a, x in g.groupby("armx")}
        W = self._c8[(scn, bt)]
        if kol not in W: return None
        j = W["CC"].join(W[kol], lsuffix="_c", rsuffix="_a", how="inner").reset_index()
        j["ind"] = j.hedef_birey.astype(str); return j[["ind", "AE_c", "AE_a", "RAE_c", "RAE_a"]]

    # ---- RF bag tables ----------------------------------------------------------
    @staticmethod
    def _pair(g, arm, keys=("ind", "bag")):
        a = g[g.arm == "CC"][list(keys) + ["AE", "RAE"]]; b = g[g.arm == arm][list(keys) + ["AE", "RAE"]]
        j = a.merge(b, on=list(keys), suffixes=("_c", "_a")); return j[["ind", "AE_c", "AE_a", "RAE_c", "RAE_a"]] if len(j) else None

    def nh(self, scn, ind, Kt):
        c = self.ctx
        nh = np.mean([self.SMB[x] for x in c.CATS if x != ind], axis=0) if scn in ("S1", "S2", "S4-Smit") else np.mean([self.DNB[x] for x in self.DCs if x != ind], axis=0)
        if Kt == 2: nh = nh[:2] / nh[:2].sum()
        return nh[:Kt]

    def nhedef(self, g0):
        scn = g0.scen.iloc[0]; cc = g0[g0.arm == "CC"]; Kt = int(cc.K.iloc[0]); p = cc[[f"p{i}" for i in range(Kt)]].values
        h = np.array([self.nh(scn, i, Kt) for i in cc.ind]); N = cc.N.values
        return pd.DataFrame(dict(ind=cc.ind.values, AE_c=cc.AE.values, AE_a=np.abs(h - p).mean(1), RAE_c=cc.RAE.values, RAE_a=rae_vec(h, p, N)))

    SRC = {"S0 (leaky)": ("lad", "S0", "30dk_ge50"), "S1": ("lad", "S1", "30dk_ge50"), "S2": ("lad", "S2", "30dk_ge50"), "S3 [exploratory]": ("v2", "S3", "kedi"),
           "S4-Smit-open": ("src5", "S4-Smit", "30dk_ge50-acik"), "S4-Smit-closed": ("src5", "S4-Smit", "30dk_ge50-kapali"), "S4-Dunford-open [exploratory]": ("v2", "S4-Dunford", "acik"),
           "S4-Dunford-closed [exploratory]": ("v2", "S4-Dunford", "kapali"), "R-D [exploratory]": ("v2", "R-D", "kedi"), "S1-closed": ("e4", "S1-kapali", "30dk_ge50")}

    def l2(self, tag, arm):
        b2 = self.r02[(self.r02.varyant == "va_f32") & (self.r02.bagdef == "30dk_ge50") & (self.r02.level == "L2") & (self.r02.basamak == tag)]
        def mk(est):
            x = b2[b2.tahminci == est]; w = x.pivot_table(index=["kedi", "torba"], columns="sinif", values=["p", "p_hat"]).dropna()
            p = w["p"].values; h = w["p_hat"].values; ix = w.index.to_frame(index=False); N = np.array([self.NL2[(str(a), str(b))] for a, b in zip(ix.kedi, ix.torba)])
            return pd.DataFrame(dict(ind=ix.kedi.astype(str).values, bag=ix.torba.astype(str).values, AE=np.abs(h - p).mean(1), RAE=rae_vec(h, p, N)))
        a, b = mk("CC"), mk(arm); j = a.merge(b, on=["ind", "bag"], suffixes=("_c", "_a")); return j[["ind", "AE_c", "AE_a", "RAE_c", "RAE_a"]]

    def get(self, scen, bagset, arm):
        """Returns (DataFrame | None, note)."""
        if arm in R08ARM:
            if scen in R08SPEC: s, bt = R08SPEC[scen]
            elif scen.endswith("within-dog") and bagset in DOGBT: s, bt = scen.split(" ")[0], DOGBT[bagset]
            else: return None, "R08 kolu bu torba türünde koşulmadı"
            return self.r08(s, bt, ARM_KOL[arm]), ""
        if scen.endswith("within-dog"):
            if bagset not in DOGBT11: return None, "köpek torba türü kayıtsız"
            if not hasattr(self, "dog11"): self.dog11 = pd.read_parquet(P.INTERIM / "r11" / "dog_rae_bags.parquet")
            g = self.dog11[(self.dog11.space == scen.split(" ")[0]) & (self.dog11.btype == DOGBT11[bagset])]
            a = g[g.arm == "CC"][["dog", "bag", "AE", "RAE"]]; b = g[g.arm == ARM_KOL.get(arm, arm)][["dog", "bag", "AE", "RAE"]]
            j = a.merge(b, on=["dog", "bag"], suffixes=("_c", "_a")).rename(columns={"dog": "ind"})
            j["ind"] = j.ind.astype(str); return j[["ind", "AE_c", "AE_a", "RAE_c", "RAE_a"]], ""
        if scen == "S2-L2 (matched)": return None, "S2-eşlenmiş L2: torba düzeyi sınıf vektörü kayıtlı değil (yalnız AE)"
        if scen in ("S1-L2 (collar)", "S1-L2 (harness)"): return self.l2("S1-tasma" if "collar" in scen else "S1-kosum", arm), ""
        if scen in ("S1 natural 2-min", "S1 APP 120-s (bridge)"):
            var = "dogal2dk" if "2-min" in scen else "APP120"; RF = pd.concat([self.lad, self.src5, self.F, self.v2], ignore_index=True)
            g = RF[(RF.scen == "S1-kopru") & (RF.variant == var) & (RF.m == 0)]
            return (self._pair(g, arm), "") if g[g.arm == arm].shape[0] or arm == "N-target" else (None, "kayıt yok")
        df, s, v = self.SRC[scen]; df = getattr(self, df); g = df[(df.scen == s) & (df.variant == v) & (df.m == 0)]
        if arm == "N-target": return self.nhedef(g), ""
        return self._pair(g, arm), ""


def boot_ind(d):
    return S6.boot_ci(d)


def s12(pairs=None):
    pr = pairs or Pairs(); L3 = pd.read_csv(T / "paper" / "table3_consolidated_results_long.csv"); rows = []
    for r in L3.itertuples():
        base = dict(Scenario=r.Scenario, Bag_set=r.Bag_set, Arm=r.Arm, AE_category=r.category, d_AE_table=r.d)
        j, note = pr.get(r.Scenario, r.Bag_set, r.Arm)
        if j is None or len(j) == 0:
            rows.append(dict(**base, n_bags=np.nan, n_individuals=np.nan, RAE_CC=np.nan, RAE_arm=np.nan, d_RAE=np.nan, lo=np.nan, hi=np.nan, d_AE_recomputed=np.nan, AE_check="", direction_mismatch="", note=note or "boş eşleşme")); continue
        dA = (j.AE_c - j.AE_a).groupby(j.ind).mean(); dR = (j.RAE_c - j.RAE_a).groupby(j.ind).mean(); lo, hi = boot_ind(dR.values)
        k = r.category; flag = bool((k in ("KAYBEDER", "ESDEGER", "BELIRSIZ") and lo > 0) or (k == "KAZANIR" and hi < 0))
        rows.append(dict(**base, n_bags=len(j), n_individuals=len(dR), RAE_CC=j.RAE_c.mean(), RAE_arm=j.RAE_a.mean(), d_RAE=dR.mean(), lo=lo, hi=hi, d_AE_recomputed=dA.mean(),
                         AE_check="OK" if abs(dA.mean() - r.d) < 1e-6 else "FARK", direction_mismatch=int(flag), note=("source=R11" if r.Scenario.endswith("within-dog") and r.Arm in ("PCC", "ACC", "PACC", "EMQ") else "")))
    return pd.DataFrame(rows)


def openset_bias(src5=None):
    """B1: S4-Smit-open, CC, care (Bakim, class 2) signed bias (pp), primary set (68 bags, 30 min, coverage ≥ 50%)."""
    s5 = src5 if src5 is not None else L7.rd(O5 / "r05_sources_bags.parquet")
    g = s5[(s5.scen == "S4-Smit") & (s5.variant == "30dk_ge50-acik") & (s5.arm == "CC") & (s5.m == 0)]
    b = ((g.h2 - g.p2) * 100).values; ind = g.ind.values; U = sorted(set(ind)); n = len(U)
    bi = pd.Series(b).groupby(ind).mean().reindex(U).values; s = np.array([b[ind == u].sum() for u in U]); c = np.array([(ind == u).sum() for u in U], float)
    lo_i, hi_i = S6.boot_ci(bi); B = np.random.default_rng(P.SEED).integers(0, n, size=(1000, n)); bt = s[B].sum(1) / c[B].sum(1)
    kd = pd.read_csv(T / "r05_bias_classes.csv"); kd = kd[(kd.scen == "S4-Smit") & (kd.variant == "acik") & (kd.level == "birey") & (kd.arm == "CC") & (kd.sinif == "Bakim")].iloc[0].bias_pp
    return pd.DataFrame([
        dict(olcu="kedi düzeyi (R05, tüm saniyeler / kedi)", bias_pp=float(kd), lo=np.nan, hi=np.nan, n_birey=n, n_torba=np.nan),
        dict(olcu="(i) birey ortalaması (birey başına torba ortalaması; R08 §4)", bias_pp=float(bi.mean()), lo=lo_i, hi=hi_i, n_birey=n, n_torba=len(g)),
        dict(olcu="(ii) torba ortalaması (68 torba)", bias_pp=float(b.mean()), lo=float(np.percentile(bt, 2.5)), hi=float(np.percentile(bt, 97.5)), n_birey=n, n_torba=len(g))])
