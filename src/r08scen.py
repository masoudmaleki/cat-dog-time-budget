r"""R08 scenario builders: produce the fold dictionaries for each scenario (input of r08lib.run_fold). Reads the stored R05–R07 outputs; no new RF (dog K-L posteriors: 48_r08_dog_post.py).
Scenarios: S1, S1-L2, S2, S4-Smit-kapali, K-L1, K-L2, S3, S4-Dunford-kapali.
"""
import pickle
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from config import paths as P
from src import r05lib as R, smit_data as D, r07clf as C07
from src import r08lib as L

SEED = P.SEED
SCENS = ["S1", "S1-L2", "S2", "S4-Smit-kapali", "K-L1", "K-L2", "S3", "S4-Dunford-kapali"]


class Ctx:
    """Common data, loaded once."""

    def __init__(self):
        self.lab, self.y1, XC, XH, XC17, XH17 = R.load_smit()
        self.XC32, self.XH32, self.XC17, self.XH17 = XC.values, XH.values, XC17.values, XH17.values
        self.y2 = self.lab.y2.values; self.cat = self.lab.Cat_id.values; self.CATS = sorted(set(self.cat))
        self.idx_all = np.where(self.y1 >= 0)[0]
        self.nat30 = R.smit_bags(self.lab, self.y1, 30, 0.5); self.nat2 = R.smit_bags(self.lab, self.y1, 2, 0.5)
        self.nat30_k = R.smit_bags(self.lab, self.y1, 30, 0.5, mask=(self.y1 < 2))
        self.NK, self.DX32, self.DX17, self.dy_L2, self.dy_L1, self.dogid = L.dog_tables()
        self.S5 = pickle.load(open(P.R05 / "sources.pkl", "rb")); self.S6 = pickle.load(open(P.R06 / "sources_v2.pkl", "rb"))
        self.dun, Xd = R.load_dunford(v2=True); self.Xd = Xd.values; self.yd = self.dun.y1.values; self.dcat = self.dun.cat.values
        self._app = {}

    # ---- APP training bags (code -> (bags, stats)); fixed seed per individual, independent of the fold
    def app(self, code):
        if code not in self._app:
            if code == 1: y, K = self.y1, 3; rows = {c: self.idx_all[self.cat[self.idx_all] == c] for c in self.CATS}
            elif code == 2: y, K = self.y2, 7; rows = {c: self.idx_all[self.cat[self.idx_all] == c] for c in self.CATS}
            else:
                y, K = (self.dy_L1, 2) if code == 3 else (self.dy_L2, 5)
                use = np.where(y >= 0)[0]; rows = {d: use[self.dogid[use] == d] for d in sorted(set(self.dogid[use]))}
            self._app[code] = L.app_train(y, K, rows, code)
        return self._app[code]

    def app_list(self, code, exclude=()):
        b, _ = self.app(code)
        return [(ind, r, p) for ind, lst in b.items() if ind not in exclude for (r, p) in lst]


def _glob(n, K, rows, P_):
    G = np.full((n, K), np.nan); G[rows] = P_; return G


def folds(ctx, scen):
    """Generator: fold dictionaries."""
    if scen in ("S1", "S2"): yield from _smit_loco(ctx, scen)
    elif scen == "S1-L2": yield from _smit_l2(ctx)
    elif scen == "S4-Smit-kapali": yield _s4_smit(ctx)
    elif scen in ("K-L1", "K-L2"): yield from _dog_within(ctx, scen)
    elif scen == "S3": yield _s3(ctx)
    elif scen == "S4-Dunford-kapali": yield _s4_dunford(ctx)
    else: raise ValueError(scen)


def _smit_loco(ctx, scen):
    N = len(ctx.lab); win2 = ctx.nat2
    for ci, c in enumerate(ctx.CATS):
        d = pickle.load(open(P.R05 / f"ladder_{scen}_{c}.pkl", "rb")); te = d["te"].astype(int); src = d["src"]
        tr = ctx.idx_all[ctx.cat[ctx.idx_all] != c]; assert len(src["oof"]) == len(tr)
        Pt = _glob(N, 3, te, d["P"].astype(np.float64)); bags = {"30dk": [(c, w, ix) for (cc, w, ix) in ctx.nat30 if cc == c]}
        if scen == "S1":
            bags["2dk"] = [(c, w, ix) for (cc, w, ix) in ctx.nat2 if cc == c]
            yte = ctx.y1[te]; cls = {k: np.where(yte == k)[0] for k in range(3)}
            ab = C07.app_draw(cls, 3, 50, 120, np.random.default_rng(np.random.SeedSequence([SEED, 600, ci])))
            bags["APP120"] = [(c, b, te[bg[0]]) for b, bg in enumerate(ab) if bg is not None]
        nat = [(cc, ix) for (cc, w, ix) in ctx.nat30 + ctx.nat2 if cc != c]
        yield dict(scen=scen, fold=c, K=3, cls=L.L1N, Xs=(ctx.XH17 if scen == "S2" else ctx.XC17), tr=tr, ys=ctx.y1, oof=src["oof"], nat=nat, app=ctx.app_list(1, (c,)),
                   Xt=ctx.XC17, Pt=Pt, yt=ctx.y1, bags=bags, C=src["C"], Cp=src["Cp"])


def _smit_l2(ctx):
    N = len(ctx.lab); res = pickle.load(open(P.INTERIM / "r02" / "raw_va_f32.pkl", "rb"))["loco"]
    for f in res:
        c = f["cat"]; inn = pickle.load(open(P.INTERIM / "r03" / f"inner_{c}.pkl", "rb")); te = f["test_idx"].astype(int); tr = inn["tr"].astype(int)
        Pt = _glob(N, 7, te, f["P_cc"].astype(np.float64)); ic = f["inner_collar"]
        yield dict(scen="S1-L2", fold=c, K=7, cls=D.L2, Xs=ctx.XC32, tr=tr, ys=ctx.y2, oof=inn["raw_collar_oof"], nat=[(cc, ix) for (cc, w, ix) in ctx.nat30 + ctx.nat2 if cc != c],
                   app=ctx.app_list(2, (c,)), Xt=ctx.XC32, Pt=Pt, yt=ctx.y2, bags={"30dk": [(c, w, ix) for (cc, w, ix) in ctx.nat30 if cc == c]}, C=ic["C"], Cp=ic["Cp"])


def _dog_source_l1(ctx):
    """S4 dog source (ANeck, D17, K=2): global NK rows."""
    tr = np.where(ctx.dy_L1 >= 0)[0]; bg = L.dog_bags(ctx.NK, ctx.dy_L1)
    nat = [(d, rr) for d, v in bg.items() for (_, rr) in v["kayit"] + v["dogal2"]]
    return tr, nat


def _s4_smit(ctx):
    N = len(ctx.lab); S = ctx.S5; tr, nat = _dog_source_l1(ctx); assert len(S["src4"]["oof"]) == len(tr)
    Pt = _glob(N, 2, S["idx"], S["P4s"].astype(np.float64))
    return dict(scen="S4-Smit-kapali", fold="kaynak45", K=2, cls=L.L1N[:2], Xs=ctx.DX17, tr=tr, ys=ctx.dy_L1, oof=S["src4"]["oof"], nat=nat, app=ctx.app_list(3),
                Xt=ctx.XC17, Pt=Pt, yt=ctx.y1, bags={"30dk": [(c, w, ix) for (c, w, ix) in ctx.nat30_k]}, C=S["src4"]["C"], Cp=S["src4"]["Cp"])


def _s3(ctx):
    S = ctx.S6; S5 = ctx.S5; tr = S5["idx"].astype(int); assert len(S["src3"]["oof"]) == len(tr)
    bags = []
    for c in sorted(set(ctx.dcat)):
        ii = np.where(ctx.dcat == c)[0]
        if len(ii) >= 60: bags.append((c, 0, ii))
    return dict(scen="S3", fold="kaynak12", K=3, cls=L.L1N, Xs=ctx.XC17, tr=tr, ys=ctx.y1, oof=S["src3"]["oof"], nat=[(cc, ix) for (cc, w, ix) in ctx.nat30 + ctx.nat2],
                app=ctx.app_list(1), Xt=ctx.Xd, Pt=S["P3"].astype(np.float64), yt=ctx.yd, bags={"kedi": bags}, C=S["src3"]["C"], Cp=S["src3"]["Cp"])


def _s4_dunford(ctx):
    S = ctx.S6; tr, nat = _dog_source_l1(ctx); assert len(S["src4"]["oof"]) == len(tr)
    bags = []
    for c in sorted(set(ctx.dcat)):
        ii = np.where((ctx.dcat == c) & (ctx.yd < 2))[0]
        if len(ii) >= 60: bags.append((c, 0, ii))
    return dict(scen="S4-Dunford-kapali", fold="kaynak45", K=2, cls=L.L1N[:2], Xs=ctx.DX17, tr=tr, ys=ctx.dy_L1, oof=S["src4"]["oof"], nat=nat, app=ctx.app_list(3),
                Xt=ctx.Xd, Pt=S["P4d"].astype(np.float64), yt=ctx.yd, bags={"kedi": bags}, C=S["src4"]["C"], Cp=S["src4"]["Cp"])


def _dog_within(ctx, scen):
    y, K, code, cls = (ctx.dy_L2, 5, 4, L.DOGL2) if scen == "K-L2" else (ctx.dy_L1, 2, 3, L.L1N[:2]); N = len(ctx.NK)
    bg = L.dog_bags(ctx.NK, y)
    for fi in range(5):
        d = pickle.load(open(P.R08 / f"dog_{scen}_f{fi}.pkl", "rb")); te = d["te"].astype(int); tr = d["tr"].astype(int); src = d["src"]; assert len(src["oof"]) == len(tr)
        tdog = sorted(set(ctx.dogid[te])); trdog = sorted(set(ctx.dogid[tr])); Pt = _glob(N, K, te, d["P"].astype(np.float64))
        bags = {"kayit": [(dg, bid, rr) for dg in tdog for (bid, rr) in bg[dg]["kayit"]], "dogal2": [(dg, bid, rr) for dg in tdog for (bid, rr) in bg[dg]["dogal2"]]}
        nat = [(dg, rr) for dg in trdog for (_, rr) in bg[dg]["kayit"] + bg[dg]["dogal2"]]
        yield dict(scen=scen, fold=f"f{fi}", K=K, cls=cls, Xs=ctx.DX32, tr=tr, ys=y, oof=src["oof"], nat=nat, app=ctx.app_list(code, exclude=tuple(tdog)), Xt=ctx.DX32, Pt=Pt, yt=y, bags=bags,
                   C=src["C"], Cp=src["Cp"])
