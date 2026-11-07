r"""55_r09_audit.py — R09 §A: leakage and bag audit (NO model training).
Reads: R05–R08 intermediates (ladder/inner/dog pkl, sources pkl), Smit/Dunford/dog feature tables (labels + individual id only), E/T parquet files (stored bag Ns), r08_bag_level.csv.
Split sets are reproduced: outer training, inner CV (GroupKFold(4), group = individual), seconds used for C/Cp/DM histograms (recomputed with the stored C), DR training bags, m seconds in the target-labelled arms (draw with the same seed).
Writes: results/tables/r09_audit.csv, r09_audit_detail.csv (full listing including only FAIL rows), logs/r09_audit.log
Checks (columns):
  k1 = the test individual is in none of the outer-training / inner-CV / C-Cp / DM-histogram / DR-training-bag sets (also at row level, in the same table)
  k2 = in study-level target arms the label pool is outside the test fold
  k3 = in individual-level target arms the m seconds are removed from the evaluation bags (stored N = |bag| − |labelled ∩ bag|)
  k4 = bag matching: the arm has the same bag ids and the same N as CC
"""
import sys, pickle, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from config import paths as P
from src import r05lib as R, quant as Q, r06lib as L6, r07strat as ST, r08lib as L8, r08scen as SC

SEED = P.SEED; T = P.TABLES; t0 = time.time()
LOG = open(P.LOGS / "r09_audit.log", "w", encoding="utf-8")


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


ctx = SC.Ctx(); log("ctx yüklendi")
DET = []     # detail: (scenario, fold, set, check, result, note)
AUD = []     # summary rows


def ind_of(tab, rows):
    rows = np.asarray(rows, int)
    if tab == "smit": return np.array(["smit:" + str(x) for x in ctx.cat[rows]])
    if tab == "dog": return np.array(["dog:" + str(x) for x in ctx.dogid[rows]])
    if tab == "dun": return np.array(["dun:" + str(x) for x in ctx.dcat[rows]])
    raise ValueError(tab)


SRC_TAB = {"S1": "smit", "S1-L2": "smit", "S2": "smit", "S3": "smit", "S4-Smit-kapali": "dog", "S4-Dunford-kapali": "dog", "K-L1": "dog", "K-L2": "dog"}
TGT_TAB = {"S1": "smit", "S1-L2": "smit", "S2": "smit", "S3": "dun", "S4-Smit-kapali": "smit", "S4-Dunford-kapali": "dun", "K-L1": "dog", "K-L2": "dog"}


def rec(scen, fold, kume, kontrol, ok, note=""):
    DET.append(dict(senaryo=scen, dis_katman=fold, kume=kume, kontrol=kontrol, sonuc="PASS" if ok else "FAIL", ayrinti=note)); return ok


# ------------------------------------------------------------------ A1: outer training / inner CV / C / DR sets
FOLD_OK = {}   # (scen, fold) -> dict(sub_check -> bool)
for scen in SC.SCENS:
    st, tt = SRC_TAB[scen], TGT_TAB[scen]; same = st == tt
    for f in SC.folds(ctx, scen):
        fold = f["fold"]; tr = np.asarray(f["tr"], int); K = f["K"]
        Pt = f["Pt"]; te = np.where(~np.isnan(Pt[:, 0]))[0]
        test_inds = set(ind_of(tt, te))
        # bag content: target individual = bag individual, rows inside te
        tes = set(te.tolist()); bag_ok = True; bag_note = ""
        for bt, lst in f["bags"].items():
            for (ind, bid, rows) in lst:
                ii = set(ind_of(tt, rows));
                if len(ii) != 1 or (tt + ":" + str(ind)) not in ii and str(ind) not in {x.split(":", 1)[1] for x in ii}: bag_ok = False; bag_note = f"{bt}/{ind}/{bid}: bireyler {sorted(ii)}"
                if not set(np.asarray(rows).tolist()) <= tes: bag_ok = False; bag_note = f"{bt}/{ind}/{bid}: torba satırı test dışında"
        # fold test individuals = bag individuals (coverage)
        src_ind = ind_of(st, tr); sset = set(src_ind)
        k_outer = len(sset & test_inds) == 0 and (not same or len(set(tr.tolist()) & tes) == 0)
        rec(scen, fold, "dış eğitim", "k1", k_outer, f"eğitim birey {len(sset)}, test birey {len(test_inds)}, kesişim {sorted(sset & test_inds)}")
        # inner CV (GroupKFold(4), group = individual)
        oof = np.asarray(f["oof"], float); ok_inner = (len(oof) == len(tr)) and np.isfinite(oof).all(); note = f"oof {oof.shape}"
        gk = list(GroupKFold(4).split(np.zeros(len(tr)), groups=src_ind))
        for a, (itr, ite) in enumerate(gk):
            gi, gt = set(src_ind[itr]), set(src_ind[ite])
            ok_inner &= len(gi & gt) == 0 and len(gt & test_inds) == 0 and len(gi & test_inds) == 0 and (gi | gt) <= sset
        ok_inner &= sum(len(ite) for _, ite in gk) == len(tr)
        rec(scen, fold, "iç CV (GroupKFold 4, grup=birey)", "k1", ok_inner, note)
        # seconds used to estimate C, Cp, prior and the DM histograms = tr (oof); recomputed with the stored C
        ys = np.asarray(f["ys"]); Cn = Q.conf_matrix(ys[tr], oof.argmax(1), K); Cpn = Q.soft_conf_matrix(ys[tr], oof, K)
        dC = float(np.abs(Cn - np.asarray(f["C"])).max()); dCp = float(np.abs(Cpn - np.asarray(f["Cp"])).max())
        rec(scen, fold, "C ve sınıf koşullu sonsal (Cp): tr oof'tan yeniden hesap", "k1", dC < 1e-9 and dCp < 1e-9 and k_outer, f"maks|ΔC|={dC:.1e}, maks|ΔCp|={dCp:.1e}")
        # DM histograms are also on the same oof/tr
        H, pres = L8.DM.class_cond_hists(oof, ys[tr], K); rec(scen, fold, "DM sınıf koşullu histogramlar (oof, tr)", "k1", bool(np.isfinite(H).all()) and k_outer, f"mevcut sınıf {int(pres.sum())}/{K}")
        # DR training bags
        for nm, lst, rowpos in (("DR-APP eğitim torbaları", f["app"], 1), ("DR-doğal eğitim torbaları", f["nat"], 1)):
            inds = set(); rws = set()
            for item in lst: inds |= set(ind_of(st, item[rowpos])); rws |= set(np.asarray(item[rowpos]).tolist())
            ok = len(inds & test_inds) == 0 and (not same or len(rws & tes) == 0)
            rec(scen, fold, nm, "k1", ok, f"torba {len(lst)}, birey {len(inds)}, test kesişimi {sorted(inds & test_inds)}")
        rec(scen, fold, "değerlendirme torbaları ⊂ test satırları, tek birey", "k1", bag_ok, bag_note)
        FOLD_OK[(scen, fold)] = True
    log("A1", scen, "%.0f s" % (time.time() - t0))

# --- R-D (Dunford LOCO, v2) and S1-closed (Smit LOCO, care removed): stored te + rebuilt tr
for c in sorted(set(ctx.dcat)):
    d = pickle.load(open(P.R06 / f"RD_v2_{c}.pkl", "rb")); te = d["te"].astype(int); tr = np.where(ctx.dcat != c)[0]; oof = np.asarray(d["src"]["oof"], float); src_ind = ind_of("dun", tr); ti = set(ind_of("dun", te))
    ok = ti == {"dun:" + str(c)} and len(set(src_ind) & ti) == 0 and len(set(tr.tolist()) & set(te.tolist())) == 0 and len(oof) == len(tr)
    gk = list(GroupKFold(4).split(np.zeros(len(tr)), groups=src_ind)); ok_in = all(len(set(src_ind[a]) & set(src_ind[b])) == 0 and len((set(src_ind[a]) | set(src_ind[b])) & ti) == 0 for a, b in gk)
    ys = ctx.yd; Cn = Q.conf_matrix(ys[tr], oof.argmax(1), 3); Cpn = Q.soft_conf_matrix(ys[tr], oof, 3)
    okC = np.abs(Cn - d["src"]["C"]).max() < 1e-9 and np.abs(Cpn - d["src"]["Cp"]).max() < 1e-9
    rec("R-D", c, "dış eğitim", "k1", ok, f"eğitim birey {len(set(src_ind))}, test {sorted(ti)}"); rec("R-D", c, "iç CV (GroupKFold 4, grup=birey)", "k1", ok_in and len(oof) == len(tr), f"oof {oof.shape}")
    rec("R-D", c, "C ve sınıf koşullu sonsal (Cp): tr oof'tan yeniden hesap", "k1", bool(okC) and ok, "")
    rec("R-D", c, "değerlendirme torbaları ⊂ test satırları, tek birey", "k1", ti == {"dun:" + str(c)}, "")
idx2 = ctx.idx_all[ctx.y1[ctx.idx_all] < 2]
for c in ctx.CATS:
    te = idx2[ctx.cat[idx2] == c]; tr = idx2[ctx.cat[idx2] != c]; src_ind = ind_of("smit", tr); ti = {"smit:" + c}
    ok = set(ind_of("smit", te)) == ti and len(set(src_ind) & ti) == 0 and len(set(tr.tolist()) & set(te.tolist())) == 0
    gk = list(GroupKFold(4).split(np.zeros(len(tr)), groups=src_ind)); ok_in = all(len(set(src_ind[a]) & set(src_ind[b])) == 0 and len((set(src_ind[a]) | set(src_ind[b])) & ti) == 0 for a, b in gk)
    rec("S1-kapali", c, "dış eğitim", "k1", ok, f"eğitim birey {len(set(src_ind))}"); rec("S1-kapali", c, "iç CV (GroupKFold 4, grup=birey)", "k1", ok_in, "kayıtlı oof yok; bölme yeniden kuruldu")
    rec("S1-kapali", c, "C ve sınıf koşullu sonsal (Cp): tr oof'tan yeniden hesap", "k1", ok and ok_in, "kayıtlı C yok; küme kontrolü")
    rec("S1-kapali", c, "değerlendirme torbaları ⊂ test satırları, tek birey", "k1", ok, "")

# ------------------------------------------------------------------ A2: target-labelled arms — reproduce the draws
lab, y1 = ctx.lab, ctx.y1; cat_full = ctx.cat; CATS = ctx.CATS; idx_all = ctx.idx_all
SEC = lab.sec.values.astype("datetime64[s]").astype(np.int64)
STR_S = [list(a) for a in np.array_split(np.random.default_rng(SEED + 1).permutation(CATS), 4)]
DCATS = sorted(set(ctx.dcat)); STR_D = [list(a) for a in np.array_split(np.random.default_rng(SEED + 2).permutation(DCATS), 3)]
dun = ctx.dun; dcid = pd.factorize(dun.chunk.astype(str) + "_" + dun["sub"].astype(str))[0]


def load_saved():
    fs = [p for p in P.R05.glob("E_*.parquet") if not p.name.endswith("_miss.parquet")] + [p for p in P.R06.glob("E1_*.parquet") if not p.name.endswith("_miss.parquet")] + \
         [p for p in (P.INTERIM / "r07").glob("T_*.parquet") if not (p.name.endswith("_miss.parquet") or p.name.endswith("_cnt.parquet"))]
    df = pd.concat([pd.read_parquet(p).assign(src_file=p.name) for p in fs], ignore_index=True)
    return df


SAVED = load_saved(); log("kayıtlı E/T parquet", SAVED.shape)
REF = SAVED[SAVED.arm.isin(["CC", "CC-tab"])]
NMAP = {}
for r in REF.itertuples(): NMAP.setdefault((r.scen, r.variant, r.level, r.arm, int(r.m), int(r.rep), str(r.ind), str(r.bag)), set()).add(int(r.N))


def chk_birey(scen, var, tag, arm, level, c, m, rep, labrows, tgt_ind_arr, bags, rem_pos):
    """labrows: rows of the target space; bags [(w, rows)] in the target space. Returns (ok_set, ok_N, n_compared)."""
    ok_set = bool(len(labrows) == 0 or set(tgt_ind_arr[labrows]) == {c})
    okN = True; nc = 0; labset = set(np.asarray(labrows).tolist())
    for w, rows in bags:
        rr = rows[rem_pos[rows]] if rem_pos is not None else rows
        if len(rr) == 0: continue
        key = (scen, var, level, arm, m, rep, str(c), str(w))
        if key in NMAP:
            nc += 1
            if NMAP[key] != {len(rr)}: okN = False
        if labset & set(rr.tolist()): okN = False
    return ok_set, okN, nc


RES = {}   # (scen,var,level,arm_tag,fold) -> [ok_set, ok_N, n_cmp, n_draws]


def add_res(key, ok_set, okN, nc):
    r = RES.setdefault(key, [True, True, 0, 0]); r[0] &= ok_set; r[1] &= okN; r[2] += nc; r[3] += 1


# --- ladder (S1, S2) individual: random block (29) and stratified (45)
BAGS30 = ctx.nat30
for ci, c in enumerate(CATS):
    for scen in ("S1", "S2"):
        d = pickle.load(open(P.R05 / f"ladder_{scen}_{c}.pkl", "rb")); te = d["te"].astype(int); pos = {int(g): i for i, g in enumerate(te)}
        yte = y1[te]; ts = SEC[te]; srt = np.argsort(ts, kind="stable"); tind = np.array([c] * len(te)); bags = [(w, np.array([pos[int(g)] for g in ix if int(g) in pos])) for cc, w, ix in BAGS30 if cc == c]
        runs = ST.make_runs(srt, ts[srt], yte[srt], 3)
        for m in (60, 300, 900, 1800):
            for rep in range(20):
                if m in (300, 1800):
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, 400 + ci, m, rep, 7 if scen == "S1" else 8]))
                    s = L6.pick_blocks(ts[srt], m // 60, rng, []); labr = np.concatenate([srt[a:a + 60] for a in s]) if s else np.array([], int)
                    rem = np.ones(len(te), bool); rem[labr] = False
                    o = chk_birey(scen, "30dk_ge50", "rb", "CC", "birey", c, m, rep, labr, tind, bags, rem); add_res((scen, "rastgele blok", "birey", c), *o)
                rng = np.random.default_rng(np.random.SeedSequence([SEED, 800 + ci, m, rep, 7 if scen == "S1" else 8]))
                lr, cap = ST.draw_strat(runs, 3, m, rng); rem = np.ones(len(te), bool); rem[lr] = False
                o = chk_birey(scen, "30dk_ge50", "tab", "CC-tab", "birey", c, m, rep, lr, tind, bags, rem); add_res((scen, "tabakalı", "birey", c), *o)
log("A2 merdiven", "%.0f s" % (time.time() - t0))

# --- S4-Smit (open/closed): individual and study level, random block (28) and stratified (45)
S5 = ctx.S5; XS = ctx.XC17
for var in ("kapali", "acik"):
    Kt = 2 if var == "kapali" else 3; acik = var == "acik"
    tg = L6.SmitTarget(lab, y1, XS, SEC, CATS, idx_all, (y1 < 2) if var == "kapali" else None, S5["P4s"])
    runs_c = {c: ST.make_runs(tg.sorted_c[c], tg.t[tg.sorted_c[c]], tg.y[tg.sorted_c[c]], Kt) for c in CATS}
    for ci, c in enumerate(CATS):
        si = [i for i, s in enumerate(STR_S) if c in s][0]; pool = [x for x in CATS if x not in STR_S[si]]; bg = tg.bags.get(c, []); strat_set = set(STR_S[si])
        for m in (60, 300, 900, 1800):
            for rep in range(20):
                vc = 2 + acik
                labI = tg.draw([c], m, np.random.default_rng(np.random.SeedSequence([SEED, 200 + ci, m, rep, vc]))); rem = np.ones(len(tg.y), bool); rem[labI] = False
                o = chk_birey("S4-Smit", var, "rb", "CC", "birey", c, m, rep, labI, tg.ind, bg, rem); add_res(("S4-Smit-" + var, "rastgele blok", "birey", c), *o)
                labW = tg.draw(pool, m, np.random.default_rng(np.random.SeedSequence([SEED, 100 + si, m, rep, vc])))
                okW = bool(len(labW) == 0 or (set(tg.ind[labW]) & strat_set) == set()) ; add_res(("S4-Smit-" + var, "rastgele blok", "çalışma", c), okW, True, 0)
                vs = 12 + acik
                labIs = ST.draw_strat(runs_c[c], Kt, m, np.random.default_rng(np.random.SeedSequence([SEED, 900 + ci, m, rep, vs])))[0]; rem2 = np.ones(len(tg.y), bool); rem2[labIs] = False
                o = chk_birey("S4-Smit", var, "tab", "CC-tab", "birey", c, m, rep, labIs, tg.ind, bg, rem2); add_res(("S4-Smit-" + var, "tabakalı", "birey", c), *o)
                labWs = ST.draw_strat(ST.merge_runs([runs_c[x] for x in pool], Kt), Kt, m, np.random.default_rng(np.random.SeedSequence([SEED, 950 + si, m, rep, vs])))[0]
                okWs = bool(len(labWs) == 0 or (set(tg.ind[labWs]) & strat_set) == set()); add_res(("S4-Smit-" + var, "tabakalı", "çalışma", c), okWs, True, 0)
    log("A2 S4-Smit", var, "%.0f s" % (time.time() - t0))

# --- Dunford (S3, S4-Dunford): study level
for scen, var, Kt, vr, vs, drop in (("S3", "-", 3, 1, 21, False), ("S4-Dunford", "kapali", 2, 2, 22, True), ("S4-Dunford", "acik", 3, 3, 23, False)):
    tgt = L6.DunTarget(ctx.yd, ctx.dcat, dcid, ctx.Xd, DCATS, drop)
    bags = {c: len(tgt.rows_c[c]) >= 60 for c in DCATS}
    runs_c = {}
    for c in DCATS:
        lst = []
        for k, rows in tgt.chunks_c[c].items(): lst.append(ST.make_runs(rows, np.arange(len(rows)), tgt.y[rows], Kt))
        runs_c[c] = ST.merge_runs(lst, Kt)
    for si, strat in enumerate(STR_D):
        pool = [c for c in DCATS if c not in strat]
        for m in (60, 300, 900, 1800):
            for rep in range(20):
                lr = tgt.draw(pool, m, np.random.default_rng(np.random.SeedSequence([SEED, 300 + si, m, rep, vr])))
                add_res((f"{scen}-{var}", "rastgele blok", "çalışma", f"katman{si}"), bool(len(lr) == 0 or (set(tgt.ind[lr]) & set(strat)) == set()), True, 0)
                ls = ST.draw_strat(ST.merge_runs([runs_c[x] for x in pool], Kt), Kt, m, np.random.default_rng(np.random.SeedSequence([SEED, 1000 + si, m, rep, vs])))[0]
                add_res((f"{scen}-{var}", "tabakalı", "çalışma", f"katman{si}"), bool(len(ls) == 0 or (set(tgt.ind[ls]) & set(strat)) == set()), True, 0)
    log("A2 Dunford", scen, var, "%.0f s" % (time.time() - t0))

ARM_TARGET = {"rastgele blok": ["T-ACC", "PACC-hedef", "PACC-buzme", "Yeniden-egitim", "Yeniden-egitim-agirlikli", "Yalniz-hedef-esl"], "tabakalı": ["T-ACC-tab", "PACC-hedef-tab", "Yalniz-hedef-tab"]}
# detail rows: k2 / k3
for (sc_, yont, level, fold), (ok_set, okN, nc, nd) in RES.items():
    if level == "birey":
        rec(sc_, fold, f"hedef etiketli {yont} / birey: etiketli saniyeler test bireyinden", "k2_birey_kaynak", ok_set, f"çekiliş {nd}")
        rec(sc_, fold, f"hedef etiketli {yont} / birey: m saniye torbalardan çıkarılmış (kayıtlı N ile {nc} karşılaştırma)", "k3", okN, f"çekiliş {nd}, kayıtlı N karşılaştırılan {nc}")
    else:
        rec(sc_, fold, f"hedef etiketli {yont} / çalışma: etiket havuzu test katmanı dışında", "k2", ok_set, f"çekiliş {nd}")

# ------------------------------------------------------------------ A3: bag matching (k4)
def pair_check(df, keys, armcol="arm", refarms=("CC",), label=""):
    """for each key group: the arm's keys are a subset of the CC keys and N is the same. Returns {arm: (ok, n_keys, n_CC)}"""
    out = {}
    ref = df[df[armcol].isin(refarms)]; rmap = {tuple(r[:-1]): r[-1] for r in ref[keys + ["N"]].itertuples(index=False, name=None)}
    for arm, g in df.groupby(armcol):
        if arm in refarms: continue
        bad = 0; nk = 0
        for r in g[keys + ["N"]].itertuples(index=False, name=None):
            nk += 1; k = tuple(r[:-1])
            if k not in rmap or rmap[k] != r[-1]: bad += 1
        out[arm] = (bad == 0, nk, bad)
    return out


K4 = []
B8 = pd.read_csv(T / "r08_bag_level.csv", usecols=["senaryo", "torba_turu", "kol", "tanimlayici", "dis_katman", "hedef_birey", "torba", "N", "sinif"]); B8 = B8[B8.sinif == B8.groupby(["senaryo", "torba_turu", "kol", "tanimlayici", "dis_katman", "hedef_birey", "torba"]).sinif.transform("first")]
B8["armx"] = B8.kol + np.where(B8.tanimlayici == "-", "", "-" + B8.tanimlayici)
for (sc_, bt), g in B8.groupby(["senaryo", "torba_turu"]):
    r = pair_check(g, ["dis_katman", "hedef_birey", "torba"], "armx", ("CC",))
    for arm, (ok, nk, bad) in r.items(): K4.append((sc_, bt, arm, ok, f"{nk} torba, uyumsuz {bad}"))
# source arms and target-labelled arms: R05/R06/R07 bag parquet files
def rd(p): return pd.read_parquet(p)
_r5 = pd.concat([rd(P.R05 / "r05_ladder_bags.parquet"), rd(P.R05 / "r05_sources_bags.parquet")], ignore_index=True)
_r5 = _r5[~_r5.scen.isin(["S3", "S4-Dunford", "R-D"])]   # Dunford: v2 only (R06 B); the v1 numbers were invalidated in R06 (same as 32_r06_C_naive.py)
SRC = pd.concat([_r5, rd(P.R06 / "B_v2_bags.parquet"), rd(P.R06 / "E4_S1kapali_bags.parquet")], ignore_index=True)
for (sc_, var), g in SRC.groupby(["scen", "variant"]):
    g = g[g.m == 0]
    if "CC" not in set(g.arm): continue
    r = pair_check(g, ["rep", "ind", "bag"], "arm", ("CC",))
    for arm, (ok, nk, bad) in r.items(): K4.append((f"{sc_}", var, arm, ok, f"{nk} torba, uyumsuz {bad}"))
for (sc_, var, lv), g in SAVED.groupby(["scen", "variant", "level"]):
    ref_arms = ["CC"] if "CC" in set(g.arm) else []
    ref = g[g.arm == "CC"]
    # there is no CC at 'calisma' (study) level in the E files (the individual CC is shared); compare with the individual CC keys
    ref = SAVED[(SAVED.scen == sc_) & (SAVED.variant == var) & (SAVED.arm == "CC")].drop_duplicates(["m", "rep", "ind", "bag"])
    rm = {(int(r.m), int(r.rep), str(r.ind), str(r.bag)): int(r.N) for r in ref.itertuples()}
    refT = SAVED[(SAVED.scen == sc_) & (SAVED.variant == var) & (SAVED.arm == "CC-tab")].drop_duplicates(["m", "rep", "ind", "bag"])
    rmT = {(int(r.m), int(r.rep), str(r.ind), str(r.bag)): int(r.N) for r in refT.itertuples()}
    for arm, a in g.groupby("arm"):
        if arm in ("CC", "CC-tab"): continue
        mp = rmT if arm.endswith("-tab") else rm
        a = a.drop_duplicates(["m", "rep", "ind", "bag"]); bad = 0; miss = 0
        for r in a.itertuples():
            k = (int(r.m), int(r.rep), str(r.ind), str(r.bag))
            if k not in mp: miss += 1
            elif mp[k] != int(r.N) and lv == "birey": bad += 1
        # some arms at study level (Dunford) have no rem in the bag; N must equal that of CC
        if lv == "calisma" and sc_ in ("S4-Smit",): pass
        K4.append((sc_, f"{var}/{lv}", arm, (bad == 0 and miss == 0), f"{len(a)} torba, CC'de bulunmayan {miss}, N uyumsuz {bad}"))
for (sc_, vv, arm, ok, note) in K4: rec(sc_, vv, f"torba eşleşmesi: {arm}", "k4", ok, note)
log("A3 torba eşleşmesi", len(K4), "%.0f s" % (time.time() - t0))

# ------------------------------------------------------------------ summary: scenario × outer fold × arm
DET_DF = pd.DataFrame(DET)
ARMS_SRC = ["CC", "PCC", "ACC", "PACC", "EMQ", "N-hedef", "DM", "DR-APP-q", "DR-dogal-q", "DR-APP-h", "DR-dogal-h"]
NEED = {"CC": ["dış eğitim", "değerlendirme"], "PCC": ["dış eğitim", "değerlendirme"], "ACC": ["dış eğitim", "iç CV", "C ve"], "PACC": ["dış eğitim", "iç CV", "C ve"], "EMQ": ["dış eğitim", "iç CV", "C ve"],
        "N-hedef": ["dış eğitim"], "DM": ["dış eğitim", "iç CV", "DM sınıf", "değerlendirme"], "DR-APP-q": ["dış eğitim", "DR-APP"], "DR-dogal-q": ["dış eğitim", "DR-doğal"], "DR-APP-h": ["dış eğitim", "iç CV", "DR-APP"], "DR-dogal-h": ["dış eğitim", "iç CV", "DR-doğal"]}
k1d = DET_DF[DET_DF.kontrol == "k1"]; rows = []
SCEN_NAME = {s: s for s in SC.SCENS}
for (scen, fold), g in k1d.groupby(["senaryo", "dis_katman"], sort=False):
    for arm in ARMS_SRC:
        if arm == "N-hedef" and scen.startswith("K-"): continue
        sub = g[g.kume.apply(lambda s: any(s.startswith(n) for n in NEED[arm]))]
        ok = bool((sub.sonuc == "PASS").all()) and len(sub) > 0
        rows.append(dict(senaryo=scen, dis_katman=fold, kol=arm, k1="PASS" if ok else "FAIL", k2="NA", k3="NA", k4="NA", muaf="", ayrinti=f"{len(sub)} küme kontrolü"))
for (sc_, yont, level, fold), (ok_set, okN, nc, nd) in RES.items():
    arms = ARM_TARGET[yont] if sc_.startswith("S4-Smit") or sc_.startswith("S3") or sc_.startswith("S4-Dunford") else (["T-ACC", "PACC-hedef", "PACC-buzme", "Yeniden-egitim", "Yeniden-egitim-agirlikli"] if yont == "rastgele blok" else ["T-ACC-tab", "PACC-hedef-tab"])
    for arm in arms:
        rows.append(dict(senaryo=sc_, dis_katman=fold, kol=f"{arm} [{level}, {yont}]", k1="NA", k2=("PASS" if ok_set else "FAIL") if level == "çalışma" else "NA", k3=("PASS" if okN else "FAIL") if level == "birey" else "NA", k4="NA", muaf="", ayrinti=f"{nd} çekiliş; kayıtlı N karşılaştırma {nc}"))
for (sc_, vv, arm, ok, note) in K4:
    rows.append(dict(senaryo=sc_, dis_katman=vv, kol=arm, k1="NA", k2="NA", k3="NA", k4="PASS" if ok else "FAIL", muaf="", ayrinti=note))
rows.append(dict(senaryo="S0 (sızıntılı)", dis_katman="rastgele %70/%30 saniye", kol="tüm kollar", k1="MUAF", k2="MUAF", k3="MUAF", k4="NA", muaf="S0 tanım gereği sızıntılı (rastgele saniye bölmesi)", ayrinti="k1–k3 uygulanmaz"))
AUDIT = pd.DataFrame(rows)
AUDIT.to_csv(T / "r09_audit.csv", index=False); DET_DF.to_csv(T / "r09_audit_detail.csv", index=False)
cnt = {k: (AUDIT[k] == "PASS").sum() for k in ("k1", "k2", "k3", "k4")}; fl = {k: (AUDIT[k] == "FAIL").sum() for k in ("k1", "k2", "k3", "k4")}
log("PASS", cnt, "FAIL", fl, "MUAF", int((AUDIT.k1 == "MUAF").sum()))
json.dump(dict(PASS={k: int(v) for k, v in cnt.items()}, FAIL={k: int(v) for k, v in fl.items()}, muaf=int((AUDIT.k1 == "MUAF").sum()), n_rows=int(len(AUDIT)), n_detail=int(len(DET_DF)), sure_s=time.time() - t0), open(T / "r09_audit_summary.json", "w"))
for r in AUDIT[(AUDIT[["k1", "k2", "k3", "k4"]] == "FAIL").any(axis=1)].itertuples(): log("FAIL SATIRI:", r)
log("BITTI", "%.0f s" % (time.time() - t0))
