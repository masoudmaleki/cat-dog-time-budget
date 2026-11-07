r"""53_r08_history.py — R08 E: r08_analysis_history_v2_<date>.csv = all rows of v1 (unchanged) + the R08 rows. The v1 file is not touched.
Reads: results/tables/r07_analysis_history_v1_2026-11-06.csv, r08_tests.csv, r08_q1.json. Writes: results/tables/r08_analysis_history_v2_2026-11-06.csv"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from config import paths as P

T = P.TABLES; v1 = pd.read_csv(T / "r07_analysis_history_v1_2026-11-06.csv"); t = pd.read_csv(T / "r08_tests.csv"); q = json.load(open(T / "r08_q1.json"))
lab = {"S1|30dk": "S1 30dk", "S1|2dk": "S1 2dk", "S1|APP120": "S1 APP120", "S1-L2|30dk": "S1-L2 30dk", "S2|30dk": "S2", "S4-Smit-kapali|30dk": "S4-Smit-kapali", "K-L1|kayit": "K-L1 kayit", "K-L1|dogal2": "K-L1 2dk",
       "K-L2|kayit": "K-L2 kayit", "K-L2|dogal2": "K-L2 2dk", "S3|kedi": "S3 [kesifsel]", "S4-Dunford-kapali|kedi": "S4-Dunford-kapali [kesifsel]"}


def summ(arm, fam="birincil"):
    x = t[(t.kol == arm) & (t.aile == fam)]; out = []
    for k in ("KAZANIR", "KAYBEDER", "ESDEGER", "BELIRSIZ"):
        s = x[x.kategori == k]
        if len(s): out.append(f"{k} {len(s)}/12: " + ", ".join(f"{lab[a + '|' + b]} {d:+.4f}" for a, b, d in zip(s.senaryo, s.torba_turu, s.d)))
    return "; ".join(out)


new = [
    ("R08", "DM - CC (dagilim esleme, 10 kutulu cok sinifli HDy)", "12 birincil aile; F karari; birey bootstrap + Holm 3 test", "onceden (gorev 08 metni; sonuc gorulmeden yazildi)", summ("DM"), "ana metin (Iddia 1; APP120 Iddia 4)"),
    ("R08", "DR-APP - CC (DR-q: 9 ondalik dilim; APP egitim torbalari)", "12 birincil aile; F karari", "onceden (gorev 08 metni)", summ("DR-APP-q"), "ana metin (Iddia 1)"),
    ("R08", "DR-dogal - CC (DR-q; dogal egitim torbalari)", "12 birincil aile; F karari", "onceden (gorev 08 metni)", summ("DR-dogal-q"), "ana metin (Iddia 1)"),
    ("R08", "DR-h (ikincil tanimlayici: sinif basina sonsal histogrami), APP ve dogal", "ikincil aile, 2 test/hucre", "onceden (gorev 08 metni)", "; ".join(f"{a}: " + summ(a, "ikincil") for a in ("DR-APP-h", "DR-dogal-h")), "ek"),
    ("R08", "Q1 olcutu (Q1-a Smit S1 30dk, Q1-b kopek K-L1/K-L2 kayit, Q1-c hicbir birincil ailede KAYBEDER yok; ayni varyant icin birlikte)", "islemsel tanim 7.2", "onceden (gorev 08 metni; sonuc gorulmeden sabitlendi)",
     "; ".join(f"{k}: Q1-a {'EVET' if v['Q1a'] else 'HAYIR'}, Q1-b {'EVET' if v['Q1b'] else 'HAYIR'}, Q1-c {'EVET' if v['Q1c'] else 'HAYIR'}" for k, v in q["q1"].items()) + f"; olcut {'EVET' if q['Q1'] else 'HAYIR'}", "ana metin"),
    ("R08", "7.3 ara durum kurali (dogal ailede KAZANIR / yalniz APP / ikincil)", "durum secimi", "onceden (gorev 08 metni 7.3)", f"gecerli durum: {q['durum']}; dogal KAZANIR hucre: {len(q['kazanir_dogal'])}; yalniz APP KAZANIR: {len(q['kazanir_APP'])}; ikincil dogal KAZANIR: {len(q['kazanir_ikincil_dogal'])}", "ana metin (Iddia 1 cumlesi)"),
]
v2 = pd.concat([v1, pd.DataFrame(new, columns=v1.columns)], ignore_index=True)
v2.to_csv(T / "r08_analysis_history_v2_2026-11-06.csv", index=False); print(len(v1), "->", len(v2)); print(v2.tail(6).sonuc.to_string())
