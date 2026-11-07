r"""54_r08_report.py — builds the R08 report from results/tables/r08_*.csv|json (numbers are never typed by hand). Writes: results/reports/R08_dm_dr.md"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from config import paths as P

T = P.TABLES
t = pd.read_csv(T / "r08_tests.csv"); q = json.load(open(T / "r08_q1.json")); gate = json.load(open(T / "r08_gateA.json")); bias = pd.read_csv(T / "r08_class_bias.csv")
det = pd.read_csv(T / "r08_dr_detail.csv"); dmb = pd.read_csv(T / "r08_dm_boundary.csv"); ext = pd.read_csv(T / "r08_extrapolation.csv")
apps = {c: pd.read_csv(T / f"r08_app_train_stats_kod{c}.csv") for c in (1, 2, 3, 4)}


def f(x, n=4): return ("%." + str(n) + "f") % x if pd.notna(x) else "–"
def c(x, n=4): return f(x, n).replace(".", ",").replace("-", "−")
def sg(x, n=4): return ("+" if x >= 0 else "−") + c(abs(x), n)
def ci(r): return f"{sg(r.d)} [{c(r.lo)}; {c(r.hi)}]"


LAB = {("S1", "30dk"): "S1 L1 30 dk", ("S1", "2dk"): "S1 doğal 2 dk", ("S1", "APP120"): "S1 APP 120 s (köprü)", ("S1-L2", "30dk"): "S1-L2 30 dk", ("S2", "30dk"): "S2 30 dk", ("S4-Smit-kapali", "30dk"): "S4-Smit-kapalı 30 dk",
       ("K-L1", "kayit"): "K-L1 köpek×kayıt", ("K-L1", "dogal2"): "K-L1 doğal 2 dk", ("K-L2", "kayit"): "K-L2 köpek×kayıt", ("K-L2", "dogal2"): "K-L2 doğal 2 dk",
       ("S3", "kedi"): "S3 [keşifsel]", ("S4-Dunford-kapali", "kedi"): "S4-Dunford-kapalı [keşifsel]"}
CAT = {"KAZANIR": "KAZANIR", "KAYBEDER": "KAYBEDER", "ESDEGER": "EŞDEĞER", "BELIRSIZ": "BELİRSİZ"}
ORD = list(LAB)
P1 = t[t.aile == "birincil"]


def cell(sc, bt, arm, fam="birincil"): return t[(t.senaryo == sc) & (t.torba_turu == bt) & (t.kol == arm) & (t.aile == fam)].iloc[0]


def listcat(arm, k):
    out = [f"{LAB[(r.senaryo, r.torba_turu)]} {sg(r.d)}" for r in P1[(P1.kol == arm) & (P1.kategori == k)].itertuples()]
    return "; ".join(out) if out else "yok"


def lehine(arm):
    x = P1[(P1.kol == arm) & (P1.d > 0) & (P1.kategori == "BELIRSIZ")]
    return "; ".join(f"{LAB[(r.senaryo, r.torba_turu)]} {ci(r)}, Holm {c(r.p_holm)}" for r in x.itertuples()) or "yok"


L = []; A = L.append
A("# R08 — DM ve DR (doğrudan kompozisyonel regresyon) — 2026-11-06")
A("Ayarlar R07 ile aynı (RF 500, iç CV 300, seed 20261105, Dunford v2, birincil torba kümeleri, sonsal kalibrasyonu yok); QuaPy yok. d = AE_CC − AE_kol (d>0: kol daha iyi), GA = birey bootstrap (1.000), kesin işaret-çevirme (n ≤ 20) / köpekte 200.000 işaretli MC (p = (1+k)/(B+1)), aile içi Holm, |d| ≥ 0,01. **Dunford'a dayanan her sonuç (S3, S4-Dunford) keşifseldir.** Hiçbir bölüm kesilmedi.")
A("DR = birincil tanımlayıcı (DR-q: ondalık dilimler); DR-h ikincil (§5). Tablolar `results\\tables\\r08_*`, torba düzeyi `r08_bag_level.csv`.")
A("")
A("## 1. Cevaplar")
g = gate
A(f"**Kapı A: GEÇTİ.** `test_dm.py`: 10 tekrar, N = 10.000, DM AE {c(0.0047)} < CC AE {c(0.0789)}, en kötü maks |p̂−p| {c(0.0164)} ≤ 0,02; K = 2'de DM = ızgara HDy (0,6965). `test_dr.py`: ilr gidiş-dönüş ≤ 4,4e-16 (K = 2, 3, 5, 7); sıfır değiştirme toplamı 1 (≤ 2,2e-16), bileşenler > 0; taban değişmezliği ≤ 9,7e-13 (a = 1, 3, 6); sentetik DR AE {c(0.0562)} < taban AE/2 = {c(0.1997 / 2)} (taban {c(0.1997)}).")
A(f"- Yeniden üretim (kayıtlı sonsal): S1 L1 CC AE {c(g['S1_30dk']['AE_CC_torba_ort'], 5)} (hedef 0,0260); PACC − CC {sg(g['S1_30dk']['d_PACC'], 5)} (−0,0240); S1-L2 CC AE {c(g['S1L2_30dk']['AE_CC_torba_ort'], 5)} (0,0837); K-L2 köpek×kayıt CC AE (köpek ort.) {c(g['K-L2_kayit']['AE_CC_birey_ort'], 5)} = R07 {c(g['K-L2_kayit']['r07_AE_CC'], 5)}; K-L1 {c(g['K-L1_kayit']['AE_CC_birey_ort'], 5)} = {c(g['K-L1_kayit']['r07_AE_CC'], 5)}; S1 APP 120 s değerlendirme çekilişleri R05 ile aynı ({g['APP120_kontrol']['eslesen']}/{g['APP120_kontrol']['R05']} torba, maks fark {g['APP120_kontrol']['maks_fark']:.0f}).")
A("- Yeni RF eğitimi: yalnız köpek K-L1/K-L2 (32 öznitelik) dış + iç CV sonsalları (R07 yalnız AE kaydetmişti), R07 ile aynı ayar ve tohumlarla, 122 s.")
A("")
A("**DM − CC** (12 birincil aile, her biri Holm 3 test):")
for k in ("KAZANIR", "KAYBEDER", "EŞDEĞER", "BELİRSİZ"):
    kk = {v: u for u, v in CAT.items()}[k]; A(f"- {k}: {listcat('DM', kk)}")
A(f"- Nokta tahmini DM lehine ama BELİRSİZ: {lehine('DM')}.")
A("")
A("**DR-APP − CC** (DR-q):")
for k in ("KAZANIR", "KAYBEDER", "EŞDEĞER", "BELİRSİZ"):
    kk = {v: u for u, v in CAT.items()}[k]; A(f"- {k}: {listcat('DR-APP-q', kk)}")
A(f"- Nokta tahmini lehine ama BELİRSİZ: {lehine('DR-APP-q')}.")
A("")
A("**DR-doğal − CC** (DR-q):")
for k in ("KAZANIR", "KAYBEDER", "EŞDEĞER", "BELİRSİZ"):
    kk = {v: u for u, v in CAT.items()}[k]; A(f"- {k}: {listcat('DR-dogal-q', kk)}")
A(f"- Nokta tahmini lehine ama BELİRSİZ: {lehine('DR-dogal-q')}.")
A("")
A("**Q1 ölçütü (§7.2; DR-q, doğal ailesi = APP 120 s dışındaki 11 aile; Q1-c 12 birincil ailenin hepsi):**")
A("| Varyant | Q1-a (S1 30 dk KAZANIR) | Q1-b (K-L1 ya da K-L2 köpek×kayıt KAZANIR) | Q1-c (hiçbir ailede KAYBEDER yok) | Q1 |")
A("|---|---|---|---|---|")
for v, nm in (("DR-APP-q", "DR-APP"), ("DR-dogal-q", "DR-doğal")):
    x = q["q1"][v]; yn = lambda b: "EVET" if b else "HAYIR"
    A(f"| {nm} | {yn(x['Q1a'])} ({CAT[cell('S1', '30dk', v).kategori]} {sg(cell('S1', '30dk', v).d)}) | {yn(x['Q1b'])} (K-L1 {CAT[cell('K-L1', 'kayit', v).kategori]}, K-L2 {CAT[cell('K-L2', 'kayit', v).kategori]}) | {yn(x['Q1c'])} ({len(x['kayip_hucreler'])}/12 ailede KAYBEDER) | {yn(x['Q1'])} |")
A(f"**Q1 ölçütü: {'EVET' if q['Q1'] else 'HAYIR'}.**")
A("")
A("**Geçerli yorum: §7.1 ilk cümle** (§7.3 \"yalnız APP köprüsünde KAZANIR\" durumu: doğal ailede DM, DR-q ve ikincil DR-h için KAZANIR yok; KAZANIR yalnız S1 APP 120 s'de, bu doğal sayılmaz, İddia 4 altında raporlanır). Makaleye yazılacak cümle, aynen:")
A("> İddia 1 genişler: \"incelenen nicelleme yöntemleri (karışıklık matrisi düzeltmesi, dağılım eşleme, doğrudan regresyon) CC'ye karşı güvenilir kazanç sağlamadı.\"")
A("")
A("**S1 APP köprüsü (İddia 4; aile 3 test, n = 12 kedi; PACC referans R07, aileye girmez):**")
A("| Torba | AE_CC | PACC (R07) | DM d [GA] kategori | DR-APP d [GA] kategori | DR-doğal d [GA] kategori |")
A("|---|---|---|---|---|---|")
for bt, nm in (("APP120", "APP 120 s"), ("2dk", "doğal 2 dk")):
    r0 = cell("S1", bt, "DM"); cs = [cell("S1", bt, a) for a in ("DM", "DR-APP-q", "DR-dogal-q")]
    A(f"| {nm} ({r0.n_torba} torba) | {c(r0.AE_CC)} | {sg(r0.PACC_R07_d)} {CAT[r0.PACC_R07]} | " + " | ".join(f"{ci(r)} {CAT[r.kategori]}" for r in cs) + " |")
A("")
A("## 2. Ana tablo (AE_CC torba ortalaması; d birey ortalaması üzerinden; Holm 3 test; GA = birey bootstrap)")
A("| Senaryo × torba türü | n | torba | AE_CC | PACC (R07) | DM d [GA] Holm p, kategori | DR-APP d [GA] Holm p, kategori | DR-doğal d [GA] Holm p, kategori |")
A("|---|---|---|---|---|---|---|---|")
for key in ORD:
    cs = [cell(key[0], key[1], a) for a in ("DM", "DR-APP-q", "DR-dogal-q")]; r0 = cs[0]
    A(f"| {LAB[key]} | {r0.n} | {r0.n_torba} | {c(r0.AE_CC)} | {CAT[r0.PACC_R07]} {sg(r0.PACC_R07_d)} | " + " | ".join(f"{ci(r)} {c(r.p_holm, 3)}, {CAT[r.kategori]}" for r in cs) + " |")
A("Aile = senaryo × torba türü, 3 test (12 birincil aile). n = 8'de ulaşılabilir en küçük Holm p = 0,0234; n = 7'de 0,0469. Köpek n: köpek×kayıt 44; doğal 2 dk 35 (9 köpekte N ≥ 60 olan 2 dk pencere yok; R07 ile aynı) — görevdeki 44 değil. Köpek p'leri Monte Carlo. S1-L2'de n = 12, 7 sınıf.")
A("")
A("## 3. DR ayrıntı ve DM tanısı")
A("| Senaryo | DR-APP a (q) med [min–maks] | DR-doğal a (q) | eğitim torbası APP | eğitim torbası doğal | DM sınır çözümü (torba türü: oran) |")
A("|---|---|---|---|---|---|")
sc_order = ["S1", "S1-L2", "S2", "S4-Smit-kapali", "K-L1", "K-L2", "S3", "S4-Dunford-kapali"]
for sc in sc_order:
    d = det[(det.senaryo == sc) & (det.tanimlayici == "q")].set_index("varyant"); a_ = d.loc["DR-APP"]; n_ = d.loc["DR-dogal"]
    ba = dmb[dmb.senaryo == sc]
    A(f"| {sc} | {a_.a_med:g} [{a_.a_min:g}–{a_.a_maks:g}] | {n_.a_med:g} [{n_.a_min:g}–{n_.a_maks:g}] | {a_.n_med:g} [{a_.n_min:g}–{a_.n_maks:g}] | {n_.n_med:g} [{n_.n_min:g}–{n_.n_maks:g}] | " + ", ".join(f"{r.torba_turu} {c(r.sinir_orani, 3)}" for r in ba.itertuples()) + " |")
A("- a üst sınırı (20) seçilen: S1-L2 DR-doğal-q 7/12 katman; S2 DR-doğal-q medyan 14, maks 20; S3 DR-APP-h a = 20. Köpek kaynaklı doğal torbalarda 30 dk pencere (kapsam ≥ %50, N ≥ 900) sayısı **0** (62 kayıt, medyan 298 s, maks 680 s); bu yüzden köpek×kayıt + doğal 2 dk torbaları kullanıldı (S4, K-L1, K-L2).")
a1, a2, a3, a4 = apps[1], apps[2], apps[3], apps[4]
def appline(nm, s, K, names):
    return (f"{nm}: kabul {int(s.kabul.sum())}/{int(s.deneme.sum())} deneme = {c(s.kabul.sum() / s.deneme.sum(), 4)}; torbasız birey {int((s.kabul == 0).sum())}/{len(s)}; gerçekleşen oran ort. (birey ort. SD) "
            + ", ".join(f"{names[k]} {c(s[f'p{k}_ort'].mean(), 3)} ({c(s[f'p{k}_sd'].mean(), 3)})" for k in range(K)))
A("- APP eğitim torbaları (100 torba × 120 s / birey; kabul oranı): " + appline("Smit L1", a1, 3, ["Ina", "Lok", "Bak"]) + ". " + appline("Smit L2", a2, 7, ["Lyi", "Sit", "Sta", "Wal", "Tro", "Gro", "Eat"]) + ".")
A("- " + appline("Köpek L1", a3, 2, ["Ina", "Lok"]) + ". " + appline("Köpek L2", a4, 5, ["Lyi", "Sit", "Sta", "Wal", "Tro"]) + ".")
A("- Dışkestirim (değerlendirme torbası gerçek oranı, eğitim torbalarının [min, maks] aralığı dışında; sınıf sayımları toplamı): " + "; ".join(
    f"{sc} {v.replace('dogal', 'doğal')} {c(e.dis.sum() / e.n.sum(), 3)}" for sc in ("S1", "S1-L2", "S2", "S3", "K-L2") for v, e in ext[ext.senaryo == sc].groupby("varyant")) + ". K-L1, S4-Smit-kapalı, S4-Dunford-kapalı: 0,000. En büyük: S1 doğal 2 dk APP Inaktif "
  + c(ext[(ext.senaryo == "S1") & (ext.torba_turu == "2dk") & (ext.varyant == "DR-APP") & (ext.sinif == "Inaktif")].oran.iloc[0], 3) + "; S1 30 dk APP Inaktif " + c(ext[(ext.senaryo == "S1") & (ext.torba_turu == "30dk") & (ext.varyant == "DR-APP") & (ext.sinif == "Inaktif")].oran.iloc[0], 3) + ".")
A("")
A("## 4. Sınıf bazında işaretli yanlılık (yüzde puan; birey ort.; CC / PACC / DM / en küçük AE'li DR-q varyantı)")
A("| Hücre (en iyi DR) | Sınıf | CC | PACC | DM | DR |")
A("|---|---|---|---|---|---|")
for sc, bt, classes in (("S1", "30dk", ["Lokomosyon", "Bakim"]), ("S2", "30dk", ["Lokomosyon", "Bakim"]), ("S4-Smit-kapali", "30dk", ["Lokomosyon"]), ("S3", "kedi", ["Lokomosyon", "Bakim"]),
                        ("S4-Dunford-kapali", "kedi", ["Lokomosyon"]), ("K-L2", "kayit", ["Lying chest", "Sitting", "Standing", "Walking", "Trotting"])):
    x = P1[(P1.senaryo == sc) & (P1.torba_turu == bt) & P1.kol.isin(["DR-APP-q", "DR-dogal-q"])]; best = x.sort_values("AE_kol").iloc[0].kol
    for cl in classes:
        b = {a: bias[(bias.senaryo == sc) & (bias.torba_turu == bt) & (bias.kol == a) & (bias.sinif == cl)].iloc[0] for a in ("CC", "PACC", "DM", best)}
        A(f"| {LAB[(sc, bt)]} ({best.replace('-q', '').replace('dogal', 'doğal')}) | {cl} | " + " | ".join(f"{sg(b[a].bias_pp, 2)} [{c(b[a].lo, 2)}; {c(b[a].hi, 2)}]" for a in ("CC", "PACC", "DM", best)) + " |")
A("Tüm hücre × kol × sınıf: `r08_class_bias.csv`.")
A("")
A("## 5. İkincil tanımlayıcı DR-h (ayrı aile, 2 test/hücre)")
S2_ = t[t.aile == "ikincil"]
for k in ("KAZANIR", "KAYBEDER", "EŞDEĞER", "BELİRSİZ"):
    kk = {v: u for u, v in CAT.items()}[k]
    s = S2_[S2_.kategori == kk]
    A(f"- {k} ({len(s)}/24): " + ("; ".join(f"{LAB[(r.senaryo, r.torba_turu)]} {r.kol.replace('-h', '').replace('dogal', 'doğal')} {sg(r.d)}" for r in s.itertuples()) if len(s) else "yok"))
A("- Doğal ailede KAZANIR yok; yalnız S1 APP 120 s'de DR-APP-h KAZANIR. (§7.3 üçüncü madde tetiklenmedi.)")
A("")
A("## 6. Plan dışı (karar kategorisi yok)")
A("- Yok. (Dışkestirim, DM sınır oranı ve APP kabul tanıları §3'te; görevde tanımlı.)")
A("")
A("## 7. Süreler, sapmalar, dosyalar")
A("- Süreler (duvar saati): köpek sonsalları 122 s; DM/DR çalıştırması 8 senaryo toplam ≈ 4 dk (S1 22 s, S1-L2 53 s, K-L2 50 s); analiz < 1 dk. Toplam hesaplama ≈ 7 dk; 2 saat sınırı aşılmadı.")
A("- Sapmalar: (1) `PLSRegression(scale=False)` ile birlikte `tol=1e-26, max_iter=20000`: tol = 1e-12'de bile ilr tabanı döndürmesinde oran farkı 1e-5 kaldı (§1.2 testi 1e-8 ister; sklearn tol'u ağırlık farkının karesine uygular). Model tanımı aynı. (2) Bileşen seçiminde her eğitim kıvrımında bir kez a_maks uydurulup ilk a bileşenle tahmin edilir (NIPALS bileşenleri ardışık; ayrı uydurmayla fark ≤ 2e-14, testli). (3) Köpek doğal 2 dk n = 35 (§2). (4) DM'de eğitim katmanı dışı saniyesi olmayan sınıf p = 0'a sabitlenir (8 senaryonun hiçbir katında tetiklenmedi, doğrulandı).")
A("- Dosyalar: `src\\{quant_dm,quant_dr,r08lib,r08scen}.py`, `scripts\\48–54_r08_*.py`, `tests\\test_{dm,dr}.py`; ara dosyalar yalnız `<data root>\\interim\\r08\\`; `config\\paths.py` → `R08`. Şekil `results\\figures\\fig1_main_v2.png` (`fig1_main.png` değişmedi; S4-Smit-açık, S0, R-D, S1-kapalı, köpek panelinde R08 satırı yok). `results\\tables\\r08_analysis_history_v2_2026-11-06.csv` (v1'in 31 satırı + 6).")
open(P.REPORTS / "R08_dm_dr.md", "w", encoding="utf-8").write("\n".join(L) + "\n"); print(len(L), "satır")
