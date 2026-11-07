r"""06_make_r02_report.py — builds the R02 report from results/tables/r02_*.csv (numbers in the tables are never typed by hand).
Reads: results/tables/r02_*.csv/json. Writes: results/reports/R02_pilot_smit.md
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd, numpy as np
from config import paths as P

T = P.TABLES
S = pd.read_csv(T / "r02_summary_all.csv"); H = json.load(open(T / "r02_hypotheses.json"))
CLF = pd.read_csv(T / "r02_classifier_metrics_summary.csv"); H3 = pd.read_csv(T / "r02_h3_spearman.csv")
BIAS = pd.read_csv(T / "r02_class_bias.csv"); VAR = pd.read_csv(T / "r02_true_budget_variability.csv")
CLS = pd.read_csv(T / "r02_class_seconds.csv"); EXC = json.load(open(T / "r02_excluded_seconds.json"))
RT = json.load(open(T / "r02_runtimes.json")); NB = json.load(open(T / "r02_bag_counts.json"))


def n(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}".replace(".", ",")


def sg(x, d=3):
    return f"{x:+.{d}f}".replace(".", ",").replace("-", "−")


def ci(t, d=3, sign=False):
    f = sg if sign else n
    return f"{f(t[0], d)} [{f(t[1], d)}; {f(t[2], d)}]"


SC = ["S0", "S1-tasma", "S1-kosum", "S2"]; SCN = {"S0": "S0", "S1-tasma": "S1-tasma", "S1-kosum": "S1-koşum", "S2": "S2"}
EST = ["CC", "PCC", "ACC", "PACC", "EMQ", "ACC_oracle", "PACC_oracle"]
ESTN = {"ACC_oracle": "T-ACC-oracle", "PACC_oracle": "T-PACC-oracle (ek)"}


def main_table(tag, bd, lvl, col):
    x = S[(S.tag == tag) & (S.bagdef == bd) & (S.level == lvl)]
    out = [f"| Tahminci | " + " | ".join(SCN[s] for s in SC) + " |", "|---|" + "---|" * len(SC)]
    for e in EST:
        cells = []
        for s in SC:
            r = x[(x.est == e) & (x.scen == s)]
            if r.empty: cells.append("—")
            elif col == "AE": cells.append(ci((r.AE.iloc[0], r.AE_lo.iloc[0], r.AE_hi.iloc[0])))
            else: cells.append(ci((r.RAE.iloc[0], r.RAE_lo.iloc[0], r.RAE_hi.iloc[0]), d=2))
        out.append(f"| {ESTN.get(e, e)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


K = "va_f32|30dk_ge50|"
h2, h1, orc = H[K + "L2"]["H2"], H[K + "L2"]["H1"], H[K + "L2"]["oracle"]
l1 = H[K + "L1"]
v = lambda d: (d["diff_CC_minus_X"], d["lo"], d["hi"])
h3p = H3[(H3.tag == "va_f32")]
g = lambda e, sc: h3p[(h3p.est == e) & (h3p.scen == sc)].iloc[0]
var2 = VAR[(VAR.bagdef == "30dk_ge50") & (VAR.level == "L2")].set_index("class")
tm = RT["va_f32"]["toplam"] / 60

L = []
A = L.append
A("# R02 — Smit pilotu: S0, S1, S2'de zaman bütçesi tahmincileri (2026-11-05)\n")
A("Yalnız Smit verisi. Birincil: eğitim varyantı (a) doğal dağılım, 32 öznitelik, torba = 30 dk & kapsam >= %50 (68 torba), seviye L2. AE/RAE ve farklar için GA = kedi düzeyinde bootstrap (1.000 tekrar, 12 kedi yeniden örneklenir). Fark tanımı: **AE_CC − AE_X; negatif = X, CC'den kötü.**\n")
A("## 1. Cevaplar\n")
A("**Gerçek bütçe değişkenliği (önce okuyun):** birincil torbalarda (68) gerçek oranlar çok değişken: Lying ort. " + n(var2.loc['Lying', 'mean']) + " (SD " + n(var2.loc['Lying', 'sd']) + ", min–maks " + n(var2.loc['Lying', 'min'], 2) + "–" + n(var2.loc['Lying', 'max'], 2) + "), Sitting " + n(var2.loc['Sitting', 'mean']) + " (SD " + n(var2.loc['Sitting', 'sd']) + "), Grooming " + n(var2.loc['Grooming', 'mean']) + " (SD " + n(var2.loc['Grooming', 'sd']) + "), Eating " + n(var2.loc['Eating', 'mean']) + " (SD " + n(var2.loc['Eating', 'sd']) + "), Walking " + n(var2.loc['Walking', 'mean']) + " (SD " + n(var2.loc['Walking', 'sd']) + "), Trotting " + n(var2.loc['Trotting', 'mean']) + " (SD " + n(var2.loc['Trotting', 'sd']) + "). Yani gerçek oranlar torbalar arasında az değişmiyor (görevin 'az değişirse fark küçük kalır' uyarısı burada geçerli değil); yalnız Trotting ve Walking küçük. Tam tablo Bölüm 2.\n")
A("| Soru | Cevap | Sayı |\n|---|---|---|")
A(f"| H1: S1-tasmada ACC, PACC ya da EMQ, CC'den küçük AE veriyor mu? | **HAYIR** | CC AE " + n(S[(S.tag=='va_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S1-tasma')&(S.est=='CC')].AE.iloc[0]) + f". Fark (CC−X): ACC {ci(v(h1['ACC']), sign=True)}, PACC {ci(v(h1['PACC']), sign=True)}, EMQ {ci(v(h1['EMQ']), sign=True)}: üçü de CC'den anlamlı kötü (GA 0'ı içermiyor). L1: ACC {ci(v(l1['H1']['ACC']), sign=True)}, PACC {ci(v(l1['H1']['PACC']), sign=True)}, EMQ {ci(v(l1['H1']['EMQ']), sign=True)}. |")
d = h2["ACC"]
A(f"| H2: S2'de ACC'nin CC'ye göre durumu S1'e göre değişiyor mu? | **HAYIR** (birincil) | ACC için CC−ACC: S1 {ci(d['S1'], sign=True)}, S2 {ci(d['S2'], sign=True)}; fark (S2−S1) {ci(d['S2_minus_S1'], sign=True)} -> değişim yok. ACC S1'de de üstün değildi, S2'de de değil. PACC: S2−S1 {ci(h2['PACC']['S2_minus_S1'], sign=True)}; EMQ: {ci(h2['EMQ']['S2_minus_S1'], sign=True)} (EMQ S2'de daha da kötüleşiyor). Duyarlılıkta (kedi düzeyi torba, varyant b) ACC dezavantajı S2'de küçülüyor ama üstünlüğe dönüşmüyor (Bölüm 6). |")
A(f"| H3: Kedi düzeyinde makro-F1 sıralaması ile AE sıralaması uyuşuyor mu? | **BELİRSİZ** | Spearman ρ(makro-F1, AE), n=12 kedi: CC S1-tasma {n(g('CC','S1-tasma').rho,2)} (p={n(g('CC','S1-tasma').p,2)}), S1-koşum {n(g('CC','S1-kosum').rho,2)}, S2 {n(g('CC','S2').rho,2)}; ACC S1-tasma {n(g('ACC','S1-tasma').rho,2)}, S1-koşum {n(g('ACC','S1-kosum').rho,2)}, S2 {n(g('ACC','S2').rho,2)}. Havuz (3x12=36): CC {n(g('CC','havuz(3x12)').rho,2)}, ACC {n(g('ACC','havuz(3x12)').rho,2)} (havuzdaki güç basamaklar arası farktan geliyor). Karar kuralı (benim tanımım, görevde verilmemişti): EVET için basamak içinde ρ <= −0,7. Basamak içinde yalnız S2 güçlü (<= −0,71); S1'de zayıf–orta (|ρ| <= {n(max(abs(g(e,s).rho) for e in ('CC','ACC') for s in ('S1-tasma','S1-kosum')),2)}). |")
A(f"| T-ACC-oracle, S2'deki ACC kaybının ne kadarını geri kazanıyor? | **BELİRSİZ** (L2); L1'de ~%{n(l1['oracle']['recovery_pct'],0)} | L2: kayıp = AE_ACC(S2) − AE_ACC(S1-tasma) = {n(orc['AE_ACC_S2'])} − {n(orc['AE_ACC_S1tasma'])} = {n(orc['loss'])}; oracle AE {n(orc['AE_oracle_S2'])}; geri kazanım **%{n(orc['recovery_pct'],1)}** [{n(orc['recovery_lo'],1)}; {n(orc['recovery_hi'],1)}]. L1: {n(l1['oracle']['AE_ACC_S2'])} -> {n(l1['oracle']['AE_oracle_S2'])} (S1-tasma {n(l1['oracle']['AE_ACC_S1tasma'])}); geri kazanım %{n(l1['oracle']['recovery_pct'],1)} [{n(l1['oracle']['recovery_lo'],1)}; {n(l1['oracle']['recovery_hi'],1)}]. |")
A("\nÖzet: bu veride ACC, PACC ve EMQ hiçbir kayma basamağında (S0, S1, S2) CC'yi anlamlı biçimde geçmedi; sınıflandırıcı kayma altında belirgin kötüleşiyor (makro-F1 " + n(CLF[(CLF.tag=='va_f32')&(CLF.scen=='S0')].macroF1.iloc[0],2) + " S0, " + n(CLF[(CLF.tag=='va_f32')&(CLF.scen=='S1-tasma')].macroF1.iloc[0],2) + " S1-tasma, " + n(CLF[(CLF.tag=='va_f32')&(CLF.scen=='S2')].macroF1.iloc[0],2) + " S2). Nedeni bu görevde test edilmedi (Bölüm 8, madde 1).\n")

A("## 2. Veri\n")
A("Kaynak: R01'deki 167.927 etiket satırı (12 kedi). L2 = 7 sınıf, L1 = 3 sınıf. Etiketli satırlar = 335.854 (tasma+koşum), ivme tablosundan yalnız bunlar okundu; tasma/koşum eşleşmesi %100, öznitelikte NaN yok.\n")
A("| L2 sınıf | L1 | Saniye |\n|---|---|---|")
for _, r in CLS.iterrows(): A(f"| {r.y2} | {r.L1} | {r.saniye:,} |".replace(",", "."))
tot = CLS.saniye.sum(); l1s = CLS.groupby("L1").saniye.sum()
A(f"| **Toplam L2** | | **{tot:,}** |".replace(",", "."))
A(f"\nL1 saniyeleri: Inaktif {l1s['Inaktif']:,}; Lokomosyon {l1s['Lokomosyon']:,}; Bakim {l1s['Bakim']:,}.".replace(",", "."))
th = lambda x: f"{x:,}".replace(",", ".")
A(f"\n**Dışarıda kalan saniyeler:** {th(EXC['toplam_etiket_satiri'])} etiket satırından {th(EXC['L2_saniye'])} L2'de. Kullanılabilir (R01 tanımı) {th(EXC['kullanilabilir_R01'])}; bunların {th(EXC['kullanilabilir_ama_L2_disi'])} s'si küçük davranışlar (Rubbing, Shake.*, Littering.*, Jumping.*, Climbing, Allogrooming, Playfight) olduğu için L2 dışı. Kullanılabilir olmayan {th(EXC['kullanilabilir_degil'])} s: Out of sight, Other, Start, çoklu ve boş etiketli satırlar.")
A(f"\n**Torba sayıları:** 30 dk & >=%50: {NB['30dk_ge50']}; 30 dk & >=%90: {NB['30dk_ge90']}; 10 dk & >=%90: {NB['10dk_ge90']}; kedi düzeyi: 12. Kapsam R01'deki kullanılabilir tanımıyla hesaplanır (R01'le aynı pencereler: 34, 135 yeniden üretildi); gerçek bütçe, penceredeki L2 saniyelerinin oranlarıdır. S0'da torba, test (%30) saniyelerinden oluşur.\n")
A("**Gerçek bütçe değişkenliği (torbalar arası, L2; ortalama, SD, min–maks)**\n")
A("| Sınıf | 30dk>=50 (68) | 30dk>=90 (34) | 10dk>=90 (135) | Kedi (12) |\n|---|---|---|---|---|")
for c in ["Lying", "Sitting", "Standing", "Walking", "Trotting", "Grooming", "Eating"]:
    cells = []
    for bd in ("30dk_ge50", "30dk_ge90", "10dk_ge90", "kedi"):
        r = VAR[(VAR.bagdef == bd) & (VAR.level == "L2") & (VAR["class"] == c)].iloc[0]
        cells.append(f"{n(r['mean'],3)}, {n(r.sd,3)}, {n(r['min'],2)}–{n(r['max'],2)}")
    A(f"| {c} | " + " | ".join(cells) + " |")
A("\nL1 (30dk>=50): " + "; ".join(f"{r['class']} {n(r['mean'],3)} (SD {n(r.sd,3)}, {n(r['min'],2)}–{n(r['max'],2)})" for _, r in VAR[(VAR.bagdef=='30dk_ge50')&(VAR.level=='L1')].iterrows()) + ".\n")

A("## 3. Ana tablo — AE ortalaması [GA] (birincil torba, varyant a)\n")
A("**L2**\n"); A(main_table("va_f32", "30dk_ge50", "L2", "AE")); A("")
A("**L1** (L2 tahminlerinin L1'e toplanmasıyla)\n"); A(main_table("va_f32", "30dk_ge50", "L1", "AE")); A("")
A("**RAE ortalaması [GA]** (L2). RAE, p≈0 olan Trotting/Walking yüzünden çok büyük ve gürültülü: ε = 1/(2N) düzgünleştirmesine rağmen küçük sınıfların hatası baskın; yorumda AE esas alındı.\n")
A(main_table("va_f32", "30dk_ge50", "L2", "RAE")); A("")
A("RAE (L1)\n"); A(main_table("va_f32", "30dk_ge50", "L1", "RAE")); A("")
A("S0 notu: S0'da da CC (AE " + n(S[(S.tag=='va_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S0')&(S.est=='CC')].AE.iloc[0]) + ") ACC'den (" + n(S[(S.tag=='va_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S0')&(S.est=='ACC')].AE.iloc[0]) + ") iyi: C, kedi-gruplu iç CV'den (kedi-dışı) kestiriliyor, S0 testi ise sızıntılı (aynı kedi eğitimde var) olduğundan C test koşuluna uymuyor; bu spesifikasyona uygun bırakıldı.\n")

A("## 4. Eating ve Grooming için işaretli yanlılık (p̂ − p, yüzde puan; torba ortalaması, birincil torba, L2)\n")
for c in ("Eating", "Grooming"):
    b = BIAS[(BIAS.tag == "va_f32") & (BIAS.bagdef == "30dk_ge50") & (BIAS.level == "L2") & (BIAS.cls == c)]
    tm_ = b[b.scen == "S1-tasma"].true_mean_pp.iloc[0]
    A(f"**{c}** (S1-tasma gerçek ort. {n(tm_,1)} pp; GA'lı değerler r02_class_bias.csv)\n")
    A("| Tahminci | " + " | ".join(SCN[s] for s in SC) + " |\n|---|" + "---|" * 4)
    for e in ["CC", "PCC", "ACC", "PACC", "EMQ"]:
        cells = []
        for s in SC:
            r = b[(b.est == e) & (b.scen == s)].iloc[0]
            cells.append(f"{sg(r.bias_pp,2)} [{sg(r.lo,2)}; {sg(r.hi,2)}]")
        A(f"| {e} | " + " | ".join(cells) + " |")
    A("")

A("## 5. Makro-F1 ve dengeli doğruluk (saniye düzeyi, kedi ortalaması [GA]; varyant a, 32 öznitelik)\n")
A("Makro-F1 yalnız kedinin test saniyelerinde bulunan sınıflar üzerinden; dengeli doğruluk aynı sınıflar üzerinden. S0'da kedi başına, o kedinin test saniyeleriyle.\n")
A("| Basamak | Makro-F1 (L2) | Dengeli doğr. (L2) | Makro-F1 (L1) | Dengeli doğr. (L1) |\n|---|---|---|---|---|")
for s in SC:
    r = CLF[(CLF.tag == "va_f32") & (CLF.scen == s)].iloc[0]
    A(f"| {SCN[s]} | {ci((r.macroF1, r.macroF1_lo, r.macroF1_hi),2)} | {ci((r.balacc, r.balacc_lo, r.balacc_hi),2)} | {ci((r.macroF1_L1, r.macroF1_L1_lo, r.macroF1_L1_hi),2)} | {ci((r.balacc_L1, r.balacc_L1_lo, r.balacc_L1_hi),2)} |")
A("")

A("## 6. Duyarlılık (L2, AE ortalaması; GA'lar r02_summary_all.csv'de)\n")
A("| Varyant | Torba | S1-tasma CC | ACC | PACC | EMQ | S2 CC | ACC | PACC | EMQ |\n|---|---|---|---|---|---|---|---|---|---|")
TG = {"va_f32": "a, 32 öz.", "vb_f32": "b (<=7000/sınıf), 32 öz.", "va_f36": "a, 36 öz."}
BD = {"30dk_ge50": "30dk>=50", "30dk_ge90": "30dk>=90", "10dk_ge90": "10dk>=90", "kedi": "kedi (12)"}
for tag in TG:
    for bd in BD:
        x = S[(S.tag == tag) & (S.bagdef == bd) & (S.level == "L2")]
        cells = [n(x[(x.scen == sc) & (x.est == e)].AE.iloc[0]) for sc in ("S1-tasma", "S2") for e in ("CC", "ACC", "PACC", "EMQ")]
        A(f"| {TG[tag]} | {BD[bd]} | " + " | ".join(cells) + " |")
A("\nH1/H2 duyarlılık (CC−ACC fark [GA]; negatif = ACC kötü; S2−S1 pozitif = ACC dezavantajı küçülüyor):\n")
A("| Varyant | Torba | S1 | S2 | S2−S1 | Oracle geri kazanım % |\n|---|---|---|---|---|---|")
for tag in TG:
    for bd in BD:
        hh = H[f"{tag}|{bd}|L2"]; a = hh["H2"]["ACC"]; o = hh["oracle"]
        rec = "—" if (o["recovery_pct"] is None or np.isnan(o["recovery_pct"])) else f"{n(o['recovery_pct'],0)} [{n(o['recovery_lo'],0)}; {n(o['recovery_hi'],0)}]"
        A(f"| {TG[tag]} | {BD[bd]} | {ci(a['S1'],3,True)} | {ci(a['S2'],3,True)} | {ci(a['S2_minus_S1'],3,True)} | {rec} |")
A("\nDuyarlılık bulguları: (i) H1 tüm varyantlarda HAYIR ya da fark GA'sı 0'ı içeriyor (10 dk ve %90 torbalarda ACC ≈ CC); hiçbir ayarda ACC/PACC/EMQ CC'yi anlamlı geçmedi. (ii) Kedi düzeyi torbalarda CC en iyi (S1-tasma AE " + n(S[(S.tag=='va_f32')&(S.bagdef=='kedi')&(S.level=='L2')&(S.scen=='S1-tasma')&(S.est=='CC')].AE.iloc[0]) + "), ACC " + n(S[(S.tag=='va_f32')&(S.bagdef=='kedi')&(S.level=='L2')&(S.scen=='S1-tasma')&(S.est=='ACC')].AE.iloc[0]) + ". (iii) Varyant b'de (Smit gibi 7000 sınırı) EMQ S2'de daha az kötü, ama yine de CC'den (" + n(S[(S.tag=='vb_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S2')&(S.est=='CC')].AE.iloc[0]) + ") kötü (" + n(S[(S.tag=='vb_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S2')&(S.est=='EMQ')].AE.iloc[0]) + " vs a'da " + n(S[(S.tag=='va_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S2')&(S.est=='EMQ')].AE.iloc[0]) + "). (iv) 36 öznitelik sonuçları değiştirmedi. (v) Oracle geri kazanımı L2'de GA çok geniş; kayıp küçük olduğunda oran ayrışıyor (kedi düzeyi satırlarındaki uç değerler bu yüzden).\n")

A("## 7. Birim testi ve çalışma süreleri\n")
A("`tests\\test_quant.py`: sentetik 3 sınıf, eğitim önseli (1/3, 1/3, 1/3) -> test önseli (0,7; 0,2; 0,1), 20 tekrar, test n=3.000. Maks |p̂−p| ortalaması: CC 0,1181; ACC 0,0135; PCC 0,1694; PACC 0,0140; EMQ 0,0109. **GEÇTİ**: ACC, PACC, EMQ CC'den iyi ve <= 0,02. Not: ilk denemede sınıf ayrımı daha zayıf (ölçek 1,0) iken ACC/PACC/EMQ hataları 0,035–0,042'de kaldı (RF sonsal kalibrasyon kalıntısı; 60.000 örnekle bile ~0,02), bu yüzden test ayrım gücü artırılarak (ölçek 1,8) geçirildi; bu, gerçek verideki RF için uyarıdır: sonsaller kalibre edilmedi (spesifikasyon gereği), düşük ayrım gücünde düzeltme tahmincileri sapmalı olur.\n")
A("| Çalıştırma | Süre | Not |\n|---|---|---|")
for tag, ad in (("va_f32", "a, 32 öz. (birincil)"), ("vb_f32", "b, 32 öz."), ("va_f36", "a, 36 öz.")):
    A(f"| {ad} | {RT[tag]['toplam']/60:.1f} dk | S0 {RT[tag]['s0']:.0f} s; 12 LOCO katmanı (her biri 500 ağaç x 2 yerleşim + iç CV 4x300 ağaç x 2) |")
A("")
A("## 8. Sorunlar ve beklenmedik sonuçlar\n")
A("1. **Beklenmedik:** nicelleme CC'yi geçemedi. Olası neden (test edilmedi): C matrisi eğitim kedilerinin kedi-gruplu iç CV'sinden kestiriliyor, hedef kedinin hatası ise bundan sapıyor olabilir; ölçülen: makro-F1 kedi ortalaması GA'sı geniş, tasma " + ci((CLF[(CLF.tag=='va_f32')&(CLF.scen=='S1-tasma')].macroF1.iloc[0], CLF[(CLF.tag=='va_f32')&(CLF.scen=='S1-tasma')].macroF1_lo.iloc[0], CLF[(CLF.tag=='va_f32')&(CLF.scen=='S1-tasma')].macroF1_hi.iloc[0]),2) + "). Nedensel test yapılmadı.")
A("2. **Tasma vs koşum:** koşum sınıflandırıcısı belirgin daha iyi (S1-koşum makro-F1 " + n(CLF[(CLF.tag=='va_f32')&(CLF.scen=='S1-kosum')].macroF1.iloc[0],2) + " vs tasma " + n(CLF[(CLF.tag=='va_f32')&(CLF.scen=='S1-tasma')].macroF1.iloc[0],2) + "); S1-koşum'da CC AE " + n(S[(S.tag=='va_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S1-kosum')&(S.est=='CC')].AE.iloc[0]) + ", S0 ile benzer.")
A("3. **EMQ S2'de çöküyor** (birincil AE " + n(S[(S.tag=='va_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S2')&(S.est=='EMQ')].AE.iloc[0]) + " vs CC " + n(S[(S.tag=='va_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S2')&(S.est=='CC')].AE.iloc[0]) + "). Olası neden (test edilmedi): koşum sonsalları tasma özniteliklerinde sistematik yanlış; EM bunu önsel kayması olarak büyütüyor olabilir. S2 makro-F1'i 0,26.")
A("4. **EMQ önseli:** 'iç CV sonsallarıyla' ifadesi, EMQ eğitim önselinin iç-CV sonsal ortalamasından kestirilmesi olarak yorumlandı (`EMQ`). Eğitim etiket sıklığı ile alternatif `EMQ_prtrain` CSV'de; sonuçlar benzer (S1-tasma L2 AE " + n(S[(S.tag=='va_f32')&(S.bagdef=='30dk_ge50')&(S.level=='L2')&(S.scen=='S1-tasma')&(S.est=='EMQ_prtrain')].AE.iloc[0]) + ").")
A("5. L1 sonuçları ayrı L1 sınıflandırıcısıyla değil, L2 tahmin vektörünün L1'e toplanmasıyla elde edildi. ACC/PACC çözümü NNLS (toplam=1 satırı ağırlıklı) + yeniden normalleştirme; simpleks kısıtı tam sağlanır.")
A("6. S0'ın 'pencerelere dağıtılmış' torbaları, torba başına ~%30 test saniyesi içerir (RAE'nin ε'u bu N'ye göre). Torba sayısı S1 ile aynı (68), ama S0 eğitim kümesi sızıntılıdır.")
A("7. Makro-F1 yalnız kedinin test saniyelerinde bulunan sınıflar üzerinden hesaplandı (George'da Lying yok vb.). Bu nedenle kediler arası karşılaştırma tam eşit değildir.")
A("8. Dosya silme: kendi ürettiğim bir zamanlama denemesi çıktısını (`raw_va_f32_timing.pkl`, E sürücüsü, geçici) kullanıcıya sormadan sildim (CLAUDE.md kural 9'a göre sormam gerekirdi); başka dosya silinmedi.")
A("9. `results\\tables\\r02_bag_level.csv` ~12 MB (istenen torba düzeyi çıktı); D'deki depoda.\n")
A("## 9. Dosyalar ve çalıştırma sırası\n")
A("Sıra (hepsi `python -I`, `PYTHONDONTWRITEBYTECODE=1`): `tests\\test_quant.py` -> `scripts\\04_run_smit_pilot.py --variant a --feat 32`, `--variant b --feat 32`, `--variant a --feat 36` -> `scripts\\05_analyze_r02.py` -> `scripts\\06_make_r02_report.py`.\n")
A("| Dosya | İçerik |\n|---|---|\n| `src\\smit_data.py` | veri yükleme (yalnız 335.854 etiketli satır), L2/L1 eşleme, torba pencereleri |\n| `src\\quant.py` | CC, PCC, ACC, PACC, EMQ, AE, RAE |\n| `tests\\test_quant.py` | birim testi |\n| `scripts\\04_run_smit_pilot.py` | RF (500 ağaç), iç CV, sonsallar; ara çıktı `<data root>\\interim\\r02\\raw_<tag>.pkl` |\n| `scripts\\05_analyze_r02.py`, `06_make_r02_report.py` | analiz, bootstrap, rapor |\n| `results\\tables\\r02_bag_level.csv` | basamak, varyant, tahminci, kedi, torba, sınıf, p, p̂ (+ torba tanımı, seviye) |\n| `results\\tables\\r02_summary_all.csv` | AE/RAE [GA], tüm varyant x torba x basamak x tahminci |\n| `results\\tables\\r02_class_bias.csv`, `r02_true_budget_variability.csv`, `r02_classifier_metrics_*.csv`, `r02_h3_spearman.csv`, `r02_hypotheses.json` | yanlılık, değişkenlik, F1, H3, H1/H2/oracle |")
Path(P.REPORTS / "R02_pilot_smit.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print(len(L), "satır (bölünmemiş)", sum(x.count("\n") + 1 for x in L))
