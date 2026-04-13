# Fund Manager Workstation v3

Bu sürüm, eski Excel → VBA → Google Sheets → Apps Script zincirini günlük kullanım açısından ortadan kaldırır.

## Bu sürümde ne var?
- Tek Streamlit web uygulaması
- Tek SQLite veritabanı
- **Tek tuşla universe refresh**
- TEFAS full universe ingest
- Nasdaq ETF universe ingest
- TRY repo / TPP kamu referansları
- Compare Desk
- Portfolio Lab
- Mandate fit
- Seçilen ETF ve TEFAS fonları için **on-demand history fetch**

## Neden bu yapı?
Tüm evreni her seferinde tam tarihçesiyle indirmek hem yavaş hem kırılgan olur. Bu yüzden sistem iki katmanlı kuruldu:

1. **Universe katmanı**
   - Binlerce ETF / yüzlerce TEFAS fonu metadata olarak içeri alınır.
   - Screener bu evren üzerinden çalışır.

2. **History katmanı**
   - Kullanıcı Compare veya Portfolio Lab'de gerçekten seçtiği ETF / TEFAS fonlarını açınca tarihçe otomatik çekilir.
   - Böylece ilk kurulum tek tuşla kalır, ama sistem yine derin analiz yapabilir.

## Klasör yapısı
- `app.py` → ana Streamlit uygulaması
- `requirements.txt` → bağımlılıklar
- `data/` → bootstrap fixed income + mandate csv + sqlite db
- `screener/` → tüm iş mantığı

## Deploy
### GitHub
Bu klasörün içindeki dosyaları repoya yükle.

### Streamlit Community Cloud
- Repository: senin repo
- Branch: `main`
- Main file path: `ultimate_fund_screener/app.py`  
  Eğer klasör adını değiştirmezsen bu path'i kullan.

## Uygulama açıldıktan sonra
1. **Veritabanını sıfırla ve temel yapıyı yeniden kur**
2. **Tüm evreni yenile (TEFAS + Nasdaq ETF + TRY referans)**

## Beklenen sonuç
İlk bootstrap sonrası küçük sayı görürsün.
Universe refresh sonrası sayı ciddi şekilde artar:
- TEFAS full fund universe
- Nasdaq ETF universe
- TRY reference satırları
- bootstrap fixed income

## Gerçekçi sınır
Bu sürüm lisanssız/public veriyle çalışır.
Bu yüzden:
- TEFAS: güçlü
- ETF universe: güçlü
- TRY kamu referansı: güçlü
- TR local bond / eurobond canlı ve eksiksiz kurumsal feed: **henüz yok**

TR tahvil/eurobond tarafında tam kurumsal canlılık için ileride lisanslı veri veya kurum içi kaynak gerekir.
