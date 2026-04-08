# Nakit Park Motoru — MVP

Bu paket, bugün çalıştırabileceğin ilk sürümüdür.

## İçindekiler
- `app.py` → Streamlit uygulaması
- `requirements.txt` → gerekli paketler
- `data/instruments_master.csv` → ürün evreni
- `data/market_quotes.csv` → günlük quote verisi
- `data/portfolio_rules.csv` → fon bazlı örnek kurallar

## Bugünkü kapsam
- Repo
- Takasbank Para Piyasası
- Para piyasası fonları
- Kısa vadeli borçlanma araçları fonları
- Manuel quote ile mevduat / katılım

## Yerel çalıştırma
Terminal'de bu klasöre gir ve şunu çalıştır:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tarayıcıda bir local URL açılır.

## Ücretsiz deploy (önerilen yol)
1. GitHub'da yeni bir repo aç.
2. Bu klasördeki tüm dosyaları repo'ya yükle.
3. `share.streamlit.io` adresine gir.
4. GitHub hesabını bağla.
5. "Create app" de.
6. Repo'yu seç.
7. Branch: `main`
8. Entrypoint file: `app.py`
9. Deploy de.

## Veri güncelleme
En kolay yöntem:
1. GitHub repo içinden `data/market_quotes.csv` dosyasını aç.
2. Sağ üstteki kalem ikonuna bas.
3. Oranları / quote saatlerini güncelle.
4. Commit changes de.
5. Uygulama birkaç dakika içinde yeni veriyle açılır.

## Hangi dosyayı ne için güncellersin?
- `market_quotes.csv` → günlük faiz/oran/quote güncellemesi
- `instruments_master.csv` → yeni araç ekleme, min tutar, cutoff, tip
- `portfolio_rules.csv` → fon bazlı kısıtlar

## Bu sürümün sınırları
- Gerçek zamanlı API yok
- Login yok
- Limit motoru basit
- CSV odaklı veri yönetimi var

## Sonraki adımlar
- Google Sheets entegrasyonu
- gelişmiş mandate / limit motoru
- kullanıcı girişi
- işlem loglama
