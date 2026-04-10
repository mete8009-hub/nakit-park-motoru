# Nakit Park Motoru — çoklu vade sürümü

Bu sürümde repo, ters repo ve Takasbank Para Piyasası tarafı artık sadece O/N satırlarıyla sınırlı değildir.

## Yeni mantık
- Google Sheets / VBA köprüsünden gelen `repo_7d`, `tersrepo_14d`, `tpp_30d` gibi enstrüman kimlikleri uygulamada otomatik tanınır.
- `instruments_master.csv` içinde açık satır olmasa bile, quote geldiğinde uygulama eksik para piyasası enstrümanlarını dinamik olarak türetebilir.
- Karar motoru artık vade uyumu kontrolü yapar:
  - park süresinden uzun enstrümanı eler
  - kısa vadeyi daha uzun park için kullanıyorsa `rollover` varsayımını gösterir
- PM görünümünde tüm canlı repo / ters repo / TPP vadeleri fonların yanında listelenir.

## Önerilen instrument_id standardı
- Repo: `repo_on`, `repo_2d`, `repo_7d`, `repo_30d`
- Ters repo: `tersrepo_on`, `tersrepo_2d`, `tersrepo_7d`, `tersrepo_30d`
- TPP: `tpp_on`, `tpp_2d`, `tpp_7d`, `tpp_30d`

## Yükleme sırası
1. `app_updated.py` dosyasını repoda `app.py` olarak değiştir.
2. `instruments_master_updated.csv`, `market_quotes_updated.csv`, `portfolio_rules_updated.csv` dosyalarını data klasörüne koy.
3. VBA modülünü eski sürümün yerine içe aktar.
4. Apps Script kodunu güncelle.
5. Repo / ters repo / TPP workbook'larından tüm dolu vadeleri Google Sheet'e gönder.

## Not
Kamuya açık referans çekimi uygulamada hâlâ sadece O/N benchmark amaçlıdır. Çoklu vade canlı akışın ana kaynağı VBA + Google Sheets köprüsüdür.
