
from __future__ import annotations

import io
from datetime import datetime, date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Nakit Park Motoru", page_icon="💸", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

RATING_ORDER = {
    "NR": 0,
    "BBB": 1,
    "A": 2,
    "AA": 3,
    "AAA": 4,
}

IST = ZoneInfo("Europe/Istanbul")


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )


@st.cache_data
def load_default_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    return df


def load_csv_or_default(uploaded_file, default_filename: str) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_csv(default_filename).copy()
    return pd.read_csv(uploaded_file)


def clean_data(
    instruments: pd.DataFrame,
    quotes: pd.DataFrame,
    rules: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    instruments = instruments.copy()
    quotes = quotes.copy()
    rules = rules.copy()

    for col in ["quote_required", "supports_same_day", "supports_forward_value", "participation_flag", "government_only_flag", "active_flag"]:
        instruments[col] = parse_bool(instruments[col])

    for col in ["quote_confirmed", "capacity_available"]:
        quotes[col] = parse_bool(quotes[col])

    for col in ["participation_only_flag", "government_only_flag"]:
        rules[col] = parse_bool(rules[col])

    numeric_cols_instruments = [
        "min_amount", "max_amount", "max_horizon_days", "base_liquidity_score", "base_operational_friction"
    ]
    for col in numeric_cols_instruments:
        instruments[col] = pd.to_numeric(instruments[col], errors="coerce")

    numeric_cols_quotes = ["gross_yield", "fee_bps", "tax_bps", "net_yield"]
    for col in numeric_cols_quotes:
        quotes[col] = pd.to_numeric(quotes[col], errors="coerce")

    quotes["quote_timestamp"] = pd.to_datetime(quotes["quote_timestamp"], errors="coerce")
    quotes["quote_valid_until"] = pd.to_datetime(quotes["quote_valid_until"], errors="coerce")

    return instruments, quotes, rules


def latest_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    quotes_sorted = quotes.sort_values(["instrument_id", "quote_timestamp"]).drop_duplicates("instrument_id", keep="last")
    return quotes_sorted


def pct_score(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(series), 70.0), index=series.index)
    min_val = series.min()
    max_val = series.max()
    return 100 * (series - min_val) / (max_val - min_val)


def normalize_value(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def split_pipe_values(value: str | None) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def score_candidates(
    instruments: pd.DataFrame,
    quotes: pd.DataFrame,
    rules: pd.DataFrame,
    portfolio_id: str,
    amount: float,
    horizon_days: int,
    planning_mode: str,
    liquidity_need: str,
    participation_pref: str,
    government_pref: str,
    min_rating_user: str,
    evaluation_dt: datetime,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = instruments.merge(latest_quotes(quotes), on="instrument_id", how="left", suffixes=("", "_quote"))
    merged = merged[merged["active_flag"]].copy()

    rule = rules.loc[rules["portfolio_id"] == portfolio_id].iloc[0]
    allowed_types = set(split_pipe_values(rule["allowed_instrument_types"]))

    require_same_day = planning_mode == "Bugün park et"
    next_day_ok = planning_mode == "Yarın başlat"

    min_rating_required = max(
        RATING_ORDER.get(normalize_value(rule["min_rating"]), 0),
        RATING_ORDER.get(normalize_value(min_rating_user), 0),
    )

    threshold_liquidity = {"T+0": 70, "T+1": 55, "Esnek": 0}[liquidity_need]

    rows = []
    rejected_rows = []

    for _, row in merged.iterrows():
        reasons = []
        blockers = []

        instrument_rating = RATING_ORDER.get(normalize_value(row.get("rating")), 0)
        quote_exists = pd.notna(row.get("net_yield"))
        quote_confirmed = bool(row.get("quote_confirmed", False))
        capacity_available = bool(row.get("capacity_available", True))
        cutoff = row.get("same_day_cutoff")
        cutoff_time = None
        if pd.notna(cutoff):
            try:
                cutoff_time = datetime.strptime(str(cutoff), "%H:%M").time()
            except ValueError:
                cutoff_time = None

        same_day_still_open = True
        if cutoff_time is not None:
            same_day_still_open = evaluation_dt.time() <= cutoff_time

        # Hard filters
        if row["instrument_type"] not in allowed_types:
            blockers.append("Fon kural setinde bu ürün tipi izinli değil.")

        if amount < row["min_amount"]:
            blockers.append(f"Minimum işlem tutarı {row['min_amount']:,.0f} TL.")
        if amount > row["max_amount"]:
            blockers.append(f"Maksimum işlem tutarı {row['max_amount']:,.0f} TL.")
        if horizon_days > row["max_horizon_days"]:
            blockers.append(f"Bu ürün için azami park süresi {int(row['max_horizon_days'])} gün.")
        if horizon_days > rule["max_horizon_days"]:
            blockers.append(f"Fon kural seti en fazla {int(rule['max_horizon_days'])} gün izin veriyor.")

        if instrument_rating < min_rating_required:
            blockers.append("Rating filtresini karşılamıyor.")

        if rule["participation_only_flag"] and not row["participation_flag"]:
            blockers.append("Fon sadece katılım uyumlu araçlara izin veriyor.")
        if participation_pref == "Sadece katılım" and not row["participation_flag"]:
            blockers.append("Kullanıcı filtresi sadece katılım istiyor.")
        if participation_pref == "Katılım hariç" and row["participation_flag"]:
            blockers.append("Kullanıcı filtresi katılım ürünlerini hariç tutuyor.")

        if rule["government_only_flag"] and not row["government_only_flag"]:
            blockers.append("Fon kural seti kamu odaklı araç istiyor.")
        if government_pref == "Sadece kamu benzeri" and not row["government_only_flag"]:
            blockers.append("Kullanıcı filtresi sadece kamu benzeri araç istiyor.")

        if row["quote_required"] and (not quote_exists or not quote_confirmed):
            blockers.append("Canlı/onaylı quote gerekli.")
        if not capacity_available:
            blockers.append("Kapasite uygun değil veya limit dolu.")

        if require_same_day:
            if not row["supports_same_day"]:
                blockers.append("Aynı gün başlangıç desteklenmiyor.")
            elif not same_day_still_open:
                blockers.append("Aynı gün cutoff saati geçmiş.")
        elif next_day_ok:
            if not (row["supports_forward_value"] or row["supports_same_day"]):
                blockers.append("İleri başlangıç için uygun değil.")

        if row["base_liquidity_score"] < threshold_liquidity:
            blockers.append("Likidite ihtiyacını karşılamıyor.")

        # Soft reasons
        if row["base_liquidity_score"] >= 80:
            reasons.append("Likidite profili güçlü.")
        elif row["base_liquidity_score"] >= 65:
            reasons.append("Likidite profili yeterli.")

        if row["base_operational_friction"] <= 12:
            reasons.append("Operasyonel sürtünmesi düşük.")
        elif row["base_operational_friction"] <= 20:
            reasons.append("Operasyonel olarak yönetilebilir.")

        if quote_exists and row["net_yield"] >= merged["net_yield"].dropna().median():
            reasons.append("Net getiri seviyesi evren ortalamasının üzerinde.")

        if require_same_day and same_day_still_open:
            reasons.append("Bugün içinde uygulanabilir.")
        elif next_day_ok and row["supports_forward_value"]:
            reasons.append("İleri başlangıç için uygun.")

        eligible = len(blockers) == 0
        days = max(int(horizon_days), 1)
        net_yield = float(row["net_yield"]) if quote_exists else np.nan
        gross_yield = float(row["gross_yield"]) if quote_exists else np.nan

        expected_net_income = amount * (net_yield / 100) * (days / 365) if quote_exists else np.nan

        row_payload = {
            "instrument_id": row["instrument_id"],
            "Araç": row["instrument_name"],
            "Tür": row["instrument_type"],
            "Sağlayıcı": row["provider_name"],
            "Net Getiri (%)": round(net_yield, 2) if quote_exists else np.nan,
            "Brüt Getiri (%)": round(gross_yield, 2) if quote_exists else np.nan,
            "Beklenen Net TL Katkı": expected_net_income,
            "Likidite Skoru": float(row["base_liquidity_score"]),
            "Operasyonel Sürtünme": float(row["base_operational_friction"]),
            "Cutoff": row["same_day_cutoff"],
            "Notlar": row.get("notes", ""),
            "Nedenler": " | ".join(reasons) if reasons else "",
            "Blokajlar": " | ".join(blockers) if blockers else "",
            "eligible": eligible,
        }

        if eligible:
            rows.append(row_payload)
        else:
            rejected_rows.append(row_payload)

    eligible_df = pd.DataFrame(rows)
    rejected_df = pd.DataFrame(rejected_rows)

    if not eligible_df.empty:
        eligible_df["Getiri Skoru"] = pct_score(eligible_df["Net Getiri (%)"].fillna(0))
        eligible_df["Uygulanabilirlik Skoru"] = 100 - eligible_df["Operasyonel Sürtünme"].clip(lower=0, upper=100)
        eligible_df["Likidite Normalized"] = eligible_df["Likidite Skoru"].clip(lower=0, upper=100)

        fit_bonus = []
        for _, row in eligible_df.iterrows():
            bonus = 80
            if planning_mode == "Bugün park et" and row["Cutoff"] and evaluation_dt.time() <= datetime.strptime(str(row["Cutoff"]), "%H:%M").time():
                bonus += 10
            if "Bugün içinde uygulanabilir." in row["Nedenler"]:
                bonus += 5
            fit_bonus.append(min(bonus, 100))
        eligible_df["Uygunluk Skoru"] = fit_bonus

        eligible_df["Toplam Skor"] = (
            0.35 * eligible_df["Getiri Skoru"]
            + 0.30 * eligible_df["Uygulanabilirlik Skoru"]
            + 0.20 * eligible_df["Likidite Normalized"]
            + 0.15 * eligible_df["Uygunluk Skoru"]
        )
        eligible_df = eligible_df.sort_values(["Toplam Skor", "Net Getiri (%)"], ascending=[False, False]).reset_index(drop=True)
        eligible_df.insert(0, "Sıra", np.arange(1, len(eligible_df) + 1))
        eligible_df["Beklenen Net TL Katkı"] = eligible_df["Beklenen Net TL Katkı"].round(0)

    return eligible_df, rejected_df


def format_currency(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f} TL".replace(",", ".")


def dataframe_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


st.title("💸 Nakit Park Motoru")
st.caption("İlk çalışan MVP • repo / TPP / PPF / KVBAF / manuel quote ile karar desteği")

with st.sidebar:
    st.subheader("Veri Kaynağı")
    st.write("İstersen örnek CSV'leri kullan, istersen kendi CSV'lerini yükle.")
    uploaded_instruments = st.file_uploader("instruments_master.csv", type=["csv"])
    uploaded_quotes = st.file_uploader("market_quotes.csv", type=["csv"])
    uploaded_rules = st.file_uploader("portfolio_rules.csv", type=["csv"])

instruments_raw = load_csv_or_default(uploaded_instruments, "instruments_master.csv")
quotes_raw = load_csv_or_default(uploaded_quotes, "market_quotes.csv")
rules_raw = load_csv_or_default(uploaded_rules, "portfolio_rules.csv")

instruments, quotes, rules = clean_data(instruments_raw, quotes_raw, rules_raw)

tab1, tab2, tab3 = st.tabs(["Karar Motoru", "Veri Yönetimi", "Nasıl Kullanılır"])

with tab1:
    st.markdown("### Talep Girişi")
    col1, col2, col3, col4 = st.columns(4)

    portfolio_options = rules[["portfolio_id", "portfolio_name"]].drop_duplicates()
    default_portfolio_idx = int(portfolio_options.index[portfolio_options["portfolio_id"] == "LIKIT_FON"][0]) if "LIKIT_FON" in portfolio_options["portfolio_id"].values else 0

    with col1:
        portfolio_id = st.selectbox(
            "Fon / kural seti",
            options=portfolio_options["portfolio_id"].tolist(),
            index=default_portfolio_idx,
            format_func=lambda x: f"{x} — {portfolio_options.loc[portfolio_options['portfolio_id'] == x, 'portfolio_name'].iloc[0]}",
        )
        amount = st.number_input("Boş nakit (TL)", min_value=1000.0, value=25000000.0, step=1000000.0, format="%.0f")
    with col2:
        horizon_days = st.number_input("Park süresi (gün)", min_value=1, max_value=180, value=1, step=1)
        planning_mode = st.selectbox("Başlangıç modu", ["Bugün park et", "Yarın başlat"])
    with col3:
        liquidity_need = st.selectbox("Likidite ihtiyacı", ["T+0", "T+1", "Esnek"])
        participation_pref = st.selectbox("Katılım filtresi", ["Hepsi", "Sadece katılım", "Katılım hariç"])
    with col4:
        government_pref = st.selectbox("Kamu filtresi", ["Hepsi", "Sadece kamu benzeri"])
        min_rating_user = st.selectbox("Minimum rating", ["NR", "BBB", "A", "AA", "AAA"], index=2)

    st.markdown("### Değerlendirme Zamanı")
    now_ist = datetime.now(IST)
    cdt1, cdt2 = st.columns(2)
    with cdt1:
        eval_date = st.date_input("Tarih", value=now_ist.date())
    with cdt2:
        eval_time = st.time_input("Saat", value=now_ist.time().replace(second=0, microsecond=0))

    evaluation_dt = datetime.combine(eval_date, eval_time)

    eligible_df, rejected_df = score_candidates(
        instruments=instruments,
        quotes=quotes,
        rules=rules,
        portfolio_id=portfolio_id,
        amount=amount,
        horizon_days=int(horizon_days),
        planning_mode=planning_mode,
        liquidity_need=liquidity_need,
        participation_pref=participation_pref,
        government_pref=government_pref,
        min_rating_user=min_rating_user,
        evaluation_dt=evaluation_dt,
    )

    st.markdown("### Sonuç")
    if eligible_df.empty:
        st.error("Bu filtrelerle uygun alternatif bulunamadı. Filtreleri gevşet veya veri setini güncelle.")
    else:
        top = eligible_df.iloc[0]
        colm1, colm2, colm3, colm4 = st.columns(4)
        colm1.metric("En iyi alternatif", top["Araç"])
        colm2.metric("Net getiri", f"%{top['Net Getiri (%)']:.2f}")
        colm3.metric("Beklenen net katkı", format_currency(top["Beklenen Net TL Katkı"]))
        colm4.metric("Toplam skor", f"{top['Toplam Skor']:.1f}")

        st.markdown("#### İlk 3 öneri")
        for _, rec in eligible_df.head(3).iterrows():
            with st.container(border=True):
                left, right = st.columns([2, 1])
                with left:
                    st.markdown(f"**{int(rec['Sıra'])}. {rec['Araç']}**")
                    st.write(f"Tür: {rec['Tür']}  |  Sağlayıcı: {rec['Sağlayıcı']}")
                    st.write(f"Nedenler: {rec['Nedenler'] or '—'}")
                    if rec["Notlar"]:
                        st.caption(rec["Notlar"])
                with right:
                    st.metric("Net getiri", f"%{rec['Net Getiri (%)']:.2f}")
                    st.metric("Net katkı", format_currency(rec["Beklenen Net TL Katkı"]))
                    st.metric("Skor", f"{rec['Toplam Skor']:.1f}")

        st.markdown("#### Sıralı tablo")
        show_cols = [
            "Sıra", "Araç", "Tür", "Sağlayıcı", "Net Getiri (%)", "Beklenen Net TL Katkı",
            "Likidite Skoru", "Toplam Skor", "Nedenler", "Cutoff"
        ]
        st.dataframe(
            eligible_df[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Beklenen Net TL Katkı": st.column_config.NumberColumn(format="%.0f"),
                "Net Getiri (%)": st.column_config.NumberColumn(format="%.2f"),
                "Toplam Skor": st.column_config.NumberColumn(format="%.1f"),
            },
        )

        chart_df = eligible_df[["Araç", "Toplam Skor"]].set_index("Araç")
        st.bar_chart(chart_df)

        dataframe_to_csv_download(eligible_df, "oneriler.csv", "Öneri tablosunu indir")

    st.markdown("### Elenen alternatifler")
    if rejected_df.empty:
        st.info("Elenen alternatif yok.")
    else:
        st.dataframe(
            rejected_df[["Araç", "Tür", "Sağlayıcı", "Blokajlar", "Cutoff"]],
            use_container_width=True,
            hide_index=True,
        )
        dataframe_to_csv_download(rejected_df, "elenenler.csv", "Elenenleri indir")

with tab2:
    st.markdown("### Yüklü veri setleri")
    st.write("İlk aşamada veriyi doğrudan CSV ile yönetiyoruz. İleride Google Sheets bağlantısı eklenebilir.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**instruments_master.csv**")
        st.dataframe(instruments, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(instruments, "instruments_master_template.csv", "Şablonu indir")
    with c2:
        st.markdown("**market_quotes.csv**")
        st.dataframe(quotes, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(quotes, "market_quotes_template.csv", "Şablonu indir")
    with c3:
        st.markdown("**portfolio_rules.csv**")
        st.dataframe(rules, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(rules, "portfolio_rules_template.csv", "Şablonu indir")

    st.markdown("### GitHub üzerinden veriyi nasıl güncellersin?")
    st.write(
        "1) Repo'da ilgili CSV dosyasını aç. 2) Sağ üstteki kalem ikonuna bas. "
        "3) Satırları düzenle. 4) Commit changes de. App otomatik yeniden deploy olur."
    )

with tab3:
    st.markdown("### Uygulama mantığı")
    st.write(
        "Bu sürüm, her aracı dört filtreden geçirir: uygunluk, uygulanabilirlik, net getiri ve likidite. "
        "Önce hard filter ile eler, sonra kalanları skorlar."
    )
    st.markdown(
        """
        **Bugünkü sürümün kapsamı**
        - Repo
        - Takasbank Para Piyasası
        - Para piyasası fonları
        - Kısa vadeli borçlanma araçları fonları
        - Manuel quote girilen mevduat / katılım

        **Bugünkü sürümün sınırları**
        - Gerçek zamanlı API yok
        - Kullanıcı girişi / login yok
        - Limit yönetimi basitleştirilmiş
        - Veri güncelleme CSV veya GitHub edit ile yapılıyor
        """
    )
    st.markdown("### Skor formülü")
    st.code(
        "Toplam Skor = 35% Net Getiri + 30% Uygulanabilirlik + 20% Likidite + 15% Uygunluk",
        language="text",
    )
    st.markdown("### Sonraki geliştirme adımları")
    st.write(
        "Google Sheets bağlantısı, karşı taraf limiti, daha detaylı mandate motoru ve admin ekranı sonraki fazda eklenebilir."
    )
