from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title="Nakit Park Motoru", page_icon="💸", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IST = ZoneInfo("Europe/Istanbul")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

RATING_ORDER = {"NR": 0, "BBB": 1, "A": 2, "AA": 3, "AAA": 4}

INSTRUMENT_TYPE_DEFAULTS = {
    "Repo": {"same_day_cutoff": "17:15", "supports_same_day": True, "supports_forward_value": True, "base_liquidity_score": 90, "base_operational_friction": 15},
    "TPP": {"same_day_cutoff": "15:30", "supports_same_day": True, "supports_forward_value": True, "base_liquidity_score": 85, "base_operational_friction": 20},
    "PPF": {"same_day_cutoff": "13:30", "supports_same_day": True, "supports_forward_value": True, "base_liquidity_score": 78, "base_operational_friction": 10},
    "KVBAF": {"same_day_cutoff": "13:30", "supports_same_day": False, "supports_forward_value": True, "base_liquidity_score": 62, "base_operational_friction": 18},
    "Mevduat": {"same_day_cutoff": "16:30", "supports_same_day": True, "supports_forward_value": True, "base_liquidity_score": 68, "base_operational_friction": 25},
    "Katılım": {"same_day_cutoff": "16:00", "supports_same_day": True, "supports_forward_value": True, "base_liquidity_score": 65, "base_operational_friction": 25},
}


# ---------- helpers ----------
def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )


def load_default_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


def load_csv_or_default(uploaded_file, default_filename: str) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_csv(default_filename).copy()
    return pd.read_csv(uploaded_file)


def ensure_columns(df: pd.DataFrame, required: dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    for col, default_value in required.items():
        if col not in df.columns:
            df[col] = default_value
    return df


def clean_data(
    instruments: pd.DataFrame,
    quotes: pd.DataFrame,
    rules: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    instruments = ensure_columns(
        instruments,
        {
            "external_code": "",
            "same_day_cutoff": "",
            "notes": "",
            "quote_required": False,
            "supports_same_day": False,
            "supports_forward_value": False,
            "participation_flag": False,
            "government_only_flag": False,
            "active_flag": True,
            "base_liquidity_score": 0,
            "base_operational_friction": 0,
        },
    )
    quotes = ensure_columns(
        quotes,
        {
            "fee_bps": 0,
            "tax_bps": 0,
            "quote_confirmed": False,
            "capacity_available": True,
            "source": "",
            "notes": "",
        },
    )
    rules = ensure_columns(
        rules,
        {
            "notes": "",
            "participation_only_flag": False,
            "government_only_flag": False,
        },
    )

    for col in [
        "quote_required",
        "supports_same_day",
        "supports_forward_value",
        "participation_flag",
        "government_only_flag",
        "active_flag",
    ]:
        instruments[col] = parse_bool(instruments[col])

    for col in ["quote_confirmed", "capacity_available"]:
        quotes[col] = parse_bool(quotes[col])

    for col in ["participation_only_flag", "government_only_flag"]:
        rules[col] = parse_bool(rules[col])

    numeric_cols_instruments = [
        "min_amount",
        "max_amount",
        "max_horizon_days",
        "base_liquidity_score",
        "base_operational_friction",
    ]
    for col in numeric_cols_instruments:
        instruments[col] = pd.to_numeric(instruments[col], errors="coerce")

    numeric_cols_quotes = ["gross_yield", "fee_bps", "tax_bps", "net_yield"]
    for col in numeric_cols_quotes:
        quotes[col] = pd.to_numeric(quotes[col], errors="coerce")

    quotes["quote_timestamp"] = pd.to_datetime(quotes["quote_timestamp"], errors="coerce")
    quotes["quote_valid_until"] = pd.to_datetime(quotes["quote_valid_until"], errors="coerce")
    rules["max_horizon_days"] = pd.to_numeric(rules["max_horizon_days"], errors="coerce")

    return instruments, quotes, rules


def latest_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    return quotes.sort_values(["instrument_id", "quote_timestamp"]).drop_duplicates("instrument_id", keep="last")


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


def format_currency(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f} TL".replace(",", ".")


def dataframe_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


# ---------- TEFAS scraping ----------
def extract_numeric_pct(text: str) -> float | None:
    if text is None:
        return None
    cleaned = text.strip().replace("%", "").replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except Exception:
        return None


def extract_numeric_amount(text: str) -> float | None:
    if text is None:
        return None
    cleaned = text.strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except Exception:
        return None


def extract_after_label(text: str, label: str, pattern: str) -> str | None:
    regex = rf"{re.escape(label)}\s*{pattern}"
    match = re.search(regex, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_tefas_fund(code: str) -> dict:
    code = code.strip().upper()
    if not code:
        raise ValueError("Fon kodu boş olamaz.")

    url = f"https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={code}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"

    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text("\n", strip=True)

    name = extract_after_label(text, "Fon Detaylı Analiz", r"([A-ZÇĞİÖŞÜ0-9\-\(\)\.\s]+FON)")
    category = extract_after_label(text, "Kategorisi", r"([A-Za-zÇĞİÖŞÜçğıöşü\s]+)")
    daily_return_text = extract_after_label(text, "Günlük Getiri (%)", r"(%?[0-9\.,-]+)")
    one_month_text = extract_after_label(text, "Son 1 Ay Getirisi", r"(%?[0-9\.,-]+)")
    start_hour = extract_after_label(text, "İşlem Başlangıç Saati", r"([0-9]{2}:[0-9]{2})")
    cutoff_hour = extract_after_label(text, "Son İşlem Saati", r"([0-9]{2}:[0-9]{2})")
    buy_valor = extract_after_label(text, "Fon Alış Valörü", r"([0-9]+)")
    sell_valor = extract_after_label(text, "Fon Satış Valörü", r"([0-9]+)")
    max_buy_text = extract_after_label(text, "Max. Alış İşlem Miktarı", r"([0-9\.,]+)")
    risk_value = extract_after_label(text, "Fonun Risk Değeri", r"([0-9]+)")

    daily_return = extract_numeric_pct(daily_return_text) or 0.0
    one_month_return = extract_numeric_pct(one_month_text) or 0.0
    max_buy_amount = extract_numeric_amount(max_buy_text)

    annualized_daily = daily_return * 365
    annualized_monthly = one_month_return * 12
    annualized_proxy = annualized_daily if daily_return > 0 else annualized_monthly

    return {
        "fund_code": code,
        "instrument_name": name or code,
        "category": category or "",
        "daily_return_pct": round(daily_return, 6),
        "one_month_return_pct": round(one_month_return, 6),
        "annualized_proxy_pct": round(annualized_proxy, 2),
        "buy_start": start_hour or "09:00",
        "cutoff": cutoff_hour or "13:30",
        "buy_valor": int(buy_valor) if buy_valor and buy_valor.isdigit() else None,
        "sell_valor": int(sell_valor) if sell_valor and sell_valor.isdigit() else None,
        "max_buy_amount": max_buy_amount,
        "risk_value": int(risk_value) if risk_value and risk_value.isdigit() else None,
        "source_url": url,
    }


def upsert_quote(quotes_df: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    quotes_df = quotes_df.copy()
    quotes_df = quotes_df[quotes_df["instrument_id"] != new_row["instrument_id"]]
    quotes_df = pd.concat([quotes_df, pd.DataFrame([new_row])], ignore_index=True)
    return quotes_df.sort_values(["instrument_id", "quote_timestamp"]).reset_index(drop=True)


def upsert_instrument(instruments_df: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    instruments_df = instruments_df.copy()
    instruments_df = instruments_df[instruments_df["instrument_id"] != new_row["instrument_id"]]
    instruments_df = pd.concat([instruments_df, pd.DataFrame([new_row])], ignore_index=True)
    return instruments_df.sort_values("instrument_id").reset_index(drop=True)


def apply_tefas_to_data(instruments_df: pd.DataFrame, quotes_df: pd.DataFrame, fund_data: dict) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    instruments_df = instruments_df.copy()
    quotes_df = quotes_df.copy()

    match = instruments_df.loc[instruments_df["external_code"].astype(str).str.upper() == fund_data["fund_code"]]
    instrument_id = None
    action_text = ""
    if not match.empty:
        instrument_id = match.iloc[0]["instrument_id"]
        row = match.iloc[0].to_dict()
        row["instrument_name"] = fund_data["instrument_name"]
        row["same_day_cutoff"] = fund_data["cutoff"]
        if fund_data["max_buy_amount"]:
            row["max_amount"] = fund_data["max_buy_amount"]
        row["notes"] = f"TEFAS otomatik güncelleme | {fund_data['source_url']}"
        instruments_df = upsert_instrument(instruments_df, row)
        action_text = f"{fund_data['fund_code']} mevcut {instrument_id} satırına işlendi."
    else:
        category_upper = fund_data["category"].upper()
        if "PARA PİYASASI" in category_upper:
            instrument_type = "PPF"
        elif "KISA VADELİ BORÇLANMA" in category_upper or "BORÇLANMA" in category_upper:
            instrument_type = "KVBAF"
        else:
            instrument_type = "PPF"

        defaults = INSTRUMENT_TYPE_DEFAULTS[instrument_type]
        instrument_id = f"tefas_{fund_data['fund_code'].lower()}"
        new_instrument = {
            "instrument_id": instrument_id,
            "instrument_name": fund_data["instrument_name"],
            "instrument_type": instrument_type,
            "provider_name": "TEFAS / Manuel Eşleme",
            "issuer_type": "Fund",
            "rating": "AA" if instrument_type == "PPF" else "A",
            "min_amount": 1,
            "max_amount": fund_data["max_buy_amount"] or 999999999,
            "max_horizon_days": 30 if instrument_type == "PPF" else 90,
            "quote_required": False,
            "supports_same_day": defaults["supports_same_day"],
            "supports_forward_value": defaults["supports_forward_value"],
            "participation_flag": False,
            "government_only_flag": False,
            "base_liquidity_score": defaults["base_liquidity_score"],
            "base_operational_friction": defaults["base_operational_friction"],
            "same_day_cutoff": fund_data["cutoff"],
            "active_flag": True,
            "notes": f"TEFAS otomatik eklendi | {fund_data['source_url']}",
            "external_code": fund_data["fund_code"],
        }
        instruments_df = upsert_instrument(instruments_df, new_instrument)
        action_text = f"{fund_data['fund_code']} için yeni {instrument_id} satırı açıldı."

    now_ist = datetime.now(IST).replace(tzinfo=None, second=0, microsecond=0)
    quote_valid_until = datetime.combine(now_ist.date(), datetime.strptime(fund_data["cutoff"], "%H:%M").time())
    new_quote = {
        "instrument_id": instrument_id,
        "quote_timestamp": now_ist,
        "quote_valid_until": quote_valid_until,
        "gross_yield": fund_data["annualized_proxy_pct"],
        "fee_bps": 0,
        "tax_bps": 0,
        "net_yield": fund_data["annualized_proxy_pct"],
        "quote_confirmed": True,
        "capacity_available": True,
        "source": f"TEFAS Auto | {fund_data['fund_code']}",
        "notes": f"Günlük getiri %{fund_data['daily_return_pct']} | Son 1Ay %{fund_data['one_month_return_pct']}",
    }
    quotes_df = upsert_quote(quotes_df, new_quote)
    return instruments_df, quotes_df, action_text


# ---------- scoring ----------
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
        reasons: list[str] = []
        blockers: list[str] = []

        instrument_rating = RATING_ORDER.get(normalize_value(row.get("rating")), 0)
        quote_exists = pd.notna(row.get("net_yield"))
        quote_confirmed = bool(row.get("quote_confirmed", False))
        capacity_available = bool(row.get("capacity_available", True))
        cutoff = row.get("same_day_cutoff")
        cutoff_time = None
        if pd.notna(cutoff) and str(cutoff).strip():
            try:
                cutoff_time = datetime.strptime(str(cutoff), "%H:%M").time()
            except ValueError:
                cutoff_time = None

        same_day_still_open = True
        if cutoff_time is not None:
            same_day_still_open = evaluation_dt.time() <= cutoff_time

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

        payload = {
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
        (rows if eligible else rejected_rows).append(payload)

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


# ---------- UI state ----------
if "instruments_df" not in st.session_state:
    st.session_state.instruments_df = None
if "quotes_df" not in st.session_state:
    st.session_state.quotes_df = None
if "rules_df" not in st.session_state:
    st.session_state.rules_df = None

st.title("💸 Nakit Park Motoru")
st.caption("MVP v2 • karar motoru + hızlı quote girişi + TEFAS otomatik çekim")

with st.sidebar:
    st.subheader("Veri Kaynağı")
    st.write("Örnek CSV kullanabilir veya kendi dosyalarını yükleyebilirsin.")
    uploaded_instruments = st.file_uploader("instruments_master.csv", type=["csv"])
    uploaded_quotes = st.file_uploader("market_quotes.csv", type=["csv"])
    uploaded_rules = st.file_uploader("portfolio_rules.csv", type=["csv"])
    if st.button("Yüklenen dosyaları uygula", use_container_width=True):
        st.session_state.instruments_df = load_csv_or_default(uploaded_instruments, "instruments_master.csv")
        st.session_state.quotes_df = load_csv_or_default(uploaded_quotes, "market_quotes.csv")
        st.session_state.rules_df = load_csv_or_default(uploaded_rules, "portfolio_rules.csv")
        st.success("Dosyalar uygulandı.")

if st.session_state.instruments_df is None:
    st.session_state.instruments_df = load_csv_or_default(uploaded_instruments, "instruments_master.csv")
if st.session_state.quotes_df is None:
    st.session_state.quotes_df = load_csv_or_default(uploaded_quotes, "market_quotes.csv")
if st.session_state.rules_df is None:
    st.session_state.rules_df = load_csv_or_default(uploaded_rules, "portfolio_rules.csv")

instruments, quotes, rules = clean_data(
    st.session_state.instruments_df,
    st.session_state.quotes_df,
    st.session_state.rules_df,
)
st.session_state.instruments_df = instruments
st.session_state.quotes_df = quotes
st.session_state.rules_df = rules


tab1, tab2, tab3, tab4 = st.tabs(["Karar Motoru", "Hızlı Quote Girişi", "TEFAS Otomatik Çek", "Veri Yönetimi"])

with tab1:
    st.markdown("### Talep Girişi")
    col1, col2, col3, col4 = st.columns(4)
    portfolio_options = rules[["portfolio_id", "portfolio_name"]].drop_duplicates()
    default_idx = int(portfolio_options.index[portfolio_options["portfolio_id"] == "LIKIT_FON"][0]) if "LIKIT_FON" in portfolio_options["portfolio_id"].values else 0

    with col1:
        portfolio_id = st.selectbox(
            "Fon / kural seti",
            options=portfolio_options["portfolio_id"].tolist(),
            index=default_idx,
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
    now_ist = datetime.now(IST).replace(tzinfo=None)
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
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("En iyi alternatif", top["Araç"])
        m2.metric("Net getiri", f"%{top['Net Getiri (%)']:.2f}")
        m3.metric("Beklenen net katkı", format_currency(top["Beklenen Net TL Katkı"]))
        m4.metric("Toplam skor", f"{top['Toplam Skor']:.1f}")

        st.markdown("#### İlk 3 öneri")
        for _, rec in eligible_df.head(3).iterrows():
            with st.container(border=True):
                left, right = st.columns([2, 1])
                with left:
                    st.markdown(f"**{int(rec['Sıra'])}. {rec['Araç']}**")
                    st.write(f"Tür: {rec['Tür']} | Sağlayıcı: {rec['Sağlayıcı']}")
                    st.write(f"Nedenler: {rec['Nedenler'] or '—'}")
                    if rec["Notlar"]:
                        st.caption(rec["Notlar"])
                with right:
                    st.metric("Net getiri", f"%{rec['Net Getiri (%)']:.2f}")
                    st.metric("Net katkı", format_currency(rec["Beklenen Net TL Katkı"]))
                    st.metric("Skor", f"{rec['Toplam Skor']:.1f}")

        st.markdown("#### Sıralı tablo")
        show_cols = ["Sıra", "Araç", "Tür", "Sağlayıcı", "Net Getiri (%)", "Beklenen Net TL Katkı", "Likidite Skoru", "Toplam Skor", "Nedenler", "Cutoff"]
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
        st.bar_chart(eligible_df[["Araç", "Toplam Skor"]].set_index("Araç"))
        dataframe_to_csv_download(eligible_df, "oneriler.csv", "Öneri tablosunu indir")

    st.markdown("### Elenen alternatifler")
    if rejected_df.empty:
        st.info("Elenen alternatif yok.")
    else:
        st.dataframe(rejected_df[["Araç", "Tür", "Sağlayıcı", "Blokajlar", "Cutoff"]], use_container_width=True, hide_index=True)
        dataframe_to_csv_download(rejected_df, "elenenler.csv", "Elenenleri indir")

with tab2:
    st.markdown("### Hızlı manuel quote girişi")
    st.write("Repo, TPP, mevduat ve katılım quote'larını burada hızlıca güncelleyebilirsin. Kaydet dediğinde uygulama içindeki veri yenilenir.")

    col_a, col_b, col_c = st.columns(3)
    instrument_options = instruments[["instrument_id", "instrument_name"]].drop_duplicates()

    with col_a:
        selected_instrument = st.selectbox(
            "Araç",
            options=instrument_options["instrument_id"].tolist(),
            format_func=lambda x: f"{x} — {instrument_options.loc[instrument_options['instrument_id'] == x, 'instrument_name'].iloc[0]}",
            key="quick_quote_instrument",
        )
        gross_yield = st.number_input("Brüt getiri (%)", value=41.00, step=0.01, format="%.2f")
        fee_bps = st.number_input("Fee (bps)", value=0, step=1)
    with col_b:
        tax_bps = st.number_input("Tax (bps)", value=0, step=1)
        quote_confirmed = st.checkbox("Quote teyitli", value=True)
        capacity_available = st.checkbox("Kapasite uygun", value=True)
    with col_c:
        source = st.text_input("Kaynak", value="Dealer Quote")
        valid_until_time = st.time_input("Geçerlilik saati", value=datetime.now(IST).time().replace(hour=16, minute=30, second=0, microsecond=0))
        notes = st.text_input("Not", value="")

    if st.button("Quote'u kaydet", type="primary"):
        now_ist = datetime.now(IST).replace(tzinfo=None, second=0, microsecond=0)
        fee_pct = fee_bps / 100
        tax_pct = tax_bps / 100
        net_yield = round(float(gross_yield) - fee_pct - tax_pct, 2)
        valid_until = datetime.combine(now_ist.date(), valid_until_time)
        new_quote = {
            "instrument_id": selected_instrument,
            "quote_timestamp": now_ist,
            "quote_valid_until": valid_until,
            "gross_yield": gross_yield,
            "fee_bps": fee_bps,
            "tax_bps": tax_bps,
            "net_yield": net_yield,
            "quote_confirmed": quote_confirmed,
            "capacity_available": capacity_available,
            "source": source,
            "notes": notes,
        }
        st.session_state.quotes_df = upsert_quote(st.session_state.quotes_df, new_quote)
        st.success(f"{selected_instrument} güncellendi. Net getiri %{net_yield:.2f}")
        st.rerun()

    st.markdown("### Toplu düzenleme")
    editable_quotes = st.data_editor(
        st.session_state.quotes_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="quotes_editor",
    )
    if st.button("Toplu düzenlemeyi uygula"):
        st.session_state.quotes_df = editable_quotes
        st.success("Quote tablosu güncellendi.")
        st.rerun()

    dataframe_to_csv_download(st.session_state.quotes_df, "market_quotes.csv", "Güncel quote CSV indir")

with tab3:
    st.markdown("### TEFAS'tan otomatik çek")
    st.write("Fon kodlarını gir. Sistem TEFAS sayfasından günlük getiri, son 1 ay getirisi, cutoff ve valör bilgilerini çekip taşıma proxy'si üretir.")

    fund_codes = st.text_input("Fon kodları", value="AC4", help="Virgülle ayır: AC4, TI1, PPN")

    if st.button("TEFAS verisini çek"):
        codes = [c.strip().upper() for c in fund_codes.split(",") if c.strip()]
        if not codes:
            st.warning("En az bir fon kodu gir.")
        else:
            fetched_rows = []
            errors = []
            for code in codes:
                try:
                    fetched_rows.append(fetch_tefas_fund(code))
                except Exception as exc:
                    errors.append(f"{code}: {exc}")

            if fetched_rows:
                fetched_df = pd.DataFrame(fetched_rows)
                st.dataframe(fetched_df, use_container_width=True, hide_index=True)
                st.session_state["tefas_last_df"] = fetched_df
            if errors:
                for err in errors:
                    st.error(err)

    tefas_last_df = st.session_state.get("tefas_last_df")
    if tefas_last_df is not None and not tefas_last_df.empty:
        st.markdown("### Uygulamaya işle")
        selected_code = st.selectbox("İşlenecek fon", options=tefas_last_df["fund_code"].tolist())
        if st.button("Seçili fonu veri setine ekle/güncelle"):
            fund_data = tefas_last_df.loc[tefas_last_df["fund_code"] == selected_code].iloc[0].to_dict()
            new_inst, new_quotes, action_text = apply_tefas_to_data(st.session_state.instruments_df, st.session_state.quotes_df, fund_data)
            st.session_state.instruments_df = new_inst
            st.session_state.quotes_df = new_quotes
            st.success(action_text)
            st.rerun()

with tab4:
    st.markdown("### Aktif veri setleri")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**instruments_master.csv**")
        st.dataframe(st.session_state.instruments_df, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(st.session_state.instruments_df, "instruments_master.csv", "Instrument CSV indir")
    with c2:
        st.markdown("**market_quotes.csv**")
        st.dataframe(st.session_state.quotes_df, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(st.session_state.quotes_df, "market_quotes.csv", "Quote CSV indir")
    with c3:
        st.markdown("**portfolio_rules.csv**")
        st.dataframe(st.session_state.rules_df, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(st.session_state.rules_df, "portfolio_rules.csv", "Rules CSV indir")

    st.markdown("### Bugün nasıl kullanacaksın?")
    st.write(
        "1) Repo ve TPP için dealer/ForInvest oranını Hızlı Quote Girişi sekmesinden gir. "
        "2) PPF/KVBAF için TEFAS kodunu yazıp otomatik çek. "
        "3) Mevduat/katılım teklifini yine Hızlı Quote Girişi'nden işle. "
        "4) Karar Motoru sekmesinde sonucu al."
    )
    st.info("Not: Streamlit Community Cloud kalıcı veritabanı değildir. Uygulama içindeki güncellemeleri kaybetmemek için CSV'leri indirip GitHub repo'na geri yükle.")
