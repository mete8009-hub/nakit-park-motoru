from __future__ import annotations

import math
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
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

RATING_ORDER = {"NR": 0, "BBB": 1, "A": 2, "AA": 3, "AAA": 4}

INSTRUMENT_TYPE_DEFAULTS = {
    "Repo": {
        "same_day_cutoff": "17:15",
        "supports_same_day": True,
        "supports_forward_value": True,
        "base_liquidity_score": 90,
        "base_operational_friction": 15,
    },
    "TPP": {
        "same_day_cutoff": "15:30",
        "supports_same_day": True,
        "supports_forward_value": True,
        "base_liquidity_score": 85,
        "base_operational_friction": 20,
    },
    "PPF": {
        "same_day_cutoff": "13:30",
        "supports_same_day": True,
        "supports_forward_value": True,
        "base_liquidity_score": 78,
        "base_operational_friction": 10,
    },
    "KVBAF": {
        "same_day_cutoff": "13:30",
        "supports_same_day": False,
        "supports_forward_value": True,
        "base_liquidity_score": 62,
        "base_operational_friction": 18,
    },
    "Mevduat": {
        "same_day_cutoff": "16:30",
        "supports_same_day": True,
        "supports_forward_value": True,
        "base_liquidity_score": 68,
        "base_operational_friction": 25,
    },
    "Katılım": {
        "same_day_cutoff": "16:00",
        "supports_same_day": True,
        "supports_forward_value": True,
        "base_liquidity_score": 65,
        "base_operational_friction": 25,
    },
}

REFERENCE_SOURCES = {
    "repo_benchmark": {
        "label": "BIST-KYD Repo (Gross)",
        "url": "https://www.borsaistanbul.com/en/index/repbr",
        "notes": "Endeksteki bugünkü değişimden yaklaşık yıllıklandırılmış referans repo seviyesi türetilir.",
    },
    "tpp_benchmark": {
        "label": "Takasbank TPP",
        "url": "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-gunluk-bulten",
        "notes": "Kamuya açık sayfadaki günlük O/N faiz bilgisi bulunursa kullanılır; bulunamazsa manuel override gerekir.",
    },
    "tlref_reference": {
        "label": "TLREF sayfası",
        "url": "https://www.borsaistanbul.com/en/indices/tlref",
        "notes": "Benchmark kutusunda gösterim içindir; doğrudan executable quote değildir.",
    },
}


# ---------- generic helpers ----------
def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def parse_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
        .fillna(False)
    )


def normalize_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip().upper()


def split_pipe_values(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def format_currency(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.0f} TL".replace(",", ".")


def format_pct(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"%{x:.2f}"


def dataframe_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def load_default_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


def load_csv_or_default(uploaded_file, default_filename: str) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_csv(default_filename).copy()
    return pd.read_csv(uploaded_file)


def ensure_columns(df: pd.DataFrame, required: dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    for col, default in required.items():
        if col not in df.columns:
            df[col] = default
    return df


def now_ist() -> datetime:
    return datetime.now(IST).replace(tzinfo=None)


def parse_datetime(value) -> pd.Timestamp | pd.NaT:
    return pd.to_datetime(value, errors="coerce")


def latest_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return quotes.copy()
    return quotes.sort_values(["instrument_id", "quote_timestamp"]).drop_duplicates("instrument_id", keep="last")


def pct_score(series: pd.Series) -> pd.Series:
    s = series.fillna(series.min() if series.notna().any() else 0)
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.full(len(s), 70.0), index=s.index)
    return 100 * (s - s.min()) / (s.max() - s.min())


def rating_value(rating: str) -> int:
    return RATING_ORDER.get(normalize_value(rating), 0)


def calculate_net_yield(gross_yield: float, fee_bps: float = 0.0, tax_bps: float = 0.0) -> float:
    if pd.isna(gross_yield):
        return np.nan
    return float(gross_yield) - float(fee_bps or 0) / 100 - float(tax_bps or 0) / 100


# ---------- data cleaning ----------
def clean_data(instruments: pd.DataFrame, quotes: pd.DataFrame, rules: pd.DataFrame):
    instruments = ensure_columns(
        instruments,
        {
            "external_code": "",
            "auto_source": "manual",
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

    for col in ["min_amount", "max_amount", "max_horizon_days", "base_liquidity_score", "base_operational_friction"]:
        instruments[col] = pd.to_numeric(instruments[col], errors="coerce")

    for col in ["gross_yield", "fee_bps", "tax_bps", "net_yield"]:
        quotes[col] = pd.to_numeric(quotes[col], errors="coerce")

    quotes["quote_timestamp"] = pd.to_datetime(quotes["quote_timestamp"], errors="coerce")
    quotes["quote_valid_until"] = pd.to_datetime(quotes["quote_valid_until"], errors="coerce")
    rules["max_horizon_days"] = pd.to_numeric(rules["max_horizon_days"], errors="coerce")

    # fill defaults by instrument type if missing
    for idx, row in instruments.iterrows():
        defaults = INSTRUMENT_TYPE_DEFAULTS.get(row["instrument_type"], {})
        for col, val in defaults.items():
            current = row.get(col)
            if (isinstance(current, str) and not current.strip()) or pd.isna(current):
                instruments.at[idx, col] = val

    return instruments, quotes, rules


# ---------- scraping helpers ----------
def fetch_url_content(url: str, prewarm_url: str | None = None) -> tuple[str, str]:
    session = requests.Session()
    session.headers.update(HEADERS)
    if prewarm_url:
        try:
            session.get(prewarm_url, timeout=20)
        except Exception:
            pass
    resp = session.get(url, timeout=20, allow_redirects=True)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text("\n", strip=True)
    return html, text


def get_page_text(url: str) -> str:
    _, text = fetch_url_content(url)
    return text


def extract_numeric_pct(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = text.strip().replace("%", "").replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except Exception:
        return None


def extract_numeric_amount(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = text.strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except Exception:
        return None


def extract_after_label(text: str, label: str, pattern: str) -> str | None:
    regex = rf"{re.escape(label)}\s*{pattern}"
    m = re.search(regex, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_after_any_label(text: str, labels: list[str], pattern: str) -> str | None:
    for label in labels:
        val = extract_after_label(text, label, pattern)
        if val:
            return val
    return None


def tefas_field(text: str, html: str, labels: list[str], pattern: str) -> str | None:
    for source in (text, html):
        val = extract_after_any_label(source, labels, pattern)
        if val:
            return val
    return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_tefas_fund(code: str) -> dict:
    code = code.strip().upper()
    if not code:
        raise ValueError("Fon kodu boş olamaz.")
    url = f"https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={code}"
    html, text = fetch_url_content(url, prewarm_url="https://www.tefas.gov.tr/")

    title_name = None
    title_match = re.search(r"##\s+([^\n]+)", text)
    if title_match:
        title_name = title_match.group(1).strip()

    name = title_name or tefas_field(text, html, ["Fon Detaylı Analiz"], r"([^\n]+)")
    category = tefas_field(text, html, ["Kategorisi"], r"([A-Za-zÇĞİÖŞÜçğıöşü()\-\s]+)")
    daily_return_text = tefas_field(text, html, ["Günlük Getiri (%)", "Gunluk Getiri (%)"], r"(%?[\-]?[0-9\.,]+)")
    one_month_text = tefas_field(text, html, ["Son 1 Ay Getirisi"], r"(%?[\-]?[0-9\.,]+)")
    start_hour = tefas_field(text, html, ["İşlem Başlangıç Saati", "Islem Baslangic Saati"], r"([0-9]{2}:[0-9]{2})")
    cutoff_hour = tefas_field(text, html, ["Son İşlem Saati", "Son Islem Saati"], r"([0-9]{2}:[0-9]{2})")
    buy_valor = tefas_field(text, html, ["Fon Alış Valörü", "Fon Alis Valoru"], r"([0-9]+)")
    sell_valor = tefas_field(text, html, ["Fon Satış Valörü", "Fon Satis Valoru"], r"([0-9]+)")
    max_buy_text = tefas_field(text, html, ["Max. Alış İşlem Miktarı", "Max. Alis Islem Miktari"], r"([0-9\.,]+)")
    risk_value = tefas_field(text, html, ["Fonun Risk Değeri", "Fonun Risk Degeri"], r"([0-9]+)")

    daily_return = extract_numeric_pct(daily_return_text)
    one_month_return = extract_numeric_pct(one_month_text)
    max_buy_amount = extract_numeric_amount(max_buy_text)

    parsed_points = sum(
        x is not None and str(x).strip() != ""
        for x in [name, category, daily_return_text, one_month_text, start_hour, cutoff_hour, buy_valor, sell_valor]
    )
    if parsed_points < 4:
        raise RuntimeError(
            "TEFAS sayfası eksik döndü veya WAF nedeniyle değerler okunamadı. "
            "Bu fonda otomatik güncelle güvenilir değil; manuel override kullan."
        )

    daily_return = 0.0 if daily_return is None else daily_return
    one_month_return = 0.0 if one_month_return is None else one_month_return

    annualized_daily = daily_return * 365
    annualized_monthly = one_month_return * 12
    if daily_return > 0:
        annualized_proxy = annualized_daily
    elif one_month_return > 0:
        annualized_proxy = annualized_monthly
    else:
        annualized_proxy = max(annualized_daily, annualized_monthly)

    if annualized_proxy <= 0 and (daily_return == 0 and one_month_return == 0):
        raise RuntimeError("TEFAS getiri alanları okunamadı; 0 değer ile quote güncellenmedi.")

    return {
        "fund_code": code,
        "instrument_name": (name or code).strip(),
        "category": (category or "").strip(),
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_borsa_index_info(index_url: str) -> dict:
    html, text = fetch_url_content(index_url)

    def _extract(label: str):
        for source in (text, html):
            m = re.search(rf"{re.escape(label)}\s*([0-9\.,]+)", source, flags=re.IGNORECASE)
            if m:
                return extract_numeric_amount(m.group(1))
        return None

    current_val = _extract("Current Value")
    previous_close = _extract("Previous Close")
    daily_change_pct = _extract("Change (%)")
    index_date_match = re.search(r"(\d{2}\.\d{2}\.\d{4}|\d{1,2}/\d{1,2}/\d{4})", text)
    index_date = index_date_match.group(1) if index_date_match else ""

    return {
        "current_value": current_val,
        "previous_close": previous_close,
        "daily_change_pct": daily_change_pct,
        "date": index_date,
        "url": index_url,
    }


@st.cache_data(ttl=900, show_spinner=False)
def fetch_tpp_public_reference() -> dict | None:
    candidates = [
        "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-gunluk-bulten",
        "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-islem-ortalamalari-raporu",
    ]
    for url in candidates:
        try:
            html, text = fetch_url_content(url)
            searchable = [text, html]
            patterns = [
                r"O/?N[^\n]{0,120}?([0-9]{1,2}[\.,][0-9]{1,4})",
                r"Gecelik[^\n]{0,120}?([0-9]{1,2}[\.,][0-9]{1,4})",
                r"Ağırlıklı Ortalama Faiz[^\n]{0,120}?([0-9]{1,2}[\.,][0-9]{1,4})",
                r"ortalama[^\n]{0,120}?faiz[^\n]{0,120}?([0-9]{1,2}[\.,][0-9]{1,4})",
            ]
            for source in searchable:
                for p in patterns:
                    m = re.search(p, source, flags=re.IGNORECASE)
                    if m:
                        rate = extract_numeric_pct(m.group(1))
                        if rate is not None:
                            return {"rate": rate, "url": url, "method": "regex"}
            try:
                tables = pd.read_html(html)
                for tbl in tables:
                    tbl = tbl.fillna("").astype(str)
                    joined = " ".join(tbl.stack().tolist())
                    for p in patterns:
                        m = re.search(p, joined, flags=re.IGNORECASE)
                        if m:
                            rate = extract_numeric_pct(m.group(1))
                            if rate is not None:
                                return {"rate": rate, "url": url, "method": "table"}
            except Exception:
                pass
        except Exception:
            continue
    return None


def derived_repo_rate_from_index(index_info: dict) -> float | None:
    current_val = index_info.get("current_value")
    previous_close = index_info.get("previous_close")
    if not current_val or not previous_close or previous_close == 0:
        return None
    daily_return = current_val / previous_close - 1
    annualized = daily_return * 365 * 100
    if annualized < 0 or annualized > 1000:
        return None
    return annualized


# ---------- quote management ----------
def build_quote_row(
    instrument_id: str,
    gross_yield: float,
    fee_bps: float = 0.0,
    tax_bps: float = 0.0,
    quote_confirmed: bool = True,
    capacity_available: bool = True,
    source: str = "",
    notes: str = "",
    valid_until: datetime | None = None,
    quote_timestamp: datetime | None = None,
) -> dict:
    ts = quote_timestamp or now_ist()
    vu = valid_until or (ts + timedelta(hours=1))
    net_yield = calculate_net_yield(gross_yield, fee_bps, tax_bps)
    return {
        "instrument_id": instrument_id,
        "quote_timestamp": ts,
        "quote_valid_until": vu,
        "gross_yield": gross_yield,
        "fee_bps": fee_bps,
        "tax_bps": tax_bps,
        "net_yield": net_yield,
        "quote_confirmed": quote_confirmed,
        "capacity_available": capacity_available,
        "source": source,
        "notes": notes,
    }


def upsert_quote(quotes_df: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    q = quotes_df.copy()
    q = q[q["instrument_id"] != new_row["instrument_id"]]
    q = pd.concat([q, pd.DataFrame([new_row])], ignore_index=True)
    return q.sort_values(["instrument_id", "quote_timestamp"]).reset_index(drop=True)


def get_instrument_meta(instruments: pd.DataFrame, instrument_id: str) -> pd.Series:
    rows = instruments[instruments["instrument_id"] == instrument_id]
    if rows.empty:
        raise KeyError(f"Instrument bulunamadı: {instrument_id}")
    return rows.iloc[0]


def valid_until_by_cutoff(cutoff_str: str | None, fallback_hours: int = 1) -> datetime:
    ts = now_ist()
    if cutoff_str:
        try:
            cutoff_time = datetime.strptime(str(cutoff_str), "%H:%M").time()
            return datetime.combine(ts.date(), cutoff_time)
        except Exception:
            pass
    return ts + timedelta(hours=fallback_hours)


def add_manual_quote(
    quotes_df: pd.DataFrame,
    instruments_df: pd.DataFrame,
    instrument_id: str,
    gross_yield: float,
    fee_bps: float,
    tax_bps: float,
    source: str,
    notes: str,
    quote_confirmed: bool,
    capacity_available: bool,
) -> pd.DataFrame:
    meta = get_instrument_meta(instruments_df, instrument_id)
    row = build_quote_row(
        instrument_id=instrument_id,
        gross_yield=gross_yield,
        fee_bps=fee_bps,
        tax_bps=tax_bps,
        quote_confirmed=quote_confirmed,
        capacity_available=capacity_available,
        source=source,
        notes=notes,
        valid_until=valid_until_by_cutoff(meta.get("same_day_cutoff")),
    )
    return upsert_quote(quotes_df, row)


def refresh_tefas_quotes(instruments_df: pd.DataFrame, quotes_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[str]]:
    quotes = quotes_df.copy()
    refreshed = []
    errors = []
    funds = instruments_df[
        instruments_df["auto_source"].astype(str).str.lower().eq("tefas") & instruments_df["external_code"].astype(str).str.strip().ne("")
    ]
    for _, row in funds.iterrows():
        try:
            info = fetch_tefas_fund(row["external_code"])
            fee_bps = 10 if row["instrument_type"] == "PPF" else 14
            quote_row = build_quote_row(
                instrument_id=row["instrument_id"],
                gross_yield=info["annualized_proxy_pct"],
                fee_bps=fee_bps,
                tax_bps=0,
                quote_confirmed=True,
                capacity_available=True,
                source=f"TEFAS {info['fund_code']}",
                notes=(
                    f"Günlük getiri %{info['daily_return_pct']:.4f}; 1A %{info['one_month_return_pct']:.2f}; "
                    f"proxy yıllıklandırılmış %{info['annualized_proxy_pct']:.2f}"
                ),
                valid_until=valid_until_by_cutoff(info.get("cutoff")),
            )
            quotes = upsert_quote(quotes, quote_row)
            refreshed.append(
                {
                    "instrument_id": row["instrument_id"],
                    "instrument_name": row["instrument_name"],
                    "fund_code": info["fund_code"],
                    "annualized_proxy_pct": info["annualized_proxy_pct"],
                    "cutoff": info["cutoff"],
                    "source_url": info["source_url"],
                }
            )
        except Exception as e:
            errors.append(f"{row['instrument_name']} ({row['external_code']}): {e}")
    return quotes, refreshed, errors


def refresh_public_reference_quotes(instruments_df: pd.DataFrame, quotes_df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], list[str], dict]:
    quotes = quotes_df.copy()
    refreshed = []
    errors = []
    benchmarks: dict[str, dict] = {}

    # Repo benchmark from BIST-KYD repo gross index
    try:
        repo_info = fetch_borsa_index_info(REFERENCE_SOURCES["repo_benchmark"]["url"])
        repo_rate = derived_repo_rate_from_index(repo_info)
        benchmarks["repo"] = repo_info | {"derived_rate": repo_rate}
        if repo_rate is not None and "repo_on" in instruments_df["instrument_id"].values:
            existing = latest_quotes(quotes)
            existing_repo = existing[existing["instrument_id"] == "repo_on"]
            has_manual = (not existing_repo.empty) and bool(existing_repo.iloc[0].get("quote_confirmed", False))
            if not has_manual:
                row = build_quote_row(
                    instrument_id="repo_on",
                    gross_yield=repo_rate,
                    fee_bps=2,
                    tax_bps=0,
                    quote_confirmed=False,
                    capacity_available=True,
                    source="BIST-KYD Repo (Gross) türetilmiş referans",
                    notes="Kamuya açık endeks değişiminden türetilen referans; executable quote yerine benchmark olarak düşün.",
                    valid_until=valid_until_by_cutoff("17:15"),
                )
                quotes = upsert_quote(quotes, row)
                refreshed.append({"instrument_id": "repo_on", "label": "Repo referansı", "gross_yield": repo_rate, "source": row["source"]})
            else:
                refreshed.append({"instrument_id": "repo_on", "label": "Repo referansı", "gross_yield": repo_rate, "source": "Benchmark bulundu ama teyitli manuel quote korunuyor"})
    except Exception as e:
        errors.append(f"Repo benchmark çekilemedi: {e}")

    # TPP reference best effort
    try:
        tpp_info = fetch_tpp_public_reference()
        if tpp_info is None:
            errors.append("TPP kamuya açık sayfasından oran okunamadı; TPP için manuel override kullan.")
        else:
            benchmarks["tpp"] = tpp_info
            if "tpp_on" in instruments_df["instrument_id"].values:
                existing = latest_quotes(quotes)
                existing_tpp = existing[existing["instrument_id"] == "tpp_on"]
                has_manual = (not existing_tpp.empty) and bool(existing_tpp.iloc[0].get("quote_confirmed", False))
                if not has_manual:
                    row = build_quote_row(
                        instrument_id="tpp_on",
                        gross_yield=tpp_info["rate"],
                        fee_bps=1,
                        tax_bps=0,
                        quote_confirmed=False,
                        capacity_available=True,
                        source="Takasbank kamuya açık referans",
                        notes="Kamuya açık sayfadan best-effort çekim; mümkünse kurum ekranıyla teyit et.",
                        valid_until=valid_until_by_cutoff("15:30"),
                    )
                    quotes = upsert_quote(quotes, row)
                    refreshed.append({"instrument_id": "tpp_on", "label": "TPP referansı", "gross_yield": tpp_info["rate"], "source": row["source"]})
                else:
                    refreshed.append({"instrument_id": "tpp_on", "label": "TPP referansı", "gross_yield": tpp_info["rate"], "source": "Benchmark bulundu ama teyitli manuel quote korunuyor"})
    except Exception as e:
        errors.append(f"TPP benchmark çekilemedi: {e}")

    return quotes, refreshed, errors, benchmarks


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
):
    latest = latest_quotes(quotes)
    universe = instruments[instruments["active_flag"]].merge(latest, on="instrument_id", how="left", suffixes=("", "_q"))

    rule_rows = rules[rules["portfolio_id"] == portfolio_id]
    if rule_rows.empty:
        raise ValueError(f"Kural seti bulunamadı: {portfolio_id}")
    rule = rule_rows.iloc[0]
    allowed_types = set(split_pipe_values(rule["allowed_instrument_types"]))
    portfolio_min_rating = max(rating_value(rule["min_rating"]), rating_value(min_rating_user))
    max_horizon = min(int(rule["max_horizon_days"]), int(horizon_days)) if not pd.isna(rule["max_horizon_days"]) else int(horizon_days)

    eligible_rows = []
    rejected_rows = []

    for _, row in universe.iterrows():
        blocks = []
        reasons = []
        inst_type = row["instrument_type"]
        cutoff = str(row["same_day_cutoff"]).strip() if pd.notna(row["same_day_cutoff"]) else ""

        if allowed_types and inst_type not in allowed_types:
            blocks.append("Fon kural setinde izinli değil")
        if amount < row["min_amount"]:
            blocks.append("Minimum tutarın altında")
        if not pd.isna(row["max_amount"]) and amount > row["max_amount"]:
            blocks.append("Maksimum tutarın üzerinde")
        if horizon_days > row["max_horizon_days"]:
            blocks.append("Araç vade sınırını aşıyor")
        if horizon_days > max_horizon:
            blocks.append("Fon vade sınırını aşıyor")
        if rating_value(row["rating"]) < portfolio_min_rating:
            blocks.append("Rating eşiğinin altında")
        if bool(rule["participation_only_flag"]) and not bool(row["participation_flag"]):
            blocks.append("Fon sadece katılım uyumlu ürün kabul ediyor")
        if participation_pref == "Sadece katılım" and not bool(row["participation_flag"]):
            blocks.append("Katılım filtresine takıldı")
        if participation_pref == "Katılım hariç" and bool(row["participation_flag"]):
            blocks.append("Katılım hariç filtresine takıldı")
        if bool(rule["government_only_flag"]) and not bool(row["government_only_flag"]):
            blocks.append("Fon kamu benzeri ürün istiyor")
        if government_pref == "Sadece kamu benzeri" and not bool(row["government_only_flag"]):
            blocks.append("Kamu filtresine takıldı")
        if liquidity_need == "T+0" and not bool(row["supports_same_day"]):
            blocks.append("T+0 likiditeyi desteklemiyor")
        if liquidity_need == "T+1" and not (bool(row["supports_same_day"]) or bool(row["supports_forward_value"])):
            blocks.append("T+1 likidite için zayıf")
        if bool(row["quote_required"]) and not bool(row["quote_confirmed"]):
            blocks.append("Teyitli quote yok")
        if not bool(row["capacity_available"]) and not pd.isna(row["capacity_available"]):
            blocks.append("Kapasite uygun değil")
        if pd.isna(row["net_yield"]):
            blocks.append("Geçerli quote yok")
        if planning_mode == "Bugün park et" and cutoff:
            try:
                cutoff_time = datetime.strptime(cutoff, "%H:%M").time()
                if evaluation_dt.time() > cutoff_time:
                    blocks.append("Bugünkü cutoff geçmiş")
                else:
                    reasons.append("Bugün içinde uygulanabilir")
            except Exception:
                pass
        if planning_mode == "Yarın başlat":
            reasons.append("Yarın başlangıç planı ile değerlendirildi")

        if inst_type in {"Repo", "TPP"}:
            reasons.append("Likidite tarafı güçlü")
        if inst_type == "PPF":
            reasons.append("Operasyonel sürtünmesi düşük")
        if inst_type == "KVBAF":
            reasons.append("Biraz daha uzun parkta anlamlı olabilir")
        if inst_type in {"Mevduat", "Katılım"}:
            reasons.append("Karşı taraf quote kalitesi kritik")

        if blocks:
            rejected_rows.append(
                {
                    "Araç": row["instrument_name"],
                    "Tür": inst_type,
                    "Sağlayıcı": row["provider_name"],
                    "Blokajlar": " | ".join(blocks),
                    "Cutoff": cutoff or "-",
                }
            )
            continue

        net_yield = float(row["net_yield"])
        net_contribution = amount * net_yield / 100 * max(horizon_days, 1) / 365
        eligible_rows.append(
            {
                "instrument_id": row["instrument_id"],
                "Araç": row["instrument_name"],
                "Tür": inst_type,
                "Sağlayıcı": row["provider_name"],
                "Gross Getiri (%)": row["gross_yield"],
                "Net Getiri (%)": net_yield,
                "Beklenen Net TL Katkı": net_contribution,
                "Likidite Skoru": float(row["base_liquidity_score"]),
                "Operasyonel Sürtünme": float(row["base_operational_friction"]),
                "Quote Teyitli": bool(row["quote_confirmed"]),
                "Nedenler": " | ".join(reasons),
                "Notlar": row["notes"] or row.get("notes_q", "") or row.get("source", ""),
                "Cutoff": cutoff or "-",
                "Source": row.get("source", ""),
            }
        )

    eligible_df = pd.DataFrame(eligible_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    if not eligible_df.empty:
        eligible_df["Getiri Skoru"] = pct_score(eligible_df["Net Getiri (%)"].fillna(0))
        eligible_df["Uygulanabilirlik Skoru"] = 100 - eligible_df["Operasyonel Sürtünme"].clip(lower=0, upper=100)
        eligible_df["Likidite Normalized"] = eligible_df["Likidite Skoru"].clip(lower=0, upper=100)

        fit_scores = []
        for _, row in eligible_df.iterrows():
            score = 80
            if row["Quote Teyitli"]:
                score += 10
            if "Bugün içinde uygulanabilir" in row["Nedenler"]:
                score += 5
            fit_scores.append(min(score, 100))
        eligible_df["Uygunluk Skoru"] = fit_scores
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


# ---------- session state ----------
def init_state():
    if "instruments_df" not in st.session_state:
        st.session_state["instruments_df"] = load_default_csv("instruments_master.csv")
    if "quotes_df" not in st.session_state:
        st.session_state["quotes_df"] = load_default_csv("market_quotes.csv")
    if "rules_df" not in st.session_state:
        st.session_state["rules_df"] = load_default_csv("portfolio_rules.csv")
    if "benchmarks" not in st.session_state:
        st.session_state["benchmarks"] = {}


# ---------- UI ----------
init_state()

st.title("💸 Nakit Park Motoru")
st.caption("Repo / TPP / PPF / KVBAF / Mevduat / Katılım için iç kullanım karar destek MVP'si")

with st.sidebar:
    st.subheader("Veri Kaynağı")
    st.write("İstersen örnek veriyle devam et, istersen kendi CSV'lerini yükleyip oturumu o veriyle başlat.")
    uploaded_instruments = st.file_uploader("instruments_master.csv", type=["csv"])
    uploaded_quotes = st.file_uploader("market_quotes.csv", type=["csv"])
    uploaded_rules = st.file_uploader("portfolio_rules.csv", type=["csv"])

    if st.button("Yüklenen dosyaları oturuma uygula", use_container_width=True):
        st.session_state["instruments_df"] = load_csv_or_default(uploaded_instruments, "instruments_master.csv")
        st.session_state["quotes_df"] = load_csv_or_default(uploaded_quotes, "market_quotes.csv")
        st.session_state["rules_df"] = load_csv_or_default(uploaded_rules, "portfolio_rules.csv")
        st.success("Yüklü dosyalar oturuma alındı.")

    if st.button("Varsayılan örnek veriye dön", use_container_width=True):
        st.session_state["instruments_df"] = load_default_csv("instruments_master.csv")
        st.session_state["quotes_df"] = load_default_csv("market_quotes.csv")
        st.session_state["rules_df"] = load_default_csv("portfolio_rules.csv")
        st.success("Varsayılan örnek veri yüklendi.")

instruments, quotes, rules = clean_data(
    st.session_state["instruments_df"],
    st.session_state["quotes_df"],
    st.session_state["rules_df"],
)
st.session_state["instruments_df"] = instruments
st.session_state["quotes_df"] = quotes
st.session_state["rules_df"] = rules


tab1, tab2, tab3, tab4 = st.tabs([
    "Karar Motoru",
    "Otomatik Güncelle",
    "Quote Yönetimi",
    "Veri / Kurulum",
])

with tab1:
    st.markdown("### Talep Girişi")
    portfolio_options = rules[["portfolio_id", "portfolio_name"]].drop_duplicates()
    default_portfolio_idx = int(portfolio_options.index[portfolio_options["portfolio_id"] == "LIKIT_FON"][0]) if "LIKIT_FON" in portfolio_options["portfolio_id"].values else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        portfolio_id = st.selectbox(
            "Fon / kural seti",
            options=portfolio_options["portfolio_id"].tolist(),
            index=default_portfolio_idx,
            format_func=lambda x: f"{x} — {portfolio_options.loc[portfolio_options['portfolio_id'] == x, 'portfolio_name'].iloc[0]}",
        )
        amount = st.number_input("Boş nakit (TL)", min_value=1000.0, value=25000000.0, step=1000000.0, format="%.0f")
    with c2:
        horizon_days = st.number_input("Park süresi (gün)", min_value=1, max_value=180, value=1, step=1)
        planning_mode = st.selectbox("Başlangıç modu", ["Bugün park et", "Yarın başlat"])
    with c3:
        liquidity_need = st.selectbox("Likidite ihtiyacı", ["T+0", "T+1", "Esnek"])
        participation_pref = st.selectbox("Katılım filtresi", ["Hepsi", "Sadece katılım", "Katılım hariç"])
    with c4:
        government_pref = st.selectbox("Kamu filtresi", ["Hepsi", "Sadece kamu benzeri"])
        min_rating_user = st.selectbox("Minimum rating", ["NR", "BBB", "A", "AA", "AAA"], index=2)

    st.markdown("### Değerlendirme Zamanı")
    current = now_ist()
    d1, d2 = st.columns(2)
    with d1:
        eval_date = st.date_input("Tarih", value=current.date())
    with d2:
        eval_time = st.time_input("Saat", value=current.time().replace(second=0, microsecond=0))
    evaluation_dt = datetime.combine(eval_date, eval_time)

    eligible_df, rejected_df = score_candidates(
        instruments=instruments,
        quotes=quotes,
        rules=rules,
        portfolio_id=portfolio_id,
        amount=float(amount),
        horizon_days=int(horizon_days),
        planning_mode=planning_mode,
        liquidity_need=liquidity_need,
        participation_pref=participation_pref,
        government_pref=government_pref,
        min_rating_user=min_rating_user,
        evaluation_dt=evaluation_dt,
    )

    st.markdown("### Referans Kutusu")
    latest_q = latest_quotes(quotes)
    ref_cols = st.columns(4)
    repo_row = latest_q[latest_q["instrument_id"] == "repo_on"]
    tpp_row = latest_q[latest_q["instrument_id"] == "tpp_on"]
    ppf_rows = latest_q.merge(instruments[["instrument_id", "instrument_type"]], on="instrument_id", how="left")
    ppf_mean = ppf_rows[ppf_rows["instrument_type"] == "PPF"]["net_yield"].mean()
    kv_mean = ppf_rows[ppf_rows["instrument_type"] == "KVBAF"]["net_yield"].mean()
    ref_cols[0].metric("Repo referansı", format_pct(repo_row["net_yield"].iloc[0]) if not repo_row.empty else "-")
    ref_cols[1].metric("TPP referansı", format_pct(tpp_row["net_yield"].iloc[0]) if not tpp_row.empty else "-")
    ref_cols[2].metric("PPF ort. net", format_pct(ppf_mean) if not pd.isna(ppf_mean) else "-")
    ref_cols[3].metric("KVBAF ort. net", format_pct(kv_mean) if not pd.isna(kv_mean) else "-")

    st.markdown("### Sonuç")
    if eligible_df.empty:
        st.error("Bu filtrelerle uygun alternatif bulunamadı. Filtreleri gevşet ya da veri setini güncelle.")
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
                left, right = st.columns([2.2, 1])
                with left:
                    st.markdown(f"**{int(rec['Sıra'])}. {rec['Araç']}**")
                    st.write(f"Tür: {rec['Tür']}  |  Sağlayıcı: {rec['Sağlayıcı']}")
                    st.write(f"Nedenler: {rec['Nedenler'] or '—'}")
                    st.caption(f"Kaynak: {rec['Source'] or '-'}")
                    if rec["Notlar"]:
                        st.caption(rec["Notlar"])
                with right:
                    st.metric("Net getiri", f"%{rec['Net Getiri (%)']:.2f}")
                    st.metric("Net katkı", format_currency(rec["Beklenen Net TL Katkı"]))
                    st.metric("Skor", f"{rec['Toplam Skor']:.1f}")

        show_cols = [
            "Sıra", "Araç", "Tür", "Sağlayıcı", "Net Getiri (%)", "Beklenen Net TL Katkı",
            "Likidite Skoru", "Toplam Skor", "Cutoff", "Source", "Nedenler"
        ]
        st.dataframe(
            eligible_df[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Net Getiri (%)": st.column_config.NumberColumn(format="%.2f"),
                "Beklenen Net TL Katkı": st.column_config.NumberColumn(format="%.0f"),
                "Toplam Skor": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        dataframe_to_csv_download(eligible_df, "oneriler.csv", "Öneri tablosunu indir")

    st.markdown("### Elenen alternatifler")
    if rejected_df.empty:
        st.info("Elenen alternatif yok.")
    else:
        st.dataframe(rejected_df, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(rejected_df, "elenenler.csv", "Elenenleri indir")

with tab2:
    st.markdown("### 1) TEFAS fonlarını topluca yenile")
    tefas_candidates = instruments[instruments["auto_source"].astype(str).str.lower().eq("tefas")][["instrument_name", "external_code"]]
    if tefas_candidates.empty:
        st.info("TEFAS otomasyonu için instruments_master.csv içinde auto_source=tefas ve external_code alanları gerekli.")
    else:
        st.dataframe(tefas_candidates, hide_index=True, use_container_width=True)
        if st.button("TEFAS fonlarını yenile", type="primary"):
            updated_quotes, refreshed, errors = refresh_tefas_quotes(instruments, quotes)
            st.session_state["quotes_df"] = updated_quotes
            quotes = updated_quotes
            if refreshed:
                st.success(f"{len(refreshed)} fon güncellendi.")
                st.dataframe(pd.DataFrame(refreshed), use_container_width=True, hide_index=True)
            if errors:
                st.warning("Bazı fonlar güncellenemedi veya güvenilir okunamadı:")
                for err in errors:
                    st.write(f"- {err}")
                st.info("Bu durumda mevcut CSV quote'larını koruyup yalnızca teyitli manuel quote ile devam et.")

    st.markdown("### 2) Kamuya açık referansları çek")
    st.caption("Repo ve TPP tarafında kamuya açık veriden best-effort referans üretir. İşlem yapılabilir oran yerine benchmark olarak düşün.")
    if st.button("Repo / TPP referansını çek"):
        updated_quotes, refreshed, errors, benchmarks = refresh_public_reference_quotes(instruments, quotes)
        st.session_state["quotes_df"] = updated_quotes
        st.session_state["benchmarks"] = benchmarks
        quotes = updated_quotes
        if refreshed:
            st.success("Referans veriler güncellendi.")
            st.dataframe(pd.DataFrame(refreshed), use_container_width=True, hide_index=True)
        if errors:
            st.warning("Bazı referanslar çekilemedi. Bu durumda manuel override kullan.")
            for err in errors:
                st.write(f"- {err}")

    st.markdown("### 3) Tek fon çekme")
    with st.form("single_tefas_form"):
        tefas_code = st.text_input("Fon kodu", value="AC4")
        single_submit = st.form_submit_button("Fon bilgisini çek")
    if single_submit:
        try:
            info = fetch_tefas_fund(tefas_code)
            if info.get("annualized_proxy_pct", 0) <= 0:
                st.warning("Fon sayfası açıldı ama getiri alanları güvenilir okunmadı. Bu fonu otomatik quote olarak kullanma.")
            else:
                st.success("Fon verisi çekildi.")
            st.json(info)
        except Exception as e:
            st.error(f"Fon verisi çekilemedi: {e}")

with tab3:
    st.markdown("### Manuel quote girişi")
    manual_instruments = instruments[instruments["instrument_type"].isin(["Repo", "TPP", "Mevduat", "Katılım"])]
    with st.form("manual_quote_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_instrument = st.selectbox(
                "Araç",
                options=manual_instruments["instrument_id"].tolist(),
                format_func=lambda x: f"{x} — {manual_instruments.loc[manual_instruments['instrument_id']==x, 'instrument_name'].iloc[0]}",
            )
            gross_yield = st.number_input("Brüt oran (%)", min_value=0.0, max_value=1000.0, value=42.0, step=0.05)
            fee_bps = st.number_input("Fee (bps)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
        with col2:
            tax_bps = st.number_input("Tax (bps)", min_value=0.0, max_value=500.0, value=0.0, step=1.0)
            source = st.text_input("Kaynak", value="Dealer Quote")
            quote_confirmed = st.checkbox("Quote teyitli", value=True)
        with col3:
            capacity_available = st.checkbox("Kapasite uygun", value=True)
            notes = st.text_area("Not", value="")
        manual_submit = st.form_submit_button("Quote'u kaydet", type="primary")

    if manual_submit:
        updated_quotes = add_manual_quote(
            quotes, instruments, selected_instrument, gross_yield, fee_bps, tax_bps, source, notes, quote_confirmed, capacity_available
        )
        st.session_state["quotes_df"] = updated_quotes
        quotes = updated_quotes
        st.success("Quote güncellendi.")

    st.markdown("### Güncel quote tablosu")
    latest_q = latest_quotes(quotes).merge(
        instruments[["instrument_id", "instrument_name", "instrument_type", "provider_name"]], on="instrument_id", how="left"
    )
    st.dataframe(
        latest_q[[
            "instrument_name", "instrument_type", "provider_name", "gross_yield", "net_yield",
            "quote_timestamp", "quote_valid_until", "quote_confirmed", "source", "notes"
        ]],
        use_container_width=True,
        hide_index=True,
    )
    dataframe_to_csv_download(quotes, "market_quotes.csv", "Güncel market_quotes.csv indir")

with tab4:
    st.markdown("### Veri dosyaları")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**instruments_master.csv**")
        st.dataframe(instruments, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(instruments, "instruments_master.csv", "instruments_master.csv indir")
    with c2:
        st.markdown("**market_quotes.csv**")
        st.dataframe(quotes, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(quotes, "market_quotes.csv", "market_quotes.csv indir")
    with c3:
        st.markdown("**portfolio_rules.csv**")
        st.dataframe(rules, use_container_width=True, hide_index=True)
        dataframe_to_csv_download(rules, "portfolio_rules.csv", "portfolio_rules.csv indir")

    st.markdown("### Senin bugün yapacağın şey")
    st.write(
        "1) Bu yeni app.py, requirements.txt ve data klasörünü GitHub repo'na yükle. "
        "2) Streamlit Cloud otomatik yeniden build edecek. "
        "3) App açılınca önce Otomatik Güncelle sekmesinde TEFAS fonlarını yenile. "
        "4) Repo/TPP referansı çek. "
        "5) Mevduat ve katılım için yalnızca gelen gerçek banka quote'unu Manuel Quote sekmesinden gir. "
        "6) İstersen güncel CSV'leri indirip GitHub'a geri koy; böylece son veri repo'da kalıcı olur."
    )

    st.markdown("### Önemli notlar")
    st.markdown(
        """
        - TEFAS tarafı otomatik.
        - Repo ve TPP tarafında kamuya açık referans çekimi **best-effort** çalışır; kesin executable quote değildir.
        - Mevduat ve katılım oranı manuel kalır; çünkü gerçek banka quote'u gerekir.
        - Streamlit Cloud kalıcı veritabanı değildir; bu yüzden istersen güncel CSV'leri indirip GitHub'a yükle.
        """
    )

