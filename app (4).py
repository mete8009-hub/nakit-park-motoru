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

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
.metric-card {border: 1px solid rgba(250,250,250,0.08); border-radius: 14px; padding: 0.9rem 1rem; background: rgba(255,255,255,0.02);}
.small-label {font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.15rem;}
.big-value {font-size: 1.55rem; font-weight: 700; line-height: 1.1;}
.status-ok {color: #22c55e; font-weight: 600;}
.status-warn {color: #f59e0b; font-weight: 600;}
.status-bad {color: #ef4444; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
IST = ZoneInfo("Europe/Istanbul")
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}

GOOGLE_QUOTES_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTJlOPHHr_7KP6NtIV5nXKXc-GWzTqA36I3XgOohTGCmY1ghtxXvS3WABwKJ_RWDm8PVbNFT9xUlUVI/pub?gid=0&single=true&output=csv"

RATING_ORDER = {"NR": 0, "BBB": 1, "A": 2, "AA": 3, "AAA": 4}


INSTRUMENT_TYPE_DEFAULTS = {
    "Repo": {
        "same_day_cutoff": "17:15",
        "supports_same_day": True,
        "supports_forward_value": True,
        "base_liquidity_score": 90,
        "base_operational_friction": 15,
    },
    "TersRepo": {
        "same_day_cutoff": "17:15",
        "supports_same_day": True,
        "supports_forward_value": True,
        "base_liquidity_score": 88,
        "base_operational_friction": 16,
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

MONEY_MARKET_TYPES = {"Repo", "TersRepo", "TPP"}

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


def format_dt_tr(value) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "-"
    return ts.strftime("%d.%m.%Y %H:%M")


def render_metric_card(label: str, value: str, help_text: str | None = None):
    help_html = f'<div class="small-label">{help_text}</div>' if help_text else ''
    st.markdown(
        f'<div class="metric-card"><div class="small-label">{label}</div><div class="big-value">{value}</div>{help_html}</div>',
        unsafe_allow_html=True,
    )


def dataframe_to_csv_download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def load_default_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / filename)


def load_csv_or_default(uploaded_file, default_filename: str) -> pd.DataFrame:
    if uploaded_file is None:
        return load_default_csv(default_filename).copy()
    return pd.read_csv(uploaded_file)


@st.cache_data(ttl=15, show_spinner=False)
def fetch_google_quotes_csv(csv_url: str) -> pd.DataFrame:
    return pd.read_csv(csv_url)


def parse_google_rate(value, source: str = "") -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().replace("%", "").replace(" ", "")
    if not s:
        return None
    # Decimal comma support
    if "," in s and "." not in s:
        s = s.replace(",", ".")
    try:
        n = float(s)
    except Exception:
        return None
    # DDE bridge sometimes sends 4000 for 40.00 or 4155 for 41.55
    if "ForInvest DDE" in str(source) and abs(n) >= 100:
        n = n / 100.0
    return round(n, 2)


def _parse_sheet_timestamp(value) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    s = str(value).strip()
    if not s:
        return pd.NaT
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _normalize_sheet_quotes(sheet_quotes: pd.DataFrame) -> pd.DataFrame:
    sheet_quotes = sheet_quotes.copy()
    needed = ["instrument_id", "instrument_name", "gross_yield", "net_yield", "quote_timestamp", "source", "notes"]
    for col in needed:
        if col not in sheet_quotes.columns:
            sheet_quotes[col] = ""
    sheet_quotes = sheet_quotes[needed]
    sheet_quotes["instrument_id"] = sheet_quotes["instrument_id"].astype(str).str.strip()
    sheet_quotes["gross_yield"] = [parse_google_rate(v, s) for v, s in zip(sheet_quotes["gross_yield"], sheet_quotes["source"])]
    sheet_quotes["net_yield"] = [parse_google_rate(v, s) for v, s in zip(sheet_quotes["net_yield"], sheet_quotes["source"])]
    sheet_quotes["quote_timestamp"] = sheet_quotes["quote_timestamp"].apply(_parse_sheet_timestamp)
    return sheet_quotes.dropna(subset=["instrument_id"])



def merge_google_quotes(base_quotes: pd.DataFrame, sheet_quotes: pd.DataFrame, instruments: pd.DataFrame) -> pd.DataFrame:
    base_quotes = ensure_columns(base_quotes, {
        "instrument_name": "", "fee_bps": 0, "tax_bps": 0, "quote_confirmed": False, "capacity_available": True,
        "source": "", "notes": "", "quote_valid_until": pd.NaT, "quote_timestamp": pd.NaT,
    }).copy()
    if sheet_quotes.empty:
        return base_quotes

    norm = _normalize_sheet_quotes(sheet_quotes)
    if norm.empty:
        return base_quotes

    inst_map = instruments.set_index("instrument_id") if not instruments.empty else pd.DataFrame()
    rows = []
    current_ts = now_ist()
    for _, row in norm.iterrows():
        iid = row["instrument_id"]
        existing = base_quotes[base_quotes["instrument_id"] == iid].sort_values("quote_timestamp").tail(1)
        out = existing.iloc[0].to_dict() if not existing.empty else {"instrument_id": iid}
        out["instrument_id"] = iid

        inst = None
        if not inst_map.empty and iid in inst_map.index:
            inst = inst_map.loc[iid]
        else:
            inferred = infer_market_meta_from_instrument_id(iid, str(row.get("instrument_name", "") or ""))
            inst = pd.Series(inferred) if inferred else pd.Series(dtype="object")

        if not inst.empty:
            out.setdefault("fee_bps", 0)
            out.setdefault("tax_bps", 0)
            out.setdefault("capacity_available", True)
            out.setdefault("quote_confirmed", True)
            cutoff = str(inst.get("same_day_cutoff", "")).strip()
            out["quote_valid_until"] = pd.to_datetime(f"{current_ts.date()} {cutoff}", errors="coerce") if cutoff else pd.NaT

        out["instrument_name"] = row.get("instrument_name", out.get("instrument_name", "")) or out.get("instrument_name", "")
        out["gross_yield"] = row["gross_yield"]
        out["net_yield"] = row["net_yield"] if pd.notna(row["net_yield"]) else row["gross_yield"]
        ts_value = row["quote_timestamp"] if pd.notna(row["quote_timestamp"]) else pd.Timestamp(current_ts)
        if pd.notna(ts_value) and iid in {"repo_on", "tpp_on", "tersrepo_on"}:
            ts_value = pd.Timestamp(ts_value)
            if ts_value.hour == 0 and ts_value.minute == 0 and ts_value.second == 0:
                ts_value = pd.Timestamp(current_ts)
        out["quote_timestamp"] = ts_value
        out["source"] = row.get("source", "Google Sheets") or "Google Sheets"
        out["notes"] = row.get("notes", "")
        out["quote_confirmed"] = True
        out["capacity_available"] = True
        rows.append(out)

    updates_df = pd.DataFrame(rows)
    rest = base_quotes[~base_quotes["instrument_id"].isin(updates_df["instrument_id"])].copy()
    merged = pd.concat([rest, updates_df], ignore_index=True, sort=False)
    return merged


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


def tenor_label(days_value) -> str:
    days = safe_float(days_value)
    if pd.isna(days) or days <= 0:
        return "-"
    days = int(days)
    return "O/N" if days == 1 else f"{days}G"


def infer_tenor_days_from_instrument_id(instrument_id: str):
    iid = str(instrument_id or "").strip().lower()
    alias_days = {"repo_on": 1, "tpp_on": 1, "tersrepo_on": 1, "reverse_repo_on": 1}
    if iid in alias_days:
        return alias_days[iid]
    m = re.match(r"^(repo|tpp|tersrepo|reverse_repo)_(\d+)d$", iid)
    if m:
        return int(m.group(2))
    return np.nan


def infer_market_meta_from_instrument_id(instrument_id: str, instrument_name: str = "") -> dict | None:
    iid = str(instrument_id or "").strip().lower()
    if not iid:
        return None

    prefix = None
    days = infer_tenor_days_from_instrument_id(iid)
    if iid.startswith("repo_") or iid == "repo_on":
        prefix = "repo"
    elif iid.startswith("tpp_") or iid == "tpp_on":
        prefix = "tpp"
    elif iid.startswith("tersrepo_") or iid.startswith("reverse_repo_") or iid in {"tersrepo_on", "reverse_repo_on"}:
        prefix = "tersrepo"

    if prefix is None or pd.isna(days):
        return None

    days = int(days)
    if prefix == "repo":
        inst_type = "Repo"
        family = "Repo"
        provider = "BIST Repo Piyasası"
        issuer_type = "Government"
        rating = "AAA"
        government_only = True
    elif prefix == "tersrepo":
        inst_type = "TersRepo"
        family = "Ters Repo"
        provider = "BIST Ters Repo Piyasası"
        issuer_type = "Market"
        rating = "AAA"
        government_only = False
    else:
        inst_type = "TPP"
        family = "TPP"
        provider = "Takasbank"
        issuer_type = "Market"
        rating = "AAA"
        government_only = False

    defaults = INSTRUMENT_TYPE_DEFAULTS.get(inst_type, {})
    liquidity = max(45, float(defaults.get("base_liquidity_score", 70)) - min(max(days - 1, 0) * 0.75, 22))
    friction = min(60, float(defaults.get("base_operational_friction", 20)) + min(max(days - 1, 0) * 0.20, 10))

    if instrument_name:
        display_name = instrument_name
    else:
        if days == 1:
            display_name = {
                "Repo": "BIST Repo O/N",
                "TersRepo": "BIST Ters Repo O/N",
                "TPP": "Takasbank Para Piyasası O/N",
            }[inst_type]
        else:
            display_name = {
                "Repo": f"BIST Repo {days} Gün",
                "TersRepo": f"BIST Ters Repo {days} Gün",
                "TPP": f"Takasbank Para Piyasası {days} Gün",
            }[inst_type]

    return {
        "instrument_id": instrument_id,
        "instrument_name": display_name,
        "instrument_type": inst_type,
        "instrument_family": family,
        "provider_name": provider,
        "issuer_type": issuer_type,
        "rating": rating,
        "min_amount": 100000,
        "max_amount": 999999999999,
        "max_horizon_days": 365,
        "tenor_days": days,
        "rollover_allowed": True,
        "quote_required": False,
        "supports_same_day": days <= 1,
        "supports_forward_value": True,
        "participation_flag": False,
        "government_only_flag": government_only,
        "base_liquidity_score": round(liquidity, 1),
        "base_operational_friction": round(friction, 1),
        "same_day_cutoff": defaults.get("same_day_cutoff", ""),
        "active_flag": True,
        "external_code": "",
        "auto_source": "manual",
        "notes": f"Dinamik türetilen piyasa enstrümanı | vade={days} gün | aile={family}",
    }


def synthesize_missing_instruments_from_quotes(instruments: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    inst = instruments.copy()
    if quotes.empty or "instrument_id" not in quotes.columns:
        return inst

    existing = set(inst["instrument_id"].astype(str).str.strip())
    q = ensure_columns(quotes, {"instrument_name": ""}).copy()
    q["instrument_id"] = q["instrument_id"].astype(str).str.strip()
    q_latest = latest_quotes(q[[c for c in q.columns if c in {"instrument_id", "instrument_name", "quote_timestamp"}]])

    new_rows = []
    for _, row in q_latest.iterrows():
        iid = str(row.get("instrument_id", "")).strip()
        if not iid or iid in existing:
            continue
        meta = infer_market_meta_from_instrument_id(iid, str(row.get("instrument_name", "") or ""))
        if meta:
            new_rows.append(meta)
            existing.add(iid)

    if new_rows:
        inst = pd.concat([inst, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    return inst


def best_live_quote_by_type(monitor_df: pd.DataFrame, instrument_type: str) -> dict:
    df = monitor_df[(monitor_df["instrument_type"] == instrument_type) & monitor_df["net_yield"].notna()].copy()
    if df.empty:
        return {}
    return df.sort_values(["net_yield", "quote_timestamp"], ascending=[False, False]).iloc[0].to_dict()


def format_quote_with_tenor(row: dict, empty: str = "-") -> str:
    if not row:
        return empty
    value = format_pct(row.get("net_yield", np.nan))
    tenor = tenor_label(row.get("tenor_days", np.nan))
    return f"{value} | {tenor}" if value != "-" else empty


# ---------- data cleaning ----------

def clean_data(instruments: pd.DataFrame, quotes: pd.DataFrame, rules: pd.DataFrame):
    if "notes" not in quotes.columns and "notes;;" in quotes.columns:
        quotes["notes"] = quotes["notes;;"]
    elif "notes;;" in quotes.columns:
        quotes["notes"] = quotes["notes"].fillna(quotes["notes;;"])

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
            "tenor_days": np.nan,
            "rollover_allowed": True,
            "instrument_family": "",
        },
    )
    quotes = ensure_columns(
        quotes,
        {
            "instrument_name": "",
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
        "rollover_allowed",
    ]:
        instruments[col] = parse_bool(instruments[col])

    for col in ["quote_confirmed", "capacity_available"]:
        quotes[col] = parse_bool(quotes[col])

    for col in ["participation_only_flag", "government_only_flag"]:
        rules[col] = parse_bool(rules[col])

    for col in ["min_amount", "max_amount", "max_horizon_days", "base_liquidity_score", "base_operational_friction", "tenor_days"]:
        instruments[col] = pd.to_numeric(instruments[col], errors="coerce")

    for col in ["gross_yield", "fee_bps", "tax_bps", "net_yield"]:
        quotes[col] = pd.to_numeric(quotes[col], errors="coerce")

    quotes["quote_timestamp"] = pd.to_datetime(quotes["quote_timestamp"], errors="coerce")
    quotes["quote_valid_until"] = pd.to_datetime(quotes["quote_valid_until"], errors="coerce")
    rules["max_horizon_days"] = pd.to_numeric(rules["max_horizon_days"], errors="coerce")

    instruments = synthesize_missing_instruments_from_quotes(instruments, quotes)

    for idx, row in instruments.iterrows():
        defaults = INSTRUMENT_TYPE_DEFAULTS.get(row["instrument_type"], {})
        for col, val in defaults.items():
            current = row.get(col)
            if (isinstance(current, str) and not current.strip()) or pd.isna(current):
                instruments.at[idx, col] = val

        if pd.isna(row.get("tenor_days")):
            inferred_days = infer_tenor_days_from_instrument_id(row.get("instrument_id", ""))
            if not pd.isna(inferred_days):
                instruments.at[idx, "tenor_days"] = inferred_days

        if not str(row.get("instrument_family", "")).strip():
            inferred = infer_market_meta_from_instrument_id(row.get("instrument_id", ""), row.get("instrument_name", ""))
            if inferred:
                instruments.at[idx, "instrument_family"] = inferred.get("instrument_family", "")
            else:
                instruments.at[idx, "instrument_family"] = row["instrument_type"]

    instruments["instrument_family"] = instruments["instrument_family"].fillna("").replace("", np.nan).fillna(instruments["instrument_type"])
    quotes["instrument_id"] = quotes["instrument_id"].astype(str).str.strip()
    instruments["instrument_id"] = instruments["instrument_id"].astype(str).str.strip()

    return instruments, quotes, rules



# ---------- scraping helpers ----------
def get_page_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text("\n", strip=True)


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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_tefas_fund(code: str) -> dict:
    code = code.strip().upper()
    if not code:
        raise ValueError("Fon kodu boş olamaz.")
    url = f"https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={code}"
    text = get_page_text(url)

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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_borsa_index_info(index_url: str) -> dict:
    text = get_page_text(index_url)

    def _extract(label: str):
        m = re.search(rf"{re.escape(label)}\s*([0-9\.,]+)", text, flags=re.IGNORECASE)
        return extract_numeric_amount(m.group(1)) if m else None

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
def fetch_tpp_public_reference() -> dict:
    # best-effort parser; page structure may change
    candidates = [
        "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-gunluk-bulten",
        "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-islem-ortalamalari-raporu",
    ]
    last_error = None
    for url in candidates:
        try:
            text = get_page_text(url)
            patterns = [
                r"O/?N[^\n]{0,80}?([0-9]{1,2}[\.,][0-9]{1,4})",
                r"Gecelik[^\n]{0,80}?([0-9]{1,2}[\.,][0-9]{1,4})",
                r"Ağırlıklı Ortalama Faiz[^\n]{0,80}?([0-9]{1,2}[\.,][0-9]{1,4})",
            ]
            for p in patterns:
                m = re.search(p, text, flags=re.IGNORECASE)
                if m:
                    rate = extract_numeric_pct(m.group(1))
                    if rate is not None:
                        return {"rate": rate, "url": url, "method": "regex"}
        except Exception as e:  # pragma: no cover - network failures are expected in some environments
            last_error = e
    raise RuntimeError(f"TPP kamuya açık referans verisi okunamadı: {last_error}")


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
    except Exception as e:
        errors.append(f"Repo benchmark çekilemedi: {e}")

    # TPP reference best effort
    try:
        tpp_info = fetch_tpp_public_reference()
        benchmarks["tpp"] = tpp_info
        if "tpp_on" in instruments_df["instrument_id"].values:
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
        tenor_days = safe_float(row.get("tenor_days"), np.nan)
        tenor_days = int(tenor_days) if not pd.isna(tenor_days) and tenor_days > 0 else np.nan
        rollover_count = 0
        liquidity_score = float(row["base_liquidity_score"] or 0)
        operational_friction = float(row["base_operational_friction"] or 0)

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

        if inst_type in MONEY_MARKET_TYPES and not pd.isna(tenor_days):
            if horizon_days < tenor_days:
                blocks.append(f"Araç vadesi ({tenor_days} gün) park süresinden uzun")
            else:
                rollover_count = max(int(math.ceil(horizon_days / tenor_days)) - 1, 0)
                if horizon_days == tenor_days:
                    reasons.append("Vade park süresiyle bire bir uyumlu")
                elif rollover_count > 0:
                    reasons.append(f"{rollover_count} kez rollover varsayımı içerir")
                if liquidity_need == "T+0" and tenor_days > 1:
                    blocks.append("T+0 ihtiyacı için vade çok uzun")
                if liquidity_need == "T+1" and tenor_days > 2:
                    blocks.append("T+1 ihtiyacı için vade çok uzun")
                liquidity_score = max(35, liquidity_score - min(max(tenor_days - 1, 0) * 0.70, 20))
                operational_friction = min(70, operational_friction + min(rollover_count * 2, 12))
        else:
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

        if inst_type == "Repo":
            reasons.append("Teminatlı repo kanalı")
        if inst_type == "TersRepo":
            reasons.append("Ters repo kanalı")
        if inst_type == "TPP":
            reasons.append("TPP ekranından fiyatlanan vade")
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
                    "Tenor": tenor_label(tenor_days),
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
                "Aile": row.get("instrument_family", inst_type),
                "Tenor (gün)": tenor_days if not pd.isna(tenor_days) else np.nan,
                "Tenor": tenor_label(tenor_days),
                "Rollover": rollover_count,
                "Sağlayıcı": row["provider_name"],
                "Gross Getiri (%)": row["gross_yield"],
                "Net Getiri (%)": net_yield,
                "Beklenen Net TL Katkı": net_contribution,
                "Likidite Skoru": float(liquidity_score),
                "Operasyonel Sürtünme": float(operational_friction),
                "Quote Teyitli": bool(row["quote_confirmed"]),
                "Nedenler": " | ".join(dict.fromkeys(reasons)),
                "Notlar": row.get("notes", "") or row.get("notes_q", "") or row.get("source", ""),
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
            score = 78
            if row["Quote Teyitli"]:
                score += 10
            if "Bugün içinde uygulanabilir" in row["Nedenler"]:
                score += 5
            if row["Rollover"] == 0:
                score += 5
            else:
                score -= min(int(row["Rollover"]) * 3, 15)
            fit_scores.append(max(min(score, 100), 30))
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




# ---------- quote monitoring ----------
def quote_stale_minutes(inst_type: str) -> int:
    return {"Repo": 30, "TersRepo": 30, "TPP": 30, "PPF": 24 * 60, "KVBAF": 24 * 60, "Mevduat": 6 * 60, "Katılım": 6 * 60}.get(inst_type, 6 * 60)



def build_quote_monitor(quotes: pd.DataFrame, instruments: pd.DataFrame) -> pd.DataFrame:
    latest_q = latest_quotes(quotes).merge(
        instruments[["instrument_id", "instrument_name", "instrument_type", "provider_name", "tenor_days", "instrument_family"]],
        on="instrument_id", how="left"
    )
    if latest_q.empty:
        return latest_q
    now = now_ist()
    latest_q["quote_timestamp"] = pd.to_datetime(latest_q["quote_timestamp"], errors="coerce")
    latest_q["quote_valid_until"] = pd.to_datetime(latest_q["quote_valid_until"], errors="coerce")
    latest_q["age_min"] = (pd.Timestamp(now) - latest_q["quote_timestamp"]).dt.total_seconds().div(60)
    latest_q["allowed_age_min"] = latest_q["instrument_type"].map(quote_stale_minutes)
    latest_q["expired_cutoff"] = latest_q["quote_valid_until"].notna() & (latest_q["quote_valid_until"] < pd.Timestamp(now))
    latest_q["stale"] = latest_q["age_min"].fillna(10**9) > latest_q["allowed_age_min"]
    latest_q["durum"] = np.where(latest_q["expired_cutoff"], "Cutoff geçti", np.where(latest_q["stale"], "Eski", "Güncel"))
    latest_q["status_icon"] = latest_q["durum"].map({"Güncel": "🟢", "Eski": "🟠", "Cutoff geçti": "🔴"}).fillna("⚪")
    latest_q["last_refresh_label"] = latest_q["quote_timestamp"].apply(format_dt_tr)
    latest_q["tenor_label"] = latest_q["tenor_days"].apply(tenor_label)
    return latest_q.sort_values(["instrument_family", "tenor_days", "instrument_name"]).reset_index(drop=True)


# ---------- session state ----------
def init_state():
    if "instruments_df" not in st.session_state:
        st.session_state["instruments_df"] = load_default_csv("instruments_master.csv")
    if "rules_df" not in st.session_state:
        st.session_state["rules_df"] = load_default_csv("portfolio_rules.csv")
    if "quotes_df" not in st.session_state:
        base_quotes = load_default_csv("market_quotes.csv")
        try:
            sheet_quotes = fetch_google_quotes_csv(GOOGLE_QUOTES_CSV_URL)
            st.session_state["quotes_df"] = merge_google_quotes(base_quotes, sheet_quotes, st.session_state["instruments_df"])
            st.session_state["quotes_source_status"] = "Google Sheets canlı veri kullanılıyor."
        except Exception as e:
            st.session_state["quotes_df"] = base_quotes
            st.session_state["quotes_source_status"] = f"Google Sheets okunamadı, yerel varsayılan quote'lar kullanılıyor: {e}"
    if "benchmarks" not in st.session_state:
        st.session_state["benchmarks"] = {}


# ---------- UI ----------
init_state()

st.title("💸 Nakit Park Motoru")
st.caption("Kısa park kararında repo, ters repo, TPP ve fonları vade bazında tek ekranda karşılaştıran iç kullanım karar destek ekranı")

with st.sidebar:
    st.subheader("Veri Kaynağı")
    st.caption("Repo ve TPP quote'ları Google Sheets köprüsünden, fon tarafı TEFAS ve repo içi veri dosyalarından gelir.")
    st.info(st.session_state.get("quotes_source_status", "-"))

    uploaded_instruments = st.file_uploader("instruments_master.csv", type=["csv"])
    uploaded_rules = st.file_uploader("portfolio_rules.csv", type=["csv"])

    if st.button("Google Sheets quote'larını şimdi yenile", use_container_width=True):
        try:
            fetch_google_quotes_csv.clear()
            sheet_quotes = fetch_google_quotes_csv(GOOGLE_QUOTES_CSV_URL)
            base_quotes = load_default_csv("market_quotes.csv")
            st.session_state["quotes_df"] = merge_google_quotes(base_quotes, sheet_quotes, st.session_state["instruments_df"])
            st.session_state["quotes_source_status"] = "Google Sheets canlı veri yenilendi."
            st.success("Google Sheets quote'ları yenilendi.")
        except Exception as e:
            st.error(f"Google Sheets verisi okunamadı: {e}")

    if st.button("Yüklenen instruments/rules dosyalarını uygula", use_container_width=True):
        st.session_state["instruments_df"] = load_csv_or_default(uploaded_instruments, "instruments_master.csv")
        st.session_state["rules_df"] = load_csv_or_default(uploaded_rules, "portfolio_rules.csv")
        try:
            fetch_google_quotes_csv.clear()
            sheet_quotes = fetch_google_quotes_csv(GOOGLE_QUOTES_CSV_URL)
            base_quotes = load_default_csv("market_quotes.csv")
            st.session_state["quotes_df"] = merge_google_quotes(base_quotes, sheet_quotes, st.session_state["instruments_df"])
            st.session_state["quotes_source_status"] = "Google Sheets canlı veri kullanılıyor."
        except Exception as e:
            st.session_state["quotes_df"] = load_default_csv("market_quotes.csv")
            st.session_state["quotes_source_status"] = f"Google Sheets okunamadı, yerel varsayılan quote'lar kullanılıyor: {e}"
        st.success("Dosyalar oturuma alındı.")

    if st.button("Yerel varsayılan veriye dön", use_container_width=True):
        st.session_state["instruments_df"] = load_default_csv("instruments_master.csv")
        st.session_state["rules_df"] = load_default_csv("portfolio_rules.csv")
        try:
            fetch_google_quotes_csv.clear()
            sheet_quotes = fetch_google_quotes_csv(GOOGLE_QUOTES_CSV_URL)
            st.session_state["quotes_df"] = merge_google_quotes(load_default_csv("market_quotes.csv"), sheet_quotes, st.session_state["instruments_df"])
            st.session_state["quotes_source_status"] = "Google Sheets canlı veri kullanılıyor."
        except Exception as e:
            st.session_state["quotes_df"] = load_default_csv("market_quotes.csv")
            st.session_state["quotes_source_status"] = f"Google Sheets okunamadı, yerel varsayılan quote'lar kullanılıyor: {e}"
        st.success("Varsayılan veri yüklendi.")

instruments, quotes, rules = clean_data(
    st.session_state["instruments_df"],
    st.session_state["quotes_df"],
    st.session_state["rules_df"],
)
st.session_state["instruments_df"] = instruments
st.session_state["quotes_df"] = quotes
st.session_state["rules_df"] = rules

monitor_df = build_quote_monitor(quotes, instruments)
money_market_monitor = monitor_df[monitor_df["instrument_type"].isin(MONEY_MARKET_TYPES)].copy()
repo_summary = best_live_quote_by_type(money_market_monitor, "Repo")
tersrepo_summary = best_live_quote_by_type(money_market_monitor, "TersRepo")
tpp_summary = best_live_quote_by_type(money_market_monitor, "TPP")
last_live_ts = money_market_monitor["quote_timestamp"].max() if not money_market_monitor.empty else pd.NaT
last_live_label = format_dt_tr(last_live_ts)
stale_money_market = money_market_monitor[money_market_monitor["durum"] != "Güncel"]

with st.sidebar:
    st.markdown("---")
    st.caption(f"Son güncelleme: **{last_live_label}**")
    if stale_money_market.empty:
        st.success("Repo / ters repo / TPP quote'ları güncel görünüyor.")
    else:
        stale_names = ", ".join(stale_money_market["instrument_name"].tolist())
        st.warning(f"Stale/expired quote uyarısı: {stale_names}")

summary_cols = st.columns(5)
with summary_cols[0]:
    render_metric_card("En iyi Repo", format_quote_with_tenor(repo_summary), repo_summary.get("durum", "-"))
with summary_cols[1]:
    render_metric_card("En iyi Ters Repo", format_quote_with_tenor(tersrepo_summary), tersrepo_summary.get("durum", "-"))
with summary_cols[2]:
    render_metric_card("En iyi TPP", format_quote_with_tenor(tpp_summary), tpp_summary.get("durum", "-"))
with summary_cols[3]:
    render_metric_card("Son güncelleme", last_live_label, "Canlı quote zamanı")
with summary_cols[4]:
    fresh_count = int((money_market_monitor["durum"] == "Güncel").sum()) if not money_market_monitor.empty else 0
    total_count = int(len(money_market_monitor)) if not money_market_monitor.empty else 0
    render_metric_card("Canlı quote oranı", f"{fresh_count}/{total_count}", "Repo / ters repo / TPP sağlığı")

if not stale_money_market.empty:
    st.warning("Repo, ters repo veya TPP quote'larından en az biri eski ya da cutoff sonrası durumda. Karar vermeden önce quote yenile.")

tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "PM Görünümü",
    "Karar Motoru",
    "Otomatik Güncelle",
    "Quote Yönetimi",
    "Veri / Kurulum",
])


with tab0:
    st.markdown("### Portföy Yöneticisi Özeti")
    pm_cols = st.columns([1.4, 1.1, 1.2])
    with pm_cols[0]:
        st.markdown("#### Canlı piyasa enstrümanları")
        core_view = money_market_monitor[["status_icon", "instrument_name", "instrument_type", "tenor_label", "net_yield", "durum", "last_refresh_label", "source"]].copy()
        if core_view.empty:
            st.info("Repo / ters repo / TPP canlı quote bulunamadı.")
        else:
            core_view = core_view.rename(columns={"status_icon": "Durum", "instrument_name": "Araç", "instrument_type": "Tür", "tenor_label": "Vade", "net_yield": "Net Getiri", "last_refresh_label": "Son Güncelleme", "source": "Kaynak"})
            st.dataframe(core_view, use_container_width=True, hide_index=True, column_config={"Net Getiri": st.column_config.NumberColumn(format="%.2f")})
    with pm_cols[1]:
        st.markdown("#### Fon tarafı")
        fund_view = monitor_df[monitor_df["instrument_type"].isin(["PPF", "KVBAF", "Katılım"])].sort_values("net_yield", ascending=False).head(8)[["instrument_name", "instrument_type", "net_yield", "last_refresh_label"]]
        if fund_view.empty:
            st.info("Fon quote'u yok.")
        else:
            fund_view = fund_view.rename(columns={"instrument_name": "Fon", "instrument_type": "Tür", "net_yield": "Net Getiri", "last_refresh_label": "Son Güncelleme"})
            st.dataframe(fund_view, use_container_width=True, hide_index=True, column_config={"Net Getiri": st.column_config.NumberColumn(format="%.2f")})
    with pm_cols[2]:
        st.markdown("#### Kullanım notu")
        st.write("1. Önce canlı quote'ları Google Sheets'ten yenileyin.")
        st.write("2. VBA köprüsü repo, ters repo ve TPP tarafında dolu gelen tüm vadeleri aynı Google Sheet'e gönderebilsin.")
        st.write("3. Karar Motoru, park süresi kısa olanı uzun vade enstrümanlarla karıştırmasın; vade uyumu ve rollover varsayımını göstersin.")
        st.info("Bu ekran tavsiye motoru değil; ilk filtreleme, canlı quote okuma ve hızlı kıyaslama ekranıdır.")

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
    ref_cols = st.columns(5)
    ppf_rows = latest_q.merge(instruments[["instrument_id", "instrument_type"]], on="instrument_id", how="left")
    ppf_mean = ppf_rows[ppf_rows["instrument_type"] == "PPF"]["net_yield"].mean()
    ref_cols[0].metric("En iyi Repo", format_quote_with_tenor(repo_summary))
    ref_cols[1].metric("En iyi Ters Repo", format_quote_with_tenor(tersrepo_summary))
    ref_cols[2].metric("En iyi TPP", format_quote_with_tenor(tpp_summary))
    ref_cols[3].metric("PPF ort. net", format_pct(ppf_mean) if not pd.isna(ppf_mean) else "-")
    ref_cols[4].metric("Canlı quote zamanı", last_live_label)

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
                    st.write(f"Tür: {rec['Tür']}  |  Vade: {rec['Tenor']}  |  Sağlayıcı: {rec['Sağlayıcı']}")
                    st.write(f"Nedenler: {rec['Nedenler'] or '—'}")
                    st.caption(f"Kaynak: {rec['Source'] or '-'}")
                    if rec["Notlar"]:
                        st.caption(rec["Notlar"])
                with right:
                    st.metric("Net getiri", f"%{rec['Net Getiri (%)']:.2f}")
                    st.metric("Net katkı", format_currency(rec["Beklenen Net TL Katkı"]))
                    st.metric("Skor", f"{rec['Toplam Skor']:.1f}")

        show_cols = [
            "Sıra", "Araç", "Tür", "Tenor", "Rollover", "Sağlayıcı", "Net Getiri (%)", "Beklenen Net TL Katkı",
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
                st.warning("Bazı fonlar güncellenemedi:")
                for err in errors:
                    st.write(f"- {err}")

    st.markdown("### 2) Kamuya açık referansları çek")
    st.caption("Kamuya açık referans çekimi O/N benchmark amaçlı kalır; çoklu vade canlı akışını VBA + Google Sheets köprüsü yönetir.")
    if st.button("Repo / TPP O/N benchmarkını çek"):
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
            st.success("Fon verisi çekildi.")
            st.json(info)
        except Exception as e:
            st.error(f"Fon verisi çekilemedi: {e}")

with tab3:
    st.markdown("### Manuel quote girişi")
    manual_instruments = instruments[instruments["instrument_type"].isin(["Repo", "TersRepo", "TPP", "Mevduat", "Katılım"])]
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
    latest_q = build_quote_monitor(quotes, instruments)
    display_q = latest_q[[
        "status_icon", "durum", "instrument_name", "instrument_type", "tenor_label", "provider_name", "gross_yield", "net_yield",
        "quote_timestamp", "quote_valid_until", "quote_confirmed", "source", "notes"
    ]].rename(columns={"status_icon": "Durum İkonu", "durum": "Durum", "tenor_label": "Vade"})
    st.dataframe(
        display_q,
        use_container_width=True,
        hide_index=True,
        column_config={
            "gross_yield": st.column_config.NumberColumn("Brüt Getiri", format="%.2f"),
            "net_yield": st.column_config.NumberColumn("Net Getiri", format="%.2f"),
            "quote_timestamp": st.column_config.DatetimeColumn("Quote Zamanı", format="DD.MM.YYYY HH:mm"),
            "quote_valid_until": st.column_config.DatetimeColumn("Geçerlilik", format="DD.MM.YYYY HH:mm"),
        },
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
        "4) Repo/ters repo/TPP tarafı Google Sheets canlı köprüsünden gelir; VBA tüm dolu vadeleri aynı sheet'e atabilir. "
        "5) Mevduat ve katılım için yalnızca gelen gerçek banka quote'unu Manuel Quote sekmesinden gir. "
        "6) İstersen güncel CSV'leri indirip GitHub'a geri koy; böylece son veri repo'da kalıcı olur."
    )

    st.markdown("### Önemli notlar")
    st.markdown(
        """
        - TEFAS tarafı otomatik.
        - Repo ve TPP tarafındaki kamuya açık referans çekimi **best-effort** O/N benchmark amaçlıdır; kesin executable quote değildir.
        - Çoklu vade repo / ters repo / TPP akışı Google Sheets köprüsünden okunur; CSV upload gerekmez.
        - Mevduat ve katılım oranı manuel kalır; çünkü gerçek banka quote'u gerekir.
        """
    )

