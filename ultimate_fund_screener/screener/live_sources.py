from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from .config import DEFAULT_HISTORY_YEARS, ETF_WATCHLIST, HTTP_HEADERS, PUBLIC_MARKET_INSTRUMENTS, TEFAS_WATCHLIST


REPO_BENCHMARK_URL = "https://www.borsaistanbul.com/en/index/repbr"
TPP_URLS = [
    "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-gunluk-bulten",
    "https://www.takasbank.com.tr/tr/istatistikler/takasbank-para-piyasasi-tpp/tpp-islem-ortalamalari-raporu",
]


def _today_str() -> str:
    return pd.Timestamp.today().normalize().strftime("%Y-%m-%d")


def _get_page_text(url: str) -> str:
    resp = requests.get(url, headers=HTTP_HEADERS, timeout=25)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text("\n", strip=True)


def _extract_numeric_pct(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = str(text).strip().replace("%", "").replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except Exception:
        return None


def _extract_numeric_amount(text: str | None) -> float | None:
    if not text:
        return None
    cleaned = str(text).strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except Exception:
        return None


def _extract_after_label(text: str, label: str, pattern: str) -> str | None:
    regex = rf"{re.escape(label)}\s*{pattern}"
    m = re.search(regex, text, flags=re.IGNORECASE)
    return m.group(1).strip() if m else None


def fetch_tefas_fund(fund_code: str) -> dict[str, Any]:
    code = str(fund_code).strip().upper()
    url = f"https://www.tefas.gov.tr/FonAnaliz.aspx?FonKod={code}"
    text = _get_page_text(url)

    name = _extract_after_label(text, "Fon Detaylı Analiz", r"([A-ZÇĞİÖŞÜ0-9\-\(\)\.\s]+FON)")
    category = _extract_after_label(text, "Kategorisi", r"([A-Za-zÇĞİÖŞÜçğıöşü\s\-\(\)]+)")
    daily_return_text = _extract_after_label(text, "Günlük Getiri (%)", r"(%?[0-9\.,-]+)")
    one_month_text = _extract_after_label(text, "Son 1 Ay Getirisi", r"(%?[0-9\.,-]+)")
    one_year_text = _extract_after_label(text, "Son 1 Yıl Getirisi", r"(%?[0-9\.,-]+)")
    price_text = _extract_after_label(text, "Son Fiyat (TL)", r"([0-9\.,]+)")
    ftv_text = _extract_after_label(text, "Fon Toplam Değer (TL)", r"([0-9\.,]+)")

    daily_return = _extract_numeric_pct(daily_return_text) or 0.0
    one_month_return = _extract_numeric_pct(one_month_text) or 0.0
    one_year_return = _extract_numeric_pct(one_year_text) or np.nan
    price = _extract_numeric_amount(price_text)
    fund_total_value = _extract_numeric_amount(ftv_text)

    annualized_daily = daily_return * 365
    annualized_monthly = one_month_return * 12
    annualized_proxy = annualized_daily if daily_return > 0 else annualized_monthly

    return {
        "fund_code": code,
        "name": name or code,
        "category": category or "",
        "price": price,
        "daily_return_pct": round(daily_return, 6),
        "one_month_return_pct": round(one_month_return, 6),
        "one_year_return_pct": round(one_year_return, 6) if not pd.isna(one_year_return) else np.nan,
        "annualized_proxy_pct": round(annualized_proxy, 2),
        "fund_total_value": fund_total_value / 1_000_000 if fund_total_value is not None else np.nan,
        "source_url": url,
        "asof_date": _today_str(),
    }


def fetch_tefas_watchlist() -> tuple[pd.DataFrame, pd.DataFrame]:
    instruments: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    for item in TEFAS_WATCHLIST:
        info = fetch_tefas_fund(item["fund_code"])
        instruments.append(
            {
                "instrument_id": item["instrument_id"],
                "symbol": item["fund_code"],
                "name": info["name"],
                "asset_class": item["asset_class"],
                "sub_asset_class": item["sub_asset_class"],
                "category": item["category"],
                "currency": item["currency"],
                "region": item["region"],
                "country": item["country"],
                "issuer": item["issuer"],
                "issuer_type": item["issuer_type"],
                "rating": item["rating"],
                "duration_years": item["duration_years"],
                "maturity_date": None,
                "coupon": 0.0,
                "expense_ratio": np.nan,
                "aum_mn": info["fund_total_value"],
                "esg_score": np.nan,
                "liquidity_score": 70.0 if item["category"] == "KVBAF" else 92.0,
                "yield_pct": info["annualized_proxy_pct"],
                "ytm_pct": info["annualized_proxy_pct"],
                "risk_score": 15.0 if item["category"] == "KVBAF" else 8.0,
                "theme": item["theme"],
                "tradable": 1,
                "source": "TEFAS",
                "is_demo": 0,
            }
        )
        snapshots.append(
            {
                "instrument_id": item["instrument_id"],
                "asof_date": info["asof_date"],
                "price": info["price"] if info["price"] is not None else 1.0,
                "nav": info["price"] if info["price"] is not None else 1.0,
                "yield_pct": info["annualized_proxy_pct"],
                "ytm_pct": info["annualized_proxy_pct"],
                "spread_bps": np.nan,
                "expense_ratio": np.nan,
                "esg_score": np.nan,
                "liquidity_score": 70.0 if item["category"] == "KVBAF" else 92.0,
                "aum_mn": info["fund_total_value"],
                "quote_source": f"TEFAS {info['fund_code']}",
                "updated_at": str(pd.Timestamp.now()),
            }
        )
    return pd.DataFrame(instruments), pd.DataFrame(snapshots)


def _extract_first_number(text: str) -> float | None:
    m = re.search(r"([0-9]{1,3}(?:[\.,][0-9]{1,6})?)", text)
    return _extract_numeric_pct(m.group(1)) if m else None


def fetch_bist_repo_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    text = _get_page_text(REPO_BENCHMARK_URL)
    current_match = re.search(r"Current Value\s*([0-9\.,]+)", text, flags=re.IGNORECASE)
    prev_match = re.search(r"Previous Close\s*([0-9\.,]+)", text, flags=re.IGNORECASE)
    current_val = _extract_numeric_amount(current_match.group(1)) if current_match else None
    previous_close = _extract_numeric_amount(prev_match.group(1)) if prev_match else None
    repo_rate = np.nan
    if current_val and previous_close and previous_close != 0:
        daily_return = current_val / previous_close - 1
        derived = daily_return * 365 * 100
        if 0 <= derived <= 1000:
            repo_rate = round(derived, 2)

    inst = pd.DataFrame(
        [
            {
                **PUBLIC_MARKET_INSTRUMENTS[0],
                "symbol": "REPO_ON",
                "maturity_date": None,
                "coupon": 0.0,
                "expense_ratio": 0.0,
                "aum_mn": np.nan,
                "esg_score": np.nan,
                "liquidity_score": 95.0,
                "yield_pct": repo_rate,
                "ytm_pct": repo_rate,
                "risk_score": 5.0,
                "tradable": 1,
                "is_demo": 0,
            }
        ]
    )
    snap = pd.DataFrame(
        [
            {
                "instrument_id": "repo_on",
                "asof_date": _today_str(),
                "price": 1.0,
                "nav": 1.0,
                "yield_pct": repo_rate,
                "ytm_pct": repo_rate,
                "spread_bps": np.nan,
                "expense_ratio": 0.0,
                "esg_score": np.nan,
                "liquidity_score": 95.0,
                "aum_mn": np.nan,
                "quote_source": "Borsa Istanbul Public Repo Benchmark",
                "updated_at": str(pd.Timestamp.now()),
            }
        ]
    )
    return inst, snap


def fetch_tpp_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    rate = np.nan
    last_error = None
    for url in TPP_URLS:
        try:
            text = _get_page_text(url)
            for pattern in [
                r"O/?N[^\n]{0,80}?([0-9]{1,2}[\.,][0-9]{1,4})",
                r"Gecelik[^\n]{0,80}?([0-9]{1,2}[\.,][0-9]{1,4})",
                r"Ağırlıklı Ortalama Faiz[^\n]{0,80}?([0-9]{1,2}[\.,][0-9]{1,4})",
            ]:
                m = re.search(pattern, text, flags=re.IGNORECASE)
                if m:
                    parsed = _extract_numeric_pct(m.group(1))
                    if parsed is not None:
                        rate = round(parsed, 2)
                        raise StopIteration(url)
        except StopIteration as done:
            src_url = str(done)
            break
        except Exception as exc:  # pragma: no cover
            last_error = exc
    else:
        src_url = TPP_URLS[0]
        if last_error:
            raise RuntimeError(f"TPP referansı çekilemedi: {last_error}")

    inst = pd.DataFrame(
        [
            {
                **PUBLIC_MARKET_INSTRUMENTS[1],
                "symbol": "TPP_ON",
                "maturity_date": None,
                "coupon": 0.0,
                "expense_ratio": 0.0,
                "aum_mn": np.nan,
                "esg_score": np.nan,
                "liquidity_score": 92.0,
                "yield_pct": rate,
                "ytm_pct": rate,
                "risk_score": 6.0,
                "tradable": 1,
                "is_demo": 0,
            }
        ]
    )
    snap = pd.DataFrame(
        [
            {
                "instrument_id": "tpp_on",
                "asof_date": _today_str(),
                "price": 1.0,
                "nav": 1.0,
                "yield_pct": rate,
                "ytm_pct": rate,
                "spread_bps": np.nan,
                "expense_ratio": 0.0,
                "esg_score": np.nan,
                "liquidity_score": 92.0,
                "aum_mn": np.nan,
                "quote_source": f"Takasbank Public {src_url}",
                "updated_at": str(pd.Timestamp.now()),
            }
        ]
    )
    return inst, snap


def _safe_info(ticker: yf.Ticker) -> dict[str, Any]:
    try:
        data = ticker.info or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _extract_expense_ratio(info: dict[str, Any], fallback: float) -> float:
    for key in ["annualReportExpenseRatio", "netExpenseRatio", "annualHoldingsTurnover"]:
        value = info.get(key)
        if value is None:
            continue
        try:
            v = float(value)
            return round(v * 100 if v < 1 else v, 4)
        except Exception:
            continue
    return fallback


def _extract_yield_pct(info: dict[str, Any]) -> float | float:
    for key in ["yield", "yieldPct", "trailingAnnualDividendYield", "dividendYield"]:
        value = info.get(key)
        if value is None:
            continue
        try:
            v = float(value)
            return round(v * 100 if v < 1 else v, 4)
        except Exception:
            continue
    return np.nan


def fetch_yfinance_watchlist(years: int = DEFAULT_HISTORY_YEARS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    instruments: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    histories: list[pd.DataFrame] = []

    for item in ETF_WATCHLIST:
        ticker = yf.Ticker(item["ticker"])
        hist = ticker.history(period=f"{max(1, int(years))}y", auto_adjust=True)
        if hist.empty:
            raise RuntimeError(f"Yahoo Finance history boş geldi: {item['ticker']}")
        hist = hist.reset_index()
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]
        hist[date_col] = pd.to_datetime(hist[date_col]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        close_col = "Close" if "Close" in hist.columns else hist.columns[-1]
        last_price = float(hist[close_col].iloc[-1])
        px = hist[[date_col, close_col]].rename(columns={date_col: "date", close_col: "value"})
        px["instrument_id"] = item["instrument_id"]
        histories.append(px[["instrument_id", "date", "value"]])

        info = _safe_info(ticker)
        long_name = info.get("longName") or info.get("shortName") or item["name"]
        currency = info.get("currency") or item["currency"]
        aum = info.get("totalAssets")
        try:
            aum_mn = round(float(aum) / 1_000_000, 2) if aum is not None else item.get("aum_mn", np.nan)
        except Exception:
            aum_mn = item.get("aum_mn", np.nan)
        yield_pct = _extract_yield_pct(info)
        expense_ratio = _extract_expense_ratio(info, item.get("expense_ratio", np.nan))
        ytm_pct = yield_pct if item["sub_asset_class"] == "Fixed Income" else np.nan
        ytm_pct = item.get("ytm_pct", ytm_pct) if pd.isna(ytm_pct) else ytm_pct

        instruments.append(
            {
                "instrument_id": item["instrument_id"],
                "symbol": item["ticker"],
                "name": long_name,
                "asset_class": item["asset_class"],
                "sub_asset_class": item["sub_asset_class"],
                "category": item["category"],
                "currency": currency,
                "region": item["region"],
                "country": item["country"],
                "issuer": item["issuer"],
                "issuer_type": item["issuer_type"],
                "rating": item["rating"],
                "duration_years": item["duration_years"],
                "maturity_date": None,
                "coupon": 0.0,
                "expense_ratio": expense_ratio,
                "aum_mn": aum_mn,
                "esg_score": item.get("esg_score", np.nan),
                "liquidity_score": item.get("liquidity_score", np.nan),
                "yield_pct": yield_pct,
                "ytm_pct": ytm_pct,
                "risk_score": 55.0 if item["sub_asset_class"] == "Equity" else 25.0,
                "theme": item.get("theme"),
                "tradable": 1,
                "source": "Yahoo Finance",
                "is_demo": 0,
            }
        )
        snapshots.append(
            {
                "instrument_id": item["instrument_id"],
                "asof_date": str(px["date"].iloc[-1]),
                "price": last_price,
                "nav": last_price,
                "yield_pct": yield_pct,
                "ytm_pct": ytm_pct,
                "spread_bps": np.nan,
                "expense_ratio": expense_ratio,
                "esg_score": item.get("esg_score", np.nan),
                "liquidity_score": item.get("liquidity_score", np.nan),
                "aum_mn": aum_mn,
                "quote_source": f"Yahoo Finance {item['ticker']}",
                "updated_at": str(pd.Timestamp.now()),
            }
        )

    history = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame(columns=["instrument_id", "date", "value"])
    return pd.DataFrame(instruments), pd.DataFrame(snapshots), history
