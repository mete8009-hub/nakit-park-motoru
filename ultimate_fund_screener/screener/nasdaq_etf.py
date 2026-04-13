from __future__ import annotations

from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import HEADERS, NASDAQ_ETF_LIST_URL
from .heuristics import classify_etf_name, detect_theme, liquidity_from_category, safe_json


def fetch_nasdaq_etf_universe() -> pd.DataFrame:
    resp = requests.get(NASDAQ_ETF_LIST_URL, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    text = resp.text
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and lines[-1].startswith("File Creation Time"):
        lines = lines[:-1]
    df = pd.read_csv(StringIO("\n".join(lines)), sep="|")
    etf = df[df["ETF"] == "Y"].copy()
    now = datetime.utcnow().isoformat(timespec="seconds")
    rows = []
    for _, row in etf.iterrows():
        ticker = str(row["Symbol"]).strip().upper()
        name = str(row["Security Name"]).strip()
        asset, sub, duration, esg = classify_etf_name(name)
        theme = detect_theme(name, sub)
        rows.append(
            {
                "instrument_id": f"ETF:{ticker}",
                "external_code": ticker,
                "source_type": "nasdaq_etf",
                "name": name,
                "asset_class": asset,
                "sub_asset_class": sub,
                "category": sub,
                "theme": theme,
                "issuer": "",
                "currency": "USD",
                "country": "US",
                "region": "Global",
                "rating": "NR",
                "maturity_date": None,
                "duration_years": duration,
                "yield_pct": None,
                "ytm_pct": None,
                "ret_1m_pct": None,
                "ret_3m_pct": None,
                "ret_6m_pct": None,
                "ret_1y_pct": None,
                "vol_1y_pct": None,
                "max_drawdown_1y_pct": None,
                "liquidity_score": liquidity_from_category(sub, "etf"),
                "esg_score": esg,
                "aum_mn": None,
                "expense_ratio": None,
                "risk_score": None,
                "last_price": None,
                "is_demo": 0,
                "last_refresh_at": now,
                "metadata_json": safe_json(
                    {
                        "listing_source": "nasdaqtrader",
                        "market_category": row.get("Market Category"),
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def fetch_etf_history(tickers: list[str], period: str = "2y") -> tuple[pd.DataFrame, pd.DataFrame]:
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    raw = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    price_rows = []
    metric_rows = []
    now = datetime.utcnow().isoformat(timespec="seconds")
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                close = raw["Close"].dropna().rename(ticker)
            else:
                close = raw[ticker]["Close"].dropna()
        except Exception:
            continue
        if close.empty:
            continue
        rets = close.pct_change().dropna()
        running = close.cummax()
        dd = close / running - 1
        price_rows.extend(
            {
                "instrument_id": f"ETF:{ticker}",
                "date": idx.date().isoformat(),
                "price": float(val),
                "source": "yfinance",
            }
            for idx, val in close.items()
        )
        metric_rows.append(
            {
                "instrument_id": f"ETF:{ticker}",
                "last_price": float(close.iloc[-1]),
                "ret_1m_pct": _period_return(close, 21),
                "ret_3m_pct": _period_return(close, 63),
                "ret_6m_pct": _period_return(close, 126),
                "ret_1y_pct": _period_return(close, 252),
                "vol_1y_pct": float(rets.tail(252).std() * np.sqrt(252) * 100) if len(rets) >= 20 else None,
                "max_drawdown_1y_pct": float(dd.tail(252).min() * 100) if len(dd) >= 20 else None,
                "last_refresh_at": now,
            }
        )
    return pd.DataFrame(price_rows), pd.DataFrame(metric_rows)


def _period_return(close: pd.Series, lookback: int) -> float | None:
    if len(close) <= lookback:
        return None
    base = close.iloc[-lookback - 1]
    if not base:
        return None
    return float((close.iloc[-1] / base - 1) * 100)
