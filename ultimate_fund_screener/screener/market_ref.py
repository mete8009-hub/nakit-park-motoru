from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import HEADERS, REPO_INDEX_URL, TPP_URLS
from .heuristics import liquidity_from_category, safe_json


def get_page_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text("\n", strip=True)


def _num(value: str | None) -> float | None:
    if not value:
        return None
    s = value.strip().replace("%", "").replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def fetch_repo_reference() -> float | None:
    text = get_page_text(REPO_INDEX_URL)

    def extract(label: str):
        m = re.search(rf"{re.escape(label)}\s*([0-9\.,]+)", text, flags=re.IGNORECASE)
        return _num(m.group(1)) if m else None

    current = extract("Current Value")
    prev = extract("Previous Close")
    if not current or not prev or prev == 0:
        return None
    daily = current / prev - 1
    annualized = daily * 365 * 100
    if annualized <= 0 or annualized > 1000:
        return None
    return round(annualized, 2)


def fetch_tpp_reference() -> float | None:
    patterns = [
        r"O/?N[^\n]{0,120}?([0-9]{1,2}[\.,][0-9]{1,4})",
        r"Gecelik[^\n]{0,120}?([0-9]{1,2}[\.,][0-9]{1,4})",
        r"Ağırlıklı Ortalama Faiz[^\n]{0,120}?([0-9]{1,2}[\.,][0-9]{1,4})",
    ]
    for url in TPP_URLS:
        try:
            text = get_page_text(url)
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE)
                if m:
                    rate = _num(m.group(1))
                    if rate is not None:
                        return round(rate, 2)
        except Exception:
            continue
    return None


def build_try_reference_rows() -> pd.DataFrame:
    now = datetime.utcnow().isoformat(timespec="seconds")
    rows = []
    repo = fetch_repo_reference()
    if repo is not None:
        rows.append(
            {
                "instrument_id": "TRY:REPO_ON",
                "external_code": "REPO_ON",
                "source_type": "public_try_ref",
                "name": "BIST Repo O/N Referans",
                "asset_class": "Money Market",
                "sub_asset_class": "Repo",
                "category": "Repo",
                "theme": "",
                "issuer": "Borsa Istanbul",
                "currency": "TRY",
                "country": "TR",
                "region": "Turkey",
                "rating": "AAA",
                "maturity_date": None,
                "duration_years": 0.01,
                "yield_pct": repo,
                "ytm_pct": repo,
                "ret_1m_pct": None,
                "ret_3m_pct": None,
                "ret_6m_pct": None,
                "ret_1y_pct": None,
                "vol_1y_pct": None,
                "max_drawdown_1y_pct": None,
                "liquidity_score": liquidity_from_category("repo"),
                "esg_score": None,
                "aum_mn": None,
                "expense_ratio": 0.0,
                "risk_score": 5.0,
                "last_price": 100.0,
                "is_demo": 0,
                "last_refresh_at": now,
                "metadata_json": safe_json({"kind": "public_reference"}),
            }
        )
    tpp = fetch_tpp_reference()
    if tpp is not None:
        rows.append(
            {
                "instrument_id": "TRY:TPP_ON",
                "external_code": "TPP_ON",
                "source_type": "public_try_ref",
                "name": "Takasbank Para Piyasası O/N Referans",
                "asset_class": "Money Market",
                "sub_asset_class": "TPP",
                "category": "TPP",
                "theme": "",
                "issuer": "Takasbank",
                "currency": "TRY",
                "country": "TR",
                "region": "Turkey",
                "rating": "AAA",
                "maturity_date": None,
                "duration_years": 0.01,
                "yield_pct": tpp,
                "ytm_pct": tpp,
                "ret_1m_pct": None,
                "ret_3m_pct": None,
                "ret_6m_pct": None,
                "ret_1y_pct": None,
                "vol_1y_pct": None,
                "max_drawdown_1y_pct": None,
                "liquidity_score": liquidity_from_category("tpp"),
                "esg_score": None,
                "aum_mn": None,
                "expense_ratio": 0.0,
                "risk_score": 5.0,
                "last_price": 100.0,
                "is_demo": 0,
                "last_refresh_at": now,
                "metadata_json": safe_json({"kind": "public_reference"}),
            }
        )
    return pd.DataFrame(rows)
