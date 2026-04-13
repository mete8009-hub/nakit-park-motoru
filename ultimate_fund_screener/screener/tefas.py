from __future__ import annotations

import json

from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import HEADERS, TEFAS_API_BASE
from .heuristics import detect_theme, liquidity_from_category, normalize_tefas_category, safe_json, tefas_duration_years

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def _post(path: str, payload: dict[str, str]) -> dict:
    resp = SESSION.post(
        f'{TEFAS_API_BASE}/{path}',
        data=payload,
        headers={
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': 'https://www.tefas.gov.tr',
            'X-Requested-With': 'XMLHttpRequest',
            **HEADERS,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_funds_fast() -> pd.DataFrame:
    data = _post('BindComparisonFundReturns', {
        'calismatipi': '2', 'fontip': 'YAT', 'sfontur': 'Tümü', 'kurucukod': '', 'fongrup': '',
        'bastarih': 'Başlangıç', 'bittarih': 'Bitiş', 'fonturkod': '', 'fonunvantip': '', 'strperiod': '1,1,1,1,1,1,1', 'islemdurum': '1'
    }).get('data', [])
    rows = []
    now = datetime.utcnow().isoformat(timespec='seconds')
    for item in data:
        code = str(item.get('FONKODU') or '').strip().upper()
        if not code:
            continue
        name = str(item.get('FONUNVAN') or '').strip()
        raw_cat = str(item.get('FONTURACIKLAMA') or '').strip()
        cat = normalize_tefas_category(raw_cat)
        theme = detect_theme(name, raw_cat)
        rows.append({
            'instrument_id': f'TEFAS:{code}',
            'external_code': code,
            'source_type': 'tefas',
            'name': name or code,
            'asset_class': 'Fund',
            'sub_asset_class': 'TEFAS Fund',
            'category': cat,
            'theme': theme,
            'issuer': '',
            'currency': 'TRY',
            'country': 'TR',
            'region': 'Turkey',
            'rating': 'NR',
            'maturity_date': None,
            'duration_years': tefas_duration_years(cat),
            'yield_pct': None,
            'ytm_pct': None,
            'ret_1m_pct': item.get('GETIRI1A'),
            'ret_3m_pct': item.get('GETIRI3A'),
            'ret_6m_pct': item.get('GETIRI6A'),
            'ret_1y_pct': item.get('GETIRI1Y'),
            'vol_1y_pct': None,
            'max_drawdown_1y_pct': None,
            'liquidity_score': liquidity_from_category(cat),
            'esg_score': 75.0 if 'green' in theme.lower() or 'esg' in theme.lower() else 55.0,
            'aum_mn': None,
            'expense_ratio': None,
            'risk_score': None,
            'last_price': None,
            'is_demo': 0,
            'last_refresh_at': now,
            'metadata_json': safe_json({'raw_category': raw_cat}),
        })
    return pd.DataFrame(rows)


def fetch_fund_detail(code: str) -> dict | None:
    data = _post('GetAllFundAnalyzeData', {'dil': 'TR', 'fonkod': code.upper()})
    info = (data.get('fundInfo') or [])
    ret = (data.get('fundReturn') or [{}])
    if not info:
        return None
    fi = info[0]
    fr = ret[0] if ret else {}
    return {
        'name': fi.get('FONUNVAN'),
        'date': fi.get('TARIH'),
        'price': fi.get('SONFIYAT'),
        'fund_size': fi.get('PORTBUYUKLUK'),
        'investor_count': fi.get('YATIRIMCISAYI'),
        'founder': fi.get('KURUCU'),
        'manager': fi.get('YONETICI'),
        'fund_type': fi.get('FONTUR'),
        'category': fi.get('FONKATEGORI'),
        'risk_value': fi.get('RISKDEGERI'),
        'daily_return': fi.get('GUNLUKGETIRI'),
        'ret_1m': fr.get('GETIRI1A'),
        'ret_3m': fr.get('GETIRI3A'),
        'ret_6m': fr.get('GETIRI6A'),
        'ret_ytd': fr.get('GETIRIYB'),
        'ret_1y': fr.get('GETIRI1Y'),
        'ret_3y': fr.get('GETIRI3Y'),
        'ret_5y': fr.get('GETIRI5Y'),
    }


def enrich_top_tefas_categories(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    target = df[df['category'].isin(['PPF', 'KVBAF'])].copy()
    if target.empty:
        return df
    rows = []
    for code in target['external_code'].tolist():
        try:
            detail = fetch_fund_detail(code)
        except Exception:
            detail = None
        if not detail:
            continue
        rows.append((code, detail))
    if not rows:
        return df
    out = df.copy()
    for code, detail in rows:
        mask = out['external_code'] == code
        out.loc[mask, 'name'] = detail.get('name') or out.loc[mask, 'name']
        out.loc[mask, 'issuer'] = detail.get('founder') or out.loc[mask, 'issuer']
        out.loc[mask, 'last_price'] = detail.get('price')
        try:
            out.loc[mask, 'aum_mn'] = float(detail.get('fund_size') or 0) / 1_000_000
        except Exception:
            pass
        out.loc[mask, 'risk_score'] = detail.get('risk_value')
        out.loc[mask, 'ret_1m_pct'] = detail.get('ret_1m')
        out.loc[mask, 'ret_3m_pct'] = detail.get('ret_3m')
        out.loc[mask, 'ret_6m_pct'] = detail.get('ret_6m')
        out.loc[mask, 'ret_1y_pct'] = detail.get('ret_1y')
        out.loc[mask, 'metadata_json'] = out.loc[mask, 'metadata_json'].apply(lambda x: safe_json({**(json.loads(x) if isinstance(x, str) and x else {}), 'detail': detail}))
    return out


def fetch_fund_history(code: str, days: int = 730) -> pd.DataFrame:
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    payload = {
        'fontip': 'YAT', 'sfontur': '', 'fonkod': code.upper(), 'fongrup': '',
        'bastarih': start.strftime('%d.%m.%Y'), 'bittarih': end.strftime('%d.%m.%Y'), 'fonturkod': '', 'fonunvantip': '', 'kurucukod': ''
    }
    data = _post('BindHistoryInfo', payload).get('data', [])
    rows = []
    for item in data:
        ts = item.get('TARIH')
        if not ts:
            continue
        dt = pd.to_datetime(ts, unit='ms', errors='coerce')
        price = item.get('FIYAT')
        if pd.isna(dt) or price in (None, ''):
            continue
        rows.append({'date': dt.date().isoformat(), 'price': float(price)})
    return pd.DataFrame(rows).drop_duplicates('date').sort_values('date')
