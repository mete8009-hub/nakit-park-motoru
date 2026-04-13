from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .config import RATING_SCALE, THEME_KEYWORDS


def safe_json(data: dict[str, Any] | None) -> str:
    return json.dumps(data or {}, ensure_ascii=False)


def detect_theme(*texts: str | None) -> str:
    text = ' '.join([t or '' for t in texts]).lower()
    for theme, words in THEME_KEYWORDS.items():
        if any(w in text for w in words):
            return theme
    return ''


def classify_etf_name(name: str) -> tuple[str, str, float, float]:
    n = (name or '').lower()
    duration = None
    esg = 55.0
    if 'treasury' in n or 'bond' in n or 'municipal' in n or 'income' in n or 'ultra short' in n:
        asset = 'ETF'
        sub = 'Bond ETF'
        duration = 3.0
    elif 'gold' in n or 'silver' in n or 'commodity' in n:
        asset = 'ETF'
        sub = 'Commodity ETF'
        duration = 0.0
    else:
        asset = 'ETF'
        sub = 'Equity ETF'
        duration = 0.0
    if any(x in n for x in ['esg', 'sustainable', 'green', 'clean', 'climate', 'low carbon']):
        esg = 80.0
    elif any(x in n for x in ['oil', 'coal']):
        esg = 35.0
    return asset, sub, duration or 0.0, esg


def normalize_tefas_category(cat: str) -> str:
    c = (cat or '').strip()
    cl = c.lower()
    mapping = [
        ('para piyasası', 'PPF'),
        ('kısa vadeli borçlanma', 'KVBAF'),
        ('değişken', 'Değişken Fon'),
        ('hisse senedi', 'Hisse Fon'),
        ('katılım', 'Katılım Fon'),
        ('borçlanma araçları', 'Borçlanma Araçları Fon'),
        ('fon sepeti', 'Fon Sepeti Fon'),
        ('serbest', 'Serbest Fon'),
        ('altın', 'Altın Fon'),
        ('eurobond', 'Eurobond Fon'),
        ('sürdürülebilir', 'Sürdürülebilirlik Temalı Fon'),
    ]
    for key, out in mapping:
        if key in cl:
            return out
    return c or 'TEFAS Fon'


def tefas_duration_years(category: str) -> float:
    c = (category or '').lower()
    if 'ppf' in c or 'para piyasası' in c:
        return 0.05
    if 'kvbaf' in c or 'kısa vadeli' in c:
        return 0.35
    if 'borçlanma' in c:
        return 1.5
    if 'eurobond' in c:
        return 4.0
    if 'altın' in c:
        return 0.0
    return 1.0


def liquidity_from_category(category: str, source_type: str = '') -> float:
    c = (category or '').lower()
    s = (source_type or '').lower()
    if 'repo' in c or 'tpp' in c:
        return 95.0
    if 'ppf' in c or 'para piyasası' in c:
        return 92.0
    if 'kvbaf' in c or 'kısa vadeli' in c:
        return 82.0
    if 'bond etf' in c or 'treasury' in c:
        return 78.0
    if 'etf' in s:
        return 72.0
    if 'eurobond' in c:
        return 58.0
    return 65.0


def rating_value(rating: str | None) -> int:
    return RATING_SCALE.get(str(rating or 'NR').upper(), 0)


def mandate_fit_row(row: pd.Series, mandate: pd.Series) -> tuple[str, str]:
    allowed_assets = {x.strip() for x in str(mandate.get('allowed_asset_classes', '')).split('|') if x.strip()}
    allowed_ccy = {x.strip() for x in str(mandate.get('allowed_currencies', '')).split('|') if x.strip()}
    notes: list[str] = []
    status = 'Pass'

    if allowed_assets and row.get('asset_class') not in allowed_assets:
        return 'Block', 'Asset class mandate dışında'
    if allowed_ccy and row.get('currency') not in allowed_ccy:
        return 'Block', 'Döviz mandate dışında'
    if mandate.get('require_investment_grade', 0) and rating_value(row.get('rating')) < rating_value('BBB'):
        return 'Block', 'Investment grade değil'
    if rating_value(row.get('rating')) < rating_value(mandate.get('min_rating')):
        return 'Block', 'Rating eşiğinin altında'
    max_dur = pd.to_numeric(mandate.get('max_duration_years'), errors='coerce')
    row_dur = pd.to_numeric(row.get('duration_years'), errors='coerce')
    if pd.notna(max_dur) and pd.notna(row_dur) and row_dur > max_dur:
        return 'Block', 'Duration limiti aşılıyor'
    min_liq = pd.to_numeric(mandate.get('min_liquidity_score'), errors='coerce')
    row_liq = pd.to_numeric(row.get('liquidity_score'), errors='coerce')
    if pd.notna(min_liq) and pd.notna(row_liq) and row_liq < min_liq:
        status = 'Warning'
        notes.append('Likidite zayıf')
    esg_floor = pd.to_numeric(mandate.get('esg_floor'), errors='coerce')
    row_esg = pd.to_numeric(row.get('esg_score'), errors='coerce')
    if pd.notna(esg_floor) and esg_floor > 0 and pd.notna(row_esg) and row_esg < esg_floor:
        status = 'Warning'
        notes.append('ESG eşiğinin altında')
    return status, ' | '.join(notes) if notes else 'Mandate ile uyumlu'
