from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .market_ref import build_try_reference_rows
from .nasdaq_etf import fetch_etf_history, fetch_nasdaq_etf_universe
from .repository import append_price_history, get_price_history, get_universe, log_refresh, update_app_meta, upsert_instruments
from .tefas import enrich_top_tefas_categories, fetch_all_funds_fast, fetch_fund_history


@dataclass
class RefreshRun:
    source: str
    status: str
    message: str


@dataclass
class RefreshSummary:
    runs: list[RefreshRun]


def refresh_tefas_universe(db_path: Path | str) -> RefreshRun:
    try:
        df = fetch_all_funds_fast()
        df = enrich_top_tefas_categories(df)
        upsert_instruments(db_path, df)
        update_app_meta(db_path, 'last_refresh_tefas', datetime.utcnow().isoformat(timespec='seconds'))
        msg = f'{len(df)} TEFAS fonu yenilendi.'
        log_refresh(db_path, 'TEFAS', 'ok', msg)
        return RefreshRun('TEFAS', 'ok', msg)
    except Exception as e:
        msg = f'TEFAS yenileme hatası: {e}'
        log_refresh(db_path, 'TEFAS', 'error', msg)
        return RefreshRun('TEFAS', 'error', msg)


def refresh_etf_universe(db_path: Path | str) -> RefreshRun:
    try:
        df = fetch_nasdaq_etf_universe()
        upsert_instruments(db_path, df)
        update_app_meta(db_path, 'last_refresh_etf_universe', datetime.utcnow().isoformat(timespec='seconds'))
        msg = f'{len(df)} Nasdaq ETF universe satırı yüklendi.'
        log_refresh(db_path, 'ETF Universe', 'ok', msg)
        return RefreshRun('ETF Universe', 'ok', msg)
    except Exception as e:
        msg = f'ETF universe hatası: {e}'
        log_refresh(db_path, 'ETF Universe', 'error', msg)
        return RefreshRun('ETF Universe', 'error', msg)


def refresh_public_try(db_path: Path | str) -> RefreshRun:
    try:
        df = build_try_reference_rows()
        upsert_instruments(db_path, df)
        update_app_meta(db_path, 'last_refresh_public_try', datetime.utcnow().isoformat(timespec='seconds'))
        msg = f'{len(df)} TRY referans satırı yenilendi.'
        log_refresh(db_path, 'TRY Public Ref', 'ok', msg)
        return RefreshRun('TRY Public Ref', 'ok', msg)
    except Exception as e:
        msg = f'TRY referans hatası: {e}'
        log_refresh(db_path, 'TRY Public Ref', 'error', msg)
        return RefreshRun('TRY Public Ref', 'error', msg)


def refresh_all(db_path: Path | str) -> RefreshSummary:
    runs = [refresh_tefas_universe(db_path), refresh_etf_universe(db_path), refresh_public_try(db_path)]
    return RefreshSummary(runs=runs)


def ensure_history_for_selection(db_path: Path | str, instrument_ids: list[str]) -> RefreshSummary:
    runs: list[RefreshRun] = []
    universe = get_universe(db_path)
    existing = get_price_history(db_path, instrument_ids)
    existing_ids = set(existing['instrument_id'].unique()) if not existing.empty else set()
    missing = [iid for iid in instrument_ids if iid not in existing_ids]
    if not missing:
        return RefreshSummary(runs=[])

    etf_tickers = [iid.split(':',1)[1] for iid in missing if iid.startswith('ETF:')]
    if etf_tickers:
        try:
            price_df, metric_df = fetch_etf_history(etf_tickers)
            append_price_history(db_path, price_df)
            if not metric_df.empty:
                merged = universe.merge(metric_df, on='instrument_id', how='left', suffixes=('', '_new'))
                for col in ['last_price','ret_1m_pct','ret_3m_pct','ret_6m_pct','ret_1y_pct','vol_1y_pct','max_drawdown_1y_pct','last_refresh_at']:
                    new_col = f'{col}_new'
                    if new_col in merged.columns:
                        merged[col] = merged[new_col].where(merged[new_col].notna(), merged[col])
                drop_cols = [c for c in merged.columns if c.endswith('_new')]
                upsert_instruments(db_path, merged.drop(columns=drop_cols))
            msg = f'{len(etf_tickers)} ETF için tarihçe getirildi.'
            runs.append(RefreshRun('ETF History', 'ok', msg))
            log_refresh(db_path, 'ETF History', 'ok', msg)
        except Exception as e:
            msg = f'ETF tarihçe hatası: {e}'
            runs.append(RefreshRun('ETF History', 'error', msg))
            log_refresh(db_path, 'ETF History', 'error', msg)

    tefas_codes = [iid.split(':',1)[1] for iid in missing if iid.startswith('TEFAS:')]
    if tefas_codes:
        ok = 0
        universe2 = get_universe(db_path)
        for code in tefas_codes:
            try:
                hist = fetch_fund_history(code)
                if hist.empty:
                    continue
                hist['instrument_id'] = f'TEFAS:{code}'
                hist['source'] = 'tefas_history'
                append_price_history(db_path, hist[['instrument_id','date','price','source']])
                s = hist['price']
                ret_1m = _ret(s, 21)
                ret_3m = _ret(s, 63)
                ret_6m = _ret(s, 126)
                ret_1y = _ret(s, 252)
                rets = s.pct_change().dropna()
                dd = s / s.cummax() - 1
                mask = universe2['instrument_id'] == f'TEFAS:{code}'
                universe2.loc[mask, 'last_price'] = float(s.iloc[-1])
                universe2.loc[mask, 'ret_1m_pct'] = ret_1m
                universe2.loc[mask, 'ret_3m_pct'] = ret_3m
                universe2.loc[mask, 'ret_6m_pct'] = ret_6m
                universe2.loc[mask, 'ret_1y_pct'] = ret_1y
                universe2.loc[mask, 'vol_1y_pct'] = float(rets.tail(252).std() * (252 ** 0.5) * 100) if len(rets) >= 20 else None
                universe2.loc[mask, 'max_drawdown_1y_pct'] = float(dd.tail(252).min() * 100) if len(dd) >= 20 else None
                ok += 1
            except Exception:
                continue
        upsert_instruments(db_path, universe2)
        msg = f'{ok} TEFAS fonu için tarihçe getirildi.'
        runs.append(RefreshRun('TEFAS History', 'ok', msg))
        log_refresh(db_path, 'TEFAS History', 'ok', msg)

    return RefreshSummary(runs=runs)


def _ret(series: pd.Series, lookback: int):
    if len(series) <= lookback:
        return None
    base = series.iloc[-lookback-1]
    return float((series.iloc[-1] / base - 1) * 100) if base else None
