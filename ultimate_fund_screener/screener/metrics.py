from __future__ import annotations

import numpy as np
import pandas as pd


def build_price_matrix(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()
    h = history.copy()
    h['date'] = pd.to_datetime(h['date'])
    return h.pivot_table(index='date', columns='instrument_id', values='price', aggfunc='last').sort_index()


def ensure_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.sort_index().pct_change().replace([np.inf, -np.inf], np.nan)


def portfolio_series(weights: dict[str, float], prices: pd.DataFrame) -> pd.Series:
    if prices.empty or not weights:
        return pd.Series(dtype=float)
    common = [k for k in weights if k in prices.columns]
    if not common:
        return pd.Series(dtype=float)
    w = pd.Series({k: weights[k] for k in common}, dtype=float)
    w = w / w.sum()
    base = prices[common].ffill().dropna(how='all')
    rets = ensure_returns(base).fillna(0.0)
    port = (rets * w).sum(axis=1)
    return (1 + port).cumprod() * 100


def portfolio_metrics(series: pd.Series) -> dict[str, float]:
    if series.empty or len(series) < 2:
        return {}
    rets = series.pct_change().dropna()
    ann_ret = (series.iloc[-1] / series.iloc[0]) ** (252 / max(len(rets), 1)) - 1
    ann_vol = rets.std() * np.sqrt(252)
    running_max = series.cummax()
    dd = series / running_max - 1
    mdd = dd.min()
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol == ann_vol else np.nan
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside and downside == downside else np.nan
    calmar = ann_ret / abs(mdd) if mdd and mdd == mdd else np.nan
    return {
        'CAGR %': ann_ret * 100,
        'Vol %': ann_vol * 100,
        'Max Drawdown %': mdd * 100,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
    }


def correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    return ensure_returns(prices).corr()


def hedge_ratios(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame()
    rets = ensure_returns(prices).dropna(how='all')
    cols = list(rets.columns)
    rows = []
    for i in cols:
        for j in cols:
            if i == j:
                continue
            x = rets[j].dropna()
            y = rets[i].reindex(x.index).dropna()
            x = x.reindex(y.index)
            if len(x) < 20 or x.var() == 0:
                continue
            beta = y.cov(x) / x.var()
            rows.append({'Long': i, 'Hedge': j, 'Beta/Hedge Ratio': beta})
    return pd.DataFrame(rows)


def weighted_portfolio_attributes(universe: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if not weights:
        return pd.DataFrame()
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    sub = universe[universe['instrument_id'].isin(w.index)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub['weight'] = sub['instrument_id'].map(w)
    out = {
        'Weighted Duration': (sub['duration_years'].fillna(0) * sub['weight']).sum(),
        'Weighted YTM': (sub['ytm_pct'].fillna(0) * sub['weight']).sum(),
        'Weighted ESG': (sub['esg_score'].fillna(0) * sub['weight']).sum(),
        'Weighted Liquidity': (sub['liquidity_score'].fillna(0) * sub['weight']).sum(),
        'Weighted 1Y Return': (sub['ret_1y_pct'].fillna(0) * sub['weight']).sum(),
    }
    return pd.DataFrame([out])
