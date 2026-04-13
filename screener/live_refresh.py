from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DB_PATH
from .db import get_connection, init_db
from .live_sources import fetch_bist_repo_reference, fetch_tefas_watchlist, fetch_tpp_reference, fetch_yfinance_watchlist


@dataclass
class RefreshRun:
    source: str
    status: str
    message: str
    rows_loaded: int


@dataclass
class RefreshSummary:
    runs: list[RefreshRun]

    @property
    def ok(self) -> bool:
        return all(r.status == "ok" for r in self.runs) if self.runs else False

    @property
    def total_rows(self) -> int:
        return sum(int(r.rows_loaded or 0) for r in self.runs)


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[cols]


def _append_log(con, run: RefreshRun) -> None:
    con.execute(
        "INSERT INTO refresh_log(run_at, source, status, message, rows_loaded) VALUES (?, ?, ?, ?, ?)",
        (str(pd.Timestamp.now()), run.source, run.status, run.message, int(run.rows_loaded or 0)),
    )


def _upsert_instruments(con, df: pd.DataFrame) -> int:
    cols = [
        "instrument_id", "symbol", "name", "asset_class", "sub_asset_class", "category", "currency", "region", "country",
        "issuer", "issuer_type", "rating", "duration_years", "maturity_date", "coupon", "expense_ratio", "aum_mn",
        "esg_score", "liquidity_score", "yield_pct", "ytm_pct", "risk_score", "theme", "tradable", "source", "is_demo",
    ]
    df = _ensure_columns(df, cols)
    if df.empty:
        return 0
    temp = "tmp_instruments_upsert"
    con.execute(f"DROP TABLE IF EXISTS {temp}")
    df.to_sql(temp, con, index=False)
    con.execute(
        f"""
        INSERT OR REPLACE INTO instruments({', '.join(cols)})
        SELECT {', '.join(cols)} FROM {temp}
        """
    )
    con.execute(f"DROP TABLE IF EXISTS {temp}")
    return len(df)


def _upsert_snapshots(con, df: pd.DataFrame) -> int:
    cols = [
        "instrument_id", "asof_date", "price", "nav", "yield_pct", "ytm_pct", "spread_bps", "expense_ratio",
        "esg_score", "liquidity_score", "aum_mn", "quote_source", "updated_at",
    ]
    df = _ensure_columns(df, cols)
    if df.empty:
        return 0
    temp = "tmp_snapshots_upsert"
    con.execute(f"DROP TABLE IF EXISTS {temp}")
    df.to_sql(temp, con, index=False)
    con.execute(
        f"""
        INSERT OR REPLACE INTO market_snapshot({', '.join(cols)})
        SELECT {', '.join(cols)} FROM {temp}
        """
    )
    con.execute(f"DROP TABLE IF EXISTS {temp}")
    return len(df)


def _replace_history(con, df: pd.DataFrame) -> int:
    cols = ["instrument_id", "date", "value"]
    df = _ensure_columns(df, cols)
    if df.empty:
        return 0
    ids = tuple(sorted(df["instrument_id"].dropna().unique().tolist()))
    if ids:
        placeholders = ",".join(["?"] * len(ids))
        con.execute(f"DELETE FROM price_history WHERE instrument_id IN ({placeholders})", ids)
    df.to_sql("price_history", con, if_exists="append", index=False)
    return len(df)


def refresh_yfinance(db_path: Path | str = DB_PATH, years: int = 3) -> RefreshRun:
    try:
        init_db(db_path)
        instruments, snapshots, history = fetch_yfinance_watchlist(years=years)
        with get_connection(db_path) as con:
            rows = _upsert_instruments(con, instruments) + _upsert_snapshots(con, snapshots) + _replace_history(con, history)
            con.execute("INSERT OR REPLACE INTO app_meta(key, value) VALUES (?, ?)", ("last_refresh_yfinance", str(pd.Timestamp.now())))
            run = RefreshRun("yfinance", "ok", f"ETF ve ETF tarihçesi yenilendi: {len(instruments)} enstrüman.", rows)
            _append_log(con, run)
        return run
    except Exception as exc:
        with get_connection(db_path) as con:
            run = RefreshRun("yfinance", "error", str(exc), 0)
            _append_log(con, run)
        return run


def refresh_tefas(db_path: Path | str = DB_PATH) -> RefreshRun:
    try:
        init_db(db_path)
        instruments, snapshots = fetch_tefas_watchlist()
        with get_connection(db_path) as con:
            rows = _upsert_instruments(con, instruments) + _upsert_snapshots(con, snapshots)
            con.execute("INSERT OR REPLACE INTO app_meta(key, value) VALUES (?, ?)", ("last_refresh_tefas", str(pd.Timestamp.now())))
            run = RefreshRun("tefas", "ok", f"TEFAS watchlist yenilendi: {len(instruments)} fon.", rows)
            _append_log(con, run)
        return run
    except Exception as exc:
        with get_connection(db_path) as con:
            run = RefreshRun("tefas", "error", str(exc), 0)
            _append_log(con, run)
        return run


def refresh_public_try(db_path: Path | str = DB_PATH) -> RefreshRun:
    try:
        init_db(db_path)
        repo_inst, repo_snap = fetch_bist_repo_reference()
        tpp_inst, tpp_snap = fetch_tpp_reference()
        instruments = pd.concat([repo_inst, tpp_inst], ignore_index=True)
        snapshots = pd.concat([repo_snap, tpp_snap], ignore_index=True)
        with get_connection(db_path) as con:
            rows = _upsert_instruments(con, instruments) + _upsert_snapshots(con, snapshots)
            con.execute("INSERT OR REPLACE INTO app_meta(key, value) VALUES (?, ?)", ("last_refresh_public_try", str(pd.Timestamp.now())))
            run = RefreshRun("public_try", "ok", "BIST repo ve Takasbank TPP referansları yenilendi.", rows)
            _append_log(con, run)
        return run
    except Exception as exc:
        with get_connection(db_path) as con:
            run = RefreshRun("public_try", "error", str(exc), 0)
            _append_log(con, run)
        return run


def refresh_all(db_path: Path | str = DB_PATH, years: int = 3) -> RefreshSummary:
    runs = [
        refresh_public_try(db_path=db_path),
        refresh_tefas(db_path=db_path),
        refresh_yfinance(db_path=db_path, years=years),
    ]
    return RefreshSummary(runs=runs)
