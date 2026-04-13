from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS instruments (
    instrument_id TEXT PRIMARY KEY,
    external_code TEXT,
    source_type TEXT,
    name TEXT,
    asset_class TEXT,
    sub_asset_class TEXT,
    category TEXT,
    theme TEXT,
    issuer TEXT,
    currency TEXT,
    country TEXT,
    region TEXT,
    rating TEXT,
    maturity_date TEXT,
    duration_years REAL,
    yield_pct REAL,
    ytm_pct REAL,
    ret_1m_pct REAL,
    ret_3m_pct REAL,
    ret_6m_pct REAL,
    ret_1y_pct REAL,
    vol_1y_pct REAL,
    max_drawdown_1y_pct REAL,
    liquidity_score REAL,
    esg_score REAL,
    aum_mn REAL,
    expense_ratio REAL,
    risk_score REAL,
    last_price REAL,
    is_demo INTEGER DEFAULT 0,
    last_refresh_at TEXT,
    metadata_json TEXT
);
CREATE TABLE IF NOT EXISTS mandates (
    mandate_id TEXT PRIMARY KEY,
    name TEXT,
    base_currency TEXT,
    allowed_asset_classes TEXT,
    allowed_currencies TEXT,
    min_rating TEXT,
    max_duration_years REAL,
    min_liquidity_score REAL,
    require_investment_grade INTEGER DEFAULT 0,
    esg_floor REAL DEFAULT 0,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS price_history (
    instrument_id TEXT,
    date TEXT,
    price REAL,
    source TEXT,
    PRIMARY KEY (instrument_id, date)
);
CREATE TABLE IF NOT EXISTS refresh_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT,
    status TEXT,
    message TEXT,
    created_at TEXT
);
CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""

def get_conn(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute('PRAGMA foreign_keys=OFF;')
    return conn


def init_db(db_path: Path | str) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def reset_db(db_path: Path | str) -> None:
    p = Path(db_path)
    if p.exists():
        p.unlink()
    init_db(db_path)
