from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .db import get_conn, init_db


def _now() -> str:
    return datetime.utcnow().isoformat(timespec='seconds')


def upsert_instruments(db_path: Path | str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        existing = pd.read_sql_query('SELECT * FROM instruments', conn)
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.sort_values('last_refresh_at').drop_duplicates('instrument_id', keep='last')
        merged.to_sql('instruments', conn, if_exists='replace', index=False)
        conn.commit()
        return len(df)
    finally:
        conn.close()


def replace_mandates(db_path: Path | str, df: pd.DataFrame) -> int:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        df.to_sql('mandates', conn, if_exists='replace', index=False)
        conn.commit()
        return len(df)
    finally:
        conn.close()


def append_price_history(db_path: Path | str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        cur = conn.cursor()
        rows = list(df[['instrument_id', 'date', 'price', 'source']].itertuples(index=False, name=None))
        cur.executemany(
            'INSERT OR REPLACE INTO price_history (instrument_id, date, price, source) VALUES (?, ?, ?, ?)',
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def update_app_meta(db_path: Path | str, key: str, value: str) -> None:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        conn.execute('INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)', (key, value))
        conn.commit()
    finally:
        conn.close()


def log_refresh(db_path: Path | str, source: str, status: str, message: str) -> None:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        conn.execute(
            'INSERT INTO refresh_log (source, status, message, created_at) VALUES (?, ?, ?, ?)',
            (source, status, message, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_universe(db_path: Path | str) -> pd.DataFrame:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        return pd.read_sql_query('SELECT * FROM instruments ORDER BY name', conn)
    finally:
        conn.close()


def get_mandates(db_path: Path | str) -> pd.DataFrame:
    init_db(db_path)
    conn = get_conn(db_path)
    try:
        return pd.read_sql_query('SELECT * FROM mandates ORDER BY name', conn)
    finally:
        conn.close()


def get_price_history(db_path: Path | str, instrument_ids: Iterable[str]) -> pd.DataFrame:
    ids = list(instrument_ids)
    if not ids:
        return pd.DataFrame(columns=['instrument_id', 'date', 'price', 'source'])
    conn = get_conn(db_path)
    try:
        placeholders = ','.join('?' for _ in ids)
        q = f'SELECT * FROM price_history WHERE instrument_id IN ({placeholders}) ORDER BY date'
        return pd.read_sql_query(q, conn, params=ids)
    finally:
        conn.close()


def get_refresh_log(db_path: Path | str, limit: int = 100) -> pd.DataFrame:
    conn = get_conn(db_path)
    try:
        return pd.read_sql_query(f'SELECT * FROM refresh_log ORDER BY id DESC LIMIT {int(limit)}', conn)
    finally:
        conn.close()


def get_app_meta(db_path: Path | str) -> pd.DataFrame:
    conn = get_conn(db_path)
    try:
        return pd.read_sql_query('SELECT * FROM app_meta', conn)
    finally:
        conn.close()
