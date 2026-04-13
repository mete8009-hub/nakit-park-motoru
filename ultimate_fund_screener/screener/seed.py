from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .db import init_db, reset_db
from .heuristics import safe_json
from .repository import append_price_history, replace_mandates, update_app_meta, upsert_instruments


@dataclass
class SeedResult:
    instruments: int
    mandates: int
    prices: int


def _load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    return pd.read_csv(path)


def _synthetic_history(rows: pd.DataFrame, years: int = 2) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * years)
    out = []
    for i, row in rows.iterrows():
        seed = abs(hash(row['instrument_id'])) % (2**32)
        rng = np.random.default_rng(seed)
        drift = (pd.to_numeric(row.get('ytm_pct'), errors='coerce') or 15) / 100 / 252
        vol = max(0.002, 0.06 / np.sqrt(252)) if row.get('asset_class') == 'Fixed Income' else 0.01
        rets = rng.normal(drift, vol, len(dates))
        prices = 100 * np.cumprod(1 + rets)
        out.extend({'instrument_id': row['instrument_id'], 'date': d.date().isoformat(), 'price': float(p), 'source': 'seed'} for d, p in zip(dates, prices))
    return pd.DataFrame(out)


def reset_and_seed(db_path: Path | str) -> SeedResult:
    reset_db(db_path)
    init_db(db_path)
    instruments = _load_csv('bootstrap_fixed_income.csv')
    mandates = _load_csv('mandates.csv')
    now = datetime.utcnow().isoformat(timespec='seconds')
    instruments['last_refresh_at'] = now
    instruments['metadata_json'] = instruments['metadata_json'].fillna('{}')
    instrument_count = upsert_instruments(db_path, instruments)
    mandate_count = replace_mandates(db_path, mandates)
    ph = _synthetic_history(instruments[instruments['is_demo'] == 1])
    price_count = append_price_history(db_path, ph)
    update_app_meta(db_path, 'last_seeded_at', now)
    return SeedResult(instrument_count, mandate_count, price_count)


def ensure_seeded(db_path: Path | str) -> None:
    if not Path(db_path).exists():
        reset_and_seed(db_path)
