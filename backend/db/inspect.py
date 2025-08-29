from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import settings
from .session import get_connection


PRICES_TABLE = "prices"


# Preferred candidates for logical columns (lower-cased for matching)
CANDIDATES: dict[str, list[str]] = {
    "code": ["stock_code", "code", "ticker", "symbol"],
    "date": ["date", "trade_date", "datetime"],
    "open": ["open", "Open", "o"],
    "high": ["high", "High", "h"],
    "low": ["low", "Low", "l"],
    "close": [
        "adj_close",
        "adjclose",
        "adj_close_price",
        "adjcloseprice",
        "close",
        "Close",
        "price",
        "Price",
    ],
    "volume": ["volume", "Volume", "vol", "shares", "turnover"],
}


@dataclass
class SchemaMeta:
    columns: List[Dict[str, Any]]
    schema_map: Dict[str, str]
    extra_numeric_columns: List[str]
    table: str = PRICES_TABLE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table": self.table,
            "columns": self.columns,
            "schema_map": self.schema_map,
            "extra_numeric_columns": self.extra_numeric_columns,
            "db_path": settings.DB_PATH,
        }


def _normalize(col: str) -> str:
    return col.strip().strip("`\" ").lower()


def inspect_db(db_path: Optional[str] = None) -> SchemaMeta:
    db_path = db_path or settings.DB_PATH
    cols: list[dict[str, Any]] = []
    with get_connection(db_path) as conn:
        cur = conn.execute("PRAGMA table_info('prices')")
        pragma_info = cur.fetchall()
        for row in pragma_info:
            cols.append({
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": row[3],
                "dflt_value": row[4],
                "pk": row[5],
            })

        # Sample a single row to infer numeric/text when PRAGMA types are vague
        sample_row = None
        try:
            cur = conn.execute("SELECT * FROM prices LIMIT 1")
            sample_row = cur.fetchone()
        except Exception:
            sample_row = None

    # Build mapping
    available_cols = [c["name"] for c in cols]
    available_lc = {_normalize(c): c for c in available_cols}

    schema_map: dict[str, str] = {}
    used_actuals: set[str] = set()

    for logical, cands in CANDIDATES.items():
        found: Optional[str] = None
        for cand in cands:
            key = _normalize(cand)
            if key in available_lc:
                found = available_lc[key]
                break
        if found:
            schema_map[logical] = found
            used_actuals.add(found)

    # If both adj_close and close exist, prefer adj_close as logical 'close'
    # above matching already tries adj_close variants first.

    # Infer numeric columns
    numeric_like_patterns = ["int", "real", "num", "dec", "float"]
    numeric_cols: set[str] = set()
    for c in cols:
        ctype = (c.get("type") or "").lower()
        if any(pat in ctype for pat in numeric_like_patterns):
            numeric_cols.add(c["name"]) 

    # Fallback to sample row type inference if PRAGMA types are vague
    if sample_row is not None:
        for k in sample_row.keys():
            v = sample_row[k]
            if isinstance(v, (int, float)):
                numeric_cols.add(k)

    # Extra numeric columns: numeric and not used as logical OHLCV/date/code
    excluded = set(schema_map.values())
    extra_numeric = [c for c in sorted(numeric_cols) if c not in excluded]

    meta = SchemaMeta(columns=cols, schema_map=schema_map, extra_numeric_columns=extra_numeric)
    return meta


def save_cache(meta: SchemaMeta, cache_path: Optional[Path] = None) -> Path:
    cache_file = cache_path or settings.CACHE_PATH
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, ensure_ascii=False, indent=2)
    return cache_file


def load_cache(cache_path: Optional[Path] = None) -> Optional[dict]:
    fp = cache_path or settings.CACHE_PATH
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def ensure_cache(force: bool = False) -> dict:
    cached = load_cache()
    if cached and not force:
        return cached
    try:
        meta = inspect_db()
        save_cache(meta)
        return meta.to_dict()
    except Exception as e:
        # Save minimal placeholder to allow /api/meta to respond
        placeholder = {
            "table": PRICES_TABLE,
            "columns": [],
            "schema_map": {},
            "extra_numeric_columns": [],
            "db_path": settings.DB_PATH,
            "error": str(e),
        }
        settings.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings.CACHE_PATH.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2), encoding="utf-8")
        return placeholder


__all__ = [
    "inspect_db",
    "save_cache",
    "load_cache",
    "ensure_cache",
]

