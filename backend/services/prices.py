from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..core.config import settings
from ..db.inspect import ensure_cache
from ..db.session import get_connection


def _qident(name: str) -> str:
    # Quote SQLite identifier safely
    return '"' + str(name).replace('"', '""') + '"'


def get_schema() -> Dict[str, Any]:
    return ensure_cache()


def suggest_stocks(q: str, limit: int = 20) -> List[Dict[str, str]]:
    meta = get_schema()
    schema_map = meta.get("schema_map", {})
    code_col = schema_map.get("code")
    if not code_col:
        return []
    # try to find a display name column
    name_col: Optional[str] = None
    lower_to_actual = {c["name"].lower(): c["name"] for c in meta.get("columns", [])}
    for cand in ["name", "company", "company_name", "security_name", "issue_name", "short_name"]:
        if cand in lower_to_actual:
            name_col = lower_to_actual[cand]
            break
    q_like = f"%{q}%"
    if name_col:
        sql = (
            f"SELECT DISTINCT {_qident(code_col)} as code, {_qident(name_col)} as name "
            f"FROM prices WHERE {_qident(code_col)} LIKE ? OR {_qident(name_col)} LIKE ? "
            f"ORDER BY 2,1 LIMIT ?"
        )
        params = (q_like, q_like, limit)
    else:
        sql = (
            f"SELECT DISTINCT {_qident(code_col)} as code, {_qident(code_col)} as name "
            f"FROM prices WHERE {_qident(code_col)} LIKE ? ORDER BY 1 LIMIT ?"
        )
        params = (q_like, limit)
    with get_connection(settings.DB_PATH) as conn:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
    return [
        {"code": str(r["code"]), "name": (str(r["name"]) if r["name"] is not None else str(r["code"]))}
        for r in rows
    ]


def _select_columns(schema_map: Dict[str, str]) -> Tuple[List[str], List[str]]:
    cols = []
    optional = []
    for key in ["date", "open", "high", "low", "close", "volume"]:
        col = schema_map.get(key)
        if not col:
            if key in ("open", "high", "low", "volume"):
                optional.append(key)
            continue
        cols.append(col)
    return cols, optional


def get_prices(
    stock: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    meta = get_schema()
    schema_map = meta.get("schema_map", {})
    code_col = schema_map.get("code")
    date_col = schema_map.get("date")
    if not code_col or not date_col or not schema_map.get("close"):
        return {"dates": [], "close": []}

    select_cols, _ = _select_columns(schema_map)
    select_part = ", ".join(_qident(c) for c in select_cols)
    sql = f"SELECT {select_part} FROM prices WHERE {_qident(code_col)} = ?"
    params: list[Any] = [stock]
    if start:
        sql += f" AND {date_col} >= ?"
        params.append(start)
    if end:
        sql += f" AND {date_col} <= ?"
        params.append(end)
    sql += f" ORDER BY {_qident(date_col)} ASC"

    with get_connection(settings.DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    # Normalize column names to logical keys for output
    out: Dict[str, Any] = {"dates": df[date_col].astype(str).tolist()}
    for logical, actual in schema_map.items():
        if logical in ("open", "high", "low", "close", "volume") and actual in df.columns:
            out[logical if logical != "close" else "close"] = df[actual].tolist()
    return out


def get_all_dates_for_stock(stock: str) -> List[str]:
    meta = get_schema()
    schema_map = meta.get("schema_map", {})
    code_col = schema_map.get("code")
    date_col = schema_map.get("date")
    if not code_col or not date_col:
        return []
    sql = f"SELECT {_qident(date_col)} as d FROM prices WHERE {_qident(code_col)} = ? ORDER BY {_qident(date_col)}"
    with get_connection(settings.DB_PATH) as conn:
        df = pd.read_sql_query(sql, conn, params=(stock,))
    return df["d"].astype(str).tolist()
