from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MetaResponse(BaseModel):
    table: str
    columns: List[Dict[str, Any]]
    schema_map: Dict[str, str]
    extra_numeric_columns: List[str]
    db_path: str
    error: Optional[str] = None

