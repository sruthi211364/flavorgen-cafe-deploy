from __future__ import annotations

import re
from typing import Dict, Optional

import pandas as pd


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def build_name_to_id_map(drinks_df: pd.DataFrame) -> Dict[str, int]:
    if drinks_df is None or drinks_df.empty:
        return {}
    if "name" not in drinks_df.columns or "drink_id" not in drinks_df.columns:
        return {}

    out: Dict[str, int] = {}
    for _, r in drinks_df.iterrows():
        name = _norm(r.get("name", ""))
        did = r.get("drink_id", None)
        if name and did is not None:
            try:
                out[name] = int(did)
            except Exception:
                continue
    return out


def lookup_drink_id_by_name(drinks_df: pd.DataFrame, name: str) -> Optional[int]:
    m = build_name_to_id_map(drinks_df)
    key = _norm(name)
    return m.get(key)
