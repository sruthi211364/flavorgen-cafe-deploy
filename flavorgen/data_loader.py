from __future__ import annotations

import ast
import os
from typing import List

import numpy as np
import pandas as pd

FLAVOR_DIMS: List[str] = [
    "sweet",
    "bitter",
    "creamy",
    "fresh",
    "fruity",
    "nutty",
    "acidic",
    "warm_spice",
]


def _data_path(*parts: str) -> str:
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base, "data", *parts)


def _safe_literal_list(x) -> List[int]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    s = str(x).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            out = []
            for item in v:
                try:
                    out.append(int(item))
                except Exception:
                    pass
            return out
    except Exception:
        pass

    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    out2: List[int] = []
    for p in parts:
        if not p:
            continue
        try:
            out2.append(int(p))
        except Exception:
            continue
    return out2


def _coerce_float01(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    return s2.clip(0.0, 1.0)


def load_ingredients() -> pd.DataFrame:
    path = _data_path("ingredients.csv")
    df = pd.read_csv(path)

    if "ingredient_id" not in df.columns:
        for alt in ["id", "ingredientId"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "ingredient_id"})
                break

    if "ingredient_id" in df.columns:
        df["ingredient_id"] = pd.to_numeric(df["ingredient_id"], errors="coerce").astype("Int64")

    for d in FLAVOR_DIMS:
        if d not in df.columns:
            df[d] = 0.0
        df[d] = _coerce_float01(df[d])

    return df


def _compute_flavor_from_ingredients(ingredients_df: pd.DataFrame, ingredient_ids: List[int]) -> np.ndarray:
    if ingredients_df is None or ingredients_df.empty or not ingredient_ids:
        return np.zeros(len(FLAVOR_DIMS), dtype=float)

    sub = ingredients_df[ingredients_df["ingredient_id"].isin([int(x) for x in ingredient_ids if x is not None])]
    if sub.empty:
        return np.zeros(len(FLAVOR_DIMS), dtype=float)

    mat = sub[FLAVOR_DIMS].to_numpy(dtype=float)
    vec = np.mean(mat, axis=0)
    return np.clip(vec, 0.0, 1.0).astype(float)


def _ensure_flavor_columns(drinks_df: pd.DataFrame) -> pd.DataFrame:
    if all(d in drinks_df.columns for d in FLAVOR_DIMS):
        for d in FLAVOR_DIMS:
            drinks_df[d] = _coerce_float01(drinks_df[d])
        return drinks_df

    ingredients_df = load_ingredients()

    if "ingredient_ids" not in drinks_df.columns:
        drinks_df["ingredient_ids"] = [[] for _ in range(len(drinks_df))]

    parsed_ids = []
    for x in drinks_df["ingredient_ids"].tolist():
        parsed_ids.append(_safe_literal_list(x))
    drinks_df["ingredient_ids"] = parsed_ids

    vectors = []
    for ids in drinks_df["ingredient_ids"].tolist():
        vectors.append(_compute_flavor_from_ingredients(ingredients_df, ids))

    mat = np.vstack(vectors) if vectors else np.zeros((len(drinks_df), len(FLAVOR_DIMS)), dtype=float)
    for i, d in enumerate(FLAVOR_DIMS):
        drinks_df[d] = mat[:, i].astype(float)

    return drinks_df


def load_drinks(filename: str = "drinks_hybrid_augmented.csv") -> pd.DataFrame:
    path = _data_path(filename)
    df = pd.read_csv(path)

    if "drink_id" in df.columns:
        df["drink_id"] = pd.to_numeric(df["drink_id"], errors="coerce").astype("Int64")

    for col in ["type", "temperature", "sugar_level", "caffeine_level"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip().str.lower()

    if "tags" in df.columns:
        df["tags"] = df["tags"].fillna("").astype(str)

    # IMPORTANT: your file uses ingredient_ids (not ingredient_ids)
    if "ingredient_ids" in df.columns:
        df["ingredient_ids"] = df["ingredient_ids"].fillna("").astype(str)

    if "description" in df.columns:
        df["description"] = df["description"].fillna("").astype(str)

    df = _ensure_flavor_columns(df)
    return df

def compute_drink_flavor_vector(drinks_df: pd.DataFrame, drink_id: int) -> np.ndarray:
    if drinks_df is None or drinks_df.empty:
        return np.zeros(len(FLAVOR_DIMS), dtype=float)

    drinks_df = _ensure_flavor_columns(drinks_df)

    if "drink_id" not in drinks_df.columns:
        return np.zeros(len(FLAVOR_DIMS), dtype=float)

    row = drinks_df.loc[drinks_df["drink_id"] == int(drink_id)]
    if row.empty:
        return np.zeros(len(FLAVOR_DIMS), dtype=float)

    r = row.iloc[0]
    vec = np.array([float(r.get(d, 0.0)) for d in FLAVOR_DIMS], dtype=float)
    return np.clip(vec, 0.0, 1.0)
