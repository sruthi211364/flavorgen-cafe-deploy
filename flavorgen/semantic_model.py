from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flavorgen.data_loader import FLAVOR_DIMS

SUGAR_LEVELS = ["any", "zero", "half", "regular"]
CAFFEINE_LEVELS = ["any", "none", "low", "medium", "high"]
TEMP_LEVELS = ["any", "iced", "hot", "blended"]

KEYWORD_MAP = {
    "iced": {"temperature": "iced"},
    "cold": {"temperature": "iced"},
    "hot": {"temperature": "hot"},
    "blended": {"temperature": "blended"},
    "fruity": {"flavor": "fruity"},
    "sweet": {"flavor": "sweet"},
    "bitter": {"flavor": "bitter"},
    "creamy": {"flavor": "creamy"},
    "nutty": {"flavor": "nutty"},
    "acidic": {"flavor": "acidic"},
    "fresh": {"flavor": "fresh"},
    "spice": {"flavor": "warm_spice"},
    "chai": {"flavor": "warm_spice"},
    "cinnamon": {"flavor": "warm_spice"},
    "nutmeg": {"flavor": "warm_spice"},
    "zero sugar": {"sugar": "zero"},
    "no sugar": {"sugar": "zero"},
    "half sugar": {"sugar": "half"},
    "low sugar": {"sugar": "half"},
    "regular sugar": {"sugar": "regular"},
    "high caffeine": {"caffeine": "high"},
    "low caffeine": {"caffeine": "low"},
    "medium caffeine": {"caffeine": "medium"},
    "no caffeine": {"caffeine": "none"},
    "decaf": {"caffeine": "none"},
    "no milk": {"exclude": "milk"},
    "without milk": {"exclude": "milk"},
    "less sweet": {"sugar": "half"},
    "not too sweet": {"sugar": "half"},
    "unsweetened": {"sugar": "zero"},
    "extra strong": {"caffeine": "high"},
    "strong": {"caffeine": "high"},
    "light caffeine": {"caffeine": "low"},
    "refreshing": {"flavor": "fresh"},
    "tart": {"flavor": "acidic"},
    "citrus": {"flavor": "acidic"},
    "vanilla": {"flavor": "creamy"},
    "chocolate": {"flavor": "bitter"},
    "caramel": {"flavor": "sweet"},
}


def _norm_text(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def parse_query_to_preferences(query: str) -> Dict[str, object]:
    q = _norm_text(query)
    prefs: Dict[str, object] = {
        "temperature": "any",
        "sugar_level": "any",
        "caffeine_level": "any",
        "flavor_focus": [],
        "exclude_terms": [],
    }

    for phrase, rule in KEYWORD_MAP.items():
        if " " in phrase and phrase in q:
            if "temperature" in rule:
                prefs["temperature"] = rule["temperature"]
            if "sugar" in rule:
                prefs["sugar_level"] = rule["sugar"]
            if "caffeine" in rule:
                prefs["caffeine_level"] = rule["caffeine"]
            if "flavor" in rule:
                prefs["flavor_focus"].append(rule["flavor"])
            if "exclude" in rule:
                prefs["exclude_terms"].append(rule["exclude"])

    tokens = q.split()
    for t in tokens:
        if t in KEYWORD_MAP and " " not in t:
            rule = KEYWORD_MAP[t]
            if "temperature" in rule:
                prefs["temperature"] = rule["temperature"]
            if "sugar" in rule:
                prefs["sugar_level"] = rule["sugar"]
            if "caffeine" in rule:
                prefs["caffeine_level"] = rule["caffeine"]
            if "flavor" in rule:
                prefs["flavor_focus"].append(rule["flavor"])
            if "exclude" in rule:
                prefs["exclude_terms"].append(rule["exclude"])

    focus = []
    for f in prefs["flavor_focus"]:
        if f in FLAVOR_DIMS and f not in focus:
            focus.append(f)
    prefs["flavor_focus"] = focus

    ex = []
    for e in prefs["exclude_terms"]:
        e2 = _norm_text(e)
        if e2 and e2 not in ex:
            ex.append(e2)
    prefs["exclude_terms"] = ex

    return prefs


def _filter_by_prefs(
    df: pd.DataFrame,
    drink_type: str = "any",
    temperature: str = "any",
    sugar_level: str = "any",
    caffeine_level: str = "any",
) -> pd.DataFrame:
    out = df.copy()

    def norm_col(c: str) -> pd.Series:
        return out[c].fillna("").astype(str).str.lower().str.strip()

    if drink_type != "any" and "type" in out.columns:
        out = out[norm_col("type") == drink_type]
    if temperature != "any" and "temperature" in out.columns:
        out = out[norm_col("temperature") == temperature]
    if sugar_level != "any" and "sugar_level" in out.columns:
        out = out[norm_col("sugar_level") == sugar_level]
    if caffeine_level != "any" and "caffeine_level" in out.columns:
        out = out[norm_col("caffeine_level") == caffeine_level]

    return out


def _build_flavor_target(flavor_focus: List[str]) -> np.ndarray:
    target = np.zeros(len(FLAVOR_DIMS), dtype=float)
    if not flavor_focus:
        return target
    for i, d in enumerate(FLAVOR_DIMS):
        if d in flavor_focus:
            target[i] = 1.0
    s = float(target.sum())
    if s > 0:
        target = target / s
    return target


def _flavor_score(df: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    if target.sum() == 0:
        return np.zeros(len(df), dtype=float)

    for d in FLAVOR_DIMS:
        if d not in df.columns:
            df[d] = 0.0

    mat = df[FLAVOR_DIMS].to_numpy(dtype=float)
    num = mat @ target
    denom = (np.linalg.norm(mat, axis=1) * (np.linalg.norm(target) + 1e-9) + 1e-9)
    return np.clip(num / denom, 0.0, 1.0)


@dataclass
class HybridModel:
    vectorizer: TfidfVectorizer
    doc_matrix: object
    drink_ids: np.ndarray


def build_hybrid_model(drinks_df: pd.DataFrame) -> HybridModel:
    text = []
    for _, r in drinks_df.iterrows():
        name = str(r.get("name", ""))
        tags = str(r.get("tags", ""))
        desc = str(r.get("description", ""))
        combo = f"{name}. {tags}. {desc}"
        text.append(_norm_text(combo))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        stop_words="english",
    )
    doc_matrix = vectorizer.fit_transform(text)
    return HybridModel(vectorizer=vectorizer, doc_matrix=doc_matrix, drink_ids=drinks_df["drink_id"].to_numpy())


def hybrid_recommend(
    drinks_df: pd.DataFrame,
    model: HybridModel,
    user_query: str,
    drink_type: str = "any",
    temperature: str = "any",
    sugar_level: str = "any",
    caffeine_level: str = "any",
    flavor_focus: Optional[List[str]] = None,
    top_k: int = 5,
) -> pd.DataFrame:
    flavor_focus = flavor_focus or []

    filtered = _filter_by_prefs(
        drinks_df,
        drink_type=drink_type,
        temperature=temperature,
        sugar_level=sugar_level,
        caffeine_level=caffeine_level,
    )

    if filtered.empty:
        return filtered

    keep_mask = np.isin(model.drink_ids, filtered["drink_id"].to_numpy())
    doc_mat = model.doc_matrix[keep_mask]

    q = _norm_text(user_query or "")
    if q:
        q_vec = model.vectorizer.transform([q])
        text_sim = cosine_similarity(q_vec, doc_mat).flatten()
        text_sim = np.clip(text_sim, 0.0, 1.0)
    else:
        text_sim = np.zeros(len(filtered), dtype=float)

    target = _build_flavor_target(flavor_focus)
    flavor_sim = _flavor_score(filtered.copy(), target)

    score = (0.35 * text_sim) + (0.65 * flavor_sim)

    out = filtered.copy()
    out["score"] = score
    out["score_text"] = text_sim
    out["score_flavor"] = flavor_sim
    out = out.sort_values("score", ascending=False).head(int(top_k))
    return out
