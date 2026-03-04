from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from flavorgen.data_loader import FLAVOR_DIMS


def _root_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _safe_str(x: object) -> str:
    return str(x).strip()


def _norm_text(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


class TextEmbedder:
    """
    Embeds text into a fixed vector.
    Backend 'auto' uses transformers if available, else TF-IDF fallback.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = backend
        self._kind = None
        self._vectorizer = None
        self._hf_model = None
        self._hf_tokenizer = None
        self._hf_dim = None

        if backend not in ["auto", "transformers", "tfidf"]:
            self.backend = "auto"

        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend in ["transformers"]:  # never auto-download on HF
            try:
                from transformers import AutoTokenizer, AutoModel  # type: ignore

                name = "sentence-transformers/all-MiniLM-L6-v2"
                tok = AutoTokenizer.from_pretrained(name)
                model = AutoModel.from_pretrained(name)
                model.eval()

                self._hf_tokenizer = tok
                self._hf_model = model
                self._hf_dim = int(model.config.hidden_size)
                self._kind = "transformers"
                return
            except Exception:
                if self.backend == "transformers":
                    raise

        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, stop_words="english")
        self._kind = "tfidf"

    def kind(self) -> str:
        return str(self._kind)

    def dim(self) -> int:
        if self._kind == "transformers":
            return int(self._hf_dim or 0)
        if self._kind == "tfidf":
            return int(getattr(self._vectorizer, "max_features", 0) or 0)
        return 0

    def fit(self, texts: List[str]) -> None:
        if self._kind == "tfidf":
            self._vectorizer.fit([_norm_text(t) for t in texts])

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._kind == "transformers":
            import torch

            tok = self._hf_tokenizer
            model = self._hf_model
            assert tok is not None and model is not None

            with torch.no_grad():
                inputs = tok(texts, padding=True, truncation=True, return_tensors="pt")
                out = model(**inputs)
                # mean pooling
                last = out.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).expand(last.size()).float()
                summed = torch.sum(last * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                emb = summed / counts
                emb = emb.detach().cpu().numpy().astype(np.float32)
                return emb

        # TF-IDF
        mat = self._vectorizer.transform([_norm_text(t) for t in texts])
        return mat.toarray().astype(np.float32)

    def to_state(self) -> Dict[str, Any]:
        if self._kind == "tfidf":
            return {
                "kind": "tfidf",
                "vectorizer": self._vectorizer,
            }
        return {"kind": "transformers"}

    @staticmethod
    def from_state(state: Dict[str, Any]) -> "TextEmbedder":
        kind = state.get("kind", "transformers")
        if kind == "tfidf":
            emb = TextEmbedder(backend="tfidf")
            emb._vectorizer = state["vectorizer"]
            emb._kind = "tfidf"
            return emb
        return TextEmbedder(backend="transformers")


def build_category_maps(drinks_df: pd.DataFrame) -> Dict[str, List[str]]:
    maps: Dict[str, List[str]] = {}
    for col in ["type", "temperature", "sugar_level", "caffeine_level"]:
        if col in drinks_df.columns:
            vals = drinks_df[col].fillna("").astype(str).str.strip().str.lower().unique().tolist()
            vals = [v for v in vals if v]
            maps[col] = sorted(list(set(vals)))
        else:
            maps[col] = []
    return maps


def build_ingredient_vocab(ingredients_df: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, str]]:
    # ingredient_id -> index, index -> name
    id_to_index: Dict[int, int] = {}
    index_to_name: Dict[int, str] = {}

    if ingredients_df is None or ingredients_df.empty:
        return id_to_index, index_to_name

    if "ingredient_id" not in ingredients_df.columns:
        return id_to_index, index_to_name

    # Build vocab by iterating rows (safe even if ingredient_id has nulls)
    idx = 0
    for _, row in ingredients_df.iterrows():
        try:
            iid = int(row.get("ingredient_id"))
        except Exception:
            continue

        nm = str(row.get("name", "")).strip()
        if not nm:
            nm = f"ingredient_{iid}"

        id_to_index[iid] = idx
        index_to_name[idx] = nm
        idx += 1

    return id_to_index, index_to_name


def build_text_for_drink(r: pd.Series, ingredients_df: pd.DataFrame) -> str:
    name = _safe_str(r.get("name", ""))
    tags = _safe_str(r.get("tags", ""))
    desc = _safe_str(r.get("description", ""))

    ingred_names: List[str] = []
    ids = r.get("ingredient_ids", [])
    try:
        if isinstance(ids, str):
            ids = ids.strip()
    except Exception:
        pass

    try:
        import ast

        if isinstance(ids, str) and ids:
            parsed = ast.literal_eval(ids)
            if isinstance(parsed, list):
                ids = parsed
    except Exception:
        pass

    if isinstance(ids, list) and "ingredient_id" in ingredients_df.columns and "name" in ingredients_df.columns:
        sub = ingredients_df[ingredients_df["ingredient_id"].isin([int(x) for x in ids if str(x).isdigit()])]
        ingred_names = sub["name"].fillna("").astype(str).tolist()

    combo = f"{name}. {tags}. {desc}. Ingredients: " + ", ".join([_safe_str(x) for x in ingred_names if _safe_str(x)])
    return _norm_text(combo)


def _one_hot(value: str, choices: List[str]) -> np.ndarray:
    v = _norm_text(value)
    out = np.zeros(len(choices), dtype=np.float32)
    for i, c in enumerate(choices):
        if v == _norm_text(c):
            out[i] = 1.0
            break
    return out


def build_training_matrices(
    drinks_df: pd.DataFrame,
    ingredients_df: pd.DataFrame,
    embedder: TextEmbedder,
    cat_maps: Dict[str, List[str]],
    id_to_index: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    texts = [build_text_for_drink(r, ingredients_df) for _, r in drinks_df.iterrows()]

    # Fit TF-IDF only if needed
    if embedder.kind() == "tfidf":
        embedder.fit(texts)

    text_emb = embedder.encode(texts)

    # Structured one-hots
    type_oh = np.vstack([_one_hot(str(r.get("type", "")), cat_maps.get("type", [])) for _, r in drinks_df.iterrows()]).astype(np.float32)
    temp_oh = np.vstack([_one_hot(str(r.get("temperature", "")), cat_maps.get("temperature", [])) for _, r in drinks_df.iterrows()]).astype(np.float32)
    sugar_oh = np.vstack([_one_hot(str(r.get("sugar_level", "")), cat_maps.get("sugar_level", [])) for _, r in drinks_df.iterrows()]).astype(np.float32)
    caf_oh = np.vstack([_one_hot(str(r.get("caffeine_level", "")), cat_maps.get("caffeine_level", [])) for _, r in drinks_df.iterrows()]).astype(np.float32)

    # Flavor vector already exists on drinks_df
    flavor = drinks_df[FLAVOR_DIMS].fillna(0.0).to_numpy(dtype=np.float32) if all(d in drinks_df.columns for d in FLAVOR_DIMS) else np.zeros((len(drinks_df), len(FLAVOR_DIMS)), dtype=np.float32)

    X = np.concatenate([text_emb, type_oh, temp_oh, sugar_oh, caf_oh, flavor], axis=1).astype(np.float32)

    # Multi-label ingredient vector
    out_dim = len(id_to_index)
    Y = np.zeros((len(drinks_df), out_dim), dtype=np.float32)

    for i, r in enumerate(drinks_df.itertuples(index=False)):
        ids = getattr(r, "ingredient_ids", [])
        # safe parse
        try:
            if isinstance(ids, str) and ids:
                import ast
                parsed = ast.literal_eval(ids)
                if isinstance(parsed, list):
                    ids = parsed
        except Exception:
            pass

        if isinstance(ids, list):
            for iid in ids:
                try:
                    iid_int = int(iid)
                except Exception:
                    continue
                if iid_int in id_to_index:
                    Y[i, id_to_index[iid_int]] = 1.0

    return X, Y


class FusionNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 192),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(192, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class FusionArtifacts:
    model: FusionNet
    embedder: TextEmbedder
    meta: Dict[str, Any]


def save_artifacts(model: FusionNet, embedder: TextEmbedder, out_dir: str, meta: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(out_dir, "fusion_model.pt"))

    # Embedder state
    emb_state = embedder.to_state()
    if emb_state.get("kind") == "tfidf":
        import joblib  # type: ignore

        joblib.dump(emb_state["vectorizer"], os.path.join(out_dir, "tfidf_vectorizer.joblib"))
        emb_state = {"kind": "tfidf"}
    with open(os.path.join(out_dir, "embedder_meta.json"), "w", encoding="utf-8") as f:
        json.dump(emb_state, f, indent=2)

    with open(os.path.join(out_dir, "fusion_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_artifacts(out_dir: str, in_dim: int, out_dim: int) -> FusionArtifacts:
    model = FusionNet(in_dim=in_dim, out_dim=out_dim)
    model.load_state_dict(torch.load(os.path.join(out_dir, "fusion_model.pt"), map_location="cpu"))
    model.eval()

    embed_meta_path = os.path.join(out_dir, "embedder_meta.json")
    embed_meta = json.load(open(embed_meta_path, "r", encoding="utf-8"))

    if embed_meta.get("kind") == "tfidf":
        import joblib  # type: ignore

        vec = joblib.load(os.path.join(out_dir, "tfidf_vectorizer.joblib"))
        emb = TextEmbedder(backend="tfidf")
        emb._vectorizer = vec
        emb._kind = "tfidf"
        embedder = emb
    else:
        embedder = TextEmbedder(backend="transformers")

    meta = json.load(open(os.path.join(out_dir, "fusion_meta.json"), "r", encoding="utf-8"))

    # FIX: JSON turns dict keys into strings, convert ingredient_index_to_name back to int keys
    m = meta.get("ingredient_index_to_name", {})
    if isinstance(m, dict):
        fixed_map: Dict[int, str] = {}
        for k, v in m.items():
            try:
                fixed_map[int(k)] = str(v)
            except Exception:
                continue
        meta["ingredient_index_to_name"] = fixed_map

    return FusionArtifacts(model=model, embedder=embedder, meta=meta)

def _row_to_feature(
    r: pd.Series,
    ingredients_df: pd.DataFrame,
    embedder: TextEmbedder,
    cat_maps: Dict[str, List[str]],
) -> np.ndarray:
    txt = build_text_for_drink(r, ingredients_df)
    tvec = embedder.encode([txt]).astype(np.float32)

    type_oh = _one_hot(str(r.get("type", "")), cat_maps.get("type", []))[None, :]
    temp_oh = _one_hot(str(r.get("temperature", "")), cat_maps.get("temperature", []))[None, :]
    sugar_oh = _one_hot(str(r.get("sugar_level", "")), cat_maps.get("sugar_level", []))[None, :]
    caf_oh = _one_hot(str(r.get("caffeine_level", "")), cat_maps.get("caffeine_level", []))[None, :]

    if all(d in r.index for d in FLAVOR_DIMS):
        flavor = np.array([float(r.get(d, 0.0)) for d in FLAVOR_DIMS], dtype=np.float32)[None, :]
    else:
        flavor = np.zeros((1, len(FLAVOR_DIMS)), dtype=np.float32)

    return np.concatenate([tvec, type_oh, temp_oh, sugar_oh, caf_oh, flavor], axis=1).astype(np.float32)


def predict_fusion(
    art: FusionArtifacts,
    drinks_df: pd.DataFrame,
    ingredients_df: pd.DataFrame,
    drink_id_a: int,
    drink_id_b: int,
    alpha: float = 0.5,
    top_k: int = 10,
) -> Dict[str, Any]:
    if "drink_id" not in drinks_df.columns:
        raise ValueError("drinks_df missing drink_id")

    row_a = drinks_df[drinks_df["drink_id"] == int(drink_id_a)]
    row_b = drinks_df[drinks_df["drink_id"] == int(drink_id_b)]
    if row_a.empty or row_b.empty:
        raise ValueError("Invalid drink ids selected")

    ra = row_a.iloc[0]
    rb = row_b.iloc[0]

    cat_maps = art.meta.get("cat_maps", {})
    index_to_name = art.meta.get("ingredient_index_to_name", {})

    xa = _row_to_feature(ra, ingredients_df, art.embedder, cat_maps)
    xb = _row_to_feature(rb, ingredients_df, art.embedder, cat_maps)

    a = float(alpha)
    xmix = ((1.0 - a) * xa) + (a * xb)

    with torch.no_grad():
        logits = art.model(torch.tensor(xmix, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    top_idx = np.argsort(-probs)[: int(top_k)].tolist()
    ingred_names = [str(index_to_name.get(int(i), f"ingredient_{i}")) for i in top_idx]
    ingred_conf = [float(probs[i]) for i in top_idx]

    # fused flavor vector, simple weighted blend
    fa = np.array([float(ra.get(d, 0.0)) for d in FLAVOR_DIMS], dtype=float)
    fb = np.array([float(rb.get(d, 0.0)) for d in FLAVOR_DIMS], dtype=float)
    fv = ((1.0 - a) * fa) + (a * fb)
    fv = np.clip(fv, 0.0, 1.0)

    name = f"Fusion of {_safe_str(ra.get('name','A'))} + {_safe_str(rb.get('name','B'))}"
    desc = f"A fusion drink blending elements of {_safe_str(ra.get('name','A'))} and {_safe_str(rb.get('name','B'))}, generated using NLP + deep learning."

    return {
        "name": name,
        "description": desc,
        "ingredient_names": ingred_names,
        "ingredient_confidence": ingred_conf,
        "flavor_vector": fv.tolist(),
        "drink_id_a": int(drink_id_a),
        "drink_id_b": int(drink_id_b),
        "alpha": float(alpha),
    }
