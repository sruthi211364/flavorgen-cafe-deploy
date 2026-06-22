from __future__ import annotations

import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SEED = 42
AUG_PER_DRINK = 5  # increase for more diversity (5 means about 6x total)
ADD_DESCRIPTION = True

random.seed(SEED)
np.random.seed(SEED)

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")

IN_DRINKS = os.path.join(DATA_DIR, "drinks_hybrid.csv")
IN_ING = os.path.join(DATA_DIR, "ingredients.csv")
OUT_DRINKS = os.path.join(DATA_DIR, "drinks_hybrid_augmented.csv")

FLAVOR_DIMS = ["sweet", "bitter", "creamy", "fresh", "fruity", "nutty", "acidic", "warm_spice"]


def norm(s: str) -> str:
    return " ".join(str(s).lower().strip().split())


def titleize(s: str) -> str:
    return " ".join(w.capitalize() for w in str(s).split())


def parse_ids(x) -> List[int]:
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
    s = s.replace("[", "").replace("]", "").replace(";", ",").replace('"', "").replace("'", "")
    parts = [p.strip() for p in s.split(",")]
    out2: List[int] = []
    for p in parts:
        if not p:
            continue
        try:
            out2.append(int(p))
        except Exception:
            pass
    return out2


def ids_to_str(ids: List[int]) -> str:
    # keep the same style as your CSV
    return "[" + ",".join(str(int(i)) for i in ids) + "]"


def tag_append(tags: str, extra: List[str]) -> str:
    t = str(tags) if tags is not None else ""
    for e in extra:
        e2 = norm(e).replace(" ", "_")
        if not e2:
            continue
        if e2 not in t:
            if t.strip():
                t = t + "; " + e2
            else:
                t = e2
    return t


def build_maps(ing_df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[str, int], Dict[int, str]]:
    # returns: id2name, name2id, id2category
    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    id2cat: Dict[int, str] = {}

    if "ingredient_id" not in ing_df.columns or "name" not in ing_df.columns:
        return id2name, name2id, id2cat

    for _, r in ing_df.iterrows():
        iid = r.get("ingredient_id")
        nm = r.get("name", "")
        cat = r.get("category", "")
        if pd.isna(iid):
            continue
        try:
            iid_int = int(iid)
        except Exception:
            continue
        nm2 = norm(nm)
        if not nm2:
            continue
        id2name[iid_int] = nm2
        name2id[nm2] = iid_int
        id2cat[iid_int] = norm(cat)
    return id2name, name2id, id2cat


def detect_groups_from_categories(ing_df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    If your ingredients.csv has a useful 'category', we can auto-build swap groups.
    Fallback is small curated name-based groups.
    """
    groups: Dict[str, List[int]] = {}

    if "category" in ing_df.columns and "ingredient_id" in ing_df.columns:
        cat_counts = ing_df["category"].fillna("").astype(str).str.strip().str.lower().value_counts()
        # pick categories that actually have enough members to swap
        good_cats = [c for c, n in cat_counts.items() if c and n >= 3]
        for c in good_cats:
            sub = ing_df[ing_df["category"].fillna("").astype(str).str.strip().str.lower() == c]
            ids = pd.to_numeric(sub["ingredient_id"], errors="coerce").dropna().astype(int).tolist()
            if len(ids) >= 3:
                groups[c] = ids

    return groups


def curated_groups(name2id: Dict[str, int]) -> Dict[str, List[int]]:
    # name-based safety net (only keeps items that exist in your ingredient list)
    base = {
        "milk": ["milk", "whole milk", "oat milk", "almond milk", "soy milk", "coconut milk", "cream"],
        "sweetener": ["sugar", "brown sugar", "honey", "stevia", "agave", "maple syrup"],
        "coffee_base": ["espresso", "coffee", "cold brew", "decaf coffee", "decaf espresso"],
        "syrup": ["vanilla syrup", "caramel syrup", "hazelnut syrup", "mocha syrup", "chocolate syrup"],
        "spice": ["cinnamon", "nutmeg", "chai spice", "cardamom"],
        "fruit": ["lemon", "lime", "orange", "strawberry", "mango", "peach", "berry"],
        "tea": ["black tea", "green tea", "matcha", "chai"],
    }
    out: Dict[str, List[int]] = {}
    for g, names in base.items():
        ids: List[int] = []
        for nm in names:
            nm2 = norm(nm)
            if nm2 in name2id:
                ids.append(int(name2id[nm2]))
        if len(ids) >= 2:
            out[g] = ids
    return out


def swap_one(ids: List[int], groups: Dict[str, List[int]]) -> List[int]:
    if not ids or not groups:
        return ids
    ids_set = set(ids)
    candidates = [(g, g_ids) for g, g_ids in groups.items() if any(i in ids_set for i in g_ids)]
    if not candidates:
        return ids
    g, g_ids = random.choice(candidates)
    present = [i for i in g_ids if i in ids_set]
    if not present:
        return ids
    to_replace = random.choice(present)
    alt = [i for i in g_ids if i != to_replace]
    if not alt:
        return ids
    new_id = random.choice(alt)
    new_ids = [i for i in ids if i != to_replace] + [new_id]
    return sorted(list(set(new_ids)))


def add_one(ids: List[int], groups: Dict[str, List[int]]) -> List[int]:
    if not groups:
        return ids
    g, g_ids = random.choice(list(groups.items()))
    new_id = random.choice(g_ids)
    return sorted(list(set(ids + [new_id])))


def temp_variant(temp: str) -> str:
    t = norm(temp)
    if t == "hot":
        return "iced"
    if t == "iced":
        return "hot"
    if t == "blended":
        return random.choice(["iced", "blended"])
    return temp


def sugar_variant(sugar: str) -> str:
    s = norm(sugar)
    order = ["zero", "half", "regular"]
    if s not in order:
        return sugar
    idx = order.index(s)
    if idx > 0 and random.random() < 0.7:
        return order[idx - 1]
    return random.choice(order)


def caffeine_variant(caf: str) -> str:
    c = norm(caf)
    order = ["none", "low", "medium", "high"]
    if c not in order:
        return caf
    idx = order.index(c)
    step = random.choice([-1, 1])
    j = max(0, min(len(order) - 1, idx + step))
    return order[j]


def bump_popularity(p: float) -> float:
    try:
        x = float(p)
    except Exception:
        x = 0.5
    noise = np.random.normal(0.0, 0.03)
    return float(np.clip(x + noise, 0.0, 1.0))


def make_name(base: str, t: str, s: str, c: str) -> str:
    base_clean = re.sub(r"\s+", " ", str(base)).strip()
    prefix = titleize(t) + " " if t in ["iced", "hot", "blended"] else ""
    suffix_bits = []
    if s in ["zero", "half"]:
        suffix_bits.append("Zero Sugar" if s == "zero" else "Half Sugar")
    if c in ["none", "high"]:
        suffix_bits.append("Decaf" if c == "none" else "Extra Caffeine")
    suffix = f" ({', '.join(suffix_bits)})" if suffix_bits else ""
    return prefix + base_clean + suffix


def build_description(drink_name: str, drink_type: str, temp: str, sugar: str, caf: str, top_ing: List[str]) -> str:
    bits = []
    if temp in ["iced", "hot", "blended"]:
        bits.append(temp)
    if drink_type:
        bits.append(drink_type)
    style = " ".join(bits).strip()

    sugar_txt = "zero sugar" if sugar == "zero" else ("half sugar" if sugar == "half" else "regular sweetness")
    caf_txt = "decaf" if caf == "none" else (f"{caf} caffeine" if caf in ["low", "medium", "high"] else "caffeine")

    ing_txt = ""
    if top_ing:
        ing_txt = " Featuring " + ", ".join(top_ing[:4]) + "."

    return f"{drink_name} is a {style} drink with {sugar_txt} and {caf_txt}.{ing_txt}".strip()


def main() -> None:
    drinks = pd.read_csv(IN_DRINKS)
    ing = pd.read_csv(IN_ING)

    # Normalize column names in drinks
    if "drink_id" not in drinks.columns:
        raise ValueError("drinks_hybrid.csv must contain 'drink_id'")
    if "ingredient_ids" not in drinks.columns:
        raise ValueError("drinks_hybrid.csv must contain 'ingredient_ids'")

    for col in ["type", "temperature", "sugar_level", "caffeine_level"]:
        if col in drinks.columns:
            drinks[col] = drinks[col].fillna("").astype(str).str.strip().str.lower()

    if "tags" in drinks.columns:
        drinks["tags"] = drinks["tags"].fillna("").astype(str)

    if "popularity_score" in drinks.columns:
        drinks["popularity_score"] = pd.to_numeric(drinks["popularity_score"], errors="coerce").fillna(0.5)

    # Build ingredient maps
    id2name, name2id, _id2cat = build_maps(ing)

    # Groups: category-based first, plus curated fallbacks
    groups = detect_groups_from_categories(ing)
    curated = curated_groups(name2id)
    for k, v in curated.items():
        if k not in groups and len(v) >= 2:
            groups[k] = v

    # New IDs
    max_id = int(pd.to_numeric(drinks["drink_id"], errors="coerce").dropna().max())
    next_id = max_id + 1

    aug_rows = []
    for _, r in drinks.iterrows():
        base_ids = parse_ids(r.get("ingredient_ids", ""))
        base_name = str(r.get("name", "Drink")).strip()
        base_temp = norm(r.get("temperature", "any"))
        base_sugar = norm(r.get("sugar_level", "any"))
        base_caf = norm(r.get("caffeine_level", "any"))
        base_tags = str(r.get("tags", "")).strip()
        base_pop = float(r.get("popularity_score", 0.5))
        base_type = norm(r.get("type", ""))

        for _k in range(AUG_PER_DRINK):
            new = r.copy()

            mode = random.choice(["temp", "sugar", "caffeine", "swap", "swap_add", "combo"])
            new_temp = base_temp
            new_sugar = base_sugar
            new_caf = base_caf
            new_ids = list(base_ids)
            extra_tags = ["augmented"]

            if mode == "temp":
                new_temp = temp_variant(base_temp)
                extra_tags.append(f"temp_{new_temp}")
            elif mode == "sugar":
                new_sugar = sugar_variant(base_sugar)
                extra_tags.append(f"sugar_{new_sugar}")
            elif mode == "caffeine":
                new_caf = caffeine_variant(base_caf)
                extra_tags.append(f"caffeine_{new_caf}")
            elif mode == "swap":
                new_ids = swap_one(new_ids, groups)
                extra_tags.append("ingredient_swap")
            elif mode == "swap_add":
                new_ids = swap_one(new_ids, groups)
                if random.random() < 0.7:
                    new_ids = add_one(new_ids, groups)
                extra_tags.append("swap_add")
            elif mode == "combo":
                # do 2 changes to create more diversity
                new_sugar = sugar_variant(base_sugar)
                new_caf = caffeine_variant(base_caf)
                new_ids = swap_one(new_ids, groups)
                if random.random() < 0.5:
                    new_ids = add_one(new_ids, groups)
                extra_tags.append("combo")

            new["drink_id"] = next_id
            next_id += 1

            new["temperature"] = new_temp
            new["sugar_level"] = new_sugar
            new["caffeine_level"] = new_caf
            new["ingredient_ids"] = ids_to_str(new_ids)

            new["name"] = make_name(base_name, new_temp, new_sugar, new_caf)
            new["tags"] = tag_append(base_tags, extra_tags)

            if "popularity_score" in drinks.columns:
                new["popularity_score"] = bump_popularity(base_pop)

            if ADD_DESCRIPTION:
                top_ing = [id2name[i] for i in new_ids if i in id2name][:4]
                new["description"] = build_description(new["name"], base_type, new_temp, new_sugar, new_caf, top_ing)

            aug_rows.append(new)

    out = pd.concat([drinks, pd.DataFrame(aug_rows)], ignore_index=True)

    out["drink_id"] = pd.to_numeric(out["drink_id"], errors="coerce").astype("Int64")

    out.to_csv(OUT_DRINKS, index=False)
    print(f"Saved augmented dataset: {OUT_DRINKS}")
    print(f"Original rows: {len(drinks)}")
    print(f"Augmented rows: {len(out)}")
    print(f"Aug factor: {len(out) / max(1, len(drinks)):.2f}x")


if __name__ == "__main__":
    main()
