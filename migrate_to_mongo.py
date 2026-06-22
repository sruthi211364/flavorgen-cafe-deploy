# migrate_to_mongo.py
# Run this ONCE to push your local data into MongoDB Atlas.
#
# Usage:
#   cd flavorgen_cafedata
#   python migrate_to_mongo.py
#
# What it does:
#   1. Loads drinks + ingredients from your local CSV/data files
#   2. Upserts them into MongoDB  (drinks, ingredients collections)
#   3. Migrates users.json → users collection
#   4. Migrates generated_drinks.json → generated_drinks collection
#   5. Prints a summary

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
FLAVORGEN_DIR = ROOT / "flavorgen"
USERS_JSON    = FLAVORGEN_DIR / "users.json"
GEN_JSON      = ROOT / "generated_drinks.json"

# ── Connect ───────────────────────────────────────────────────
URI = os.environ.get("MONGODB_URI", "")
if not URI:
    print("ERROR: MONGODB_URI not found in environment. Add it to your .env file.")
    sys.exit(1)

print("Connecting to MongoDB Atlas...")
client = MongoClient(URI, serverSelectionTimeoutMS=8000)
db     = client["flavorgen_cafe"]
print(f"Connected. Database: flavorgen_cafe\n")


# ════════════════════════════════════════════════════════════
# 1. DRINKS + INGREDIENTS  (from flavorgen data loaders)
# ════════════════════════════════════════════════════════════
print("── Step 1: Migrating drinks and ingredients ──")

# Add flavorgen package to path
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FLAVORGEN_DIR))

try:
    from flavorgen.data_loader import load_drinks, load_ingredients
    drinks_df      = load_drinks()
    ingredients_df = load_ingredients()
    print(f"  Loaded {len(drinks_df)} drinks, {len(ingredients_df)} ingredients from local files")
except Exception as e:
    print(f"  WARNING: Could not load via data_loader ({e})")
    print("  Skipping drinks/ingredients migration.")
    drinks_df = ingredients_df = None


def _safe_val(v):
    """Convert numpy/pandas types to plain Python for MongoDB."""
    if hasattr(v, "item"):       return v.item()
    if hasattr(v, "tolist"):     return v.tolist()
    if isinstance(v, float) and (v != v):  return None   # NaN → None
    return v


def df_to_docs(df, id_col: str) -> list[dict]:
    docs = []
    for _, row in df.iterrows():
        doc = {k: _safe_val(v) for k, v in row.to_dict().items()}
        docs.append(doc)
    return docs


if drinks_df is not None and not drinks_df.empty:
    drink_docs = df_to_docs(drinks_df, "drink_id")
    ops = [
        UpdateOne({"drink_id": d["drink_id"]}, {"$set": d}, upsert=True)
        for d in drink_docs if "drink_id" in d
    ]
    result = db["drinks"].bulk_write(ops)
    print(f"  drinks → upserted: {result.upserted_count}, modified: {result.modified_count}")
else:
    print("  drinks → skipped (no data)")

if ingredients_df is not None and not ingredients_df.empty:
    ing_docs = df_to_docs(ingredients_df, "ingredient_id")
    ops = [
        UpdateOne({"ingredient_id": d["ingredient_id"]}, {"$set": d}, upsert=True)
        for d in ing_docs if "ingredient_id" in d
    ]
    result = db["ingredients"].bulk_write(ops)
    print(f"  ingredients → upserted: {result.upserted_count}, modified: {result.modified_count}")
else:
    print("  ingredients → skipped (no data)")


# ════════════════════════════════════════════════════════════
# 2. USERS  (from users.json)
# ════════════════════════════════════════════════════════════
print("\n── Step 2: Migrating users.json ──")

if USERS_JSON.exists():
    try:
        users_data = json.loads(USERS_JSON.read_text(encoding="utf-8"))
        count = 0
        for username, profile in users_data.items():
            doc = {
                "username":     username,
                "pw_hash":      profile.get("pw_hash", ""),
                "display_name": profile.get("display_name", username),
                "interests":    profile.get("interests", []),
                "favourites":   [int(x) for x in profile.get("favorites", [])],
                "created_at":   profile.get("created_at", int(time.time())),
            }
            db["users"].update_one(
                {"username": username},
                {"$setOnInsert": doc},
                upsert=True,
            )
            count += 1
        print(f"  Migrated {count} user(s) (existing users not overwritten)")
    except Exception as e:
        print(f"  ERROR migrating users: {e}")
else:
    print("  users.json not found — skipping")


# ════════════════════════════════════════════════════════════
# 3. GENERATED DRINKS  (from generated_drinks.json)
# ════════════════════════════════════════════════════════════
print("\n── Step 3: Migrating generated_drinks.json ──")

if GEN_JSON.exists():
    try:
        items = json.loads(GEN_JSON.read_text(encoding="utf-8")) or []
        count = 0
        for item in items:
            doc = {**item, "username": "__migrated__", "created_at": int(time.time()) - count}
            if "flavor_vector" in doc and hasattr(doc["flavor_vector"], "tolist"):
                doc["flavor_vector"] = doc["flavor_vector"].tolist()
            db["generated_drinks"].insert_one(doc)
            count += 1
        print(f"  Migrated {count} generated drink(s)")
    except Exception as e:
        print(f"  ERROR migrating generated drinks: {e}")
else:
    print("  generated_drinks.json not found — skipping")


# ════════════════════════════════════════════════════════════
# 4. INDEXES
# ════════════════════════════════════════════════════════════
print("\n── Step 4: Creating indexes ──")
try:
    db["users"].create_index("username", unique=True)
    db["drinks"].create_index("drink_id", unique=True)
    db["generated_drinks"].create_index([("username", 1), ("created_at", -1)])
    print("  Indexes created/confirmed")
except PyMongoError as e:
    print(f"  Index warning (may already exist): {e}")


# ════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════
print("\n── Migration complete ──")
print(f"  users:            {db['users'].count_documents({})}")
print(f"  drinks:           {db['drinks'].count_documents({})}")
print(f"  ingredients:      {db['ingredients'].count_documents({})}")
print(f"  generated_drinks: {db['generated_drinks'].count_documents({})}")
print("\nYou can now remove users.json and generated_drinks.json from your project.")
print("Update LOAD_FROM_MONGO=true in your .env to use MongoDB in the app.")
