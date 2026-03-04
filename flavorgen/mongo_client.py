from __future__ import annotations

import hashlib
import os
import time
from typing import List, Optional

import streamlit as st
from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

# Optional: local dev only
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ── Connection ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_db():
    """Return the flavorgen_cafe database. Cached for the process lifetime."""
    print(">>> get_db() called", flush=True)
    uri = os.environ.get("MONGO_URI", "") or os.environ.get("MONGODB_URI", "")
    if not uri:
        raise RuntimeError(
            "Mongo URI not set. Add MONGO_URI in Hugging Face Space Secrets."
        )
    print(">>> URI found, connecting...", flush=True)
    client = MongoClient(
        uri,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
        socketTimeoutMS=5000,
    )
    print(">>> MongoClient created", flush=True)
    db = client["flavorgen_cafe"]
    _ensure_indexes(db)
    print(">>> get_db() done", flush=True)
    return db


def _ensure_indexes(db) -> None:
    """Create indexes once on first connect. Safe to call multiple times."""
    try:
        db["users"].create_index("username", unique=True)
        db["generated_drinks"].create_index(
            [("username", ASCENDING), ("created_at", ASCENDING)]
        )
        db["drinks"].create_index("drink_id", unique=True)
    except PyMongoError:
        pass


# ── Helpers ───────────────────────────────────────────────────
def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


# ════════════════════════════════════════════════════════════
# USERS
# ════════════════════════════════════════════════════════════
def db_register_user(username: str, pw: str, display: str) -> tuple[bool, str]:
    """Create a new user. Returns (success, message)."""
    if not username.strip() or not pw.strip():
        return False, "Username and password cannot be empty."
    try:
        db = get_db()
        db["users"].insert_one(
            {
                "username": username.strip(),
                "pw_hash": _hash_pw(pw),
                "display_name": display.strip() or username.strip(),
                "interests": [],
                "created_at": int(time.time()),
            }
        )
        return True, "Account created! You can now sign in."
    except DuplicateKeyError:
        return False, "Username already taken."
    except PyMongoError as e:
        return False, f"Database error: {e}"


def db_login_user(username: str, pw: str) -> Optional[dict]:
    """Return user doc if credentials match, else None."""
    try:
        db = get_db()
        user = db["users"].find_one({"username": username.strip()})
        if user and user.get("pw_hash") == _hash_pw(pw):
            return user
        return None
    except PyMongoError:
        return None


def db_save_interests(username: str, interests: List[str]) -> None:
    """Update a user's taste interests list."""
    try:
        get_db()["users"].update_one(
            {"username": username},
            {"$set": {"interests": interests, "updated_at": int(time.time())}},
        )
    except PyMongoError:
        pass


def db_get_user(username: str) -> Optional[dict]:
    try:
        return get_db()["users"].find_one({"username": username})
    except PyMongoError:
        return None


# ════════════════════════════════════════════════════════════
# FAVOURITES
# ════════════════════════════════════════════════════════════
def db_save_favourites(username: str, favourite_ids: List[int]) -> None:
    if username in (None, "__guest__"):
        return
    try:
        get_db()["users"].update_one(
            {"username": username},
            {"$set": {"favourites": sorted(favourite_ids), "updated_at": int(time.time())}},
        )
    except PyMongoError:
        pass


def db_load_favourites(username: str) -> List[int]:
    if username in (None, "__guest__"):
        return []
    try:
        user = get_db()["users"].find_one({"username": username}, {"favourites": 1})
        if user:
            return [int(x) for x in user.get("favourites", [])]
        return []
    except PyMongoError:
        return []


# ════════════════════════════════════════════════════════════
# GENERATED / FUSION DRINKS
# ════════════════════════════════════════════════════════════
def db_save_generated_drink(username: str, drink: dict) -> None:
    try:
        doc = {**drink, "username": username, "created_at": int(time.time())}
        if "flavor_vector" in doc and hasattr(doc["flavor_vector"], "tolist"):
            doc["flavor_vector"] = doc["flavor_vector"].tolist()
        get_db()["generated_drinks"].insert_one(doc)
    except PyMongoError as e:
        raise RuntimeError(f"Could not save drink: {e}")


def db_load_generated_drinks(username: str, limit: int = 50) -> List[dict]:
    try:
        cursor = (
            get_db()["generated_drinks"]
            .find({"username": username}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        return list(cursor)
    except PyMongoError:
        return []


def db_delete_generated_drink(username: str, created_at: int) -> None:
    try:
        get_db()["generated_drinks"].delete_one({"username": username, "created_at": created_at})
    except PyMongoError:
        pass


def db_load_all_generated_drinks(limit: int = 50) -> List[dict]:
    try:
        cursor = (
            get_db()["generated_drinks"]
            .find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(limit)
        )
        return list(cursor)
    except PyMongoError:
        return []


# ════════════════════════════════════════════════════════════
# DRINKS CATALOGUE
# ════════════════════════════════════════════════════════════
def db_load_drinks() -> List[dict]:
    try:
        return list(get_db()["drinks"].find({}, {"_id": 0}))
    except PyMongoError:
        return []


def db_load_ingredients() -> List[dict]:
    try:
        return list(get_db()["ingredients"].find({}, {"_id": 0}))
    except PyMongoError:
        return []