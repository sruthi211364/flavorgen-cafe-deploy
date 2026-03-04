from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError


def get_mongo_collection() -> Tuple[bool, Optional[Collection], str]:
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        return False, None, "Not connected (favorites are local)."

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=4000)
        _ = client.server_info()
        db = client["flavorgen"]
        col = db["events"]
        return True, col, "Connected (favorites persist)."
    except PyMongoError as e:
        return False, None, f"Not connected (favorites are local). ({type(e).__name__})"


def log_favorite_event(
    col: Collection,
    user_id: str,
    session_id: str,
    drink_id: int,
    action: str,
) -> None:
    doc = {
        "event_type": "favorite",
        "user_id": user_id,
        "session_id": session_id,
        "drink_id": int(drink_id),
        "action": action,
        "created_at": datetime.now(timezone.utc),
    }
    col.insert_one(doc)


def get_current_favorites(col: Collection, user_id: str, session_id: str, limit: int = 500) -> Set[int]:
    cur = (
        col.find({"event_type": "favorite", "user_id": user_id, "session_id": session_id})
        .sort("created_at", 1)
        .limit(int(limit))
    )
    favs: Set[int] = set()
    for d in cur:
        did = int(d.get("drink_id"))
        act = str(d.get("action", "")).lower()
        if act == "add":
            favs.add(did)
        elif act == "remove":
            favs.discard(did)
    return favs


def clear_session_favorites(col: Collection, user_id: str, session_id: str) -> int:
    res = col.delete_many({"event_type": "favorite", "user_id": user_id, "session_id": session_id})
    return int(res.deleted_count)


class MongoFavoritesStore:
    """
    Small wrapper used by app_streamlit.py.
    MongoDB is optional at runtime.
    """

    def __init__(self) -> None:
        ok, col, _msg = get_mongo_collection()
        if not ok or col is None:
            raise RuntimeError("Mongo not available")
        self.col = col
        self.user_id = "local_user"

    def add_favorite(self, session_id: str, drink_id: int) -> None:
        log_favorite_event(self.col, self.user_id, session_id, int(drink_id), "add")

    def remove_favorite(self, session_id: str, drink_id: int) -> None:
        log_favorite_event(self.col, self.user_id, session_id, int(drink_id), "remove")

    def get_favorites(self, session_id: str) -> List[int]:
        favs = get_current_favorites(self.col, self.user_id, session_id)
        return sorted([int(x) for x in favs])

    def clear_favorites(self, session_id: str) -> int:
        return clear_session_favorites(self.col, self.user_id, session_id)
