# app_streamlit.py
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — faster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from flavorgen.data_loader import (
    FLAVOR_DIMS,
    compute_drink_flavor_vector,
    load_drinks,
    load_ingredients,
)
from flavorgen.semantic_model import (
    CAFFEINE_LEVELS,
    SUGAR_LEVELS,
    TEMP_LEVELS,
    build_hybrid_model,
    hybrid_recommend,
    parse_query_to_preferences,
)
from flavorgen.fusion_model import (
    load_artifacts,
    predict_fusion,
)


# ────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flavorgen Café Lab",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent

USERS_PATH  = Path(__file__).resolve().parent / "users.json"
GEN_PATH    = project_root() / "generated_drinks.json"


# ────────────────────────────────────────────────────────────
# Cached heavy resources  (loaded once per process)
# ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_fusion_artifacts_cached(models_dir: str, in_dim: int, out_dim: int):
    return load_artifacts(models_dir, in_dim=in_dim, out_dim=out_dim)


@st.cache_data(show_spinner=False)
def _load_fusion_meta(meta_path: str) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return load_ingredients(), load_drinks()


@st.cache_data(show_spinner=False)
def _cached_hybrid_model(df_hash: str, _df: pd.DataFrame):
    """Cache the hybrid model keyed by a hash of the dataframe — avoids rebuilding on every rerun."""
    return build_hybrid_model(_df)


@st.cache_data(show_spinner=False)
def _cached_flavor_vector(drink_id: int) -> np.ndarray:
    """Per-drink flavor vector cached globally."""
    return compute_drink_flavor_vector(drinks_df, drink_id)


def _df_hash(df: pd.DataFrame) -> str:
    """Hash a dataframe by shape + column names + row count.
    Avoids pd.util.hash_pandas_object which chokes on list-valued columns."""
    key = f"{len(df)}|{list(df.columns)}|{df.shape}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


# ────────────────────────────────────────────────────────────
# Data  (loaded at module level, cached)
# ────────────────────────────────────────────────────────────
ingredients_df, drinks_df = load_data()


# ════════════════════════════════════════════════════════════
# USER / PROFILE SYSTEM
# ════════════════════════════════════════════════════════════
def _load_users() -> dict:
    if USERS_PATH.exists():
        try:
            with open(USERS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_users(users: dict) -> None:
    try:
        with open(USERS_PATH, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)
    except Exception:
        pass


def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _blank_profile(username: str) -> dict:
    return {
        "username":   username,
        "pw_hash":    "",
        "favorites":  [],
        "interests":  [],          # list of flavor dims or tags user likes
        "created_at": int(time.time()),
        "display_name": username,
    }


def login_user(username: str, pw: str) -> bool:
    users = _load_users()
    u = users.get(username)
    if u and u.get("pw_hash") == _hash_pw(pw):
        st.session_state.current_user     = username
        st.session_state.favorites        = set(int(x) for x in u.get("favorites", []))
        st.session_state.user_interests   = u.get("interests", [])
        st.session_state.user_display     = u.get("display_name", username)
        return True
    return False


def register_user(username: str, pw: str, display: str) -> Tuple[bool, str]:
    if not username.strip() or not pw.strip():
        return False, "Username and password cannot be empty."
    users = _load_users()
    if username in users:
        return False, "Username already taken."
    p = _blank_profile(username)
    p["pw_hash"]      = _hash_pw(pw)
    p["display_name"] = display or username
    users[username]   = p
    _save_users(users)
    return True, "Account created!"


def logout_user() -> None:
    for k in ["current_user", "favorites", "user_interests", "user_display",
              "ai_pref_results", "ai_pref_meta", "ai_text_results", "ai_text_meta",
              "fusion_last", "active_page", "_random_pick_id", "_random_pick_name"]:
        st.session_state.pop(k, None)


def save_user_profile() -> None:
    """Persist current session favorites & interests back to users.json."""
    u = st.session_state.get("current_user")
    if not u:
        return
    users = _load_users()
    if u not in users:
        return
    users[u]["favorites"]  = sorted(list(st.session_state.get("favorites", set())))
    users[u]["interests"]  = st.session_state.get("user_interests", [])
    _save_users(users)


def toggle_favorite(drink_id: int) -> None:
    did = int(drink_id)
    favs = st.session_state.get("favorites", set())
    if did in favs:
        favs.discard(did)
    else:
        favs.add(did)
    st.session_state.favorites = favs
    save_user_profile()
    st.session_state["_fav_toast"] = did


def clear_favorites() -> None:
    st.session_state.favorites = set()
    save_user_profile()


def load_favorites_ids() -> List[int]:
    return sorted(int(x) for x in st.session_state.get("favorites", set()))


# ────────────────────────────────────────────────────────────
# Session bootstrap
# ────────────────────────────────────────────────────────────
def ensure_session() -> None:
    defaults = {
        "current_user":    None,
        "favorites":       set(),
        "user_interests":  [],
        "user_display":    "Guest",
        "_fav_toast":      None,
        "ai_pref_results": [],
        "ai_pref_meta":    {},
        "ai_text_results": [],
        "ai_text_meta":    {},
        "fusion_last":     None,
        "active_page":     "home",
        "_show_login":     True,   # True = login tab, False = register tab
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_session()


# ════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════
def inject_global_css() -> None:
    st.markdown(r"""
<style>
/* ── TOKENS ─────────────────────────────────────────── */
:root {
  --cream-50:   #FFFEF9;
  --cream-100:  #FFF8EE;
  --amber-400:  #F5A623;
  --amber-500:  #D4891A;
  --caramel:    #C97B2E;
  --espresso:   #2C1505;
  --mocha:      #6B3F1E;
  --brown-600:  #8B5E3C;
  --success:    #2E7D52;

  --ink-900: rgba(28,12,2,.92);
  --ink-700: rgba(28,12,2,.68);
  --ink-500: rgba(28,12,2,.45);
  --ink-300: rgba(28,12,2,.16);
  --ink-100: rgba(28,12,2,.06);

  --surface-glass:   rgba(255,255,255,.82);
  --surface-tinted:  rgba(255,246,230,.75);

  --shadow-xs: 0 2px  8px  rgba(44,21,5,.07);
  --shadow-sm: 0 6px  18px rgba(44,21,5,.09);
  --shadow-md: 0 14px 32px rgba(44,21,5,.12);
  --shadow-lg: 0 24px 56px rgba(44,21,5,.18);
  --shadow-amber: 0 8px 28px rgba(212,137,26,.32);

  --r-sm:   10px;
  --r-md:   18px;
  --r-lg:   24px;
  --r-xl:   30px;
  --r-pill: 999px;

  --tx-xs:   11px;
  --tx-sm:   13px;
  --tx-base: 15px;
  --tx-lg:   18px;
  --tx-hero: 48px;

  --font-display: 'Georgia','Palatino Linotype',serif;
  --font-body: -apple-system,'Segoe UI',sans-serif;
}

/* ── LAYOUT ──────────────────────────────────────────── */
section.main > div.block-container {
  padding-top: 1.2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1200px;
}

/* ── BACKGROUND ──────────────────────────────────────── */
html,body,[data-testid="stAppViewContainer"]{ height:100%; }

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(ellipse 900px 550px at 5% 10%,  rgba(255,224,160,.70),transparent 55%),
    radial-gradient(ellipse 700px 480px at 95% 15%, rgba(230,248,255,.55),transparent 55%),
    radial-gradient(ellipse 600px 400px at 48% 98%, rgba(205,155,90,.14), transparent 55%),
    linear-gradient(170deg,var(--cream-50) 0%,rgba(255,250,240,.90) 100%) !important;
  background-attachment:fixed !important;
  overflow-x:hidden;
}
@keyframes bgBreathe{0%,100%{filter:saturate(1) brightness(1)}50%{filter:saturate(1.07) brightness(1.02)}}
[data-testid="stAppViewContainer"]{ animation:bgBreathe 11s ease-in-out infinite; }

/* ── SIDEBAR ─────────────────────────────────────────── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(255,252,245,.94),rgba(255,245,228,.97));
  border-right:1px solid var(--ink-300);
}

/* ── HERO ────────────────────────────────────────────── */
.hero-wrap{
  position:relative; border-radius:var(--r-xl);
  border:1px solid rgba(201,123,46,.22);
  padding:26px 26px 20px; overflow:hidden;
  background:
    radial-gradient(ellipse 850px 400px at 10% 15%,rgba(255,215,140,.62),transparent 52%),
    radial-gradient(ellipse 750px 380px at 90% 12%,rgba(255,242,210,.74),transparent 52%),
    linear-gradient(180deg,rgba(255,255,255,.97),rgba(255,250,238,.84));
  box-shadow:var(--shadow-md),inset 0 1px 0 rgba(255,255,255,.9);
}
.hero-title{font-family:var(--font-display);font-size:var(--tx-hero);font-weight:700;
  letter-spacing:-1px;color:var(--espresso);margin:0;line-height:1.08;}
.hero-sub{margin-top:9px;color:var(--ink-700);font-size:var(--tx-sm);line-height:1.5;}
.hero-chiprow{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px;}
.hero-chip{display:inline-flex;align-items:center;gap:6px;padding:5px 13px;
  border-radius:var(--r-pill);border:1px solid rgba(201,123,46,.28);
  background:rgba(255,255,255,.78);font-size:var(--tx-xs);font-weight:700;color:var(--mocha);}

@keyframes floatA{0%,100%{transform:translateY(0) rotate(0);opacity:.28}50%{transform:translateY(-10px) rotate(3deg);opacity:.40}}
@keyframes floatB{0%,100%{transform:translateY(0) rotate(0);opacity:.16}50%{transform:translateY(12px) rotate(-4deg);opacity:.26}}
.floaty{position:absolute;pointer-events:none;user-select:none;font-size:28px;}
.floaty.s1{right:22px;top:16px;   animation:floatA 3.8s ease-in-out infinite;}
.floaty.s2{right:64px;top:26px;   animation:floatB 4.6s ease-in-out infinite;}
.floaty.s3{right:22px;bottom:14px;animation:floatB 4.2s ease-in-out infinite;}
.floaty.s4{right:76px;bottom:22px;animation:floatA 4.0s ease-in-out infinite;}
.floaty.s5{right:116px;top:18px;  animation:floatA 4.8s ease-in-out infinite;}

/* ── NAV TILES ───────────────────────────────────────── */
.navtile-outer{position:relative;margin-bottom:0;}
.navtile-card{
  position:relative; border-radius:var(--r-lg);
  border:1.5px solid rgba(201,123,46,.18);
  padding:24px 22px 20px; overflow:hidden; cursor:pointer;
  transition:transform .26s cubic-bezier(.34,1.56,.64,1),box-shadow .22s ease,border-color .2s ease;
  background:
    radial-gradient(ellipse 500px 280px at 0% 0%,  rgba(255,228,155,.48),transparent 58%),
    radial-gradient(ellipse 400px 260px at 100% 0%, rgba(255,244,215,.58),transparent 58%),
    linear-gradient(155deg,rgba(255,255,255,.97) 0%,rgba(255,249,237,.90) 100%);
  box-shadow:var(--shadow-sm);
  min-height:200px; display:flex; flex-direction:column; justify-content:space-between; gap:14px;
}
.navtile-card:hover{
  transform:translateY(-9px) scale(1.015);
  box-shadow:var(--shadow-lg),0 0 0 1.5px rgba(201,123,46,.30),0 0 40px rgba(245,166,35,.12);
  border-color:rgba(201,123,46,.45);
}
.navtile-card:active{transform:translateY(-3px) scale(1.004);box-shadow:var(--shadow-md);transition-duration:.08s;}
.navtile-card::after{
  content:attr(data-emoji);position:absolute;right:16px;bottom:14px;
  font-size:58px;opacity:.09;pointer-events:none;user-select:none;
  animation:floatA 4.2s ease-in-out infinite;line-height:1;transition:opacity .22s ease;
}
.navtile-card:hover::after{opacity:.16;}
.navtile-icon{font-size:30px;margin-bottom:9px;display:block;
  filter:drop-shadow(0 2px 5px rgba(0,0,0,.12));
  transition:transform .22s cubic-bezier(.34,1.56,.64,1);}
.navtile-card:hover .navtile-icon{transform:scale(1.18) rotate(-4deg);}
.navtile-title{font-family:var(--font-display);font-size:20px;font-weight:700;
  color:var(--espresso);letter-spacing:-.4px;line-height:1.2;margin:0 0 6px 0;}
.navtile-sub{font-size:var(--tx-sm);color:var(--ink-500);line-height:1.45;margin:0;}
.navtile-tags{display:flex;flex-wrap:wrap;gap:6px;}
.navtile-tag{
  display:inline-flex;align-items:center;gap:4px;padding:4px 11px;
  border-radius:var(--r-pill);border:1px solid rgba(201,123,46,.22);
  background:rgba(255,248,235,.90);color:var(--mocha);
  font-size:var(--tx-xs);font-weight:700;letter-spacing:.2px;white-space:nowrap;
  transition:background .16s ease,border-color .16s ease;
}
.navtile-card:hover .navtile-tag{background:rgba(255,240,210,.95);border-color:rgba(201,123,46,.36);}
.navtile-btn-wrap [data-testid="stButton"]>button{
  width:100% !important;border-radius:var(--r-pill) !important;
  background:rgba(201,123,46,.10) !important;border:1px solid rgba(201,123,46,.28) !important;
  color:var(--mocha) !important;font-weight:700 !important;font-size:var(--tx-xs) !important;
  padding:.38rem 1rem !important;letter-spacing:.4px !important;
  transition:background .18s ease,border-color .18s ease,transform .18s ease !important;
}
.navtile-btn-wrap [data-testid="stButton"]>button:hover{
  background:rgba(201,123,46,.18) !important;border-color:rgba(201,123,46,.50) !important;
  transform:translateY(-1px) !important;
}
@keyframes tileIn{from{opacity:0;transform:translateY(28px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}
.navtile-outer{animation:tileIn .55s cubic-bezier(.22,1,.36,1) both;}
.tile-d1{animation-delay:.04s}.tile-d2{animation-delay:.10s}.tile-d3{animation-delay:.16s}
.tile-d4{animation-delay:.22s}.tile-d5{animation-delay:.28s}.tile-d6{animation-delay:.34s}
.tile-d7{animation-delay:.40s}

/* ── RECIPE CARD ─────────────────────────────────────── */
.recipe-card{
  border-radius:var(--r-lg);border:1.5px solid rgba(201,123,46,.22);
  padding:28px 28px 22px;
  background:
    radial-gradient(ellipse 600px 300px at 5% 5%,rgba(255,228,155,.35),transparent 55%),
    linear-gradient(160deg,rgba(255,255,255,.97),rgba(255,249,237,.92));
  box-shadow:var(--shadow-md);
}
.recipe-title{font-family:var(--font-display);font-size:26px;font-weight:700;
  color:var(--espresso);letter-spacing:-.4px;margin:0 0 4px 0;}
.recipe-desc{color:var(--ink-700);font-size:var(--tx-sm);margin:0 0 18px 0;line-height:1.5;}
.recipe-meta-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:20px;}
.recipe-meta-pill{
  padding:5px 13px;border-radius:var(--r-pill);
  border:1px solid rgba(201,123,46,.26);background:rgba(255,248,235,.90);
  font-size:var(--tx-xs);font-weight:700;color:var(--mocha);
}
.recipe-section-label{
  font-size:var(--tx-xs);font-weight:800;letter-spacing:.8px;
  text-transform:uppercase;color:var(--caramel);margin:18px 0 10px 0;
}
.ingredient-row{
  display:flex;align-items:center;justify-content:space-between;
  padding:9px 14px;border-radius:var(--r-sm);
  border:1px solid var(--ink-100);background:var(--surface-glass);
  margin-bottom:6px;
}
.ingredient-name{font-weight:700;font-size:var(--tx-sm);color:var(--ink-900);}
.ingredient-conf{font-size:var(--tx-xs);color:var(--ink-500);font-weight:600;}
.conf-bar-wrap{width:90px;height:6px;background:var(--ink-100);border-radius:3px;overflow:hidden;}
.conf-bar{height:100%;border-radius:3px;
  background:linear-gradient(90deg,var(--amber-400),var(--caramel));transition:width .4s ease;}
.step-row{
  display:flex;gap:14px;align-items:flex-start;
  padding:12px 14px;border-radius:var(--r-sm);
  border:1px solid var(--ink-100);background:var(--surface-glass);
  margin-bottom:8px;
}
.step-num{
  flex-shrink:0;width:28px;height:28px;border-radius:50%;
  background:linear-gradient(135deg,var(--amber-400),var(--amber-500));
  color:#fff;font-size:var(--tx-xs);font-weight:800;
  display:flex;align-items:center;justify-content:center;
}
.step-text{font-size:var(--tx-sm);color:var(--ink-900);line-height:1.5;padding-top:4px;}
.flavor-note-pill{
  display:inline-block;padding:5px 13px;margin:0 5px 5px 0;
  border-radius:var(--r-pill);border:1px solid rgba(201,123,46,.26);
  background:rgba(255,248,235,.90);font-size:var(--tx-xs);font-weight:700;color:var(--mocha);
}

/* ── SECTION CARDS ───────────────────────────────────── */
.section-card{padding:16px 18px;border-radius:var(--r-md);border:1px solid var(--ink-300);
  background:var(--surface-tinted);box-shadow:var(--shadow-xs);margin-bottom:4px;}
.section-card-title{font-weight:800;font-size:var(--tx-lg);color:var(--ink-900);}
.section-card-sub{color:var(--ink-500);font-size:var(--tx-sm);margin-top:4px;}

/* ── DRINK CARDS ─────────────────────────────────────── */
.drink-card{padding:14px;border-radius:var(--r-md);border:1px solid var(--ink-300);
  background:var(--surface-glass);box-shadow:var(--shadow-xs);}
.drink-card-name{font-weight:800;font-size:var(--tx-base);color:var(--ink-900);}
.drink-card-meta{color:var(--ink-500);font-size:var(--tx-xs);margin-top:5px;}

/* ── TAG BADGES ──────────────────────────────────────── */
.tag-badge{display:inline-block;margin:0 5px 5px 0;padding:4px 10px;
  border-radius:var(--r-pill);border:1px solid var(--ink-300);
  background:var(--surface-glass);font-size:var(--tx-xs);font-weight:700;color:var(--ink-700);}

/* ── ROLL RESULT ─────────────────────────────────────── */
.roll-result{padding:14px 18px;border-radius:var(--r-md);
  border:1px solid rgba(201,123,46,.28);background:rgba(255,243,210,.60);
  font-size:var(--tx-base);color:var(--espresso);}
.roll-drink-name{font-family:var(--font-display);font-size:22px;font-weight:700;
  color:var(--espresso);margin:4px 0 8px 0;}

/* ── MYSTERY HERO ────────────────────────────────────── */
.mystery-hero{
  text-align:center;padding:40px 20px;
  border-radius:var(--r-xl);border:1.5px solid rgba(201,123,46,.20);
  background:
    radial-gradient(ellipse 700px 400px at 50% 40%,rgba(255,215,130,.55),transparent 55%),
    linear-gradient(180deg,rgba(255,255,255,.97),rgba(255,249,232,.90));
  box-shadow:var(--shadow-md);margin-bottom:24px;
}
.mystery-hero-title{font-family:var(--font-display);font-size:36px;font-weight:700;
  color:var(--espresso);margin:16px 0 8px 0;}
.mystery-hero-sub{color:var(--ink-700);font-size:var(--tx-base);margin:0;}

/* ── BACK BUTTON ─────────────────────────────────────── */
.backhome [data-testid="stButton"]>button{
  border-radius:var(--r-pill) !important;font-weight:700 !important;
  font-size:var(--tx-sm) !important;padding:.4rem 1rem !important;
  background:var(--surface-glass) !important;border:1px solid var(--ink-300) !important;
  box-shadow:var(--shadow-xs);color:var(--ink-700) !important;
  transition:background .15s ease,box-shadow .15s ease;
}
.backhome [data-testid="stButton"]>button:hover{
  background:rgba(255,255,255,.96) !important;box-shadow:var(--shadow-sm);}

/* ── PRIMARY BUTTONS ─────────────────────────────────── */
[data-testid="stButton"]>button[kind="primary"]{
  border-radius:var(--r-pill) !important;
  background:linear-gradient(135deg,var(--amber-400),var(--amber-500)) !important;
  border:none !important;color:#fff !important;font-weight:800 !important;
  box-shadow:var(--shadow-amber);transition:box-shadow .18s ease,transform .18s ease;
}
[data-testid="stButton"]>button[kind="primary"]:hover{
  box-shadow:0 12px 36px rgba(212,137,26,.55);transform:translateY(-2px);}

/* ── SECONDARY BUTTONS ───────────────────────────────── */
[data-testid="stButton"]>button:not([kind="primary"]){
  border-radius:var(--r-pill) !important;background:var(--surface-glass) !important;
  border:1px solid var(--ink-300) !important;color:var(--ink-900) !important;
  font-weight:700 !important;font-size:var(--tx-sm) !important;
  transition:background .15s ease,box-shadow .15s ease;
}
[data-testid="stButton"]>button:not([kind="primary"]):hover{
  background:rgba(255,255,255,.96) !important;box-shadow:var(--shadow-sm);}

/* ── INPUTS ──────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"]>div{
  border-radius:var(--r-sm) !important;border-color:var(--ink-300) !important;
  background:var(--surface-glass) !important;font-size:var(--tx-sm) !important;
}

/* ── DATAFRAMES ──────────────────────────────────────── */
[data-testid="stDataFrame"]{
  border-radius:var(--r-md) !important;overflow:hidden;
  border:1px solid var(--ink-300) !important;box-shadow:var(--shadow-xs);
}

/* ── TABS ────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"]{font-weight:700 !important;
  font-size:var(--tx-sm) !important;color:var(--ink-700) !important;}
[data-testid="stTabs"] [aria-selected="true"]{
  color:var(--brown-600) !important;border-bottom-color:var(--amber-400) !important;}

/* ── MISC ────────────────────────────────────────────── */
hr{border-color:var(--ink-300) !important;margin:1.4rem 0 !important;}
[data-testid="stExpander"]{border-radius:var(--r-md) !important;
  border:1px solid var(--ink-300) !important;background:var(--surface-tinted) !important;}

/* ── LOGIN CARD ──────────────────────────────────────── */
.login-card{
  max-width:460px;margin:40px auto;padding:36px 32px;
  border-radius:var(--r-xl);border:1.5px solid rgba(201,123,46,.22);
  background:
    radial-gradient(ellipse 500px 300px at 20% 20%,rgba(255,225,150,.45),transparent 55%),
    linear-gradient(160deg,rgba(255,255,255,.97),rgba(255,249,237,.92));
  box-shadow:var(--shadow-lg);
}
.login-title{font-family:var(--font-display);font-size:28px;font-weight:700;
  color:var(--espresso);text-align:center;margin:0 0 6px 0;}
.login-sub{color:var(--ink-500);font-size:var(--tx-sm);text-align:center;margin:0 0 24px 0;}

/* ── PROFILE BADGE ───────────────────────────────────── */
.profile-badge{
  display:flex;align-items:center;gap:10px;
  padding:10px 14px;border-radius:var(--r-pill);
  border:1px solid rgba(201,123,46,.28);background:rgba(255,248,235,.90);
}
.profile-avatar{width:34px;height:34px;border-radius:50%;
  background:linear-gradient(135deg,var(--amber-400),var(--caramel));
  display:flex;align-items:center;justify-content:center;
  font-size:16px;font-weight:800;color:#fff;flex-shrink:0;}
.profile-name{font-weight:700;font-size:var(--tx-sm);color:var(--espresso);}
.profile-sub{font-size:var(--tx-xs);color:var(--ink-500);}

/* ── SITE FLOATERS ───────────────────────────────────── */
@keyframes siteDriftA{0%,100%{transform:translateY(0) translateX(0) rotate(0)}50%{transform:translateY(-14px) translateX(7px) rotate(4deg)}}
@keyframes siteDriftB{0%,100%{transform:translateY(0) translateX(0) rotate(0)}50%{transform:translateY(16px) translateX(-9px) rotate(-5deg)}}
.site-float{position:fixed;z-index:0;font-size:32px;opacity:.08;pointer-events:none;user-select:none;}
.site-float.sA{left:18px;  top:130px;   animation:siteDriftA 6.2s ease-in-out infinite;}
.site-float.sB{right:24px; top:160px;   animation:siteDriftB 7.0s ease-in-out infinite;}
.site-float.sC{left:30px;  bottom:80px; animation:siteDriftB 6.6s ease-in-out infinite;}
.site-float.sD{right:52px; bottom:110px;animation:siteDriftA 7.4s ease-in-out infinite;}
.site-float.sE{right:210px;top:98px;    animation:siteDriftA 8.0s ease-in-out infinite;}
</style>
""", unsafe_allow_html=True)


def render_site_floaters() -> None:
    st.markdown("""
<div class="site-float sA">☕</div>
<div class="site-float sB">🧋</div>
<div class="site-float sC">🍵</div>
<div class="site-float sD">🥐</div>
<div class="site-float sE">🍫</div>
""", unsafe_allow_html=True)


inject_global_css()
render_site_floaters()


# ════════════════════════════════════════════════════════════
# LOGIN / REGISTER PAGE
# ════════════════════════════════════════════════════════════
def page_auth() -> None:
    st.markdown("""
<div class="login-card">
  <div class="login-title">☕ Flavorgen Café</div>
  <div class="login-sub">Sign in to save favourites, track your taste profile, and get personalised picks.</div>
</div>
""", unsafe_allow_html=True)

    # Centre the form
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_login, tab_reg = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            uname = st.text_input("Username", key="login_uname")
            pw    = st.text_input("Password", type="password", key="login_pw")
            if st.button("Sign In", key="btn_login", use_container_width=True, type="primary"):
                if login_user(uname.strip(), pw):
                    st.session_state.active_page = "home"
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")
            st.markdown("")
            if st.button("Continue as Guest", key="btn_guest", use_container_width=True):
                st.session_state.current_user  = "__guest__"
                st.session_state.user_display  = "Guest"
                st.session_state.active_page   = "home"
                st.rerun()

        with tab_reg:
            r_uname   = st.text_input("Username",     key="reg_uname")
            r_display = st.text_input("Display name (optional)", key="reg_display")
            r_pw      = st.text_input("Password",     type="password", key="reg_pw")
            r_pw2     = st.text_input("Confirm password", type="password", key="reg_pw2")
            if st.button("Create Account", key="btn_register", use_container_width=True, type="primary"):
                if r_pw != r_pw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(r_uname.strip(), r_pw, r_display.strip())
                    if ok:
                        st.success(msg + " You can now sign in.")
                    else:
                        st.error(msg)


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
def _safe_ids_list(x: object) -> List[int]:
    if x is None: return []
    if isinstance(x, list):
        return [int(v) for v in x if str(v).strip().lstrip("-").isdigit()]
    s = str(x).replace("[","").replace("]","").replace(";",",").replace('"',"").replace("'","")
    return [int(p.strip()) for p in s.split(",") if p.strip().lstrip("-").isdigit()]


def top_ingredients_for_drink(drink_row: pd.Series, k: int = 4) -> List[str]:
    if ingredients_df is None or ingredients_df.empty: return []
    ids = _safe_ids_list(drink_row.get("ingredient_ids",""))
    if not ids: return []
    sub = ingredients_df[ingredients_df["ingredient_id"].isin(ids)].copy()
    if sub.empty: return []
    nc = "name" if "name" in sub.columns else ("ingredient_name" if "ingredient_name" in sub.columns else None)
    if nc is None: return []
    dims = [d for d in FLAVOR_DIMS if d in sub.columns]
    if dims:
        sub["_impact"] = sub[dims].sum(axis=1)
        sub = sub.sort_values("_impact", ascending=False)
    return [n for n in sub[nc].astype(str).head(k).tolist() if n.strip()]


def safe_unique_values(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns: return ["any"]
    vals = sorted(set(df[col].dropna().astype(str).str.strip().str.lower().tolist()))
    return ["any"] + [v for v in vals if v]


def get_name_col(df: pd.DataFrame) -> str:
    return "name" if "name" in df.columns else ("drink_name" if "drink_name" in df.columns else df.columns[0])


def _parse_tags(tags_val: object) -> List[str]:
    if not tags_val: return []
    s = str(tags_val).replace(";",",").replace("[","").replace("]","").replace("'","").replace('"',"")
    return [p.replace("_"," ").strip() for p in s.split(",") if p.strip()]


def _format_meta(drink_row: pd.Series) -> None:
    parts = []
    for lbl, key in [("Type","type"),("Temp","temperature"),("Sugar","sugar_level"),("Caffeine","caffeine_level")]:
        v = str(drink_row.get(key,"")).strip()
        if v: parts.append(f"**{lbl}: {v}**")
    if parts: st.write(" | ".join(parts))
    tags = _parse_tags(drink_row.get("tags",""))
    if tags:
        st.markdown("".join([f"<span class='tag-badge'>{t}</span>" for t in tags]), unsafe_allow_html=True)


# ── Charts (lazy — only rendered when called) ────────────
def radar_chart(
    flavor_vec: Union[Dict, np.ndarray, List],
    user_vec: Optional[Union[Dict, np.ndarray, List]] = None,
) -> plt.Figure:
    dims = list(FLAVOR_DIMS)
    values = ([float(flavor_vec.get(d,0)) for d in dims] if isinstance(flavor_vec, dict)
              else np.zeros(len(dims)) if len(np.array(flavor_vec).flatten()) != len(dims)
              else np.array(flavor_vec, dtype=float).flatten().tolist())
    values = np.clip(values, 0, 1).tolist()
    angles  = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist()
    v2 = values + values[:1];  a2 = angles + angles[:1]
    fig = plt.figure(figsize=(3.8, 3.8), dpi=90)
    ax  = plt.subplot(111, polar=True)
    ax.plot(a2, v2, linewidth=1.8, label="Drink"); ax.fill(a2, v2, alpha=0.18)
    if user_vec is not None:
        uv = ([float(user_vec.get(d,0)) for d in dims] if isinstance(user_vec, dict)
              else np.array(user_vec, dtype=float).flatten().tolist())
        uv = np.clip(uv, 0, 1).tolist(); uv2 = uv + uv[:1]
        ax.plot(a2, uv2, linestyle="dashed", linewidth=1.5, label="Your pref")
    ax.set_xticks(angles); ax.set_xticklabels([d.replace("_"," ") for d in dims], fontsize=9)
    ax.set_yticklabels([]); ax.legend(loc="upper right", bbox_to_anchor=(1.25,1.15), fontsize=8)
    fig.tight_layout(); return fig


def flavor_bar_chart(avg_vec: np.ndarray, title: str = "Your flavor profile") -> plt.Figure:
    dims = list(FLAVOR_DIMS)
    vals = np.clip(np.array(avg_vec, dtype=float).flatten()
                   if len(np.array(avg_vec).flatten()) == len(dims)
                   else np.zeros(len(dims)), 0, 1)
    fig = plt.figure(figsize=(7.2, 3.2)); ax = plt.gca()
    ax.bar([d.replace("_"," ") for d in dims], vals)
    ax.set_ylim(0,1); ax.set_title(title); ax.set_ylabel("Intensity (0–1)")
    ax.set_xticklabels([d.replace("_"," ") for d in dims], rotation=35, ha="right")
    fig.tight_layout(); return fig


# ── Filters / scoring helpers ────────────────────────────
def _apply_excludes(df: pd.DataFrame, terms: List[str]) -> pd.DataFrame:
    if not terms: return df
    text = (df.get("name", pd.Series([""]* len(df))).astype(str).fillna("") + " " +
            df.get("tags", pd.Series([""]*len(df))).astype(str).fillna("")).str.lower()
    mask = ~text.str.contains("|".join(terms), na=False)
    return df[mask]


def smooth_fill_results(base: pd.DataFrame, full: pd.DataFrame, top_k: int) -> pd.DataFrame:
    base = pd.DataFrame() if (base is None) else base.copy()
    full = pd.DataFrame() if (full is None) else full.copy()
    if not base.empty: base["_is_suggestion"] = 0
    if len(base) >= top_k: return base.head(top_k)
    if "drink_id" in full.columns and "drink_id" in base.columns and not base.empty:
        used = set(base["drink_id"].dropna().astype(int).tolist())
        rem  = full[~full["drink_id"].isin(list(used))].copy()
    else:
        rem = full.copy()
    if rem.empty: return base
    rem["_is_suggestion"] = 1
    filler = rem.head(top_k - len(base))
    return pd.concat([base, filler], ignore_index=True) if not base.empty else filler


def dedupe_by_drink_id(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "drink_id" not in df.columns: return df
    out = df.copy()
    out["drink_id"] = pd.to_numeric(out["drink_id"], errors="coerce")
    return out.dropna(subset=["drink_id"]).drop_duplicates(subset=["drink_id"], keep="first")


# ── Why-this-matches ─────────────────────────────────────
def _why_bullets(drink_row, sugar_pref, caffeine_pref, temp_pref, flavor_focus,
                 score=None, score_text=None, score_flavor=None, flavor_vector=None) -> List[str]:
    b: List[str] = []
    if score        is not None: b.append(f"Overall match: {score:.3f}")
    if score_text   is not None: b.append(f"Text similarity: {score_text:.3f}")
    if score_flavor is not None: b.append(f"Flavor similarity: {score_flavor:.3f}")
    if sugar_pref and sugar_pref != "any":    b.append(f"Sugar: {sugar_pref}")
    if caffeine_pref and caffeine_pref != "any": b.append(f"Caffeine: {caffeine_pref}")
    dt = str(drink_row.get("temperature","")).strip().lower()
    if temp_pref and temp_pref != "any":
        b.append(f"Temp {'✓' if dt==temp_pref else '≠'}: {temp_pref}")
    if flavor_focus and flavor_vector is not None:
        dims = list(FLAVOR_DIMS)
        for d in flavor_focus:
            if d in dims:
                v = float(flavor_vector[dims.index(d)]) if dims.index(d) < len(flavor_vector) else 0.0
                b.append(f"{d.replace('_',' ').title()}: {v:.2f}")
    ing = top_ingredients_for_drink(drink_row, k=4)
    if ing: b.append("Key ingredients: " + ", ".join(ing))
    return b or ["Broadly matches your filters and preferences."]


# ── Render: hero, nav, back ──────────────────────────────
def render_hero(title: str, subtitle: str, chips: List[str]) -> None:
    chip_html = "".join([f"<div class='hero-chip'>{c}</div>" for c in chips])
    st.markdown(f"""
<div class="hero-wrap">
  <div class="floaty s1">☕</div><div class="floaty s2">🧋</div>
  <div class="floaty s3">🍵</div><div class="floaty s4">🥤</div>
  <div class="floaty s5">🍫</div>
  <div class="hero-title">{title}</div>
  <div class="hero-sub">{subtitle}</div>
  <div class="hero-chiprow">{chip_html}</div>
</div>
""", unsafe_allow_html=True)
    st.write("")


def goto(page_key: str) -> None:
    st.session_state.active_page = page_key
    st.rerun()


def top_back_button() -> None:
    st.markdown("<div class='backhome'>", unsafe_allow_html=True)
    if st.button("← Home", key=f"back_{st.session_state.get('active_page','x')}"):
        goto("home")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Reco card ────────────────────────────────────────────
def show_reco_card(drink_row, sugar_pref, caffeine_pref, flavor_focus,
                   score=None, score_text=None, score_flavor=None,
                   user_pref_vec=None, context="default", idx=0) -> None:
    name     = str(drink_row.get("name",""))
    drink_id = int(drink_row.get("drink_id",-1))
    fv       = _cached_flavor_vector(drink_id)
    left, right = st.columns([2, 2])
    with left:
        st.subheader(name)
        if int(drink_row.get("_is_suggestion",0)) == 1:
            st.caption("Suggestion (closest match outside one or more filters)")
        _format_meta(drink_row)
        st.markdown("**Why this matches**")
        temp_pl = "any"
        if context == "ai_pref":  temp_pl = st.session_state.get("ai_pref_meta",{}).get("temp_pref","any")
        elif context == "ai_text": temp_pl = st.session_state.get("ai_text_meta",{}).get("temp_b","any")
        for b in _why_bullets(drink_row, sugar_pref, caffeine_pref, temp_pl, flavor_focus,
                               score, score_text, score_flavor, fv):
            st.write(f"• {b}")
        is_fav = drink_id in st.session_state.get("favorites", set())
        if st.button("❤️ Remove" if is_fav else "🤍 Favourite", key=f"fav_{context}_{drink_id}_{idx}"):
            toggle_favorite(drink_id)
    with right:
        _r, _mid, _r2 = st.columns([0.5, 2.5, 0.5])
        with _mid: st.pyplot(radar_chart(fv, user_vec=user_pref_vec), clear_figure=True, use_container_width=False)


# ── Interactive Recipe Card ──────────────────────────────
def render_recipe_card(out: dict) -> None:
    """Render fusion result as a human-readable interactive recipe."""
    name   = out.get("name", "Fusion Drink")
    desc   = out.get("description", "")
    fv     = np.array(out.get("flavor_vector", np.zeros(len(FLAVOR_DIMS))), dtype=float)
    dims   = list(FLAVOR_DIMS)
    top3   = [dims[i].replace("_"," ").title() for i in np.argsort(-fv)[:3]]
    names  = out.get("ingredient_names", [])
    confs  = [float(c) for c in out.get("ingredient_confidence", [])]

    # Sort by confidence descending
    paired = sorted(zip(names, confs), key=lambda x: -x[1])
    primary   = [(n, c) for n, c in paired if c >= 0.10]
    secondary = [(n, c) for n, c in paired if c < 0.10]

    # Meta pills
    meta_pills = "".join([
        f"<span class='recipe-meta-pill'>🌿 {t}</span>" for t in top3
    ])
    meta_pills += f"<span class='recipe-meta-pill'>🧪 {len(paired)} ingredients</span>"

    # Ingredient rows HTML
    def _ing_rows(items):
        html = ""
        for n, c in items:
            pct = int(c * 100)
            bar_w = max(4, min(100, int(c * 120)))
            html += f"""
<div class="ingredient-row">
  <span class="ingredient-name">{n.title()}</span>
  <div style="display:flex;align-items:center;gap:8px;">
    <div class="conf-bar-wrap"><div class="conf-bar" style="width:{bar_w}px;"></div></div>
    <span class="ingredient-conf">{pct}% conf</span>
  </div>
</div>"""
        return html

    # Recipe steps (inferred from top ingredients)
    def _steps(primary_items):
        steps = []
        if primary_items:
            base = primary_items[0][0].title()
            steps.append(f"Start with your base: add <strong>{base}</strong> to a chilled glass or blender.")
        if len(primary_items) > 1:
            mids = ", ".join([n.title() for n, _ in primary_items[1:3]])
            steps.append(f"Layer in your core flavours: <strong>{mids}</strong>. Stir or blend gently.")
        if len(primary_items) > 3:
            accents = ", ".join([n.title() for n, _ in primary_items[3:]])
            steps.append(f"Add the finishing touches: <strong>{accents}</strong> for depth and complexity.")
        steps.append("Taste and adjust sweetness or intensity to your preference.")
        steps.append("Serve immediately over ice, or chilled. Garnish as desired. Enjoy! ☕")
        return steps

    steps_html = ""
    for i, s in enumerate(_steps(primary), 1):
        steps_html += f"""
<div class="step-row">
  <div class="step-num">{i}</div>
  <div class="step-text">{s}</div>
</div>"""

    flavor_pills = "".join([f"<span class='flavor-note-pill'>✦ {t}</span>" for t in top3])

    st.markdown(f"""
<div class="recipe-card">
  <p class="recipe-title">🧪 {name}</p>
  <p class="recipe-desc">{desc}</p>
  <div class="recipe-meta-row">{meta_pills}</div>

  <div class="recipe-section-label">✦ Dominant Flavour Notes</div>
  <div style="margin-bottom:18px;">{flavor_pills}</div>

  <div class="recipe-section-label">🧂 Primary Ingredients</div>
  {_ing_rows(primary)}

  {"<div class='recipe-section-label'>🌿 Supporting Ingredients</div>" + _ing_rows(secondary) if secondary else ""}

  <div class="recipe-section-label">📋 How to Make It</div>
  {steps_html}
</div>
""", unsafe_allow_html=True)

    # Interactive: expandable radar chart
    with st.expander("📊 View Flavour Radar", expanded=False):
        st.pyplot(radar_chart(fv), clear_figure=True, use_container_width=False)


# ── Nav tile ─────────────────────────────────────────────
def _nav_tile(col, icon, title, subtitle, tags, key, page,
              delay_class, watermark="", card_style="", title_color="", tag_style="") -> None:
    tag_html   = "".join([f"<span class='navtile-tag' style='{tag_style}'>{t}</span>" for t in tags])
    title_attr = f"style='color:{title_color};'" if title_color else ""
    with col:
        st.markdown(f"""
<div class="navtile-outer {delay_class}">
  <div class="navtile-card" data-emoji="{watermark or icon}" style="{card_style}">
    <div class="navtile-top">
      <span class="navtile-icon">{icon}</span>
      <p class="navtile-title" {title_attr}>{title}</p>
      <p class="navtile-sub">{subtitle}</p>
    </div>
    <div class="navtile-tags">{tag_html}</div>
    <div class="navtile-btn-wrap">
""", unsafe_allow_html=True)
        if st.button(f"Open {title} →", key=key, use_container_width=True):
            goto(page)
        st.markdown("</div></div></div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# HOME NAVIGATION
# ════════════════════════════════════════════════════════════
def render_home_navigation() -> None:
    c1, c2, c3 = st.columns(3, gap="large")
    _nav_tile(c1,"☕","Browse Menu","Explore and filter the full drink catalogue.",
              ["🧭 Filters","🔎 Search","📋 Full list"],"nav_browse","browse","tile-d1")
    _nav_tile(c2,"🧠","AI Recommender","Tell us your mood — ranked picks with explanations.",
              ["🎚️ Preferences","💬 Free-text","✅ Explainable"],"nav_ai","ai","tile-d2")
    _nav_tile(c3,"❤️","My Favorites","Saved drinks and personal flavour profile.",
              ["📊 Flavour profile","⬇️ Export","⭐ Saved list"],"nav_favorites","favorites","tile-d3")
    st.write("")
    c4, c5 = st.columns(2, gap="large")
    _nav_tile(c4,"🎲","Mystery Roller","One tap — a surprise drink just for you.",
              ["🎲 Instant roll","❤️ Favourite it","📊 View flavour"],"nav_mystery","mystery","tile-d4")
    _nav_tile(c5,"🧪","Fusion Lab","Blend two drinks into a brand-new creation.",
              ["🎚️ Blend ratio","🧾 Recipe card","💾 Save"],"nav_fusion","fusion","tile-d5")
    st.write("")
    c6, c7 = st.columns(2, gap="large")
    _nav_tile(c6,"📂","Generated Drinks","Review, compare, and download your saved fusions.",
              ["🗂️ Saved fusions","📈 Flavour chart","🗑️ Manage"],"nav_generated","generated","tile-d6")
    _nav_tile(c7,"✨","Today's Special","Hand-picked daily highlight just for you.",
              ["🌟 Daily pick","☕ Curated","🍂 Seasonal"],"nav_today_special","browse","tile-d7",
              card_style="""background:
                radial-gradient(ellipse 420px 240px at 0% 0%,rgba(255,200,70,.52),transparent 58%),
                radial-gradient(ellipse 360px 220px at 100% 0%,rgba(255,232,155,.60),transparent 58%),
                linear-gradient(150deg,rgba(255,249,225,.98) 0%,rgba(255,238,190,.92) 100%);
                border-color:rgba(212,137,26,.30);""",
              title_color="#7a3e00",
              tag_style="background:rgba(245,166,35,.20);border-color:rgba(245,166,35,.42);color:#7a3e00;")


# ════════════════════════════════════════════════════════════
# SIDEBAR  (profile + nav)
# ════════════════════════════════════════════════════════════
def render_sidebar() -> None:
    with st.sidebar:
        user = st.session_state.get("current_user")
        display = st.session_state.get("user_display", "Guest")
        initials = (display[0].upper() if display else "G")

        if user and user != "__guest__":
            st.markdown(f"""
<div class="profile-badge">
  <div class="profile-avatar">{initials}</div>
  <div>
    <div class="profile-name">{display}</div>
    <div class="profile-sub">❤️ {len(load_favorites_ids())} saved</div>
  </div>
</div>
""", unsafe_allow_html=True)
            st.write("")

            # Interests editor
            with st.expander("🎨 My Taste Interests", expanded=False):
                all_dims = list(FLAVOR_DIMS)
                current  = st.session_state.get("user_interests", [])
                selected = st.multiselect("Flavours I love", all_dims, default=current,
                                          key="sidebar_interests")
                if selected != current:
                    st.session_state.user_interests = selected
                    save_user_profile()

            if st.button("Sign Out", key="btn_signout"):
                logout_user()
                st.rerun()
        else:
            st.markdown(f"""
<div class="profile-badge">
  <div class="profile-avatar">👤</div>
  <div>
    <div class="profile-name">Guest</div>
    <div class="profile-sub">Sign in to save data</div>
  </div>
</div>
""", unsafe_allow_html=True)
            st.write("")
            if st.button("Sign In / Register", key="btn_goto_login"):
                st.session_state.active_page = "auth"
                st.rerun()

        st.divider()
        st.caption("Navigate")
        pages = [("🏠 Home","home"),("☕ Browse","browse"),("🎲 Mystery","mystery"),
                 ("🧠 AI Pick","ai"),("❤️ Favourites","favorites"),
                 ("🧪 Fusion","fusion"),("📂 Generated","generated")]
        for label, key in pages:
            if st.button(label, key=f"sb_{key}", use_container_width=True):
                goto(key)


# ════════════════════════════════════════════════════════════
# PAGE: MYSTERY ROLL  (pure roll — no menu table)
# ════════════════════════════════════════════════════════════
def page_mystery() -> None:
    render_hero(
        "Mystery Roller 🎲",
        "Feeling adventurous? Let the café choose for you.",
        ["🎲 Random pick", "❤️ Favourite it", "📊 Flavour radar"],
    )

    st.markdown("""
<div class="mystery-hero">
  <div style="font-size:64px;">🎲</div>
  <div class="mystery-hero-title">What'll it be today?</div>
  <div class="mystery-hero-sub">Hit Roll and let the café surprise you.</div>
</div>
""", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🎲 Roll a drink", type="primary", key="mystery_roll_btn", use_container_width=True):
            if drinks_df is not None and not drinks_df.empty:
                r = drinks_df.sample(1).iloc[0]
                st.session_state["_random_pick_id"]   = int(r.get("drink_id", -1))
                st.session_state["_random_pick_name"] = str(r.get("name", r.get("drink_name","")))
                try: st.toast("✨ Drink rolled!")
                except Exception: pass

    picked_id   = st.session_state.get("_random_pick_id")
    picked_name = st.session_state.get("_random_pick_name","")

    if picked_id and picked_id != -1:
        row = drinks_df[drinks_df["drink_id"] == picked_id]
        if not row.empty:
            r = row.iloc[0]
            st.markdown(f"""
<div class="roll-result">
  <div style="font-size:11px;font-weight:700;letter-spacing:.8px;text-transform:uppercase;
    color:var(--caramel);margin-bottom:4px;">You rolled</div>
  <div class="roll-drink-name">{picked_name}</div>
</div>
""", unsafe_allow_html=True)
            st.write("")
            meta_col, chart_col = st.columns([2, 1])
            with meta_col:
                _format_meta(r)
                is_fav = int(picked_id) in st.session_state.get("favorites", set())
                if st.button("❤️ Remove favourite" if is_fav else "🤍 Add to favourites",
                             key="mystery_fav_btn"):
                    toggle_favorite(int(picked_id))
                ings = top_ingredients_for_drink(r, k=6)
                if ings:
                    st.markdown("**Key ingredients:** " + ", ".join(ings))
            with chart_col:
                fv = _cached_flavor_vector(int(picked_id))
                st.pyplot(radar_chart(fv), clear_figure=True, use_container_width=False)


# ════════════════════════════════════════════════════════════
# PAGE: BROWSE MENU  (filters + search — no mystery roller)
# ════════════════════════════════════════════════════════════
def page_menu() -> None:
    render_hero(
        "Browse Menu ☕",
        "Filter, search, and discover drinks from the full catalogue.",
        ["🧭 Filters", "🔎 Search", f"❤️ {len(load_favorites_ids())} saved"],
    )

    st.markdown("""
<div class="section-card">
  <div class="section-card-title">🧭 Filter the Menu</div>
  <div class="section-card-sub">Use the dropdowns below to narrow down drinks instantly.</div>
</div>
""", unsafe_allow_html=True)
    st.write("")

    t_vals = safe_unique_values(drinks_df, "type")
    s_vals = safe_unique_values(drinks_df, "sugar_level")
    c_vals = safe_unique_values(drinks_df, "caffeine_level")

    f1, f2, f3, f4 = st.columns([1,1,1,.8], gap="large")
    with f1: drink_type     = st.selectbox("Drink type",      t_vals, index=0, key="menu_type")
    with f2: sugar_level    = st.selectbox("Sugar level",     s_vals, index=0, key="menu_sugar")
    with f3: caffeine_level = st.selectbox("Caffeine level",  c_vals, index=0, key="menu_caf")
    with f4:
        st.write(""); st.write("")
        if st.button("Reset", key="menu_reset"):
            for k in ["menu_type","menu_sugar","menu_caf"]: st.session_state[k] = "any"
            st.rerun()

    out_df = drinks_df.copy()
    def nc(df, col): return df[col].fillna("").astype(str).str.lower().str.strip()
    if drink_type     != "any" and "type"           in out_df.columns: out_df = out_df[nc(out_df,"type")           == drink_type]
    if sugar_level    != "any" and "sugar_level"    in out_df.columns: out_df = out_df[nc(out_df,"sugar_level")    == sugar_level]
    if caffeine_level != "any" and "caffeine_level" in out_df.columns: out_df = out_df[nc(out_df,"caffeine_level") == caffeine_level]

    st.divider()
    st.markdown("### Menu items")
    name_col = get_name_col(out_df)
    search   = st.text_input("Search", value="", key="menu_search", placeholder="e.g. matcha, iced, caramel…").strip().lower()
    view_df  = out_df.copy()
    if search:
        txt = (view_df.get(name_col, pd.Series([""]* len(view_df))).astype(str).fillna("") + " " +
               view_df.get("tags", pd.Series([""]*len(view_df))).astype(str).fillna("")).str.lower()
        view_df = view_df[txt.str.contains(search, na=False)]

    st.caption(f"Showing {len(view_df)} drinks")
    show_cols = [c for c in ["drink_id", name_col,"type","temperature","sugar_level","caffeine_level","tags"] if c in view_df.columns]
    st.dataframe(view_df[show_cols].reset_index(drop=True), use_container_width=True, height=260)

    st.divider()
    st.markdown("### Quick picks")
    top_n = min(6, len(view_df))
    if top_n == 0:
        st.warning("No drinks match those filters.")
    else:
        grid = st.columns(3)
        for i in range(top_n):
            r   = view_df.iloc[i]
            did = int(r.get("drink_id", -1))
            with grid[i % 3]:
                st.markdown(f"""
<div class="drink-card">
  <div class="drink-card-name">{str(r.get(name_col,"")).strip()}</div>
  <div class="drink-card-meta">{str(r.get("type","")).strip()} | {str(r.get("temperature","")).strip()}</div>
</div>
""", unsafe_allow_html=True)
                is_fav = did in st.session_state.get("favorites", set())
                if st.button(("❤️" if is_fav else "🤍") + " Favourite", key=f"fav_quick_{did}_{i}"):
                    toggle_favorite(did)

    st.divider()
    with st.expander("🔍 Drink detail view", expanded=False):
        if view_df.empty:
            st.info("No drinks available.")
            return
        names = sorted(set(view_df[name_col].fillna("").astype(str).str.strip().replace("nan","").tolist()))
        names = [n for n in names if n]
        sig   = (drink_type, sugar_level, caffeine_level, len(names))
        if st.session_state.get("_menu_filter_sig") != sig:
            st.session_state["_menu_filter_sig"]    = sig
            st.session_state["_menu_selected_name"] = ""
        choice = st.selectbox("Select drink", [""] + names, index=0, key="_menu_selected_name")
        if choice:
            picked = view_df[view_df[name_col].fillna("").astype(str).str.strip() == choice]
            if picked.empty: st.warning("Drink not found."); return
            row = picked.iloc[0]; did = int(row.get("drink_id",-1))
            _format_meta(row)
            st.pyplot(radar_chart(_cached_flavor_vector(did)), clear_figure=True, use_container_width=False)
            is_fav = did in st.session_state.get("favorites", set())
            if st.button("❤️ Remove" if is_fav else "🤍 Favourite", key=f"fav_menu_pick_{did}"):
                toggle_favorite(did)


# ════════════════════════════════════════════════════════════
# PAGE: AI RECOMMENDER
# ════════════════════════════════════════════════════════════
def page_ai_recommender() -> None:
    render_hero("AI Drink Recommender 🧠",
                "Pick preferences or describe your mood. Get ranked, explainable picks.",
                ["🎚️ Preferences","💬 Free-text","✅ Explained"])

    dims = list(FLAVOR_DIMS)
    tabs = st.tabs(["Start","Preferences","Free-text"])

    with tabs[0]:
        st.markdown("**Preferences** — choose sugar, caffeine, temperature and flavour focus.\n\n"
                    "**Free-text** — describe what you feel like in plain language.")
        st.caption("Tip: Set taste interests in the sidebar to personalise results.")

    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        with c1: sugar_pref    = st.selectbox("Sugar",       SUGAR_LEVELS,    0, key="sugar_pref_a")
        with c2: caffeine_pref = st.selectbox("Caffeine",    CAFFEINE_LEVELS, 0, key="caf_pref_a")
        with c3: temp_pref     = st.selectbox("Temperature", TEMP_LEVELS,     0, key="temp_pref_a")

        # Auto-seed from user interests
        interests = st.session_state.get("user_interests", [])
        default_focus = [d for d in interests if d in dims]
        flavor_focus  = st.multiselect("Flavour focus", dims, default=default_focus, key="focus_a")
        top_k         = st.slider("Results", 3, 12, 8, key="top_k_a")
        user_pref_vec = {d: (1.0 if d in flavor_focus else 0.0) for d in dims} if flavor_focus else None

        if st.button("Clear", key="clear_ai_pref"):
            st.session_state.ai_pref_results = []; st.session_state.ai_pref_meta = {}

        if st.button("Recommend", key="btn_pref", type="primary"):
            cand = drinks_df.copy()
            for col, val in [("sugar_level",sugar_pref),("caffeine_level",caffeine_pref),("temperature",temp_pref)]:
                if val != "any" and col in cand.columns:
                    cand = cand[cand[col].fillna("").astype(str).str.lower().str.strip() == val]
            full_rank = drinks_df.sort_values("popularity_score",ascending=False) if "popularity_score" in drinks_df.columns else drinks_df
            if flavor_focus and not cand.empty:
                tgt = np.array([user_pref_vec[d] for d in dims], dtype=float)
                tgt = tgt / (np.linalg.norm(tgt) + 1e-9)
                mat = cand[dims].to_numpy(dtype=float) if all(d in cand.columns for d in dims) else np.zeros((len(cand),len(dims)))
                sim = np.clip((mat @ tgt) / (np.linalg.norm(mat,axis=1)*(np.linalg.norm(tgt)+1e-9)+1e-9), 0, 1)
                cand2 = cand.copy(); cand2["score"] = cand2["score_flavor"] = sim
                cand2 = cand2.sort_values("score",ascending=False)
                filled = smooth_fill_results(cand2, full_rank, top_k)
            else:
                if "popularity_score" in cand.columns: cand = cand.sort_values("popularity_score",ascending=False)
                filled = smooth_fill_results(cand, full_rank, top_k)
            filled = dedupe_by_drink_id(filled)
            st.session_state.ai_pref_results = filled.to_dict("records")
            st.session_state.ai_pref_meta    = {"sugar_pref":sugar_pref,"caffeine_pref":caffeine_pref,
                "temp_pref":temp_pref,"flavor_focus":flavor_focus,"user_pref_vec":user_pref_vec}

        if st.session_state.ai_pref_results:
            meta = st.session_state.ai_pref_meta
            st.markdown("### Results")
            for i, rec in enumerate(st.session_state.ai_pref_results):
                r = pd.Series(rec)
                show_reco_card(r, meta.get("sugar_pref","any"), meta.get("caffeine_pref","any"),
                    meta.get("flavor_focus",[]),
                    score=float(r["score"]) if "score" in r and r["score"] is not None else None,
                    score_flavor=float(r["score_flavor"]) if "score_flavor" in r and r["score_flavor"] is not None else None,
                    user_pref_vec=meta.get("user_pref_vec"), context="ai_pref", idx=i)

    with tabs[2]:
        st.caption("e.g. iced caramel, low sugar, creamy, not bitter")
        free_text = st.text_input("Describe your drink", key="free_text")
        top_k_b   = st.slider("Results", 3, 12, 8, key="top_k_b")

        if st.button("Clear", key="clear_ai_text"):
            st.session_state.ai_text_results = []; st.session_state.ai_text_meta = {}

        if st.button("Recommend", key="btn_text", type="primary"):
            prefs    = parse_query_to_preferences(free_text or "")
            sugar_b  = str(prefs.get("sugar_level","any"))
            caf_b    = str(prefs.get("caffeine_level","any"))
            temp_b   = str(prefs.get("temperature","any"))
            focus_b  = list(prefs.get("flavor_focus",[]) or [])
            base_df  = _apply_excludes(drinks_df.copy(), list(prefs.get("exclude_terms",[]) or []))
            h        = _df_hash(base_df)
            model    = _cached_hybrid_model(h, base_df)
            recs     = hybrid_recommend(drinks_df=base_df, model=model, user_query=free_text or "",
                         drink_type="any", temperature=temp_b, sugar_level=sugar_b,
                         caffeine_level=caf_b, flavor_focus=focus_b, top_k=top_k_b)
            full_rank = base_df.sort_values("popularity_score",ascending=False) if "popularity_score" in base_df.columns else base_df
            filled    = dedupe_by_drink_id(smooth_fill_results(recs, full_rank, top_k_b))
            st.session_state.ai_text_results = filled.to_dict("records")
            st.session_state.ai_text_meta    = {"sugar_b":sugar_b,"caf_b":caf_b,"temp_b":temp_b,
                "focus_b":focus_b,"user_pref_vec":{d:(1. if d in focus_b else 0.) for d in dims} if focus_b else None}

        if st.session_state.ai_text_results:
            meta = st.session_state.ai_text_meta
            st.markdown("### Results")
            for i, rec in enumerate(st.session_state.ai_text_results):
                r = pd.Series(rec)
                show_reco_card(r, meta.get("sugar_b","any"), meta.get("caf_b","any"),
                    meta.get("focus_b",[]),
                    score=float(r["score"]) if "score" in r and r["score"] is not None else None,
                    score_text=float(r["score_text"]) if "score_text" in r and r["score_text"] is not None else None,
                    score_flavor=float(r["score_flavor"]) if "score_flavor" in r and r["score_flavor"] is not None else None,
                    user_pref_vec=meta.get("user_pref_vec"), context="ai_text", idx=i)


# ════════════════════════════════════════════════════════════
# PAGE: FAVORITES
# ════════════════════════════════════════════════════════════
def page_favorites() -> None:
    render_hero("My Favourites ❤️",
                "Your saved drinks, personal flavour profile, and taste summary.",
                ["❤️ Saved","📊 Profile","⬇️ Export"])

    if st.session_state.get("current_user") in (None, "__guest__"):
        st.info("Sign in to save and sync your favourites across sessions.")
        if st.button("Sign In", key="fav_signin_btn"): goto("auth")
        return

    if st.session_state.get("_fav_toast"):
        st.success("Favourites updated."); st.session_state._fav_toast = None

    fav_ids = load_favorites_ids()
    if not fav_ids:
        st.info("No favourites yet — browse or roll a drink and hit ❤️")
        return

    fav_df = drinks_df[drinks_df["drink_id"].isin(fav_ids)].copy() if "drink_id" in drinks_df.columns else pd.DataFrame()
    if fav_df.empty:
        st.info("Favourites could not be matched to menu items."); return

    st.markdown("### Your saved drinks")
    show_cols = [c for c in ["drink_id","name","type","temperature","sugar_level","caffeine_level","tags"] if c in fav_df.columns]
    st.dataframe(fav_df[show_cols].reset_index(drop=True), use_container_width=True)
    st.download_button("⬇️ Export as CSV", fav_df[show_cols].to_csv(index=False).encode(),
                       "flavorgen_favourites.csv", "text/csv", key="dl_favs_csv")
    st.divider()

    vecs = [_cached_flavor_vector(int(r["drink_id"])) for _, r in fav_df.iterrows()]
    if vecs:
        avg_vec = np.mean(np.vstack(vecs), axis=0)
        st.markdown("### Your flavour profile")
        st.pyplot(flavor_bar_chart(avg_vec), clear_figure=True, use_container_width=False)

    st.divider()
    for i, (_, r) in enumerate(fav_df.iterrows()):
        show_reco_card(r, "any","any",[], context="favorites", idx=i)
    st.divider()
    if st.button("🗑️ Clear all favourites", key="btn_clear_favs"):
        clear_favorites(); st.success("Cleared.")


# ════════════════════════════════════════════════════════════
# PAGE: FUSION LAB
# ════════════════════════════════════════════════════════════
def page_fusion_lab() -> None:
    render_hero("Fusion Lab 🧪",
                "Blend two drinks into a brand-new creation — get an interactive recipe card.",
                ["🧪 Blend","🎚️ Ratio","📋 Recipe","💾 Save"])

    models_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")),"models","fusion")
    meta_path  = os.path.join(models_dir,"fusion_meta.json")

    if not os.path.exists(meta_path):
        st.warning("Fusion model not trained yet. Run: python train_fusion.py"); return
    if "drink_id" not in drinks_df.columns:
        st.error("drinks_df is missing drink_id."); return

    meta    = _load_fusion_meta(meta_path)
    in_dim  = int(meta.get("in_dim",0)); out_dim = int(meta.get("out_dim",0))
    if in_dim <= 0 or out_dim <= 0:
        st.error("fusion_meta.json corrupt. Re-run train_fusion.py."); return

    art = _load_fusion_artifacts_cached(models_dir, in_dim=in_dim, out_dim=out_dim)
    dfv = drinks_df.copy()
    for col in ["name","temperature","sugar_level","caffeine_level"]:
        if col in dfv.columns: dfv[col] = dfv[col].fillna("").astype(str).str.strip()

    base_names = sorted(dfv["name"].dropna().astype(str).unique().tolist())
    if not base_names: st.warning("No drink names available."); return

    cA, cB = st.columns(2)

    def _variant_df(name):
        d = dfv[dfv["name"].astype(str) == str(name)].copy()
        d["_vl"] = (d.get("temperature","").astype(str)
                    + " | sugar: " + d.get("sugar_level","").astype(str)
                    + " | caffeine: " + d.get("caffeine_level","").astype(str))
        return d.drop_duplicates(subset=["_vl"], keep="first")

    with cA:
        base_a = st.selectbox("Drink A", base_names, 0, key="fusion_base_a")
        avdf   = _variant_df(base_a)
        avl    = avdf["_vl"].tolist()
        if not avl: st.warning("No variants for Drink A."); return
        va = st.selectbox("Variant A", avl, 0, key="fusion_variant_a")

    with cB:
        base_b = st.selectbox("Drink B", base_names, min(1,len(base_names)-1), key="fusion_base_b")
        bvdf   = _variant_df(base_b)
        bvl    = bvdf["_vl"].tolist()
        if not bvl: st.warning("No variants for Drink B."); return
        vb = st.selectbox("Variant B", bvl, 0, key="fusion_variant_b")

    ida = int(avdf[avdf["_vl"]==va]["drink_id"].iloc[0])
    idb = int(bvdf[bvdf["_vl"]==vb]["drink_id"].iloc[0])

    alpha = st.slider("Blend ratio  (← more A  |  more B →)", 0.0, 1.0, 0.5, 0.05, key="fusion_alpha")
    top_k = st.slider("Ingredients to generate", 5, 18, 10, 1, key="fusion_topk")

    cg, cs = st.columns(2)
    with cg:
        if st.button("🧪 Generate Recipe", key="btn_fusion_generate", type="primary"):
            out = predict_fusion(art=art, drinks_df=drinks_df, ingredients_df=ingredients_df,
                                 drink_id_a=ida, drink_id_b=idb, alpha=float(alpha), top_k=int(top_k))
            st.session_state.fusion_last = out

    with cs:
        if st.button("💾 Save this fusion", key="btn_fusion_save"):
            if not st.session_state.get("fusion_last"):
                st.warning("Generate first.")
            else:
                try:
                    existing = json.loads(GEN_PATH.read_text()) if GEN_PATH.exists() else []
                    existing.append(st.session_state.fusion_last)
                    GEN_PATH.write_text(json.dumps(existing, indent=2))
                    st.success("Saved to Generated Drinks!")
                except Exception as e:
                    st.error(f"Save failed: {e}")

    out = st.session_state.get("fusion_last")
    if out:
        st.write("")
        render_recipe_card(out)


# ════════════════════════════════════════════════════════════
# PAGE: GENERATED DRINKS
# ════════════════════════════════════════════════════════════
def page_generated_drinks() -> None:
    render_hero("Generated Drinks 📂",
                "Your saved fusion creations as interactive recipe cards.",
                ["🗂️ Saved","📋 Recipe view","⬇️ Download"])

    if not GEN_PATH.exists():
        st.info("No generated drinks yet. Create one in Fusion Lab."); return
    try:
        items = json.loads(GEN_PATH.read_text()) or []
    except Exception as e:
        st.error(f"Could not read file: {e}"); return
    if not items:
        st.info("No generated drinks yet."); return

    st.download_button("⬇️ Download all (JSON)",
                       json.dumps(items, indent=2).encode(),
                       "generated_drinks.json","application/json",key="dl_gen_json")
    st.divider()

    for item in reversed(items[-50:]):
        render_recipe_card(item)

        # Favourite button
        gen_name = str(item.get("name","")).strip().lower()
        match_id = None
        if gen_name and "name" in drinks_df.columns and "drink_id" in drinks_df.columns:
            m = drinks_df[drinks_df["name"].fillna("").astype(str).str.lower().str.strip() == gen_name]
            if not m.empty: match_id = int(m.iloc[0]["drink_id"])

        col_fav, col_del, _ = st.columns([1,1,3])
        with col_fav:
            if match_id is not None:
                is_fav = match_id in st.session_state.get("favorites", set())
                if st.button("❤️ Remove" if is_fav else "🤍 Favourite",
                             key=f"fav_gen_{match_id}_{gen_name}"): toggle_favorite(match_id)
            else:
                st.caption("Not in menu — can't favourite")

        with col_del:
            del_key = f"del_{abs(hash(json.dumps(item,sort_keys=True,default=str)))}"
            if st.button("🗑️ Delete", key=del_key):
                items = [x for x in items if x != item]
                try:
                    GEN_PATH.write_text(json.dumps(items, indent=2))
                    st.success("Deleted."); st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {e}")
        st.divider()


# ════════════════════════════════════════════════════════════
# SIDEBAR (always rendered)
# ════════════════════════════════════════════════════════════
render_sidebar()


# ════════════════════════════════════════════════════════════
# ROUTER
# ════════════════════════════════════════════════════════════
# Gate: if not logged in and not on auth page, show auth
ap = st.session_state.active_page

if ap == "auth":
    page_auth()

elif st.session_state.get("current_user") is None:
    page_auth()

elif ap == "home":
    render_hero("Flavorgen Café Lab ☕",
                "A cozy café experience to browse, recommend, and generate drinks.",
                ["☕ Browse","🧠 AI","❤️ Favourites","🧪 Fusion"])
    render_home_navigation()

elif ap == "browse":
    top_back_button(); page_menu()

elif ap == "mystery":
    top_back_button(); page_mystery()

elif ap == "ai":
    top_back_button(); page_ai_recommender()

elif ap == "favorites":
    top_back_button(); page_favorites()

elif ap == "fusion":
    top_back_button(); page_fusion_lab()

elif ap == "generated":
    top_back_button(); page_generated_drinks()

else:
    st.session_state.active_page = "home"
    st.rerun()