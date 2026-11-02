"""
SEQUORA — Streamlit Frontend (v6)
=====================================
Pure UI layer over the existing, unmodified recommendation pipeline.
v6: real product names (prod_name), colour on cards, descriptions in the
detail pop-up, 300 recommendations requested so category grids fill up.
"""

from __future__ import annotations

import base64
import json
import uuid
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.catalog import classify_category  # noqa: E402  (shared with the backend)

API_URL = "http://127.0.0.1:8000"
RAW_DIR = Path("raw_data/hm")
PROCESSED_DIR = Path("processed_data")

st.set_page_config(page_title="SEQUORA", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
html { scroll-behavior: smooth; }
#MainMenu, footer, header {visibility: hidden;}
.block-container { max-width: 1240px; padding-top: 0; }

.sq-logo { font-size: 20px; font-weight: 800; letter-spacing: 2px; color: #242120; }
.sq-nav-tag { font-size:12px; font-weight:600; padding:4px 2px; }

.sq-hero {
    padding: 2.5rem 1.5rem; background: linear-gradient(180deg,#F6EDE4 0%,#FBF8F3 100%);
    border-radius: 0 0 24px 24px; margin-bottom: 1.75rem;
}
.sq-hero-eyebrow { font-size:11px; letter-spacing:2px; text-transform:uppercase; color:#A8614B; font-weight:700; margin-bottom:10px; }
.sq-hero h1 { font-size:38px; font-weight:800; color:#242120; margin:0 0 12px; letter-spacing:-1px; line-height:1.1; }
.sq-hero p { font-size:14px; color:#6b6258; margin:0 0 1rem; }
.sq-hero-tagline { font-size:12.5px; font-weight:600; color:#7B2D3E; margin:0 0 1rem; font-style:italic; }
.sq-hero-img { border-radius:14px; overflow:hidden; aspect-ratio:1/1; background:#F4EFE7; }
.sq-hero-img img { width:100%; height:100%; object-fit:cover; }

.sq-cat-icon { text-align:center; }
.sq-cat-circle { width:60px; height:60px; border-radius:50%; background:#F4EFE7; overflow:hidden; margin:0 auto 6px; border:2px solid transparent; }
.sq-cat-circle.active { border-color:#7B2D3E; }
.sq-cat-circle img { width:100%; height:100%; object-fit:cover; }
.sq-cat-label { font-size:11px; font-weight:600; color:#5c584f; }

.sq-section-label { font-size:11px; letter-spacing:1.5px; text-transform:uppercase; color:#A8614B; font-weight:700; margin:0 0 2px; }
.sq-section-title { font-size:22px; font-weight:800; color:#242120; margin:0 0 2px; }
.sq-section-sub { font-size:13px; color:#8a8072; margin:0 0 1.25rem; }

.sq-card { background:#FFFFFF; border:1px solid #ece4da; border-radius:14px; overflow:hidden;
    box-shadow:0 1px 3px rgba(36,33,32,0.05); margin-bottom:0.4rem; }
.sq-card-img-wrap { width:100%; aspect-ratio:3/4; background:#F4EFE7; position:relative; overflow:hidden; }
.sq-card-img-wrap img { width:100%; height:100%; object-fit:contain; padding:6px; }
.sq-card-placeholder { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; font-size:11px; color:#b8ad9c; }
.sq-newtag { position:absolute; top:9px; left:9px; font-size:9.5px; font-weight:700; letter-spacing:0.5px;
    text-transform:uppercase; padding:3px 9px; border-radius:6px; background:#5A7A53; color:#fff; }
.sq-card-body { padding:0.65rem 0.8rem 0.4rem; }
.sq-card-category { font-size:10px; text-transform:uppercase; letter-spacing:0.5px; color:#A8614B; font-weight:700; margin:0 0 3px; }
.sq-card-name { font-size:13.5px; font-weight:700; color:#242120; margin:0 0 4px; line-height:1.3; min-height:34px; }
.sq-card-reason { font-size:10.5px; color:#8a8072; margin:0; font-style:italic; }

div.stButton > button { border-radius:9px; border:1px solid #ece4da; background:#fff; color:#242120; font-size:12px; font-weight:600; }
div.stButton > button:hover { border-color:#7B2D3E; color:#7B2D3E; }
div.stButton > button[kind="primary"] { background:#242120; border-color:#242120; color:#fff; }

.sq-why-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:0.5rem 0 2rem; }
.sq-why-card { background:#fff; border:1px solid #ece4da; border-radius:12px; padding:0.8rem 0.6rem; text-align:center; }
.sq-why-icon { font-size:16px; color:#7B2D3E; margin-bottom:6px; }
.sq-why-title { font-size:10.5px; font-weight:700; color:#242120; margin:0 0 3px; }
.sq-why-desc { font-size:9.5px; color:#8a8072; margin:0; line-height:1.3; }

.sq-explain-strip { background:#FBEDE9; border-radius:12px; padding:0.6rem 1rem; font-size:12px; color:#7B2D3E; font-weight:600; margin-bottom:1.25rem; }

@media (max-width:1100px) { .sq-grid-5 { grid-template-columns:repeat(3,1fr) !important; } }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

defaults = {
    "session_id": str(uuid.uuid4()), "session_clicks": [], "mode": "existing",
    "category_filter": "All", "wishlist": set(), "search_query": "",
    "nav_section": "women", "jfy_count": 10, "nfy_count": 10, "trend_count": 10,
    "evolved_msg": False, "new_session_prefs": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

CATEGORIES = ["All", "Dresses", "Tops", "Trousers", "Footwear", "Outerwear", "Accessories", "Activewear", "Underwear", "Other"]

# --------------------------------------------------------------------------- #
# Data + image lookup
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def load_article_lookup() -> dict:
    article_id_map = json.loads((PROCESSED_DIR / "article_id_map.json").read_text())
    idx_to_article_id = {v: k for k, v in article_id_map.items()}
    articles = pd.read_parquet(PROCESSED_DIR / "articles_clean.parquet")
    has_price = "price" in articles.columns
    has_gender = "index_group_name" in articles.columns
    articles_lookup = articles.set_index("article_id").to_dict(orient="index")

    # Real product names and descriptions from the raw articles file
    extra = {}
    try:
        raw_articles = pd.read_csv(
            RAW_DIR / "articles.csv",
            usecols=["article_id", "prod_name", "detail_desc", "product_group_name"],
            dtype={"article_id": str},
        ).fillna("")
        extra = raw_articles.set_index("article_id").to_dict(orient="index")
    except Exception:
        pass

    lookup = {}
    for idx, article_id in idx_to_article_id.items():
        info = articles_lookup.get(article_id, {})
        raw_type = info.get("product_type_name", "Item")
        more = extra.get(str(article_id), {})
        category = classify_category(raw_type, info.get("product_group_name") or more.get("product_group_name", ""), info.get("section_name", ""))
        if category is None:
            continue  # swimwear — excluded entirely
        lookup[idx] = {
            "article_id": article_id,
            "name": more.get("prod_name") or raw_type,
            "product_type": raw_type,
            "description": more.get("detail_desc", ""),
            "category": category,
            "image_path": info.get("image_path", ""),
            "price": info.get("price") if has_price else None,
            "colour": info.get("colour_group_name", ""),
            "department": info.get("department_name", ""),
            "section": info.get("section_name", ""),
            "garment_group": info.get("garment_group_name", ""),
            "gender": info.get("index_group_name", "") if has_gender else "",
        }
    return lookup, has_gender


@st.cache_data(show_spinner=False)
def get_product_image(image_path: str) -> Optional[str]:
    if not image_path:
        return None
    full_path = RAW_DIR / image_path
    if not full_path.exists():
        return None
    try:
        img = Image.open(full_path).convert("RGB")
        img.thumbnail((320, 420))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_cold_start_items(limit: int = 300) -> list:
    path = PROCESSED_DIR / "cold_start_items.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())[:limit]


@st.cache_data(show_spinner=False)
def load_personas() -> list:
    path = PROCESSED_DIR / "personas.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@st.cache_data(show_spinner=False)
def category_icon_images(_lookup: dict) -> dict:
    icons = {}
    for cat in CATEGORIES:
        if cat == "All":
            continue
        for info in _lookup.values():
            if info["category"] == cat and info.get("image_path"):
                uri = get_product_image(info["image_path"])
                if uri:
                    icons[cat] = uri
                    break
    return icons


def call_api(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=5, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Can't reach the backend. Make sure uvicorn is running."
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


SOURCE_LABELS = {
    "sasrec": "Because of your recent activity",
    "two_tower": "Matches your style",
    "popularity_fallback": "Popular this week",
}
SOURCE_SIGNAL = {
    "sasrec": "Sequential preference",
    "two_tower": "Visual + content similarity",
    "popularity_fallback": "Category match",
}

health, err = call_api("GET", "/health")
if err:
    st.error(err)
    st.stop()

lookup, has_gender_field = load_article_lookup()
icons = category_icon_images(lookup)

# --------------------------------------------------------------------------- #
# Product detail modal
# --------------------------------------------------------------------------- #

@st.dialog("Product details")
def show_product_detail(item_id: int, reason: str, source: str, section: str) -> None:
    info = lookup.get(item_id, {})
    uri = get_product_image(info.get("image_path", ""))
    if uri:
        st.image(uri, use_container_width=True)

    st.markdown(f"**{info.get('category', 'Other')}**")
    st.subheader(info.get("name", f"Item {item_id}"))
    st.caption(f"Article ID: {info.get('article_id', item_id)}")
    st.caption(f"Type: {info.get('product_type', '—')}")
    if info.get("description"):
        st.write(info["description"])

    st.markdown("**Why SEQUORA recommended this**")
    st.info(f"{reason}. Signal: {SOURCE_SIGNAL.get(source, 'Catalogue match')}.")

    meta_cols = st.columns(2)
    with meta_cols[0]:
        st.caption(f"Department: {info.get('department', '—')}")
        st.caption(f"Section: {info.get('section', '—')}")
    with meta_cols[1]:
        st.caption(f"Garment group: {info.get('garment_group', '—')}")
        st.caption(f"Colour: {info.get('colour', '—')}")

    wishlisted = item_id in st.session_state.wishlist
    btn_cols = st.columns(2)
    with btn_cols[0]:
        label = "Remove from wishlist" if wishlisted else "♡ Add to wishlist"
        if st.button(label, key=f"detail_wish_{item_id}", use_container_width=True):
            _toggle_wishlist(item_id)
            st.rerun()
    with btn_cols[1]:
        if st.button("View similar items", key=f"detail_similar_{item_id}", use_container_width=True):
            st.session_state.session_clicks.append(item_id)
            st.session_state.evolved_msg = True
            st.rerun()


def _toggle_wishlist(item_id: int) -> None:
    if item_id in st.session_state.wishlist:
        st.session_state.wishlist.discard(item_id)
    else:
        st.session_state.wishlist.add(item_id)
        st.session_state.session_clicks.append(item_id)
        st.session_state.evolved_msg = True
        call_api(
            "POST", "/events",
            params={"session_id": st.session_state.session_id},
            json={"user_id": None, "item_id": item_id, "event_type": "click"},
        )

# --------------------------------------------------------------------------- #
# Render helpers
# --------------------------------------------------------------------------- #

def render_card(item_id: int, reason: str, source: str, section: str, position: int, is_new: bool = False) -> None:
    unique_key = f"{section}_{item_id}_{position}"
    info = lookup.get(item_id, {})
    name = escape(str(info.get("name", f"Item {item_id}")))
    category = escape(str(info.get("category", "Other")))
    colour = escape(str(info.get("colour", "") or ""))
    uri = get_product_image(info.get("image_path", ""))

    img_html = f'<img src="{uri}" alt="{name}" />' if uri else '<div class="sq-card-placeholder">image unavailable</div>'
    new_tag_html = '<span class="sq-newtag">New</span>' if is_new else ""
    sub = f"{category} · {colour}" if colour else category

    st.markdown(
        f'<div class="sq-card"><div class="sq-card-img-wrap">{img_html}{new_tag_html}</div>'
        f'<div class="sq-card-body"><p class="sq-card-category">{sub}</p>'
        f'<p class="sq-card-name">{name}</p><p class="sq-card-reason">{reason}</p></div></div>',
        unsafe_allow_html=True,
    )
    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("View", key=f"view_{unique_key}", use_container_width=True):
            show_product_detail(item_id, reason, source, section)
    with btn_cols[1]:
        wishlisted = item_id in st.session_state.wishlist
        heart = "♥" if wishlisted else "♡"
        if st.button(heart, key=f"wishlist_{item_id}_{section}_{position}", use_container_width=True):
            _toggle_wishlist(item_id)
            st.rerun()


def render_grid(items: list, reason_fn, source_fn, section: str, is_new: bool = False) -> None:
    cols = st.columns(5)
    for i, item in enumerate(items):
        item_id = item["item_id"] if isinstance(item, dict) else item
        with cols[i % 5]:
            render_card(item_id, reason_fn(item), source_fn(item), section=section, position=i, is_new=is_new)


def render_history(item_ids: list) -> None:
    shown = [i for i in item_ids if i in lookup][-6:]
    if not shown:
        return
    st.markdown(
        '<p class="sq-section-label">Input</p><p class="sq-section-title">Recent purchases</p>'
        '<p class="sq-section-sub">What this customer bought during the training period (oldest to newest). '
        'The recommendations below are predicted from this history.</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(6)
    for col, item_id in zip(cols, shown):
        info = lookup[item_id]
        uri = get_product_image(info.get("image_path", ""))
        with col:
            if uri:
                st.image(uri, use_container_width=True)
            st.caption(f"{info.get('name', 'Item')} · {info.get('colour', '')}")


def apply_gender_filter(item_ids: list) -> list:
    if not has_gender_field or st.session_state.nav_section not in ("women", "men"):
        return item_ids
    target = "Ladieswear" if st.session_state.nav_section == "women" else "Menswear"
    return [i for i in item_ids if lookup.get(i, {}).get("gender") == target] or item_ids


def filter_by_category(recs: list) -> list:
    if st.session_state.category_filter == "All":
        return recs
    return [r for r in recs if lookup.get(r["item_id"] if isinstance(r, dict) else r, {}).get("category") == st.session_state.category_filter]


def filter_by_search(recs: list) -> list:
    query = st.session_state.search_query.strip().lower()
    if not query:
        return recs
    out = []
    for r in recs:
        item_id = r["item_id"] if isinstance(r, dict) else r
        info = lookup.get(item_id, {})
        if query in f"{info.get('name','')} {info.get('category','')} {info.get('product_type','')}".lower():
            out.append(r)
    return out

# --------------------------------------------------------------------------- #
# Header + functional nav
# --------------------------------------------------------------------------- #

h_cols = st.columns([2, 4, 3, 1.3])
with h_cols[0]:
    st.markdown('<div class="sq-logo">SEQUORA</div>', unsafe_allow_html=True)
with h_cols[1]:
    nav_cols = st.columns(4)
    nav_items = [("women", "Women"), ("men", "Men"), ("new_in", "New In"), ("trending", "Trending")]
    for i, (key, label) in enumerate(nav_items):
        with nav_cols[i]:
            active = st.session_state.nav_section == key
            if st.button(label, key=f"nav_{key}", type="primary" if active else "secondary", use_container_width=True):
                st.session_state.nav_section = key
                st.rerun()
with h_cols[2]:
    st.session_state.search_query = st.text_input(
        "Search", value=st.session_state.search_query,
        placeholder="Search dresses, tops, trousers...", label_visibility="collapsed",
    )
with h_cols[3]:
    if st.button(f"♡ Wishlist ({len(st.session_state.wishlist)})", key="open_wishlist", use_container_width=True):
        st.session_state["_show_wishlist"] = not st.session_state.get("_show_wishlist", False)

st.markdown('<div style="border-bottom:1px solid #ece4da; margin-bottom:1rem;"></div>', unsafe_allow_html=True)

if not has_gender_field and st.session_state.nav_section in ("women", "men"):
    st.caption("Gender metadata not found in this dataset build — showing the full catalogue instead.")

if st.session_state.get("_show_wishlist"):
    st.markdown('<p class="sq-section-label">Saved</p><p class="sq-section-title">My wishlist</p>', unsafe_allow_html=True)
    if not st.session_state.wishlist:
        st.caption("Your wishlist is empty.")
    else:
        render_grid(list(st.session_state.wishlist), lambda item_id: "Saved by you", lambda item_id: "wishlist", section="wishlist_view")
    st.stop()

if st.session_state.evolved_msg:
    st.toast("Your style profile just evolved.", icon="✨")
    st.session_state.evolved_msg = False

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #

hero_images = []
for info in list(lookup.values())[:300]:
    uri = get_product_image(info.get("image_path", ""))
    if uri:
        hero_images.append(uri)
    if len(hero_images) == 3:
        break

hero_cols = st.columns([3, 2])
with hero_cols[0]:
    st.markdown(
        '<div class="sq-hero"><p class="sq-hero-eyebrow">AI-Powered Discovery</p>'
        '<h1>Your style.<br>Your next find.</h1>'
        '<p>Personalized fashion recommendations shaped around your style, activity and preferences.</p>'
        '<p class="sq-hero-tagline">SEQUORA learns from what you explore — not just what is popular.</p></div>',
        unsafe_allow_html=True,
    )
with hero_cols[1]:
    if hero_images:
        img_cols = st.columns(len(hero_images))
        for i, uri in enumerate(hero_images):
            with img_cols[i]:
                st.markdown(f'<div class="sq-hero-img"><img src="{uri}" /></div>', unsafe_allow_html=True)

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Shop by category
# --------------------------------------------------------------------------- #

st.markdown('<p class="sq-section-label">Browse</p><p class="sq-section-title">Shop by category</p>', unsafe_allow_html=True)
cat_cols = st.columns(len(CATEGORIES))
for i, cat in enumerate(CATEGORIES):
    with cat_cols[i]:
        active_class = "active" if st.session_state.category_filter == cat else ""
        icon_uri = icons.get(cat)
        icon_html = f'<img src="{icon_uri}" />' if icon_uri else ""
        st.markdown(
            f'<div class="sq-cat-icon"><div class="sq-cat-circle {active_class}">{icon_html}</div>'
            f'<div class="sq-cat-label">{cat}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Select", key=f"chip_{cat}", use_container_width=True):
            st.session_state.category_filter = cat
            st.rerun()

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Customer mode
# --------------------------------------------------------------------------- #

mode_col1, mode_col2, _ = st.columns([1, 1, 4])
with mode_col1:
    if st.button("Existing customer", type="primary" if st.session_state.mode == "existing" else "secondary", use_container_width=True):
        st.session_state.mode = "existing"
        st.rerun()
with mode_col2:
    if st.button("New session", type="primary" if st.session_state.mode == "new" else "secondary", use_container_width=True):
        st.session_state.mode = "new"
        st.rerun()

user_id = None
if st.session_state.mode == "existing":
    personas = load_personas()
    if personas:
        st.caption("Demo personas are real (anonymised) customers, named only by what they actually bought.")
        choice = st.selectbox(
            "Demo persona", list(range(len(personas))),
            format_func=lambda i: f"{personas[i]['name']} — {personas[i]['blurb']}",
        )
        user_id = personas[choice]["user_id"]
        render_history(personas[choice]["last_items"])
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    else:
        st.caption("Using your previous activity to personalize recommendations.")
        user_id = st.selectbox("Demo customer", list(range(1, 31)), format_func=lambda u: f"Customer #{u}")
else:
    st.caption("No previous activity — tell us what you're looking for.")
    pref_options = CATEGORIES[1:]  # fixed: always matches column count below
    pref_cols = st.columns(len(pref_options))
    for i, cat in enumerate(pref_options):
        with pref_cols[i]:
            selected = cat in st.session_state.new_session_prefs
            if st.button(cat, key=f"pref_{cat}", type="primary" if selected else "secondary", use_container_width=True):
                if selected:
                    st.session_state.new_session_prefs.remove(cat)
                else:
                    st.session_state.new_session_prefs.append(cat)
                st.rerun()
    if st.session_state.new_session_prefs:
        st.caption(f"Looking for: {', '.join(st.session_state.new_session_prefs)}")

st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Section title + subtitle per nav mode
# --------------------------------------------------------------------------- #

NAV_COPY = {
    "women": ("Just for you", "Picked based on your recent activity"),
    "men": ("Just for you", "Picked based on your recent activity"),
    "new_in": ("New in", "Fresh arrivals you haven't interacted with yet"),
    "trending": ("Trending now", "Popular styles from the catalogue"),
}
title, subtitle = NAV_COPY[st.session_state.nav_section]

st.markdown(
    f'<p class="sq-section-label">{title}</p><p class="sq-section-title">{title}</p>'
    f'<p class="sq-section-sub">{subtitle}</p>',
    unsafe_allow_html=True,
)

if st.session_state.nav_section in ("women", "men"):
    with st.spinner("Finding pieces for you..."):
        payload = {
            "user_id": int(user_id) if user_id is not None else None,
            "session_item_ids": st.session_state.session_clicks,
            "top_k": 300,
            "category": None if st.session_state.category_filter == "All" else st.session_state.category_filter,
        }
        result, err = call_api("POST", "/recommend", json=payload)

    if err:
        st.error(err)
    elif not result["recommendations"]:
        st.caption("No recommendations yet.")
    else:
        all_items = [r for r in result["recommendations"] if r["item_id"] in lookup]
        gendered_ids = set(apply_gender_filter([r["item_id"] for r in all_items]))
        all_items = [r for r in all_items if r["item_id"] in gendered_ids]
        items = filter_by_search(filter_by_category(all_items))
        st.caption(f"{len(items)} recommendations")
        shown = items[: st.session_state.jfy_count]
        if not shown:
            st.caption("No matches — try a different category or search term.")
        else:
            render_grid(shown, lambda item: SOURCE_LABELS.get(item["source"], "Recommended for you"),
                        lambda item: item["source"], section="just_for_you")
            if len(items) > st.session_state.jfy_count:
                if st.button("Load more", key="load_more_jfy"):
                    st.session_state.jfy_count += 10
                    st.rerun()

elif st.session_state.nav_section == "new_in":
    cold_all = [i for i in load_cold_start_items(limit=300) if i in lookup]
    cold_all = apply_gender_filter(cold_all)
    cold_filtered = filter_by_search(filter_by_category(cold_all))
    if not cold_filtered:
        st.caption("No new-in items match this filter right now.")
    else:
        shown = cold_filtered[: st.session_state.nfy_count]
        render_grid(shown, lambda item_id: "New-item discovery", lambda item_id: "cold_start", section="new_in", is_new=True)
        if len(cold_filtered) > st.session_state.nfy_count:
            if st.button("Load more", key="load_more_nin"):
                st.session_state.nfy_count += 10
                st.rerun()

elif st.session_state.nav_section == "trending":
    trending_result, _ = call_api("POST", "/recommend", json={"user_id": None, "session_item_ids": [], "top_k": 300,
                                                      "category": None if st.session_state.category_filter == "All" else st.session_state.category_filter})
    if trending_result:
        trend_items = [r for r in trending_result["recommendations"] if r["item_id"] in lookup]
        gendered_ids = set(apply_gender_filter([r["item_id"] for r in trend_items]))
        trend_items = [r for r in trend_items if r["item_id"] in gendered_ids]
        trend_items = filter_by_search(filter_by_category(trend_items))
        shown = trend_items[: st.session_state.trend_count]
        if not shown:
            st.caption("No trending items match this filter right now.")
        else:
            render_grid(shown, lambda item: "Trending in your catalogue", lambda item: item["source"], section="trending")
            if len(trend_items) > st.session_state.trend_count:
                if st.button("Load more", key="load_more_trend"):
                    st.session_state.trend_count += 10
                    st.rerun()

# --------------------------------------------------------------------------- #
# New for you — always shown under Women/Men, demonstrates cold-start
# --------------------------------------------------------------------------- #

if st.session_state.nav_section in ("women", "men"):
    cold_all = [i for i in load_cold_start_items(limit=200) if i in lookup]
    cold_all = apply_gender_filter(cold_all)
    cold_filtered = filter_by_search(filter_by_category(cold_all))
    if cold_filtered:
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown(
            '<p class="sq-section-label">New for you</p><p class="sq-section-title">New for you</p>'
            '<p class="sq-section-sub">Discover pieces you haven\'t interacted with yet</p>',
            unsafe_allow_html=True,
        )
        render_grid(cold_filtered[:10], lambda item_id: "New-item discovery", lambda item_id: "cold_start", section="new_for_you", is_new=True)

# --------------------------------------------------------------------------- #
# Why these picks
# --------------------------------------------------------------------------- #

st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
st.markdown(
    '<p class="sq-section-title">Why these picks?</p>'
    '<p class="sq-section-sub">SEQUORA combines sequential activity modeling, product content and visual understanding to personalize recommendations.</p>'
    '<div class="sq-why-grid">'
    '<div class="sq-why-card"><div class="sq-why-icon">&#9679;</div><p class="sq-why-title">Your activity</p><p class="sq-why-desc">Views, clicks, wishlist</p></div>'
    '<div class="sq-why-card"><div class="sq-why-icon">&#9671;</div><p class="sq-why-title">Sequential AI</p><p class="sq-why-desc">Predicts what\'s next</p></div>'
    '<div class="sq-why-card"><div class="sq-why-icon">&#10022;</div><p class="sq-why-title">Visual &amp; content</p><p class="sq-why-desc">Image + metadata matching</p></div>'
    '<div class="sq-why-card"><div class="sq-why-icon">&#9733;</div><p class="sq-why-title">Candidate retrieval</p><p class="sq-why-desc">Fast similarity search</p></div>'
    '<div class="sq-why-card"><div class="sq-why-icon">&#9825;</div><p class="sq-why-title">Personalized ranking</p><p class="sq-why-desc">Ranked just for you</p></div>'
    '</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Developer diagnostics
# --------------------------------------------------------------------------- #

with st.expander("Developer diagnostics"):
    diag, _ = call_api("GET", "/diagnostics")
    if diag:
        st.json(diag)
    st.caption(f"Backend health: {health}")
    st.caption(f"Catalogue size after swimwear exclusion: {len(lookup)} items")
    st.caption(f"Gender field available: {has_gender_field}")