"""
Adaptive Personalized Energy Advisor - main entry point.
Run with:  python -m streamlit run app.py

Flow:  Welcome page  ->  Menu of cards  ->  a section (views/*.py)  ->  Home / All sections
Navigation is handled here with st.session_state, so Streamlit's default
sidebar page list is not used.
"""
import runpy
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Adaptive Personalized Energy Advisor",
                   page_icon="⚡", layout="wide", initial_sidebar_state="auto")

from utils import inject_css, get_settings, go, btn, SECTIONS, ROUTE_FILES  # noqa: E402

VIEWS_DIR = Path(__file__).resolve().parent / "views"

inject_css()
get_settings()
st.session_state.setdefault("route", "welcome")
route = st.session_state["route"]
if route not in SECTIONS and route not in ("welcome", "menu"):
    route = st.session_state["route"] = "welcome"

# Streamlit's own page list is never shown. On the welcome and card pages the whole sidebar is hidden;
# on section pages the sidebar shows our own "All sections" panel.
st.markdown("<style>[data-testid='stSidebarNav'] {display: none !important;} "
            "header[data-testid='stHeader'] {background: transparent;}</style>", unsafe_allow_html=True)
if route in ("welcome", "menu"):
    st.markdown("<style>[data-testid='stSidebar'], [data-testid='collapsedControl'], "
                "[data-testid='stSidebarCollapsedControl'] {display: none !important;}</style>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- welcome ---
def show_welcome():
    st.markdown("""
<style>
.block-container {max-width: 860px; padding-top: 6vh;}
.w-card {background:#ffffff; border:1px solid #e4e9df; border-radius:22px; text-align:center;
         padding: 34px 28px 26px 28px; box-shadow: 0 6px 24px rgba(60,80,60,.08); margin-bottom: 18px;}
.w-bolt {font-size: 3.6rem; line-height:1;}
.w-tag {display:inline-block; background:#e8f3ea; color:#2f6b4f; border:1px solid #d3e6d8;
        padding:4px 14px; border-radius:999px; font-size:.8rem; letter-spacing:1.2px; margin:12px 0 10px 0;}
.w-title {font-size: clamp(1.8rem, 4.6vw, 2.9rem); font-weight: 800; line-height:1.15; margin: 4px 0; color:#1f2937;}
.w-title span {color:#3f8f6b;}
.w-sub {font-size: 1.08rem; color:#4b5b52; margin: 6px auto 4px auto;}
.w-desc {max-width: 620px; margin: 12px auto 18px auto; color:#374151; font-size:1.02rem; line-height:1.65;}
.w-chips {display:flex; gap:8px; justify-content:center; flex-wrap:wrap;}
.w-chip {background:#e9f1fa; color:#27496d; padding:6px 14px; border-radius:12px; font-size:.9rem;}
.w-chip.g {background:#e8f3ea; color:#2f6b4f;}
.w-foot {text-align:center; color:#6b7a70; font-size:.85rem; margin-top: 18px;}
</style>
<div class="w-card">
  <div class="w-bolt">⚡</div>
  <div class="w-tag">SOFT COMPUTING PROJECT</div>
  <div class="w-title">Adaptive Personalized<br><span>Energy Advisor</span></div>
  <div class="w-sub">Using an ANN (a computer model that learns from data) to save electricity</div>
  <div class="w-desc">This website learns how your home uses electricity, guesses how much you will use
  in the next hour, and gives you simple tips to use less power and pay less.</div>
  <div class="w-chips">
    <span class="w-chip g">🔮 Predict your power use</span><span class="w-chip">📊 See your usage</span>
    <span class="w-chip g">💡 Get simple tips</span><span class="w-chip">🧮 Work out your cost</span>
  </div>
</div>
""", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        btn("Get Started  →", type="primary", on_click=go, args=("menu",), key="get_started")
    st.markdown('<div class="w-foot">Prajna • Kalpana • Ananya &nbsp;|&nbsp; Faculty: Dr. Hari Seetha</div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------------- menu ---
def show_menu():
    st.markdown("""
<style>
.block-container {max-width: 820px;}
[data-testid="stVerticalBlockBorderWrapper"] {background:#ffffff; border-radius:14px; border:1px solid #e4e9df !important;
    box-shadow:0 1px 4px rgba(60,80,60,.07);}
.c-icon {font-size:1.9rem; line-height:1.1;}
.c-title {font-size:1.05rem; font-weight:700; color:#1f2937; margin-top:4px;}
.c-desc {font-size:.9rem; color:#55645b; min-height:2.9em; margin: 2px 0 6px 0; line-height:1.4;}
</style>
""", unsafe_allow_html=True)
    top_l, _ = st.columns([1, 3])
    with top_l:
        btn("🏡 Home", on_click=go, args=("welcome",), key="menu_home")
    st.markdown('<div class="hero"><h1>What would you like to do?</h1>'
                '<p>Click <b>Open</b> on a card to go to that section.</p></div>',
                unsafe_allow_html=True)
    keys = list(SECTIONS)                       # order is set in utils.SECTIONS
    for i in range(0, len(keys), 2):            # 2 cards per row
        cols = st.columns(2)
        for col, key in zip(cols, keys[i:i + 2]):
            icon, title, desc = SECTIONS[key]
            with col:
                with st.container(border=True):
                    st.markdown(f'<div class="c-icon">{icon}</div><div class="c-title">{title}</div>'
                                f'<div class="c-desc">{desc}</div>', unsafe_allow_html=True)
                    btn("Open →", key=f"card_{key}", on_click=go, args=(key,))


# ---------------------------------------------------------------- section ---
def show_side_panel(current):
    """The 'All sections' box, placed at the side."""
    with st.sidebar:
        st.markdown("### ☰ All sections")
        btn("🏡 Home", on_click=go, args=("welcome",), key="side_home")
        btn("🗂️ Card menu", on_click=go, args=("menu",), key="side_menu")
        st.divider()
        for key, (icon, title, _) in SECTIONS.items():
            btn(f"{icon}  {title}", on_click=go, args=(key,), key=f"side_{key}",
                type="primary" if key == current else "secondary")


def show_section(key):
    icon, title, _ = SECTIONS[key]
    show_side_panel(key)
    a, b = st.columns([1, 4])
    with a:
        btn("🏡 Home", on_click=go, args=("welcome",), key="sec_home")
    b.markdown(f"<div style='padding-top:8px;color:#55645b'>{icon} <b>{title}</b> &nbsp;·&nbsp; "
               "use the side panel (arrow at the top left on a phone) to switch sections</div>",
               unsafe_allow_html=True)
    runpy.run_path(str(VIEWS_DIR / ROUTE_FILES[key]), run_name="__main__")


if route == "welcome":
    show_welcome()
elif route == "menu":
    show_menu()
else:
    show_section(route)
