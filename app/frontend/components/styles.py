"""Shared CSS for the Streamlit dashboard (works in light and dark themes)."""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }

.app-hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 .25rem 0; padding: 0; }
.app-hero p  { opacity: .7; margin: 0 0 1rem 0; font-size: 1rem; }

/* Transcript turns */
.turn { padding: .9rem 1.1rem; margin-bottom: .75rem; border-radius: .6rem;
        border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.04); }
.turn-head { display: flex; align-items: center; gap: .6rem; margin-bottom: .45rem; font-size: .85rem; }
.spk { font-weight: 600; padding: .1rem .55rem; border-radius: 999px; color: #fff; font-size: .8rem; }
.ts  { opacity: .6; font-variant-numeric: tabular-nums; }
.turn-text { margin: 0; line-height: 1.75; }
.w-low { border-bottom: 2px solid var(--low); background: color-mix(in srgb, var(--low) 16%, transparent);
         border-radius: 3px; padding: 0 2px; font-weight: 600; }
.w-mid { border-bottom: 2px dotted var(--mid); }
.w-high { }
/* Tier symbols: filled with the tier colour */
.sym { font-weight: 700; }
.sym-low  { color: var(--low); }
.sym-mid  { color: var(--mid); }
.sym-high { color: var(--high); }
.turn-text sup.sym { font-size: .6em; margin-left: 1px; }

/* Legend chips */
.legend { display: flex; flex-wrap: wrap; gap: .5rem 1rem; font-size: .85rem; opacity: .85; margin: .25rem 0 .9rem; }
.legend span { display: inline-flex; align-items: center; gap: .35rem; }
.legend .sym { font-size: 1.05rem; }
.dot { width: .7rem; height: .7rem; border-radius: 2px; display: inline-block; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
