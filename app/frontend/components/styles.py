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

/* ---------------- Extension insights (cards from connected services) ---------------- */
.xi-divider { height: 1px; margin: 2rem 0 .5rem; background: linear-gradient(90deg, rgba(128,128,128,.45), transparent); }
.xi-source { display: flex; flex-wrap: wrap; align-items: baseline; gap: .5rem; padding-left: .6rem;
             border-left: 3px solid var(--high); margin-bottom: .35rem; }
.xi-name { font-weight: 700; font-size: .8rem; letter-spacing: .04em; text-transform: uppercase; }
.xi-meta { font-size: .75rem; opacity: .6; font-variant-numeric: tabular-nums; }
.xi-outdated { font-size: .78rem; opacity: .8; margin: .3rem 0 .2rem; }
.xi-card { margin-top: .2rem; }
.xi-heading { margin: .2rem 0 .15rem; padding: 0; font-size: 1.05rem; font-weight: 650; }
.xi-note { margin: 0 0 .6rem; opacity: .8; line-height: 1.5; }
.xi-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(7.5rem, 1fr)); gap: .5rem; margin: .4rem 0 .7rem; }
.xi-stat { display: flex; flex-direction: column; padding: .45rem .6rem; border-radius: .45rem;
           background: rgba(128,128,128,.07); border: 1px dashed rgba(128,128,128,.35); }
.xi-stat-name { font-size: .72rem; opacity: .7; }
.xi-stat-value { font-size: 1.2rem; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.3; }
.xi-hint { font-size: .7rem; opacity: .55; }
.xi-flags { list-style: none; padding: 0; margin: .2rem 0 .6rem; }
.xi-flag { display: grid; grid-template-columns: 1.2rem auto 1fr; column-gap: .45rem; row-gap: .1rem;
           align-items: baseline; padding: .35rem 0; border-bottom: 1px solid rgba(128,128,128,.15); font-size: .9rem; }
.xi-flag:last-child { border-bottom: 0; }
.xi-flag .sym { grid-row: span 2; font-size: 1rem; }
.xi-level { font-weight: 600; font-size: .78rem; }
.xi-where { font-size: .78rem; opacity: .65; font-variant-numeric: tabular-nums; }
.xi-text { grid-column: 2 / 4; line-height: 1.45; }
.xi-flag-low .xi-level { text-decoration: underline; text-decoration-color: var(--low); text-underline-offset: 3px; }
.xi-failed { padding: .5rem .7rem; border-radius: .45rem; border: 1px solid var(--low);
             background: color-mix(in srgb, var(--low) 10%, transparent); font-size: .9rem; }
.xi-row-head { display: flex; align-items: baseline; gap: .6rem; }
.xi-state { font-size: .78rem; opacity: .75; }
.xi-url { font-family: ui-monospace, monospace; font-size: .78rem; opacity: .65; word-break: break-all; }
.xi-team { font-size: .75rem; font-weight: 600; padding: .05rem .5rem; border-radius: 999px;
           border: 1px solid rgba(128,128,128,.5); white-space: nowrap; }
.xi-ok, .xi-bad { font-size: .78rem; font-weight: 600; margin-left: auto; }
.xi-steps { margin: .2rem 0 1rem 1.1rem; padding: 0; line-height: 1.7; }
.xi-summary { display: flex; gap: 1.2rem; font-size: .9rem; margin: .2rem 0 .8rem; }
.xi-summary .sym { margin-right: .25rem; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
