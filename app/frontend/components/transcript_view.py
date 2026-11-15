"""Accessible Transcript View and Interactive Review Components."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

# Distinct, colourblind-friendly speaker chip colours (Okabe-Ito / Tol subsets).
SPEAKER_COLORS = ["#0072B2", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#E69F00", "#332288", "#882255"]


def _fmt_ts(seconds: float) -> str:
    s = max(0.0, float(seconds))
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _speaker_name(raw: str) -> str:
    if not raw or raw.upper() in ("SPEAKER_UNASSIGNED", "UNKNOWN"):
        return "Speaker 1"
    return raw.replace("_", " ").title()


def render_accessible_transcript(
    segs_df: pd.DataFrame,
    words_df: pd.DataFrame,
    colors: dict[str, str],
    tier_label: dict[str, str],
    tier_symbol: dict[str, str],
    low_t: float,
    high_t: float,
    font_px: int = 18,
) -> None:
    """Render word-level transcript with multi-sensory (colour + shape + underline) confidence cues."""
    if segs_df.empty:
        st.info("The transcript is empty: no speech was detected.")
        return

    css_vars = f'--low:{colors["low"]};--mid:{colors["mid"]};--high:{colors["high"]}'

    top_l, top_r = st.columns([3, 2], vertical_alignment="center")
    with top_l:
        st.markdown(
            f'<div class="legend" style="{css_vars}">'
            f'<span><b class="sym sym-low">{tier_symbol["low"]}</b> Low (&lt; {low_t:.2f})</span>'
            f'<span><b class="sym sym-mid">{tier_symbol["mid"]}</b> Medium</span>'
            f'<span><b class="sym sym-high">{tier_symbol["high"]}</b> High (≥ {high_t:.2f})</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with top_r:
        view = st.radio(
            "Transcript format",
            ["Highlighted", "Plain text (screen readers)"],
            horizontal=True,
            label_visibility="collapsed",
        )
    plain = view.startswith("Plain")

    speakers = list(dict.fromkeys(str(s) for s in segs_df.get("speaker", pd.Series(dtype=str)).dropna()))
    spk_color = {s: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, s in enumerate(speakers)}

    for _, seg in segs_df.iterrows():
        seg_id = str(seg.get("turn_id", seg.get("segment", "turn")))
        speaker = str(seg.get("speaker", "Speaker"))
        start_t, end_t = float(seg["start"]), float(seg["end"])
        sw = words_df[(words_df["start"] >= start_t - 1e-4) & (words_df["end"] <= end_t + 1e-4)] if not words_df.empty else words_df

        if plain:
            body = html.escape(" ".join(f"[{w['word']}?]" if w["tier"] == "low" else str(w["word"]) for _, w in sw.iterrows()))
        else:
            parts = []
            for _, w in sw.iterrows():
                txt = html.escape(str(w["word"]))
                t = w["tier"]
                title = html.escape(f"{tier_label[t]}: {w['confidence']:.2f}")
                sym = f'<sup class="sym sym-{t}" aria-hidden="true">{tier_symbol[t]}</sup>'
                parts.append(f'<span class="w-{t}" title="{title}">{txt}{sym}</span>')
            body = " ".join(parts)

        st.markdown(
            f'<div class="turn" style="{css_vars}">'
            f'<div class="turn-head"><span class="spk" style="background:{spk_color.get(speaker, SPEAKER_COLORS[0])}">'
            f"{html.escape(_speaker_name(speaker))}</span>"
            f'<span class="ts">{_fmt_ts(start_t)} – {_fmt_ts(end_t)}</span></div>'
            f'<p class="turn-text" style="font-size:{font_px}px">{body}</p></div>',
            unsafe_allow_html=True,
        )


    if plain:
        st.caption("Words in [brackets?] were recognised with low confidence.")
