"""Accessible Data Tables and CSV Export Components."""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render_accessible_tables(segs_df: pd.DataFrame, words_df: pd.DataFrame) -> None:
    """Tabular (screen-reader friendly) alternatives to the charts."""
    with st.container(border=True):
        st.markdown("**Turns**")
        if not segs_df.empty:
            # st.table (not st.dataframe) so long turn text wraps and is shown in full.
            headers = {
                "segment": "Turn",
                "speaker": "Speaker",
                "start": "Start (s)",
                "end": "End (s)",
                "duration": "Duration (s)",
                "n_words": "Words",
                "avg_confidence": "Avg confidence",
                "text": "Text",
            }
            cols = [c for c in headers if c in segs_df.columns]
            turns = segs_df[cols].copy()
            for c in ("start", "end", "duration", "avg_confidence"):
                if c in turns.columns:
                    turns[c] = turns[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
            st.table(turns.rename(columns=headers), hide_index=True)
        else:
            st.caption("No turns available.")

    with st.container(border=True):
        st.markdown("**Words**")
        if words_df.empty:
            st.caption("No words available.")
            return

        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        selected_spk = c1.selectbox("Speaker", ["All"] + sorted(words_df["speaker"].astype(str).unique().tolist()))
        only_low = c2.checkbox("Only low-confidence words", key="chk_only_low")

        filtered = words_df
        if only_low:
            filtered = filtered[filtered["tier"] == "low"]
        if selected_spk != "All":
            filtered = filtered[filtered["speaker"].astype(str) == selected_spk]

        cols = [c for c in ["word", "start", "end", "duration", "confidence", "raw_confidence", "speaker", "Confidence level"] if c in filtered.columns]
        st.dataframe(
            filtered[cols],
            width="stretch",
            hide_index=True,
            column_config={
                "start": st.column_config.NumberColumn("Start (s)", format="%.2f"),
                "end": st.column_config.NumberColumn("End (s)", format="%.2f"),
                "duration": st.column_config.NumberColumn("Duration (s)", format="%.2f"),
                "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
                "raw_confidence": st.column_config.NumberColumn("Raw confidence", format="%.3f"),
            },
        )
        st.download_button(
            "Download these words as CSV",
            filtered[cols].to_csv(index=False).encode("utf-8"),
            file_name="words_confidence.csv",
            mime="text/csv",
            icon="📥",
        )
