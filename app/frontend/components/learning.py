"""Adaptive Learning UI: correct transcript turns and see what the system has learned."""
from __future__ import annotations

from typing import Any, Callable

import altair as alt
import pandas as pd
import streamlit as st

ERROR_TYPE_LABEL = {
    "acoustic_lexical": "Wrong words",
    "diarization": "Wrong speaker",
    "compound": "Wrong words and speaker",
    "none": "Verified correct",
}


def _fmt_ts(seconds: float) -> str:
    s = max(0.0, float(seconds))
    return f"{int(s // 60)}:{s % 60:04.1f}"


def _show_last_result() -> None:
    res = st.session_state.pop("last_feedback_result", None)
    if not res:
        return
    kind = ERROR_TYPE_LABEL.get(res.get("error_type"), res.get("error_type", ""))
    terms = res.get("extracted_terms") or []
    cal = res.get("calibration") or {}
    lines = [f"**Saved: {kind}.** {res.get('words_changed', 0)} of {res.get('words_total', 0)} words changed."]
    if terms:
        lines.append("Learned new terms: " + ", ".join(f"`{t}`" for t in terms) + ". They will be used as hints for the next transcription.")
    if cal.get("fitted"):
        lines.append(
            f"Confidence calibration re-fitted on {cal['n_samples']} reviewed words "
            f"(ECE {cal['ece_prior']:.3f} → {cal['ece_learned']:.3f})."
        )
    else:
        lines.append(
            f"Calibration has {cal.get('n_samples', 0)} reviewed words; it starts learning at {cal.get('min_samples', 20)}."
        )
    st.success("\n\n".join(lines), icon="🧠")


def render_correction_form(
    client: Any,
    recording_id: str,
    raw_segs: list[dict[str, Any]],
    speaker_name: Callable[[str], str],
) -> None:
    """Let the user fix the text or speaker of one turn. This is the input of the adaptive loop."""
    _show_last_result()
    raw_segs = [s for s in raw_segs if s.get("turn_id")]  # corrections are addressed by turn_id
    if not raw_segs:
        return

    with st.container(border=True):
        st.markdown("**✏️ Correct the transcript**")
        st.caption(
            "Fix any words the system got wrong, or the speaker. The system learns from each fix: "
            "new words become hints for future transcriptions, and its confidence scores are re-calibrated. "
            "Saving a turn without changes marks it as correct, which also helps calibration."
        )

        idx = st.selectbox(
            "Turn to correct",
            range(len(raw_segs)),
            format_func=lambda i: (
                f"Turn {i + 1} · {speaker_name(raw_segs[i].get('speaker'))} · "
                f"{_fmt_ts(raw_segs[i]['start'])} · {raw_segs[i].get('text', '')[:60]}"
            ),
            key=f"fix_turn_{recording_id}",
        )
        seg = raw_segs[idx]
        speakers_raw = list(dict.fromkeys(str(s.get("speaker")) for s in raw_segs if s.get("speaker")))
        current_spk = str(seg.get("speaker"))
        if current_spk not in speakers_raw:
            speakers_raw.append(current_spk)

        with st.form(key=f"fix_form_{recording_id}_{seg['turn_id']}", border=False):
            new_text = st.text_area("Corrected text", value=seg.get("text", ""), height=110)
            new_spk = st.selectbox(
                "Speaker",
                speakers_raw,
                index=speakers_raw.index(current_spk),
                format_func=lambda s: speaker_name(s),
            )
            submitted = st.form_submit_button("Save correction", type="primary")

        if submitted:
            if not new_text.strip():
                st.error("The corrected text cannot be empty.")
                return
            try:
                res = client.submit_feedback(
                    recording_id=recording_id,
                    turn_id=seg["turn_id"],
                    original_text=seg.get("text", ""),
                    corrected_text=new_text.strip(),
                    corrected_speaker=new_spk,
                )
                st.session_state["last_feedback_result"] = res
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save the correction: {exc}")


def render_calibration_curve(curve: list[dict[str, Any]], colors: dict[str, str]) -> None:
    """Prior vs learned calibration mapping. Lines differ by colour *and* dash pattern."""
    if not curve:
        st.caption("No calibration curve available.")
        return
    df = pd.DataFrame(curve).melt(id_vars="raw", value_vars=["prior", "current"], var_name="curve", value_name="calibrated")
    df["curve"] = df["curve"].map({"prior": "Starting curve (prior)", "current": "Current curve (learned)"})
    domain = ["Starting curve (prior)", "Current curve (learned)"]
    chart = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("raw:Q", title="Raw model confidence", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("calibrated:Q", title="Calibrated confidence", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("curve:N", scale=alt.Scale(domain=domain, range=[colors["mid"], colors["high"]]),
                            legend=alt.Legend(title="Curve", orient="bottom")),
            strokeDash=alt.StrokeDash("curve:N", scale=alt.Scale(domain=domain, range=[[6, 4], [1, 0]]), legend=None),
            shape=alt.Shape("curve:N", scale=alt.Scale(domain=domain, range=["square", "circle"]), legend=None),
            tooltip=[alt.Tooltip("curve:N", title="Curve"), alt.Tooltip("raw:Q", title="Raw", format=".2f"),
                     alt.Tooltip("calibrated:Q", title="Calibrated", format=".2f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")


def render_learning_panel(
    client: Any,
    recording_id: str,
    colors: dict[str, str],
    transcript_summary: dict[str, Any],
    on_retranscribe: Callable[[], None],
) -> None:
    """Dashboard of the adaptive core: what it learned, how it changed, and how to use it."""
    try:
        status = client.get_adaptive_status()
    except Exception as exc:
        st.info(f"Learning status unavailable: {exc}")
        return
    if not isinstance(status, dict):
        st.info("Learning status unavailable: unexpected response from the backend.")
        return

    cal = status.get("calibration", {})
    report = cal.get("report") or {}
    vocab = status.get("vocabulary", [])

    st.markdown(
        "The system improves from your corrections in the **Transcript** tab. "
        "It learns **domain vocabulary** (passed to Whisper as a hint) and **re-calibrates confidence** "
        "using the words you kept or changed as ground truth."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Corrections saved", status.get("corrections", 0), border=True)
    m2.metric("Words reviewed", status.get("reviewed_words", 0),
              help="Original words in corrected turns, labelled kept (correct) or changed (wrong).", border=True)
    m3.metric("Learned terms", len(vocab), border=True)
    if cal.get("is_fitted"):
        m4.metric("Calibration", "Learned", help=f"Fitted on {cal.get('n_samples')} reviewed words.", border=True)
    else:
        m4.metric("Calibration", "Prior",
                  help=f"Starts learning after {cal.get('min_samples', 20)} reviewed words "
                       f"(currently {cal.get('n_samples', 0)}).", border=True)

    used = (transcript_summary or {}).get("adaptive") or {}
    if used.get("learned_terms_used"):
        st.caption(
            f"This transcript was produced with {len(used['learned_terms_used'])} learned terms "
            f"and calibration `{used.get('calibration_version', 'prior')}`."
        )

    col_v, col_c = st.columns(2)
    with col_v, st.container(border=True):
        st.markdown("**Learned vocabulary**")
        if vocab:
            st.dataframe(
                pd.DataFrame(vocab).rename(columns={"term": "Term", "count": "Times corrected"}),
                hide_index=True, width="stretch", height=min(300, 38 + 35 * len(vocab)),
            )
            st.markdown("Hint sent to Whisper on the next transcription:")
            st.code(status.get("whisper_prompt", ""), language=None, wrap_lines=True)
        else:
            st.caption("Nothing learned yet. Correct a misrecognised word in the Transcript tab to start.")

        st.button(
            "Re-transcribe this recording with what was learned",
            on_click=on_retranscribe,
            disabled=not vocab,
            help="Runs the pipeline again with the learned vocabulary and current calibration. "
                 "Your earlier manual edits to this recording will be replaced.",
        )

    with col_c, st.container(border=True):
        st.markdown("**Confidence calibration**")
        st.caption("How raw model confidence is mapped to calibrated confidence. Dashed: starting curve. Solid: learned curve.")
        render_calibration_curve(cal.get("curve", []), colors)
        if report.get("fitted"):
            st.table(pd.DataFrame(
                [
                    {"Metric": "ECE (lower is better)", "Starting curve": f"{report['ece_prior']:.4f}",
                     "Learned curve": f"{report['ece_learned']:.4f}"},
                    {"Metric": "Brier score (lower is better)", "Starting curve": f"{report['brier_prior']:.4f}",
                     "Learned curve": f"{report['brier_learned']:.4f}"},
                ]
            ), hide_index=True)
            st.caption(
                f"Measured on {report['n_samples']} reviewed words ({report['observed_accuracy']:.0%} were correct). "
                f"Learned data weight: {report['data_weight']:.0%}, the rest comes from the starting curve."
            )

    with st.container(border=True):
        st.markdown("**Corrections for this recording**")
        try:
            history = client.list_feedback(recording_id)
        except Exception as exc:
            history = []
            st.caption(f"Could not load corrections: {exc}")
        if not history:
            st.caption("No corrections for this recording yet.")
            return
        hist_df = pd.DataFrame(history)
        hist_df["When"] = pd.to_datetime(hist_df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        hist_df["Type"] = hist_df["notes"].map(lambda n: ERROR_TYPE_LABEL.get(n, n))
        st.table(
            hist_df[["When", "Type", "original_text", "corrected_text"]].rename(
                columns={"original_text": "Before", "corrected_text": "After"}
            ),
            hide_index=True,
        )
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        fid = c1.selectbox(
            "Forget a correction",
            hist_df["feedback_id"].tolist(),
            format_func=lambda f: (
                f"{hist_df.set_index('feedback_id').loc[f, 'When']} · "
                f"{str(hist_df.set_index('feedback_id').loc[f, 'corrected_text'])[:50]}"
            ),
            help="The system un-learns it. The transcript text stays as you edited it.",
        )
        if c2.button("Forget", width="stretch"):
            try:
                client.delete_feedback(fid)
                st.toast("Correction forgotten; vocabulary and calibration re-learned.", icon="🗑️")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not delete: {exc}")
