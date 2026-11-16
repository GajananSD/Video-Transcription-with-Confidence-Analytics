"""Streamlit Web Application: Video Transcription with Confidence Analytics."""
from __future__ import annotations

import time
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

from frontend.api_client import APIClient
from frontend.components.charts import (
    render_confidence_histogram,
    render_confidence_scatter,
    render_reliability_diagram,
    render_segment_confidence_bar,
    render_speech_timeline,
)
from frontend.components.extensions import render_plugin_manager, render_plugin_results
from frontend.components.learning import render_correction_form, render_learning_panel
from frontend.components.styles import inject_styles
from frontend.components.tables import render_accessible_tables
from frontend.components.transcript_view import render_accessible_transcript

st.set_page_config(
    page_title="Video Transcription · Confidence Analytics",
    page_icon="🎙️",
    layout="wide",
)
inject_styles()

# ---------------------------------------------------------------- Constants
PALETTES = {
    "Okabe-Ito (colourblind-safe)": {"low": "#D55E00", "mid": "#E69F00", "high": "#0072B2"},
    "Blue / Orange / Purple (colourblind-safe)": {"low": "#E66100", "mid": "#5D3A9B", "high": "#005AB5"},
    "Greyscale (high contrast / print)": {"low": "#000000", "mid": "#666666", "high": "#AAAAAA"},
}
TIER_LABEL = {"low": "Low confidence", "mid": "Medium confidence", "high": "High confidence"}
TIER_SHAPE = {"low": "triangle-up", "mid": "square", "high": "circle"}
TIER_SYMBOL = {"low": "▲", "mid": "■", "high": "●"}
TIERS = ["low", "mid", "high"]

MEDIA_TYPES = ["mp4", "mov", "wav", "mp3", "m4a", "aac", "flac"]


def assign_tier(conf: float, low_t: float, high_t: float) -> str:
    if conf < low_t:
        return "low"
    if conf < high_t:
        return "mid"
    return "high"


def speaker_number_map(*label_lists) -> dict[str, str]:
    """Map raw speaker labels (e.g. SPEAKER_00, SPEAKER_UNASSIGNED) to 'Speaker 1', 'Speaker 2', ...

    Numbers follow order of first appearance. Display-only: the backend data is untouched.
    """
    mapping: dict[str, str] = {}
    for labels in label_lists:
        for raw in labels:
            key = "SPEAKER_UNASSIGNED" if raw is None or str(raw).strip() == "" else str(raw)
            if key not in mapping:
                mapping[key] = f"Speaker {len(mapping) + 1}"
    return mapping


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def recording_label(rec: dict) -> str:
    return f"{rec['filename']} · {fmt_duration(float(rec.get('duration') or 0))} · {rec['recording_id'][:8]}"


def open_recording(rec_id: str, filename: str) -> None:
    """Show an existing recording (no new job)."""
    st.session_state["recording_id"] = rec_id
    st.session_state["filename"] = filename
    st.session_state["job_id"] = None


def dispatch_job(rec_id: str, filename: str, force: bool = False) -> None:
    """Start a transcription job and remember it in session state.

    The backend automatically adds vocabulary learned from past corrections to the Whisper prompt.
    """
    st.session_state["recording_id"] = rec_id
    st.session_state["filename"] = filename
    job_res = client.trigger_transcription(rec_id, use_mock=False, initial_prompt=None, force=force)
    st.session_state["job_id"] = job_res.get("job_id")
    st.session_state["job_started_at"] = time.time()


# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### 🎙️ Transcription Studio")

    api_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
    client = APIClient(api_url)
    try:
        health = client.health()
        st.badge(f"Backend online · {health.get('device', 'cpu').upper()}", icon="✅", color="green")
    except Exception:
        st.badge("Backend offline", icon="⚠️", color="red")
        st.caption("Start it from `app/` with:\n\n`uvicorn backend.main:app --port 8000`")

    st.divider()
    st.markdown("**Display**")
    theme_choice = st.segmented_control(
        "Theme", ["☀️ Light", "🌙 Dark"],
        default="🌙 Dark" if st._config.get_option("theme.base") == "dark" else "☀️ Light",
        key="theme_choice",
    )
    wanted_base = "dark" if theme_choice == "🌙 Dark" else "light"
    if theme_choice and st._config.get_option("theme.base") != wanted_base:
        st._config.set_option("theme.base", wanted_base)
        st.rerun()
    palette_name = st.selectbox("Colour scheme", list(PALETTES))
    colors = PALETTES[palette_name]
    font_px = st.slider("Transcript text size", 14, 28, 17, format="%dpx")

    with st.expander("Confidence thresholds"):
        low_t = st.slider("Low confidence below", 0.0, 1.0, 0.30, 0.05)
        high_t = st.slider("High confidence from", 0.0, 1.0, 0.60, 0.05)
        if high_t < low_t:
            st.warning("High threshold is below the low threshold; using the low value for both.")
            high_t = low_t

    with st.expander("About the pipeline"):
        st.markdown(
            "- **ASR:** Faster-Whisper `large-v3`\n"
            "- **Alignment:** Wav2Vec2 word timestamps\n"
            "- **Speakers:** NeMo MSDD or Pyannote 3.1\n"
            "- **Confidence:** isotonic calibration (Brier, ECE)\n"
            "- **Adaptive:** learns vocabulary and re-fits calibration from your corrections"
        )

# ---------------------------------------------------------------- Header
st.markdown(
    '<div class="app-hero"><h1>Video Transcription with Confidence Analytics</h1>'
    "<p>Upload a video or audio file to get a speaker-attributed transcript with calibrated, "
    "word-level confidence.</p></div>",
    unsafe_allow_html=True,
)

# Top-level sections: transcription (default) and plugin registration (always available).
SECTION_TRANSCRIBE, SECTION_PLUGINS = "🎬 Transcribe", "🧩 Plugins"
section = st.segmented_control(
    "Section", [SECTION_TRANSCRIBE, SECTION_PLUGINS], default=SECTION_TRANSCRIBE,
    key="main_section", label_visibility="collapsed",
) or SECTION_TRANSCRIBE

if section == SECTION_PLUGINS:
    render_plugin_manager(client, st.session_state.get("recording_id"), colors)
    st.stop()

col_upload, col_recent = st.columns([3, 2], gap="medium")

with col_upload, st.container(border=True):
    st.markdown("**New transcription**")
    uploaded_file = st.file_uploader(
        "Choose a video or audio file",
        type=MEDIA_TYPES,
        help="MP4, MOV, WAV, MP3, M4A, AAC or FLAC. Audio is converted to 16 kHz mono automatically.",
    )
    if st.button("Upload and transcribe", type="primary", disabled=uploaded_file is None):
        with st.spinner("Uploading and validating media..."):
            try:
                upload_res = client.upload_media(uploaded_file.getvalue(), uploaded_file.name)
                rec_id, fname = upload_res["recording_id"], upload_res["filename"]
                # Same file uploaded before and already transcribed? Show the saved result.
                existing = None
                try:
                    existing = client.get_transcript(rec_id)
                except Exception:
                    existing = None
                if existing:
                    open_recording(rec_id, fname)
                    st.toast("This file was already transcribed. Showing the saved result.", icon="♻️")
                else:
                    dispatch_job(rec_id, fname)
                st.session_state["past_recording_selector"] = rec_id
                st.rerun()
            except Exception as e:
                st.error(f"Upload failed: {e}")


def _on_pick_recording() -> None:
    rec_id = st.session_state.get("past_recording_selector")
    if rec_id and rec_id != st.session_state.get("recording_id"):
        open_recording(rec_id, _recordings_by_id.get(rec_id, {}).get("filename", "recording"))


with col_recent, st.container(border=True):
    st.markdown("**Recent recordings**")
    _recordings_by_id: dict[str, dict] = {}
    try:
        recordings = client.list_recordings()
    except Exception as e:
        recordings = []
        st.warning(f"Could not load the recording list: {e}")
    if recordings:
        _recordings_by_id = {r["recording_id"]: r for r in recordings}
        ids = list(_recordings_by_id)
        if st.session_state.get("past_recording_selector") not in ids:
            st.session_state.pop("past_recording_selector", None)
        extra = {} if "past_recording_selector" in st.session_state else {"index": None}
        st.selectbox(
            "Open a previous recording",
            ids,
            format_func=lambda rid: recording_label(_recordings_by_id[rid]),
            placeholder="Choose a recording...",
            key="past_recording_selector",
            on_change=_on_pick_recording,
            **extra,
        )
        st.caption(f"{len(ids)} saved, newest first.")
    else:
        st.caption("Nothing yet. Your transcribed files will appear here.")

current_recording_id = st.session_state.get("recording_id")
current_job_id = st.session_state.get("job_id")
current_filename = st.session_state.get("filename") or "recording"

if not current_recording_id:
    st.info("Upload media above, or open a recent recording, to see results here.")
    st.stop()

# ---------------------------------------------------------------- Transcript / job resolution
def show_progress(job_status: dict) -> None:
    started = st.session_state.get("job_started_at")
    elapsed = f" · {fmt_duration(time.time() - started)} elapsed" if started else ""
    with st.container(border=True):
        st.markdown(f"**Transcribing `{current_filename}`**{elapsed}")
        st.progress(float(job_status.get("progress", 0.0)), text=job_status.get("stage", "Working..."))
        st.caption(
            "Large models run slowly on CPU (roughly 3–4× the media length). "
            "The first run also downloads the model. You can leave this page open."
        )
    time.sleep(1.5)
    st.rerun()


# A re-transcription keeps the old transcript until it finishes, so check the active job first.
if current_job_id:
    try:
        _active = client.get_job_status(current_job_id)
    except Exception:
        _active = None
    if _active and _active.get("status") in ("queued", "processing"):
        show_progress(_active)

try:
    transcript_res = client.get_transcript(current_recording_id)
except requests.exceptions.HTTPError as http_err:
    st.error(f"Failed to retrieve transcript: {http_err}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected error while retrieving transcript: {exc}")
    st.stop()

if not transcript_res:
    poll_target = current_job_id or current_recording_id
    try:
        job_status = client.get_job_status(poll_target)
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Failed to query job status: {http_err}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error checking job status: {exc}")
        st.stop()

    if job_status is not None:
        status_val = job_status.get("status", "unknown")
        if status_val in ("queued", "processing"):
            show_progress(job_status)
        elif status_val == "failed":
            st.error(f"Processing failed: {job_status.get('error')}")
            if st.button("Retry transcription", type="primary"):
                dispatch_job(current_recording_id, current_filename)
                st.rerun()
            st.stop()
        elif status_val == "completed":
            transcript_res = client.get_transcript(current_recording_id)
            if not transcript_res:
                time.sleep(0.5)
                st.rerun()
    else:
        with st.container(border=True):
            st.info(f"**Not transcribed yet.** `{current_filename}` has been uploaded but not processed.")
            if st.button("Transcribe this recording", type="primary"):
                try:
                    dispatch_job(current_recording_id, current_filename)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to submit transcription job: {exc}")
        st.stop()

if not transcript_res:
    st.info("No transcript found yet.")
    st.stop()

if st.session_state.get("job_id"):
    _final = client.get_job_status(st.session_state["job_id"]) or {}
    if _final.get("status") == "failed":
        st.error(f"Re-transcription failed: {_final.get('error')}. Showing the previous transcript.")
    st.session_state["job_id"] = None

# ---------------------------------------------------------------- DataFrames
raw_words = transcript_res.get("words", [])
raw_segs = transcript_res.get("segments", [])

words_df = pd.DataFrame(raw_words)
if not words_df.empty:
    if "speaker" not in words_df.columns:
        words_df["speaker"] = "Speaker"
    words_df["duration"] = words_df["end"] - words_df["start"]
    words_df["tier"] = words_df["confidence"].apply(lambda c: assign_tier(c, low_t, high_t))
    words_df["Confidence level"] = words_df["tier"].map(TIER_LABEL)
    words_df["Marker"] = words_df["tier"].map(TIER_SYMBOL)

segs_df = pd.DataFrame([{k: v for k, v in s.items() if k != "words"} for s in raw_segs])
if not segs_df.empty:
    segs_df["segment"] = [f"Turn {i + 1}" for i in range(len(segs_df))]
    segs_df["duration"] = segs_df["end"] - segs_df["start"]
    segs_df["n_words"] = [len(s.get("words", [])) for s in raw_segs]

# Display-only speaker numbering (raw labels such as SPEAKER_UNASSIGNED -> "Speaker 1").
_seg_spk = segs_df["speaker"].tolist() if "speaker" in segs_df.columns else []
_word_spk = words_df["speaker"].tolist() if "speaker" in words_df.columns else []
spk_map = speaker_number_map(_seg_spk, _word_spk)


def _num_speaker(raw) -> str:
    key = "SPEAKER_UNASSIGNED" if raw is None or str(raw).strip() == "" else str(raw)
    return spk_map.get(key, "Speaker 1")


if "speaker" in segs_df.columns:
    segs_df["speaker"] = segs_df["speaker"].map(_num_speaker)
if "speaker" in words_df.columns:
    words_df["speaker"] = words_df["speaker"].map(_num_speaker)

color_scale = alt.Scale(domain=[TIER_LABEL[t] for t in TIERS], range=[colors[t] for t in TIERS])
shape_scale = alt.Scale(domain=[TIER_LABEL[t] for t in TIERS], range=[TIER_SHAPE[t] for t in TIERS])

# ---------------------------------------------------------------- Overview
total_dur = float(transcript_res.get("duration", 0.0))
n_words = len(words_df)
mean_conf = float(words_df["confidence"].mean()) if not words_df.empty else 0.0
n_low = int((words_df["tier"] == "low").sum()) if not words_df.empty else 0
speakers_count = len(set(segs_df["speaker"])) if "speaker" in segs_df.columns else len(spk_map)

with st.container(border=True):
    head_l, head_r = st.columns([3, 2], vertical_alignment="center")
    with head_l:
        st.markdown(f"#### {transcript_res.get('filename', current_filename)}")
        st.caption(f"Recording `{current_recording_id[:8]}` · {fmt_duration(total_dur)} long")
    with head_r:
        st.audio(client.get_stream_url(current_recording_id))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Duration", fmt_duration(total_dur), border=True)
c2.metric("Total Words", n_words, border=True)
c3.metric("Avg Confidence", f"{mean_conf:.0%}", border=True)
c4.metric("Low-Confidence Words", f"{n_low} ({n_low / max(1, n_words):.0%})", border=True)
c5.metric("Speakers", speakers_count or "—", border=True)

# ---------------------------------------------------------------- Tabs
tab_tx, tab_charts, tab_learn, tab_data, tab_export, tab_plugins = st.tabs(
    ["📝 Transcript", "📊 Analytics", "🧠 Learning", "📋 Data", "💾 Export", "🧩 Plugin results"]
)

with tab_tx:
    render_accessible_transcript(
        segs_df=segs_df,
        words_df=words_df,
        colors=colors,
        tier_label=TIER_LABEL,
        tier_symbol=TIER_SYMBOL,
        low_t=low_t,
        high_t=high_t,
        font_px=font_px,
    )
    render_correction_form(client, current_recording_id, raw_segs, _num_speaker)

with tab_charts:
    with st.container(border=True):
        st.markdown("**Word confidence over time**")
        st.caption("Each point is a word. ▲ low, ■ medium, ● high confidence. Drag to pan, scroll to zoom.")
        render_confidence_scatter(words_df, color_scale, shape_scale)

    col_a, col_b = st.columns(2)
    with col_a, st.container(border=True):
        st.markdown("**Confidence distribution**")
        render_confidence_histogram(words_df, color_scale)
    with col_b, st.container(border=True):
        st.markdown("**Average confidence per turn**")
        render_segment_confidence_bar(segs_df, colors["high"])

    col_c, col_d = st.columns(2)
    with col_c, st.container(border=True):
        st.markdown("**Speech turns**")
        st.caption("Bars are speech; gaps are silence.")
        render_speech_timeline(segs_df, colors["high"])
    with col_d, st.container(border=True):
        st.markdown("**Calibration**")
        try:
            analytics = client.get_analytics(current_recording_id)
            m1, m2 = st.columns(2)
            m1.metric("Brier score", f"{analytics.get('brier_score', 0.0):.4f}", help="Lower is better.")
            m2.metric("ECE", f"{analytics.get('ece', 0.0):.4f}", help="Expected calibration error; lower is better.")
            if analytics.get("label_source") == "user_corrections":
                st.caption(f"Measured against {analytics.get('n_labels', 0)} words you reviewed (kept = correct).")
            else:
                st.caption(
                    "Estimated: no reviewed words yet, so these use a confidence-based proxy. "
                    "Correct a few turns to get real measurements."
                )
            render_reliability_diagram(analytics.get("reliability_diagram", []))
        except Exception as e:
            st.info(f"Calibration metrics unavailable: {e}")

def _retranscribe() -> None:
    try:
        dispatch_job(current_recording_id, current_filename, force=True)
    except Exception as exc:
        st.session_state["retranscribe_error"] = str(exc)


with tab_learn:
    if err := st.session_state.pop("retranscribe_error", None):
        st.error(f"Could not start re-transcription: {err}")
    render_learning_panel(
        client, current_recording_id, colors, transcript_res.get("summary", {}), on_retranscribe=_retranscribe
    )

with tab_data:
    render_accessible_tables(segs_df, words_df)

with tab_export:
    st.markdown("Download the speaker-attributed transcript.")
    stem = Path(current_filename).stem or "transcript"
    exports = [
        ("json", "JSON", "application/json", "Full result with word timings and confidence"),
        ("txt", "Plain text", "text/plain", "Readable transcript"),
        ("srt", "SubRip (SRT)", "text/plain", "Subtitles for video players"),
        ("vtt", "WebVTT", "text/vtt", "Subtitles for the web"),
    ]
    cols = st.columns(len(exports))
    for col, (fmt, label, mime, desc) in zip(cols, exports):
        with col, st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(desc)
            try:
                resp = requests.get(client.get_export_url(current_recording_id, fmt), timeout=15)
                col_data = resp.text
                # Apply the same display-only speaker numbering to the downloaded text.
                for raw, nice in sorted(spk_map.items(), key=lambda kv: -len(kv[0])):
                    col_data = col_data.replace(raw, nice)
                st.download_button(
                    "Download", col_data, f"{stem}.{fmt}", mime, key=f"dl_{fmt}", width="stretch"
                )
            except Exception:
                st.button("Unavailable", key=f"dl_{fmt}_disabled", disabled=True, width="stretch")

with tab_plugins:
    render_plugin_results(
        client,
        current_recording_id,
        colors,
        _num_speaker,
        is_greyscale=palette_name.startswith("Greyscale"),
    )
