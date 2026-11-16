"""Plugins UI: register other teams' plugins (top-level page) and show what they sent (results tab).

Cards are drawn from plain data with this dashboard's own styles. Extensions cannot send
HTML or colours: flag levels reuse the confidence tiers (attention ▲ / info ■ / good ●), so
every card follows the colour scheme chosen in the sidebar and never relies on colour alone.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import altair as alt
import pandas as pd
import streamlit as st

from .transcript_view import SPEAKER_COLORS

EVENT_LABEL = {
    "transcript.ready": "Transcript ready",
    "transcript.updated": "Transcript corrected or re-run",
}
LEVEL_TIER = {"attention": "low", "info": "mid", "good": "high"}
LEVEL_LABEL = {"attention": "Needs attention", "info": "Note", "good": "Looks good"}
GREY_SERIES = ["#222222", "#777777", "#AAAAAA", "#444444", "#999999", "#555555"]
SERIES_SHAPES = ["circle", "square", "triangle-up", "diamond", "cross", "triangle-down"]


def _safe(text: Any) -> str:
    """Escape for HTML and neutralise Markdown, so extension text is shown literally."""
    s = html.escape(str(text), quote=True)
    for ch, ent in (("[", "&#91;"), ("]", "&#93;"), ("*", "&#42;"), ("_", "&#95;"), ("`", "&#96;"),
                    ("#", "&#35;"), ("~", "&#126;"), ("|", "&#124;"), ("$", "&#36;")):
        s = s.replace(ch, ent)
    return s


def _mmss(seconds: float) -> str:
    s = max(0, int(round(float(seconds))))
    return f"{s // 60}:{s % 60:02d}"


def _ago(iso: str) -> str:
    try:
        secs = (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()
    except Exception:
        return ""
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)} min ago"
    if secs < 86400:
        return f"{int(secs // 3600)} h ago"
    return f"{int(secs // 86400)} d ago"


# ====================================================================== Plugins page
def render_plugin_manager(
    client,
    recording_id: Optional[str] = None,
    colors: Optional[dict[str, str]] = None,
    speaker_name: Callable[[str], str] = lambda s: s,
) -> None:
    """The top-level Plugins page: register other teams' plugins, test, pause and remove them."""
    st.markdown("### 🧩 Plugins")
    st.markdown(
        "A plugin is **another team's web service**, usually running on their laptop. "
        "Register it here once; after that it works automatically:"
    )
    st.markdown(
        '<ol class="xi-steps">'
        "<li><strong>Register</strong> the team's plugin and its URL below.</li>"
        "<li><strong>Transcribe</strong> a video as usual.</li>"
        "<li>When the transcript is ready, it is <strong>sent to every active plugin</strong> "
        "(and again after you correct it).</li>"
        "<li>Open the recording's <strong>🧩 Plugin results</strong> tab to see what each plugin sent back.</li>"
        "</ol>",
        unsafe_allow_html=True,
    )

    if msg := st.session_state.pop("ext_flash", None):
        (st.success if msg[0] == "ok" else st.error)(msg[1])

    left, right = st.columns([2, 3], gap="medium")

    with left, st.container(border=True):
        st.markdown("**Register a plugin**")
        with st.form("ext_add", clear_on_submit=True, border=False):
            team = st.text_input("Team name", placeholder="Team VibeCheck", max_chars=60)
            name = st.text_input("Plugin name", placeholder="Conversation Insights", max_chars=60)
            url = st.text_input(
                "Plugin URL", placeholder="http://192.168.1.30:8300/insight",
                help="The address of their service, e.g. http://<their-laptop-ip>:<port>/<path>. "
                     "The transcript is sent here with an HTTP POST.",
            )
            events = st.multiselect(
                "Send the transcript when", list(EVENT_LABEL), default=list(EVENT_LABEL),
                format_func=lambda e: EVENT_LABEL[e],
            )
            if st.form_submit_button("Register plugin", type="primary"):
                if not team.strip() or not name.strip() or not url.strip():
                    st.session_state["ext_flash"] = ("err", "Please fill in the team name, plugin name and URL.")
                else:
                    try:
                        client.register_extension(name.strip(), url.strip(), events, team=team.strip())
                        st.session_state["ext_flash"] = (
                            "ok", f"Registered “{name.strip()}” from {team.strip()}. Click **Send test** to check it.")
                    except Exception as exc:
                        st.session_state["ext_flash"] = ("err", f"Could not register: {exc}")
                st.rerun()
        with st.expander("For plugin authors: what is sent and what to answer"):
            try:
                contract = client.get_extension_contract()
                st.caption("The plugin URL receives this JSON by POST (shortened):")
                ev = dict(contract["example_event"])
                ev["words"] = ev["words"][:2] + [{"…": f"{len(contract['example_event']['words']) - 2} more"}]
                st.code(json.dumps(ev, indent=2), language="json")
                st.caption("…and must answer with an insight card. Only `heading` is required:")
                st.code(json.dumps(contract["example_card"], indent=2), language="json")
                for rule in contract.get("rules", []):
                    st.markdown(f"- {rule}")
            except Exception as exc:
                st.info(f"Contract unavailable: {exc}")

    with right:
        try:
            exts = client.list_extensions()
        except Exception as exc:
            st.warning(f"Could not load plugins: {exc}")
            return
        st.markdown(f"**Registered plugins** · {len(exts)}")
        if not exts:
            with st.container(border=True):
                st.markdown("Nothing registered yet.")
                st.caption(
                    "To try it alone, run the bundled example from `app/`: "
                    "`uvicorn extensions_examples.talk_balance:app --port 8300`, then register "
                    "`http://127.0.0.1:8300/insight`."
                )
            return
        for ext in exts:
            _render_plugin_row(client, ext, recording_id, colors or {}, speaker_name)


def _render_plugin_row(
    client, ext: dict, recording_id: Optional[str], colors: dict[str, str], speaker_name: Callable[[str], str]
) -> None:
    eid = ext["extension_id"]
    with st.container(border=True):
        top_l, top_r = st.columns([3, 1], vertical_alignment="center")
        with top_l:
            state = "● Active" if ext["enabled"] else "○ Paused"
            team = f'<span class="xi-team">{_safe(ext.get("team") or "No team")}</span>'
            st.markdown(
                f'<div class="xi-row-head">{team}<span class="xi-name">{_safe(ext["name"])}</span>'
                f'<span class="xi-state">{state}</span></div>'
                f'<div class="xi-url">{_safe(ext["url"])}</div>',
                unsafe_allow_html=True,
            )
            st.caption("Sent when: " + ", ".join(EVENT_LABEL.get(e, e).lower() for e in ext["events"]))
        with top_r:
            on = st.toggle("Active", value=ext["enabled"], key=f"ext_on_{eid}")
            if on != ext["enabled"]:
                client.update_extension(eid, enabled=on)
                st.rerun()

        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("Send test", key=f"ext_test_{eid}", width="stretch",
                     help="Send a small sample transcript and check the answer. Nothing is saved."):
            with st.spinner("Waiting for the plugin…"):
                st.session_state[f"ext_test_res_{eid}"] = client.test_extension(eid, recording_id)
        if b2.button("Remove", key=f"ext_del_{eid}", width="stretch"):
            client.delete_extension(eid)
            st.session_state.pop(f"ext_test_res_{eid}", None)
            st.session_state["ext_flash"] = ("ok", f"Removed “{ext['name']}”.")
            st.rerun()

        res = st.session_state.get(f"ext_test_res_{eid}")
        if res:
            if res["ok"]:
                st.success(f"● Connected: a valid answer arrived in {res['elapsed_ms']} ms.")
                with st.expander("Preview of the answer"):
                    _card_body(res["card"], colors, speaker_name, f"pv_{eid}")
            else:
                st.error(f"▲ Test failed: {res['error']}")


# ====================================================================== Plugin results tab
def render_plugin_results(
    client,
    recording_id: str,
    colors: dict[str, str],
    speaker_name: Callable[[str], str],
    is_greyscale: bool = False,
) -> None:
    """The recording's 'Plugin results' tab: what each registered plugin sent back."""
    try:
        exts = client.list_extensions()
    except Exception as exc:
        st.warning(f"Could not load plugins: {exc}")
        return
    active = [e for e in exts if e["enabled"]]
    if not active:
        st.info("No active plugins. Register one in **🧩 Plugins** at the top of the page, "
                "then come back here (or transcribe a new video).")
        return

    head_l, head_r = st.columns([4, 1], vertical_alignment="bottom")
    with head_l:
        st.caption(
            f"The transcript is sent to {len(active)} active plugin{'s' if len(active) != 1 else ''} "
            "when it is ready and after each correction. Below is what each one answered."
        )
    with head_r:
        resend = st.button("↻ Send again", width="stretch", help="Send this transcript to every active plugin now.")

    try:
        with st.spinner("Waiting for plugins…" if resend else "Loading plugin results…"):
            results = client.get_insights(recording_id, refresh=resend)
    except Exception as exc:
        st.warning(f"Could not load plugin results: {exc}")
        return
    if not results:
        st.caption("No plugin has answered for this recording yet. Click **↻ Send again**.")
        return

    css_vars = f'--low:{colors["low"]};--mid:{colors["mid"]};--high:{colors["high"]}'
    n_ok = sum(1 for r in results if r["ok"])
    st.markdown(
        f'<div class="xi-summary" style="{css_vars}">'
        f'<span><span class="sym sym-high">●</span> {n_ok} answered</span>'
        f'<span><span class="sym sym-low">▲</span> {len(results) - n_ok} failed</span></div>',
        unsafe_allow_html=True,
    )

    for item in results:
        with st.container(border=True):
            meta = [EVENT_LABEL.get(item["event"], item["event"]), _ago(item["ran_at"])]
            if item.get("elapsed_ms") is not None:
                meta.append(f"answered in {item['elapsed_ms']} ms")
            status = ('<span class="xi-ok"><span class="sym sym-high">●</span> Received</span>' if item["ok"]
                      else '<span class="xi-bad"><span class="sym sym-low">▲</span> Failed</span>')
            outdated = (
                '<div class="xi-outdated">■ Based on an earlier version of the transcript. '
                "Click ↻ Send again to update.</div>" if item.get("outdated") else ""
            )
            st.markdown(
                f'<div class="xi-source" style="{css_vars}">'
                f'<span class="xi-team">{_safe(item.get("team") or "No team")}</span>'
                f'<span class="xi-name">{_safe(item["name"])}</span>{status}</div>'
                f'<div class="xi-url">{_safe(item.get("url", ""))} · {_safe(" · ".join(m for m in meta if m))}</div>'
                f"{outdated}",
                unsafe_allow_html=True,
            )
            if item["ok"] and item.get("card"):
                _card_body(item["card"], colors, speaker_name, item["extension_id"], is_greyscale)
            else:
                st.markdown(
                    f'<div class="xi-failed" style="{css_vars}"><span class="sym sym-low">▲</span> '
                    f'<strong>No usable answer.</strong> {_safe(item.get("error") or "Unknown error")}</div>',
                    unsafe_allow_html=True,
                )
            if item.get("raw"):
                with st.expander("Raw response received"):
                    try:
                        st.json(json.loads(item["raw"]), expanded=False)
                    except Exception:
                        st.code(item["raw"][:5000])


def _card_body(
    card: dict[str, Any],
    colors: dict[str, str],
    speaker_name: Callable[[str], str],
    key: str,
    is_greyscale: bool = False,
) -> None:
    colors = colors or {"low": "#D55E00", "mid": "#E69F00", "high": "#0072B2"}
    css_vars = f'--low:{colors["low"]};--mid:{colors["mid"]};--high:{colors["high"]}'
    parts = [f'<div class="xi-card" style="{css_vars}"><h4 class="xi-heading">{_safe(card["heading"])}</h4>']
    if card.get("note"):
        parts.append(f'<p class="xi-note">{_safe(card["note"])}</p>')
    if card.get("stats"):
        parts.append('<div class="xi-stats">')
        for s in card["stats"]:
            hint = f'<span class="xi-hint">{_safe(s["hint"])}</span>' if s.get("hint") else ""
            parts.append(
                f'<div class="xi-stat"><span class="xi-stat-name">{_safe(s["name"])}</span>'
                f'<span class="xi-stat-value">{_safe(s["value"])}</span>{hint}</div>'
            )
        parts.append("</div>")
    if card.get("flags"):
        parts.append('<ul class="xi-flags">')
        sym = {"low": "▲", "mid": "■", "high": "●"}
        for f in card["flags"]:
            tier = LEVEL_TIER.get(f.get("level", "info"), "mid")
            where = []
            if f.get("at") is not None:
                where.append(_mmss(f["at"]))
            if f.get("speaker"):
                where.append(speaker_name(f["speaker"]))
            loc = f'<span class="xi-where">{_safe(" · ".join(where))}</span>' if where else ""
            parts.append(
                f'<li class="xi-flag xi-flag-{tier}"><span class="sym sym-{tier}" aria-hidden="true">{sym[tier]}</span>'
                f'<span class="xi-level">{LEVEL_LABEL[f.get("level", "info")]}</span>{loc}'
                f'<span class="xi-text">{_safe(f["text"])}</span></li>'
            )
        parts.append("</ul>")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)

    if card.get("series"):
        _series_chart(card["series"], speaker_name, key, is_greyscale)
    if card.get("table"):
        t = card["table"]
        width = len(t["columns"])
        rows = [(list(r) + [None] * width)[:width] for r in t["rows"]]
        df = pd.DataFrame(rows, columns=t["columns"])
        for col in df.columns:  # speaker labels in tables get the same friendly names
            df[col] = df[col].map(lambda v: speaker_name(v) if isinstance(v, str) and v.startswith("SPEAKER_") else v)
        st.dataframe(df, hide_index=True, width="stretch", key=f"xi_tbl_{key}")


def _series_chart(series: dict[str, Any], speaker_name: Callable[[str], str], key: str, is_greyscale: bool) -> None:
    df = pd.DataFrame(series["points"])
    if "group" not in df.columns:
        df["group"] = series["title"]
    df["group"] = df["group"].fillna(series["title"]).map(
        lambda g: speaker_name(g) if isinstance(g, str) and g.startswith("SPEAKER_") else g
    )
    groups = list(dict.fromkeys(df["group"]))
    palette = GREY_SERIES if is_greyscale else SPEAKER_COLORS
    color = alt.Color("group:N", title=None, scale=alt.Scale(domain=groups, range=palette[: len(groups)] or palette),
                      legend=alt.Legend(orient="bottom") if len(groups) > 1 else None)
    x = alt.X("x:Q", title=series.get("x_label", ""))
    y = alt.Y("y:Q", title=series.get("y_label", ""))
    tooltip = [alt.Tooltip("group:N", title="Series"), alt.Tooltip("x:Q", title=series.get("x_label", "x")),
               alt.Tooltip("y:Q", title=series.get("y_label", "y"))]
    base = alt.Chart(df)
    if series.get("kind") == "bar":
        # Each x value is its own labelled slot (e.g. a time window), with grouped bars side by side.
        # Groups are named in the legend and tooltip, so colour is not the only cue.
        is_time = "time" in series.get("x_label", "").lower() or "second" in series.get("x_label", "").lower()
        df["x_label"] = df["x"].map(lambda v: _mmss(v) if is_time else f"{v:g}")
        order = sorted(set(df["x_label"]), key=lambda lbl: df.loc[df["x_label"] == lbl, "x"].iloc[0])
        base = alt.Chart(df)
        chart = base.mark_bar(cornerRadiusEnd=2).encode(
            x=alt.X("x_label:N", title=series.get("x_label", ""), sort=order, axis=alt.Axis(labelAngle=0)),
            y=y, color=color, xOffset=alt.XOffset("group:N", sort=groups), tooltip=tooltip,
        )
    else:
        shape = alt.Shape("group:N", scale=alt.Scale(domain=groups, range=SERIES_SHAPES[: len(groups)]), legend=None)
        chart = base.mark_line(point=True).encode(x=x, y=y, color=color, shape=shape,
                                                  strokeDash="group:N", tooltip=tooltip)
    st.markdown(f"**{_safe(series['title'])}**", unsafe_allow_html=True)
    st.altair_chart(chart.properties(height=200), width="stretch", key=f"xi_chart_{key}")
    with st.expander("Chart data as a table"):
        st.dataframe(
            df.drop(columns=["x_label"], errors="ignore").rename(columns={"x": series.get("x_label", "x"), "y": series.get("y_label", "y"), "group": "Series"}),
            hide_index=True, width="stretch", key=f"xi_chart_tbl_{key}",
        )
