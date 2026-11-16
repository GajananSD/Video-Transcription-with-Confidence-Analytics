"""Example extension: "Talk balance".

A separate tiny web service (not part of the dashboard) that shows how an extension works.
It receives a TranscriptEvent and answers with an InsightCard about who talked how much,
speaking pace, filler words and stretches the ASR was unsure about.

Run (from app/):
    uvicorn extensions_examples.talk_balance:app --port 8300
Then register it in the dashboard's Extensions tab with the URL
    http://127.0.0.1:8300/insight
"""
from __future__ import annotations

import re
from collections import defaultdict

from fastapi import FastAPI

app = FastAPI(title="Talk balance (example extension)")

FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "uhm", "like", "basically", "actually"}
WINDOW_S = 15.0
LOW_CONF = 0.30


def _mmss(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def _clean(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


@app.get("/")
def about() -> dict:
    return {"extension": "talk_balance", "post_events_to": "/insight"}


@app.post("/insight")
def insight(event: dict) -> dict:
    turns = event.get("turns") or []
    words = event.get("words") or []
    if not turns:
        return {"heading": "Talk balance", "note": "No speech turns in this transcript yet."}

    talk = defaultdict(float)
    n_words = defaultdict(int)
    fillers = defaultdict(int)
    for t in turns:
        talk[t["speaker"]] += max(0.0, t["end"] - t["start"])
    for w in words:
        n_words[w["speaker"]] += 1
        if _clean(w["word"]) in FILLERS:
            fillers[w["speaker"]] += 1

    total_talk = sum(talk.values()) or 1.0
    speakers = sorted(talk, key=talk.get, reverse=True)
    lead = speakers[0]
    lead_share = talk[lead] / total_talk
    longest = max(turns, key=lambda t: t["end"] - t["start"])
    wpm_all = 60.0 * len(words) / total_talk

    flags = []
    if len(speakers) > 1 and lead_share > 0.7:
        flags.append({"level": "attention", "speaker": lead,
                      "text": f"One speaker holds {lead_share:.0%} of the talk time."})
    elif len(speakers) > 1:
        flags.append({"level": "good", "text": "Talk time is shared fairly evenly."})
    for spk in speakers:
        if n_words[spk] >= 20 and fillers[spk] / n_words[spk] > 0.05:
            flags.append({"level": "info", "speaker": spk,
                          "text": f"{fillers[spk]} filler words ({fillers[spk] / n_words[spk]:.0%} of their words)."})

    # Unsure stretches: 3+ low-confidence words within 5 seconds -> worth listening again.
    low = [w for w in words if w["confidence"] < LOW_CONF]
    i = 0
    while i < len(low) and len([f for f in flags if f.get("at") is not None]) < 8:
        j = i
        while j + 1 < len(low) and low[j + 1]["start"] - low[i]["start"] <= 5.0:
            j += 1
        if j - i + 1 >= 3:
            flags.append({"level": "attention", "at": round(low[i]["start"], 1), "speaker": low[i]["speaker"],
                          "text": f"{j - i + 1} low-confidence words close together: worth re-listening."})
            i = j + 1
        else:
            i += 1

    # Words per 15-second window, per speaker.
    windows = defaultdict(int)
    for w in words:
        windows[(int(w["start"] // WINDOW_S) * WINDOW_S, w["speaker"])] += 1
    points = [{"x": x, "y": y, "group": spk} for (x, spk), y in sorted(windows.items())]

    rows = [
        [spk, _mmss(talk[spk]), f"{talk[spk] / total_talk:.0%}",
         round(60.0 * n_words[spk] / talk[spk]) if talk[spk] else 0, fillers[spk]]
        for spk in speakers
    ]

    return {
        "heading": "Talk balance",
        "note": f"{len(speakers)} speaker(s). The most active one talked for {_mmss(talk[lead])} "
                f"({lead_share:.0%} of all speech).",
        "stats": [
            {"name": "Speaking pace", "value": f"{wpm_all:.0f} wpm", "hint": "Words per minute of speech"},
            {"name": "Longest turn", "value": _mmss(longest["end"] - longest["start"]),
             "hint": f"Starts at {_mmss(longest['start'])}"},
            {"name": "Filler words", "value": sum(fillers.values())},
        ],
        "flags": flags,
        "series": {"title": f"Words per {WINDOW_S:.0f}-second window", "kind": "bar",
                   "x_label": "Time (seconds)", "y_label": "Words", "points": points} if points else None,
        "table": {"columns": ["Speaker", "Talk time", "Share", "Words/min", "Fillers"], "rows": rows},
    }
