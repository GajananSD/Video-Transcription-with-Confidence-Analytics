"""The extension contract: what we send (TranscriptEvent) and what we accept back (InsightCard).

Extensions return data only, never HTML or colours. The dashboard decides how it looks,
so every card follows the user's colourblind-safe palette and the ▲ ■ ● symbols.
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "cia-extension/1"

# transcript.ready   -> first time an extension sees a recording's transcript
# transcript.updated -> the transcript changed afterwards (user correction, re-transcription)
EVENTS = ("transcript.ready", "transcript.updated")
EventName = Literal["transcript.ready", "transcript.updated"]


# ------------------------------------------------------------------ outgoing event
class EventRecording(BaseModel):
    recording_id: str
    filename: str
    duration: float


class EventTurn(BaseModel):
    turn_id: str
    speaker: str
    start: float
    end: float
    text: str
    avg_confidence: float


class EventWord(BaseModel):
    word: str
    start: float
    end: float
    speaker: str
    confidence: float


class TranscriptEvent(BaseModel):
    contract: str = CONTRACT_VERSION
    event: EventName
    recording: EventRecording
    speakers: list[str]
    summary: dict[str, Any]
    turns: list[EventTurn]
    words: list[EventWord]


def build_event(event: str, transcript: dict[str, Any]) -> TranscriptEvent:
    """Build the JSON body sent to extensions from a stored transcript result."""
    summary = transcript.get("summary") or {}
    words = transcript.get("words") or []
    confs = [float(w.get("confidence") or 0.0) for w in words]
    return TranscriptEvent(
        event=event,
        recording=EventRecording(
            recording_id=transcript.get("recording_id", ""),
            filename=transcript.get("filename", ""),
            duration=float(transcript.get("duration") or 0.0),
        ),
        speakers=list(transcript.get("speakers") or []),
        summary={
            "word_count": len(words),
            "turn_count": len(transcript.get("segments") or []),
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
            "low_confidence_word_count": summary.get("low_confidence_word_count", 0),
            "diarization_model": summary.get("diarization_model"),
        },
        turns=[
            EventTurn(
                turn_id=str(s.get("turn_id", f"turn_{i:03d}")),
                speaker=str(s.get("speaker") or "SPEAKER_UNASSIGNED"),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")),
                avg_confidence=float(s.get("avg_confidence") or 0.0),
            )
            for i, s in enumerate(transcript.get("segments") or [])
        ],
        words=[
            EventWord(
                word=str(w.get("word", "")),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                speaker=str(w.get("speaker") or "SPEAKER_UNASSIGNED"),
                confidence=float(w.get("confidence") or 0.0),
            )
            for w in words
        ],
    )


# ------------------------------------------------------------------ incoming card
class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class Stat(_Strict):
    name: str = Field(..., max_length=40)
    value: Union[str, int, float] = Field(...)
    hint: Optional[str] = Field(None, max_length=120)


class Flag(_Strict):
    """One observation. `level` maps onto the confidence tiers: attention ▲, info ■, good ●."""
    level: Literal["attention", "info", "good"] = "info"
    text: str = Field(..., max_length=300)
    at: Optional[float] = Field(None, ge=0, description="Seconds into the recording, if it points at a moment.")
    speaker: Optional[str] = Field(None, max_length=60)


class SeriesPoint(_Strict):
    x: float
    y: float
    group: Optional[str] = Field(None, max_length=60)


class Series(_Strict):
    title: str = Field(..., max_length=80)
    kind: Literal["line", "bar"] = "line"
    x_label: str = Field("Time (seconds)", max_length=40)
    y_label: str = Field("Value", max_length=40)
    points: list[SeriesPoint] = Field(..., min_length=1, max_length=1000)


class Table(_Strict):
    columns: list[str] = Field(..., min_length=1, max_length=8)
    rows: list[list[Union[str, int, float, None]]] = Field(..., max_length=200)


class InsightCard(_Strict):
    """What an extension answers. Only `heading` is required."""
    heading: str = Field(..., min_length=1, max_length=80)
    note: Optional[str] = Field(None, max_length=600)
    stats: list[Stat] = Field(default_factory=list, max_length=6)
    flags: list[Flag] = Field(default_factory=list, max_length=40)
    series: Optional[Series] = None
    table: Optional[Table] = None


def example_card() -> InsightCard:
    return InsightCard(
        heading="Talk balance",
        note="Speaker 1 did most of the talking.",
        stats=[
            Stat(name="Words per minute", value=148, hint="Across all speakers"),
            Stat(name="Longest monologue", value="0:42"),
        ],
        flags=[
            Flag(level="attention", text="Overlapping speech around here", at=73.2),
            Flag(level="good", text="Both speakers took part", speaker="SPEAKER_01"),
        ],
        series=Series(
            title="Words per 10-second window",
            kind="bar",
            y_label="Words",
            points=[SeriesPoint(x=0, y=21, group="SPEAKER_00"), SeriesPoint(x=10, y=18, group="SPEAKER_01")],
        ),
        table=Table(columns=["Speaker", "Talk time"], rows=[["SPEAKER_00", "1:12"], ["SPEAKER_01", "0:31"]]),
    )
