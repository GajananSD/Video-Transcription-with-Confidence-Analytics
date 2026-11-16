"""Extension endpoints: register external services that add insight cards under a transcript."""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from backend.api.v1.responses import error
from backend.extensions import EVENTS, ExtensionRepository, build_event, example_card, insights_for, send_test_event
from backend.extensions.runner import _SAMPLE_TRANSCRIPT

router = APIRouter()


def _check_url(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    v = v.strip()
    p = urlparse(v)
    if p.scheme not in ("http", "https") or not p.netloc:
        raise ValueError("must be an http:// or https:// URL")
    return v


def _check_events(v: Optional[list[str]]) -> Optional[list[str]]:
    if v is None:
        return v
    bad = [e for e in v if e not in EVENTS]
    if bad:
        raise ValueError(f"unknown event(s) {bad}; choose from {list(EVENTS)}")
    if not v:
        raise ValueError("subscribe to at least one event")
    return list(dict.fromkeys(v))


class ExtensionIn(BaseModel):
    team: str = Field("", max_length=60, examples=["Team VibeCheck"])
    name: str = Field(..., min_length=1, max_length=60, examples=["Talk balance"])
    url: str = Field(..., examples=["http://127.0.0.1:8300/insight"])
    events: list[str] = Field(default_factory=lambda: list(EVENTS), examples=[list(EVENTS)])

    _u = field_validator("url")(_check_url)
    _e = field_validator("events")(_check_events)


class ExtensionPatch(BaseModel):
    team: Optional[str] = Field(None, max_length=60)
    name: Optional[str] = Field(None, min_length=1, max_length=60)
    url: Optional[str] = None
    events: Optional[list[str]] = None
    enabled: Optional[bool] = None

    _u = field_validator("url")(_check_url)
    _e = field_validator("events")(_check_events)


_NOT_FOUND = {404: error("Unknown extension", "Extension '<extension_id>' not found.")}


def _get_or_404(extension_id: str):
    ext = ExtensionRepository.get(extension_id)
    if not ext:
        raise HTTPException(status_code=404, detail=f"Extension '{extension_id}' not found.")
    return ext


@router.get("/extensions", tags=["Extensions"])
def list_extensions() -> list[dict[str, Any]]:
    """All registered extensions, oldest first."""
    return [e.to_dict() for e in ExtensionRepository.list_all()]


@router.post("/extensions", tags=["Extensions"], status_code=201)
def register_extension(body: ExtensionIn) -> dict[str, Any]:
    """Register a service. It will receive a TranscriptEvent (see `/extensions/contract`) by POST."""
    return ExtensionRepository.create(body.name.strip(), body.url, body.events, team=body.team.strip()).to_dict()


@router.get("/extensions/contract", tags=["Extensions"])
def get_contract() -> dict[str, Any]:
    """An example of the event we send and of the insight card an extension must answer with."""
    return {
        "events": list(EVENTS),
        "example_event": build_event("transcript.ready", _SAMPLE_TRANSCRIPT).model_dump(),
        "example_card": example_card().model_dump(exclude_none=True),
        "rules": [
            "Answer within 8 seconds with HTTP 200 and a JSON insight card.",
            "Only `heading` is required. Send data only: no HTML, Markdown or colours.",
            "Flag levels map to the dashboard's confidence tiers: attention ▲, info ■, good ●.",
        ],
    }


@router.patch("/extensions/{extension_id}", tags=["Extensions"], responses=_NOT_FOUND)
def update_extension(extension_id: str, body: ExtensionPatch) -> dict[str, Any]:
    """Rename, move, re-subscribe, or turn an extension on/off."""
    _get_or_404(extension_id)
    return ExtensionRepository.update(extension_id, **body.model_dump()).to_dict()


@router.delete("/extensions/{extension_id}", tags=["Extensions"], responses=_NOT_FOUND)
def delete_extension(extension_id: str) -> dict[str, Any]:
    """Remove an extension and the cards it produced."""
    if not ExtensionRepository.delete(extension_id):
        raise HTTPException(status_code=404, detail=f"Extension '{extension_id}' not found.")
    return {"status": "deleted", "extension_id": extension_id}


@router.post("/extensions/{extension_id}/test", tags=["Extensions"], responses=_NOT_FOUND)
def test_extension(extension_id: str, recording_id: Optional[str] = Query(default=None)) -> dict[str, Any]:
    """Send a sample event (or a real recording's) and report whether the answer is a valid card."""
    return send_test_event(_get_or_404(extension_id), recording_id)


@router.get("/extensions/insights/{recording_id}", tags=["Extensions"])
def get_insights(recording_id: str, refresh: bool = Query(default=False)) -> list[dict[str, Any]]:
    """Insight cards for a recording. Extensions are only called when their answer is missing or
    outdated (or `refresh=true`); otherwise the stored answer is returned."""
    return insights_for(recording_id, refresh=refresh)
