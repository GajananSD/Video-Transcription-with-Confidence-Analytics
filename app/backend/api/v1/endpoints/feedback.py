"""User Corrections and Adaptive Learning Endpoints."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.api.v1.responses import error
from backend.services.feedback_service import FeedbackError, FeedbackService
from engine.schemas import FeedbackResponse, UserFeedbackItem

router = APIRouter()


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    responses={404: error("Unknown recording or turn", "Turn '<turn_id>' not found in this transcript.")},
)
def submit_feedback(feedback: UserFeedbackItem) -> FeedbackResponse:
    """Submit a correction to one transcript turn (text and/or speaker).

    The transcript is updated, the corrected words are learned as domain vocabulary
    (used in the Whisper prompt of future jobs), and every original word becomes a
    labelled sample that re-fits the confidence calibrator. Submitting the text
    unchanged marks the turn as verified-correct.
    """
    try:
        result = FeedbackService.record_feedback(feedback)
    except FeedbackError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FeedbackResponse(
        status="recorded",
        message="Correction recorded; vocabulary and calibration updated.",
        **result,
    )


@router.get("/feedback", tags=["Feedback"])
def list_feedback(recording_id: Optional[str] = Query(default=None)) -> list[dict[str, Any]]:
    """List corrections submitted by users, newest first."""
    return FeedbackService.list_feedback(recording_id)


@router.delete(
    "/feedback/{feedback_id}",
    tags=["Feedback"],
    responses={404: error("Unknown feedback_id", "Feedback '<feedback_id>' not found.")},
)
def delete_feedback(feedback_id: str) -> dict[str, Any]:
    """Forget a correction. Vocabulary and calibration are re-learned without it."""
    if not FeedbackService.delete_feedback(feedback_id):
        raise HTTPException(status_code=404, detail=f"Feedback '{feedback_id}' not found.")
    return {"status": "deleted", "feedback_id": feedback_id}


@router.get("/feedback/vocabulary", tags=["Feedback"])
def get_verified_vocabulary() -> list[str]:
    """Domain terms learned from corrections, most frequently corrected first."""
    return FeedbackService.get_accumulated_vocabulary()


@router.get("/adaptive/status", tags=["Feedback"])
def get_adaptive_status() -> dict[str, Any]:
    """What the adaptive core has learned: vocabulary, Whisper prompt, and calibration fit."""
    return FeedbackService.get_status()
