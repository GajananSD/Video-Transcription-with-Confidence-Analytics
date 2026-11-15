"""Confidence Analytics, Calibration Metrics, and Reliability Diagram Endpoints."""
from __future__ import annotations

from typing import Any
from fastapi import APIRouter, HTTPException

from adaptive.confidence import ConfidenceCalibrator
from backend.api.v1.responses import error, json_example
from backend.services.feedback_service import FeedbackService
from backend.services.transcription_service import TranscriptionService

router = APIRouter()


@router.get(
    "/analytics/{recording_id}",
    tags=["Analytics"],
    responses={
        200: json_example(
            "Confidence diagnostics. `calibration` holds the calibrator's benchmark metrics; the top-level "
            "`brier_score`/`ece` are computed against words users reviewed (`label_source: user_corrections`), "
            "or against confidence-derived proxy labels until enough reviews exist (`label_source: proxy`). "
            "`reliability_diagram` has one entry per confidence bin (10 bins).",
            {
                "recording_id": "4d595f83-4dda-44f3-95c2-12f2d1228309",
                "filename": "meeting.mp4",
                "total_words": 197,
                "mean_confidence": 0.716,
                "mean_raw_confidence": 0.687,
                "calibration": {
                    "method": "isotonic_regression",
                    "version": "1.0.0",
                    "is_calibrated": True,
                    "brier_score": 0.088,
                    "ece": 0.042,
                },
                "brier_score": 0.0421,
                "ece": 0.1614,
                "label_source": "user_corrections",
                "n_labels": 240,
                "tiers": {
                    "low": {"count": 28, "ratio": 0.142},
                    "mid": {"count": 24, "ratio": 0.122},
                    "high": {"count": 145, "ratio": 0.736},
                },
                "reliability_diagram": [
                    {"bin_start": 0.0, "bin_end": 0.1, "bin_mid": 0.05, "confidence": 0.071, "accuracy": 0.0, "count": 6}
                ],
                "speakers_count": 3,
                "duration": 83.77,
            },
        ),
        404: error("No transcript exists for this recording", "Transcript not found for analytics."),
    },
)
def get_analytics(recording_id: str) -> dict[str, Any]:
    """Retrieve detailed confidence diagnostics and calibration metrics."""
    transcript = TranscriptionService.get_transcript(recording_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found for analytics.")

    confidences = [w.confidence for w in transcript.words]
    raw_confidences = [w.raw_confidence for w in transcript.words]

    # Calibration quality needs ground-truth labels. Use the words users have reviewed
    # (kept = correct, changed = wrong), scored with the current calibrator. Until there
    # are enough of them, fall back to a confidence-derived proxy and say so.
    calibrator = TranscriptionService.get_calibrator()
    FeedbackService.ensure_loaded()
    sample_raw, sample_labels = FeedbackService.get_samples()
    if len(sample_labels) >= calibrator.MIN_FIT_SAMPLES:
        label_source = "user_corrections"
        eval_conf = [calibrator.calibrate_score(r) for r in sample_raw]
        eval_labels = sample_labels
    else:
        label_source = "proxy"
        eval_conf = confidences
        eval_labels = [1 if c >= 0.50 else 0 for c in confidences]

    ece = ConfidenceCalibrator.compute_ece(eval_conf, eval_labels, n_bins=10)
    brier = ConfidenceCalibrator.compute_brier_score(eval_conf, eval_labels)
    diagram = ConfidenceCalibrator.compute_reliability_diagram(eval_conf, eval_labels, n_bins=10)

    # Tier breakdown
    low_count = sum(1 for c in confidences if c < 0.30)
    mid_count = sum(1 for c in confidences if 0.30 <= c < 0.60)
    high_count = sum(1 for c in confidences if c >= 0.60)

    return {
        "recording_id": recording_id,
        "filename": transcript.filename,
        "total_words": len(transcript.words),
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
        "mean_raw_confidence": round(sum(raw_confidences) / len(raw_confidences), 3) if raw_confidences else 0.0,
        "calibration": transcript.calibration.model_dump(),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "label_source": label_source,
        "n_labels": len(eval_labels),
        "tiers": {
            "low": {"count": low_count, "ratio": round(low_count / max(1, len(confidences)), 3)},
            "mid": {"count": mid_count, "ratio": round(mid_count / max(1, len(confidences)), 3)},
            "high": {"count": high_count, "ratio": round(high_count / max(1, len(confidences)), 3)},
        },
        "reliability_diagram": diagram,
        "speakers_count": len(transcript.speakers),
        "duration": transcript.duration,
    }
