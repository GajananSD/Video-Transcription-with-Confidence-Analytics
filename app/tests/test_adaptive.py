"""Tests for the adaptive core: learning vocabulary and calibration from user corrections."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adaptive.confidence import ConfidenceCalibrator
from adaptive.feedback_loop import FeedbackLoopManager, correctness_labels, introduced_terms, tokenize
from backend.database.db import get_db_connection
from backend.database.models import RecordingRecord, RecordingRepository, TranscriptRepository
from backend.services.audio_service import AudioService
from backend.services.feedback_service import FeedbackService
from backend.services.transcription_service import TranscriptionService
from engine.base import BaseSpeechEngine
from engine.schemas import AlignedWord, ConversationTurn, TranscriptResult


# ---------------------------------------------------------------- pure adaptive logic
def test_introduced_terms_learns_only_the_fixed_word():
    orig = tokenize("Drop it or I'll use the laser!")
    corr = tokenize("Drop it or I'll use the Taser!")
    assert introduced_terms(orig, corr) == ["Taser"]  # not "Drop", punctuation stripped


def test_correctness_labels_mark_changed_words_wrong():
    labels = correctness_labels(tokenize("use the laser now"), tokenize("use the Taser now"))
    assert labels == [1, 1, 0, 1]


def test_vocabulary_is_ranked_by_frequency_and_feeds_prompt():
    mgr = FeedbackLoopManager()
    mgr.register_correction("the laser", "the Taser")
    mgr.register_correction("code for", "Code-4")
    mgr.register_correction("a laser", "a Taser")
    assert mgr.get_vocabulary()[0] == "Taser"
    assert mgr.get_whisper_prompt() == "Domain terms: Taser, Code-4."
    mgr.rebuild([("the laser", "the Taser")])
    assert mgr.get_vocabulary() == ["Taser"]


def test_calibrator_stays_on_prior_with_too_few_samples():
    cal = ConfidenceCalibrator()
    before = cal.calibrate_score(0.5)
    report = cal.fit([0.5] * 5, [1] * 5)
    assert report["fitted"] is False
    assert cal.calibrate_score(0.5) == before


def test_calibrator_learns_from_labels_and_improves_ece():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, 400)
    # The underlying model is over-confident: true accuracy is raw**2.
    labels = (rng.uniform(0, 1, 400) < raw ** 2).astype(int)

    cal = ConfidenceCalibrator()
    prior_mid = cal.calibrate_score(0.6)
    report = cal.fit(raw.tolist(), labels.tolist())

    assert report["fitted"] is True
    assert report["ece_learned"] < report["ece_prior"]
    assert cal.calibrate_score(0.6) < prior_mid  # learned to be less confident
    ys = [p["current"] for p in cal.curve()]
    assert ys == sorted(ys)  # mapping stays monotonic


# ---------------------------------------------------------------- end to end through the API
@pytest.fixture
def clean_feedback():
    def _wipe():
        conn = get_db_connection()
        with conn:
            conn.execute("DELETE FROM calibration_samples")
            conn.execute("DELETE FROM feedback")
        conn.close()
        FeedbackService.relearn()

    _wipe()
    yield
    _wipe()


def _seed_transcript() -> str:
    rec_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    RecordingRepository.create(RecordingRecord(rec_id, "stop.wav", "/tmp/none.wav", 3.0, 10, rec_id, now))
    texts = ["Drop", "it", "or", "I'll", "use", "the", "laser!"]
    words = [
        AlignedWord(word=w, start=i * 0.4, end=(i + 1) * 0.4, speaker="SPEAKER_00",
                    confidence=0.8, raw_confidence=0.3 if w == "laser!" else 0.9)
        for i, w in enumerate(texts)
    ]
    turn = ConversationTurn(turn_id="turn_001", speaker="SPEAKER_00", start=0.0, end=2.8,
                            text=" ".join(texts), words=words, avg_confidence=0.8)
    result = TranscriptResult(recording_id=rec_id, filename="stop.wav", duration=3.0,
                              speakers=["SPEAKER_00", "SPEAKER_01"], segments=[turn], words=words)
    TranscriptRepository.save(rec_id, result.model_dump(), now)
    return rec_id


def test_correction_updates_transcript_vocabulary_and_samples(client: TestClient, clean_feedback):
    rec_id = _seed_transcript()
    resp = client.post("/api/v1/feedback", json={
        "recording_id": rec_id, "turn_id": "turn_001", "original_text": "",
        "corrected_text": "Drop it or I'll use the Taser!", "corrected_speaker": "SPEAKER_01",
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error_type"] == "compound"
    assert body["extracted_terms"] == ["Taser"]
    assert body["words_changed"] == 1 and body["samples_added"] == 7

    tx = client.get(f"/api/v1/transcriptions/{rec_id}").json()
    turn = tx["segments"][0]
    assert turn["text"] == "Drop it or I'll use the Taser!"
    assert turn["speaker"] == "SPEAKER_01"
    fixed = [w for w in tx["words"] if w["word"] == "Taser!"][0]
    assert fixed["alignment_status"] == "corrected" and fixed["confidence"] == 1.0
    assert len(tx["words"]) == 7

    status = client.get("/api/v1/adaptive/status").json()
    assert status["corrections"] == 1
    assert status["reviewed_words"] == 7 and status["wrong_words"] == 1
    assert status["whisper_prompt"] == "Domain terms: Taser."
    assert client.get("/api/v1/feedback/vocabulary").json() == ["Taser"]

    # Forgetting the correction un-learns it.
    fid = client.get("/api/v1/feedback", params={"recording_id": rec_id}).json()[0]["feedback_id"]
    assert client.delete(f"/api/v1/feedback/{fid}").status_code == 200
    assert client.get("/api/v1/feedback/vocabulary").json() == []
    assert client.delete(f"/api/v1/feedback/{fid}").status_code == 404


def test_unknown_turn_is_rejected(client: TestClient, clean_feedback):
    rec_id = _seed_transcript()
    resp = client.post("/api/v1/feedback", json={
        "recording_id": rec_id, "turn_id": "nope", "original_text": "", "corrected_text": "x",
    })
    assert resp.status_code == 404


def test_learned_vocabulary_is_sent_to_the_next_transcription(clean_feedback, monkeypatch):
    rec_id = _seed_transcript()
    from engine.schemas import UserFeedbackItem

    FeedbackService.record_feedback(UserFeedbackItem(
        recording_id=rec_id, turn_id="turn_001", original_text="", corrected_text="Drop it or I'll use the Taser!",
    ))

    seen: dict = {}

    class CapturingEngine(BaseSpeechEngine):
        def process(self, audio_path, recording_id, filename=None, duration=None, progress_callback=None, options=None):
            seen.update(options or {})
            return TranscriptResult(recording_id=recording_id, filename="stop.wav", duration=3.0)

    monkeypatch.setattr(AudioService, "prepare_16k_mono_audio", staticmethod(lambda rid: "/tmp/none.wav"))
    previous = TranscriptionService._engine
    TranscriptionService.set_engine(CapturingEngine())
    try:
        result = TranscriptionService.execute_transcription(rec_id)
    finally:
        TranscriptionService._engine = previous

    assert seen["initial_prompt"] == "Domain terms: Taser."
    assert result.summary["adaptive"]["learned_terms_used"] == ["Taser"]
