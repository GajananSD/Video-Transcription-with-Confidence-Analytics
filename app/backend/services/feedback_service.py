"""User Corrections and the Adaptive Learning Loop.

A correction flows through the system like this::

    user edits a turn ──► feedback table (what changed)
                      ──► calibration_samples (each original word: kept=1 / changed=0)
                      ──► transcript is updated in place (corrected words get confidence 1.0)
                      ──► vocabulary re-learned  ──► Whisper initial_prompt for the next job
                      ──► calibrator re-fitted   ──► confidence of the next job

Everything learned is rebuilt from the database, so it survives restarts.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from adaptive.feedback_loop import FeedbackLoopManager, diff_words, tokenize
from backend.config import settings
from backend.database.db import get_db_connection
from backend.database.models import TranscriptRepository
from engine.schemas import AlignedWord, ConversationTurn, TranscriptResult, UserFeedbackItem

_EPS = 1e-4


class FeedbackError(ValueError):
    """Raised when a correction cannot be applied (unknown recording or turn)."""


class FeedbackService:
    """Records corrections and keeps the adaptive vocabulary and calibrator in sync with them."""

    _feedback_manager = FeedbackLoopManager()
    _loaded = False
    _lock = threading.RLock()

    # ------------------------------------------------------------------ state loading
    @classmethod
    def get_manager(cls) -> FeedbackLoopManager:
        cls.ensure_loaded()
        return cls._feedback_manager

    @classmethod
    def ensure_loaded(cls) -> None:
        """Rebuild learned state from the database once per process (e.g. after a restart)."""
        if not cls._loaded:
            cls.relearn()

    @classmethod
    def relearn(cls) -> dict[str, Any]:
        """Re-derive vocabulary and re-fit the calibrator from all stored corrections."""
        from .transcription_service import TranscriptionService

        with cls._lock:
            conn = get_db_connection()
            pairs = conn.execute(
                "SELECT original_text, corrected_text FROM feedback ORDER BY created_at"
            ).fetchall()
            samples = conn.execute("SELECT raw_confidence, label FROM calibration_samples").fetchall()
            conn.close()

            cls._feedback_manager.rebuild((p["original_text"] or "", p["corrected_text"] or "") for p in pairs)
            report = TranscriptionService.get_calibrator().fit(
                [s["raw_confidence"] for s in samples], [s["label"] for s in samples]
            )
            cls._loaded = True
            return report

    # ------------------------------------------------------------------ write path
    @classmethod
    def record_feedback(cls, feedback: UserFeedbackItem) -> dict[str, Any]:
        """Store a correction, apply it to the transcript, and update what the system has learned."""
        from .transcription_service import TranscriptionService

        cls.ensure_loaded()
        with cls._lock:
            feedback_id = feedback.feedback_id or str(uuid.uuid4())
            now_iso = datetime.now(timezone.utc).isoformat()

            transcript = TranscriptionService.get_transcript(feedback.recording_id)
            turn = None
            if feedback.turn_id is not None:
                if transcript is None:
                    raise FeedbackError(f"No transcript for recording '{feedback.recording_id}'.")
                turn = next((t for t in transcript.segments if t.turn_id == feedback.turn_id), None)
                if turn is None:
                    raise FeedbackError(f"Turn '{feedback.turn_id}' not found in this transcript.")

            original_text = turn.text if turn is not None else feedback.original_text
            original_speaker = turn.speaker if turn is not None else feedback.original_speaker

            analysis = cls._feedback_manager.register_correction(
                original_text=original_text,
                corrected_text=feedback.corrected_text,
                original_speaker=original_speaker,
                corrected_speaker=feedback.corrected_speaker,
            )

            # Per-word correctness labels for calibration, from the turn's real word tokens.
            samples: list[tuple[Any, ...]] = []
            if turn is not None and turn.words:
                orig_tokens = [w.word for w in turn.words]
                corr_tokens = tokenize(feedback.corrected_text)
                kept = set()
                for tag, i1, i2, _j1, _j2 in diff_words(orig_tokens, corr_tokens):
                    if tag == "equal":
                        kept.update(range(i1, i2))
                for i, w in enumerate(turn.words):
                    if w.alignment_status == "corrected":
                        continue  # already human-verified; not a model prediction
                    samples.append((
                        feedback_id, feedback.recording_id, w.word, float(w.raw_confidence),
                        float(w.confidence), 1 if i in kept else 0, now_iso,
                    ))

            conn = get_db_connection()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO feedback (
                        feedback_id, recording_id, turn_id, word_index,
                        original_text, corrected_text, original_speaker, corrected_speaker,
                        notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback_id, feedback.recording_id, feedback.turn_id, feedback.word_index,
                        original_text, feedback.corrected_text, original_speaker,
                        feedback.corrected_speaker, feedback.notes or analysis["error_type"], now_iso,
                    ),
                )
                conn.execute("DELETE FROM calibration_samples WHERE feedback_id = ?", (feedback_id,))
                conn.executemany(
                    """
                    INSERT INTO calibration_samples (
                        feedback_id, recording_id, word, raw_confidence, calibrated_confidence, label, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    samples,
                )
            conn.close()

            if transcript is not None and turn is not None:
                cls._apply_to_transcript(transcript, turn, feedback.corrected_text, feedback.corrected_speaker)

            calibration = cls.relearn()

        return {
            "feedback_id": feedback_id,
            "error_type": analysis["error_type"],
            "extracted_terms": analysis["extracted_terms"],
            "words_changed": analysis["words_changed"],
            "words_total": analysis["words_total"],
            "samples_added": len(samples),
            "vocabulary_size": analysis["total_verified_vocabulary"],
            "calibration": calibration,
        }

    @classmethod
    def delete_feedback(cls, feedback_id: str) -> bool:
        """Forget a correction: the system un-learns it (the transcript text is left as edited)."""
        conn = get_db_connection()
        with conn:
            conn.execute("DELETE FROM calibration_samples WHERE feedback_id = ?", (feedback_id,))
            cur = conn.execute("DELETE FROM feedback WHERE feedback_id = ?", (feedback_id,))
        conn.close()
        if cur.rowcount:
            cls.relearn()
        return bool(cur.rowcount)

    @staticmethod
    def _apply_to_transcript(
        transcript: TranscriptResult, turn: ConversationTurn, corrected_text: str, corrected_speaker: Optional[str]
    ) -> None:
        """Rewrite one turn with the user's text. Unchanged words keep their timing and confidence;
        new words are spread over the time of the words they replace and marked human-verified."""
        speaker = corrected_speaker or turn.speaker
        old = turn.words
        new_tokens = tokenize(corrected_text)
        new_words: list[AlignedWord] = []

        for tag, i1, i2, j1, j2 in diff_words([w.word for w in old], new_tokens):
            if tag == "equal":
                for w in old[i1:i2]:
                    new_words.append(w.model_copy(update={"speaker": speaker}))
            elif tag in ("replace", "insert"):
                if i1 < i2:
                    span_start, span_end = old[i1].start, old[i2 - 1].end
                else:
                    span_start = new_words[-1].end if new_words else (old[i1].start if i1 < len(old) else turn.start)
                    span_end = old[i1].start if i1 < len(old) else turn.end
                    span_end = max(span_start, span_end)
                count = j2 - j1
                step = (span_end - span_start) / count if count else 0.0
                for k, tok in enumerate(new_tokens[j1:j2]):
                    new_words.append(AlignedWord(
                        word=tok,
                        start=round(span_start + k * step, 3),
                        end=round(span_start + (k + 1) * step, 3),
                        speaker=speaker,
                        confidence=1.0,
                        raw_confidence=1.0,
                        is_calibrated=True,
                        alignment_status="corrected",
                    ))
            # "delete": the user removed these words, drop them

        t0, t1 = turn.start, turn.end
        turn.words = new_words
        turn.speaker = speaker
        turn.text = " ".join(w.word for w in new_words)
        turn.avg_confidence = round(sum(w.confidence for w in new_words) / len(new_words), 3) if new_words else 0.0

        others = [w for w in transcript.words if not (w.start >= t0 - _EPS and w.end <= t1 + _EPS)]
        transcript.words = sorted(others + new_words, key=lambda w: (w.start, w.end))
        if speaker not in transcript.speakers:
            transcript.speakers.append(speaker)
        transcript.summary["human_corrected_turns"] = int(transcript.summary.get("human_corrected_turns", 0)) + 1

        data = transcript.model_dump()
        TranscriptRepository.save(transcript.recording_id, data, datetime.now(timezone.utc).isoformat())
        json_path = settings.results_dir / f"{transcript.recording_id}.json"
        if json_path.parent.exists():
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

    # ------------------------------------------------------------------ read path
    @classmethod
    def list_feedback(cls, recording_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Retrieve recorded corrections, newest first."""
        conn = get_db_connection()
        if recording_id:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE recording_id = ? ORDER BY created_at DESC",
                (recording_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM feedback ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]

    @classmethod
    def get_accumulated_vocabulary(cls) -> list[str]:
        return cls.get_manager().get_vocabulary()

    @classmethod
    def get_samples(cls) -> tuple[list[float], list[int]]:
        conn = get_db_connection()
        rows = conn.execute("SELECT raw_confidence, label FROM calibration_samples").fetchall()
        conn.close()
        return [r["raw_confidence"] for r in rows], [r["label"] for r in rows]

    @classmethod
    def get_status(cls) -> dict[str, Any]:
        """Everything the system has learned so far, for the Learning dashboard."""
        from .transcription_service import TranscriptionService

        manager = cls.get_manager()
        calibrator = TranscriptionService.get_calibrator()
        conn = get_db_connection()
        n_feedback = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        by_type = {
            r["notes"]: r["n"]
            for r in conn.execute("SELECT notes, COUNT(*) AS n FROM feedback GROUP BY notes").fetchall()
        }
        n_words, n_wrong = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(1 - label), 0) FROM calibration_samples"
        ).fetchone()
        conn.close()

        return {
            "corrections": n_feedback,
            "corrections_by_type": by_type,
            "reviewed_words": n_words,
            "wrong_words": n_wrong,
            "vocabulary": manager.get_vocabulary_counts(),
            "whisper_prompt": manager.get_whisper_prompt(),
            "calibration": {
                "method": calibrator.method,
                "version": calibrator.version,
                "is_fitted": calibrator.is_fitted,
                "n_samples": calibrator.n_samples,
                "min_samples": calibrator.MIN_FIT_SAMPLES,
                "report": calibrator.fit_report,
                "curve": calibrator.curve(),
            },
        }
