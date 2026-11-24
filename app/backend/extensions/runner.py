"""Storing extensions, calling them, and caching their insight cards per transcript version."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import requests
from pydantic import ValidationError

from backend.database.db import get_db_connection
from backend.database.models import TranscriptRepository

from .contract import EVENTS, InsightCard, build_event

logger = logging.getLogger("backend.extensions")

CONNECT_TIMEOUT_S = 3.0   # an unreachable laptop should fail fast
TIMEOUT_S = 8.0           # hard upper bound for one whole call (connect + answer)
MAX_WORKERS = 6
MAX_RAW_CHARS = 50_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExtensionRecord:
    extension_id: str
    team: str
    name: str
    url: str
    events: list[str]
    enabled: bool
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _row_to_ext(r) -> ExtensionRecord:
    return ExtensionRecord(
        extension_id=r["extension_id"],
        team=r["team"] if "team" in r.keys() else "",
        name=r["name"],
        url=r["url"],
        events=json.loads(r["events"]),
        enabled=bool(r["enabled"]),
        created_at=r["created_at"],
    )


class ExtensionRepository:
    @staticmethod
    def create(name: str, url: str, events: list[str], team: str = "") -> ExtensionRecord:
        rec = ExtensionRecord(uuid.uuid4().hex[:12], team, name, url, events, True, _now())
        conn = get_db_connection()
        with conn:
            conn.execute(
                "INSERT INTO extensions (extension_id, team, name, url, events, enabled, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (rec.extension_id, rec.team, rec.name, rec.url, json.dumps(rec.events), 1, rec.created_at),
            )
        conn.close()
        return rec

    @staticmethod
    def get(extension_id: str) -> Optional[ExtensionRecord]:
        conn = get_db_connection()
        row = conn.execute("SELECT * FROM extensions WHERE extension_id = ?", (extension_id,)).fetchone()
        conn.close()
        return _row_to_ext(row) if row else None

    @staticmethod
    def list_all(enabled_only: bool = False) -> list[ExtensionRecord]:
        conn = get_db_connection()
        q = "SELECT * FROM extensions" + (" WHERE enabled = 1" if enabled_only else "") + " ORDER BY created_at"
        rows = conn.execute(q).fetchall()
        conn.close()
        return [_row_to_ext(r) for r in rows]

    @staticmethod
    def update(extension_id: str, **fields: Any) -> Optional[ExtensionRecord]:
        allowed = {"team", "name", "url", "events", "enabled"}
        sets, vals = [], []
        for k, v in fields.items():
            if k not in allowed or v is None:
                continue
            if k == "events":
                v = json.dumps(v)
            elif k == "enabled":
                v = 1 if v else 0
            sets.append(f"{k} = ?")
            vals.append(v)
        conn = get_db_connection()
        with conn:
            if sets:
                conn.execute(f"UPDATE extensions SET {', '.join(sets)} WHERE extension_id = ?", (*vals, extension_id))
            if "url" in fields and fields["url"] is not None:
                # A new URL is a different service: drop its cached answers.
                conn.execute("DELETE FROM extension_runs WHERE extension_id = ?", (extension_id,))
        conn.close()
        return ExtensionRepository.get(extension_id)

    @staticmethod
    def delete(extension_id: str) -> bool:
        conn = get_db_connection()
        with conn:
            cur = conn.execute("DELETE FROM extensions WHERE extension_id = ?", (extension_id,))
            conn.execute("DELETE FROM extension_runs WHERE extension_id = ?", (extension_id,))
        conn.close()
        return cur.rowcount > 0

    # -------------------------------------------------------------- cached runs
    @staticmethod
    def get_run(extension_id: str, recording_id: str) -> Optional[dict[str, Any]]:
        conn = get_db_connection()
        row = conn.execute(
            "SELECT * FROM extension_runs WHERE extension_id = ? AND recording_id = ?",
            (extension_id, recording_id),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def save_run(extension_id: str, recording_id: str, event: str, fingerprint: str, result: dict[str, Any]) -> None:
        conn = get_db_connection()
        with conn:
            conn.execute(
                """INSERT OR REPLACE INTO extension_runs
                   (extension_id, recording_id, event, fingerprint, ok, card_json, error, raw_json, elapsed_ms, ran_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    extension_id, recording_id, event, fingerprint, 1 if result["ok"] else 0,
                    json.dumps(result["card"]) if result.get("card") else None,
                    result.get("error"), result.get("raw"), result.get("elapsed_ms"), _now(),
                ),
            )
        conn.close()


# ------------------------------------------------------------------ calling
def _post(url: str, body: dict[str, Any]) -> Any:
    """POST the event and return the decoded JSON. Separate function so tests can replace it."""
    resp = requests.post(url, json=body, timeout=(CONNECT_TIMEOUT_S, TIMEOUT_S))
    resp.raise_for_status()
    return resp.json()


# requests' timeouts apply per socket operation, so a half-dead peer can stretch a call far
# past TIMEOUT_S. Calls run on this pool and are abandoned after a wall-clock deadline instead.
_CALL_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS * 2, thread_name_prefix="plugin-call")


def _post_with_deadline(url: str, body: dict[str, Any]) -> Any:
    future = _CALL_POOL.submit(_post, url, body)
    try:
        return future.result(timeout=TIMEOUT_S + 1)
    except FutureTimeout:
        future.cancel()
        raise requests.exceptions.Timeout() from None


def call_extension(ext: ExtensionRecord, body: dict[str, Any]) -> dict[str, Any]:
    """Call one extension. Never raises: failures come back as {'ok': False, 'error': ...}."""
    t0 = time.perf_counter()
    raw = None
    try:
        answer = _post_with_deadline(ext.url, body)
        raw = json.dumps(answer, ensure_ascii=False)[:MAX_RAW_CHARS]
        card = InsightCard.model_validate(answer)
        out = {"ok": True, "card": card.model_dump(exclude_none=True), "error": None}
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "card"
        out = {"ok": False, "card": None, "error": f"The answer is not a valid insight card ({where}: {first['msg']})."}
    except requests.exceptions.ConnectTimeout:
        out = {"ok": False, "card": None,
               "error": f"Could not connect within {CONNECT_TIMEOUT_S:.0f} seconds. Is that laptop on and reachable?"}
    except requests.exceptions.Timeout:
        out = {"ok": False, "card": None, "error": f"No answer within {TIMEOUT_S:.0f} seconds."}
    except requests.exceptions.ConnectionError:
        out = {"ok": False, "card": None,
               "error": "Could not connect. Is the plugin service running and is that laptop on the same network?"}
    except Exception as exc:  # HTTP errors, bad JSON, ...
        out = {"ok": False, "card": None, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    out["raw"] = raw
    return out


def fingerprint(transcript: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(transcript, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _view(ext: ExtensionRecord, run: dict[str, Any], current_fp: str) -> dict[str, Any]:
    return {
        "extension_id": ext.extension_id,
        "team": ext.team,
        "name": ext.name,
        "url": ext.url,
        "event": run["event"],
        "ok": bool(run["ok"]),
        "card": json.loads(run["card_json"]) if run.get("card_json") else None,
        "error": run.get("error"),
        "raw": run.get("raw_json"),
        "elapsed_ms": run.get("elapsed_ms"),
        "ran_at": run["ran_at"],
        "outdated": run["fingerprint"] != current_fp,
    }


def insights_for(recording_id: str, refresh: bool = False) -> list[dict[str, Any]]:
    """Return one insight entry per enabled extension for this recording.

    An extension is (re)called only when needed: it has never seen this recording, the
    transcript changed and it subscribed to `transcript.updated`, or `refresh` is set.
    Otherwise the cached answer is returned, so the dashboard can ask on every rerun.
    """
    transcript = TranscriptRepository.get(recording_id)
    if not transcript:
        return []
    fp = fingerprint(transcript)
    exts = ExtensionRepository.list_all(enabled_only=True)

    todo: list[tuple[ExtensionRecord, str]] = []
    for ext in exts:
        last = ExtensionRepository.get_run(ext.extension_id, recording_id)
        event = "transcript.updated" if last else "transcript.ready"
        if refresh or last is None or (last["fingerprint"] != fp and "transcript.updated" in ext.events):
            if event in ext.events or refresh:
                todo.append((ext, event))

    if todo:
        bodies = {e: build_event(e, transcript).model_dump() for e in {ev for _, ev in todo}}
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(todo))) as pool:
            results = list(pool.map(lambda pair: call_extension(pair[0], bodies[pair[1]]), todo))
        for (ext, event), res in zip(todo, results):
            ExtensionRepository.save_run(ext.extension_id, recording_id, event, fp, res)

    out = []
    for ext in exts:
        run = ExtensionRepository.get_run(ext.extension_id, recording_id)
        if run:
            out.append(_view(ext, run, fp))
    return out


def notify_plugins(recording_id: str) -> int:
    """Send a (new or changed) transcript to every active plugin. Never raises.

    Called right after a transcription job finishes and after a user correction, so plugin
    results are ready before anyone opens the results tab. Returns how many plugins answered OK.
    """
    try:
        return sum(1 for r in insights_for(recording_id) if r["ok"] and not r["outdated"])
    except Exception as exc:  # a broken plugin must never break transcription
        logger.warning("Sending transcript %s to plugins failed: %s", recording_id, exc)
        return 0


def send_test_event(ext: ExtensionRecord, recording_id: Optional[str] = None) -> dict[str, Any]:
    """Send a sample event (a real transcript if given, else a tiny built-in one). Nothing is stored."""
    transcript = TranscriptRepository.get(recording_id) if recording_id else None
    transcript = transcript or _SAMPLE_TRANSCRIPT
    body = build_event(EVENTS[0], transcript).model_dump()
    return {"extension_id": ext.extension_id, "team": ext.team, "name": ext.name, **call_extension(ext, body)}


_SAMPLE_TRANSCRIPT: dict[str, Any] = {
    "recording_id": "sample",
    "filename": "sample_interview.mp4",
    "duration": 9.0,
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "summary": {"low_confidence_word_count": 1},
    "segments": [
        {"turn_id": "turn_000", "speaker": "SPEAKER_00", "start": 0.4, "end": 3.1,
         "text": "Thanks for joining us today.", "avg_confidence": 0.91},
        {"turn_id": "turn_001", "speaker": "SPEAKER_01", "start": 3.6, "end": 8.7,
         "text": "Happy to be here, um, let's talk about the project.", "avg_confidence": 0.74},
    ],
    "words": [
        {"word": w, "start": s, "end": s + 0.4, "speaker": spk, "confidence": c}
        for w, s, spk, c in [
            ("Thanks", 0.4, "SPEAKER_00", 0.95), ("for", 0.9, "SPEAKER_00", 0.93),
            ("joining", 1.3, "SPEAKER_00", 0.9), ("us", 1.8, "SPEAKER_00", 0.88),
            ("today.", 2.4, "SPEAKER_00", 0.89), ("Happy", 3.6, "SPEAKER_01", 0.86),
            ("to", 4.1, "SPEAKER_01", 0.8), ("be", 4.4, "SPEAKER_01", 0.82),
            ("here,", 4.8, "SPEAKER_01", 0.79), ("um,", 5.4, "SPEAKER_01", 0.22),
            ("let's", 6.0, "SPEAKER_01", 0.71), ("talk", 6.5, "SPEAKER_01", 0.77),
            ("about", 7.0, "SPEAKER_01", 0.8), ("the", 7.5, "SPEAKER_01", 0.75),
            ("project.", 8.0, "SPEAKER_01", 0.68),
        ]
    ],
}
