"""Tests for the extension system: registration, the contract, calling, caching and the example service."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.database.models import RecordingRecord, RecordingRepository, TranscriptRepository
from backend.extensions import runner
from backend.extensions.contract import InsightCard, build_event
from backend.extensions.runner import _SAMPLE_TRANSCRIPT
from extensions_examples.talk_balance import app as talk_balance_app


@pytest.fixture
def stored_recording() -> str:
    rid = f"ext_{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    RecordingRepository.create(RecordingRecord(rid, "talk.mp4", "/tmp/talk.mp4", 9.0, 1, uuid.uuid4().hex, now))
    TranscriptRepository.save(rid, {**_SAMPLE_TRANSCRIPT, "recording_id": rid}, now)
    return rid


@pytest.fixture
def calls(monkeypatch):
    """Route extension POSTs to the example service in-process and record them."""
    seen: list[tuple[str, dict]] = []
    tb = TestClient(talk_balance_app)

    def fake_post(url, body):
        seen.append((url, body))
        if "down" in url:
            import requests
            raise requests.exceptions.ConnectionError("refused")
        if "bad" in url:
            return {"note": "missing heading"}
        r = tb.post("/insight", json=body)
        r.raise_for_status()
        return r.json()

    monkeypatch.setattr(runner, "_post", fake_post)
    return seen


@pytest.fixture(autouse=True)
def _clean_extensions(client: TestClient):
    yield
    for e in client.get("/api/v1/extensions").json():
        client.delete(f"/api/v1/extensions/{e['extension_id']}")


def _register(client, name="Talk balance", url="http://ext.local/insight", **kw):
    r = client.post("/api/v1/extensions", json={"name": name, "url": url, **kw})
    assert r.status_code == 201, r.text
    return r.json()


def test_register_validates_url_and_events(client: TestClient):
    assert client.post("/api/v1/extensions", json={"name": "x", "url": "ftp://nope"}).status_code == 422
    assert client.post("/api/v1/extensions", json={"name": "x", "url": "http://a/b", "events": ["boom"]}).status_code == 422
    ext = _register(client)
    assert ext["enabled"] is True and ext["events"] == ["transcript.ready", "transcript.updated"]
    assert [e["name"] for e in client.get("/api/v1/extensions").json()] == ["Talk balance"]


def test_contract_example_card_is_valid(client: TestClient):
    c = client.get("/api/v1/extensions/contract").json()
    InsightCard.model_validate(c["example_card"])
    assert c["example_event"]["event"] == "transcript.ready"
    assert c["example_event"]["words"]


def test_example_service_answers_a_valid_card():
    body = build_event("transcript.ready", _SAMPLE_TRANSCRIPT).model_dump()
    card = InsightCard.model_validate(TestClient(talk_balance_app).post("/insight", json=body).json())
    assert card.heading == "Talk balance"
    assert card.table and card.table.columns[0] == "Speaker"
    assert card.series and card.series.points


def test_insights_are_cached_until_transcript_changes(client, stored_recording, calls):
    _register(client)
    first = client.get(f"/api/v1/extensions/insights/{stored_recording}").json()
    assert len(first) == 1 and first[0]["ok"] and first[0]["event"] == "transcript.ready"
    assert len(calls) == 1

    client.get(f"/api/v1/extensions/insights/{stored_recording}")  # cached: no new call
    assert len(calls) == 1

    t = TranscriptRepository.get(stored_recording)
    t["segments"][0]["text"] = "Thanks for joining us this morning."
    TranscriptRepository.save(stored_recording, t, datetime.now(timezone.utc).isoformat())
    again = client.get(f"/api/v1/extensions/insights/{stored_recording}").json()
    assert len(calls) == 2 and again[0]["event"] == "transcript.updated" and not again[0]["outdated"]

    client.get(f"/api/v1/extensions/insights/{stored_recording}?refresh=true")
    assert len(calls) == 3


def test_ready_only_subscriber_keeps_old_card_marked_outdated(client, stored_recording, calls):
    _register(client, events=["transcript.ready"])
    client.get(f"/api/v1/extensions/insights/{stored_recording}")
    t = TranscriptRepository.get(stored_recording)
    t["duration"] = 10.0
    TranscriptRepository.save(stored_recording, t, datetime.now(timezone.utc).isoformat())
    res = client.get(f"/api/v1/extensions/insights/{stored_recording}").json()
    assert len(calls) == 1 and res[0]["outdated"] is True


def test_failures_are_reported_not_raised(client, stored_recording, calls):
    _register(client, name="Down", url="http://down.local/x")
    _register(client, name="Bad", url="http://bad.local/x")
    res = {r["name"]: r for r in client.get(f"/api/v1/extensions/insights/{stored_recording}").json()}
    assert not res["Down"]["ok"] and "connect" in res["Down"]["error"].lower()
    assert not res["Bad"]["ok"] and "heading" in res["Bad"]["error"]


def test_paused_and_deleted_extensions_are_skipped(client, stored_recording, calls):
    ext = _register(client)
    client.patch(f"/api/v1/extensions/{ext['extension_id']}", json={"enabled": False})
    assert client.get(f"/api/v1/extensions/insights/{stored_recording}").json() == []
    assert calls == []
    assert client.delete(f"/api/v1/extensions/{ext['extension_id']}").status_code == 200
    assert client.delete(f"/api/v1/extensions/{ext['extension_id']}").status_code == 404


def test_send_test_uses_sample_and_stores_nothing(client, stored_recording, calls):
    ext = _register(client)
    r = client.post(f"/api/v1/extensions/{ext['extension_id']}/test").json()
    assert r["ok"] and r["card"]["heading"] == "Talk balance"
    assert calls[0][1]["recording"]["recording_id"] == "sample"
    assert runner.ExtensionRepository.get_run(ext["extension_id"], stored_recording) is None


def test_card_rejects_oversized_or_unknown_levels():
    with pytest.raises(Exception):
        InsightCard.model_validate({"heading": "x", "flags": [{"level": "danger", "text": "t"}]})
    with pytest.raises(Exception):
        InsightCard.model_validate({"heading": "x" * 81})
    # unknown extra keys (e.g. html, colour) are ignored, not rendered
    card = InsightCard.model_validate({"heading": "ok", "html": "<script>", "color": "#f00"})
    assert "html" not in card.model_dump()


# --------------------------------------------------------------------------- team, raw answer, automatic calls
def test_team_url_and_raw_answer_are_reported(client, stored_recording, calls):
    ext = _register(client, team="Team VibeCheck")
    assert ext["team"] == "Team VibeCheck"
    res = client.get(f"/api/v1/extensions/insights/{stored_recording}").json()[0]
    assert res["team"] == "Team VibeCheck" and res["url"] == "http://ext.local/insight"
    assert json.loads(res["raw"])["heading"] == "Talk balance"


def test_bad_answer_keeps_raw_text_for_debugging(client, stored_recording, calls):
    _register(client, name="Bad", url="http://bad.local/x")
    res = client.get(f"/api/v1/extensions/insights/{stored_recording}").json()[0]
    assert not res["ok"] and json.loads(res["raw"]) == {"note": "missing heading"}


def test_plugins_are_called_by_the_job_and_after_corrections(client, calls):
    """Upload -> transcription job -> plugin called automatically -> correction -> called again."""
    import io
    import time

    from backend.services.transcription_service import TranscriptionService
    from tests.test_e2e_workflow import TestStubEngine

    ext = _register(client, team="Team B")
    TranscriptionService.set_engine(TestStubEngine())
    try:
        up = client.post("/api/v1/media/upload",
                         files={"file": ("plug.wav", io.BytesIO(b"RIFF" + uuid.uuid4().bytes), "audio/wav")}).json()
        rid = up["recording_id"]
        job = client.post("/api/v1/transcriptions", json={"recording_id": rid}).json()
        for _ in range(150):
            if client.get(f"/api/v1/transcriptions/{job['job_id']}/status").json()["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        # The job itself called the plugin: the result exists before anyone asked for it.
        run = runner.ExtensionRepository.get_run(ext["extension_id"], rid)
        assert run and run["ok"] and run["event"] == "transcript.ready"
        assert calls[-1][1]["recording"]["recording_id"] == rid

        r = client.post("/api/v1/feedback", json={"recording_id": rid, "turn_id": "turn_002",
                                                  "original_text": "license and registration please",
                                                  "corrected_text": "licence and registration please"})
        assert r.status_code == 200
        for _ in range(50):  # the correction notifies plugins in a background thread
            run = runner.ExtensionRepository.get_run(ext["extension_id"], rid)
            if run["event"] == "transcript.updated":
                break
            time.sleep(0.1)
        assert run["event"] == "transcript.updated"
        assert "licence" in calls[-1][1]["turns"][1]["text"]
    finally:
        TranscriptionService.set_engine(None)


def test_old_database_gets_new_columns(tmp_path, monkeypatch):
    import sqlite3

    from backend.database import db as dbmod

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE extensions (extension_id TEXT PRIMARY KEY, name TEXT NOT NULL, url TEXT NOT NULL,"
                " events TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL)")
    con.execute("INSERT INTO extensions VALUES ('x1','Old','http://a/b','[\"transcript.ready\"]',1,'2026-01-01')")
    con.commit()
    con.close()
    monkeypatch.setattr(dbmod, "DB_PATH", path)
    dbmod.init_db()
    con = sqlite3.connect(path)
    assert "team" in {r[1] for r in con.execute("PRAGMA table_info(extensions)")}
    assert "raw_json" in {r[1] for r in con.execute("PRAGMA table_info(extension_runs)")}
    assert con.execute("SELECT team FROM extensions").fetchone()[0] == ""
