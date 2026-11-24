"""Integration tests for FastAPI REST API endpoints."""
from __future__ import annotations

import io
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "CS698B" in data["app"]


def test_upload_and_list_media(client: TestClient):
    file_bytes = b"RIFFmockwavcontent"
    file_tuple = ("test_audio.wav", io.BytesIO(file_bytes), "audio/wav")
    
    resp = client.post("/api/v1/media/upload", files={"file": file_tuple})
    assert resp.status_code == 200
    data = resp.json()
    assert "recording_id" in data
    assert data["filename"] == "test_audio.wav"
    assert data["status"] == "ingested"

    # List recordings
    list_resp = client.get("/api/v1/media")
    assert list_resp.status_code == 200
    records = list_resp.json()
    assert any(r["recording_id"] == data["recording_id"] for r in records)



def test_delete_recording(client: TestClient):
    from pathlib import Path

    from backend.database.db import get_db_connection
    from backend.database.models import RecordingRepository

    file_tuple = ("to_delete.wav", io.BytesIO(b"RIFFdeletemecontent"), "audio/wav")
    rec_id = client.post("/api/v1/media/upload", files={"file": file_tuple}).json()["recording_id"]
    media_path = Path(RecordingRepository.get(rec_id).filepath)
    assert media_path.exists()

    conn = get_db_connection()
    with conn:
        conn.execute(
            "INSERT INTO transcripts (recording_id, result_json, created_at) VALUES (?, '{}', 'now')", (rec_id,)
        )
    conn.close()

    resp = client.delete(f"/api/v1/media/{rec_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"
    assert not media_path.exists()
    assert RecordingRepository.get(rec_id) is None
    conn = get_db_connection()
    assert conn.execute("SELECT 1 FROM transcripts WHERE recording_id = ?", (rec_id,)).fetchone() is None
    conn.close()
    assert all(r["recording_id"] != rec_id for r in client.get("/api/v1/media").json())

    assert client.delete(f"/api/v1/media/{rec_id}").status_code == 404


def test_delete_recording_blocked_while_job_running(client: TestClient):
    from backend.database.db import get_db_connection

    file_tuple = ("busy.wav", io.BytesIO(b"RIFFbusyjobcontent"), "audio/wav")
    rec_id = client.post("/api/v1/media/upload", files={"file": file_tuple}).json()["recording_id"]
    conn = get_db_connection()
    with conn:
        conn.execute(
            "INSERT INTO jobs (job_id, recording_id, status, created_at, updated_at) "
            "VALUES ('busy-job', ?, 'processing', 'now', 'now')",
            (rec_id,),
        )
    conn.close()
    assert client.delete(f"/api/v1/media/{rec_id}").status_code == 409

    conn = get_db_connection()
    with conn:
        conn.execute("UPDATE jobs SET status = 'completed' WHERE job_id = 'busy-job'")
    conn.close()
    assert client.delete(f"/api/v1/media/{rec_id}").status_code == 200
