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

