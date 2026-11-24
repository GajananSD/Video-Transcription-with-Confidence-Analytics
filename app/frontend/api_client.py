"""HTTP Client for communicating with the FastAPI backend."""
from __future__ import annotations

from typing import Any, Optional
import requests


class APIClient:
    """Encapsulates REST communication with the FastAPI backend."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def upload_media(self, file_bytes: bytes, filename: str) -> dict[str, Any]:
        files = {"file": (filename, file_bytes)}
        resp = requests.post(f"{self.base_url}/api/v1/media/upload", files=files, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def list_recordings(self) -> list[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/media", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def delete_recording(self, recording_id: str) -> dict[str, Any]:
        resp = requests.delete(f"{self.base_url}/api/v1/media/{recording_id}", timeout=30)
        if resp.status_code in (404, 409):
            raise RuntimeError(resp.json().get("detail", resp.text))
        resp.raise_for_status()
        return resp.json()

    def trigger_transcription(
        self,
        recording_id: str,
        use_mock: bool = False,
        initial_prompt: Optional[str] = None,
        force: bool = False,
        use_learned_vocabulary: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "recording_id": recording_id,
            "use_mock": use_mock,
            "initial_prompt": initial_prompt,
            "force": force,
            "use_learned_vocabulary": use_learned_vocabulary,
        }
        resp = requests.post(f"{self.base_url}/api/v1/transcriptions", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_job_status(self, job_or_rec_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/transcriptions/{job_or_rec_id}/status", timeout=5)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def get_transcript(self, recording_id: str) -> Optional[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/transcriptions/{recording_id}", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code in (202, 404):
            return None  # In progress (202) or not yet transcribed / not found (404)
        resp.raise_for_status()
        return None

    def get_analytics(self, recording_id: str) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/analytics/{recording_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- adaptive feedback loop
    def submit_feedback(
        self,
        recording_id: str,
        turn_id: str,
        corrected_text: str,
        corrected_speaker: Optional[str] = None,
        original_text: str = "",
    ) -> dict[str, Any]:
        payload = {
            "recording_id": recording_id,
            "turn_id": turn_id,
            "original_text": original_text,
            "corrected_text": corrected_text,
            "corrected_speaker": corrected_speaker,
        }
        resp = requests.post(f"{self.base_url}/api/v1/feedback", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def list_feedback(self, recording_id: Optional[str] = None) -> list[dict[str, Any]]:
        params = {"recording_id": recording_id} if recording_id else None
        resp = requests.get(f"{self.base_url}/api/v1/feedback", params=params, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def delete_feedback(self, feedback_id: str) -> dict[str, Any]:
        resp = requests.delete(f"{self.base_url}/api/v1/feedback/{feedback_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_adaptive_status(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/adaptive/status", timeout=5)
        resp.raise_for_status()
        return resp.json()

    # ---------------------------------------------------------- extensions
    def list_extensions(self) -> list[dict[str, Any]]:
        resp = requests.get(f"{self.base_url}/api/v1/extensions", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def register_extension(self, name: str, url: str, events: list[str], team: str = "") -> dict[str, Any]:
        resp = requests.post(
            f"{self.base_url}/api/v1/extensions",
            json={"team": team, "name": name, "url": url, "events": events},
            timeout=5,
        )
        if resp.status_code == 422:
            detail = resp.json().get("detail", [{}])
            raise ValueError(detail[0].get("msg", "Invalid input") if isinstance(detail, list) else str(detail))
        resp.raise_for_status()
        return resp.json()

    def update_extension(self, extension_id: str, **fields: Any) -> dict[str, Any]:
        resp = requests.patch(f"{self.base_url}/api/v1/extensions/{extension_id}", json=fields, timeout=5)
        resp.raise_for_status()
        return resp.json()

    def delete_extension(self, extension_id: str) -> dict[str, Any]:
        resp = requests.delete(f"{self.base_url}/api/v1/extensions/{extension_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def test_extension(self, extension_id: str, recording_id: Optional[str] = None) -> dict[str, Any]:
        params = {"recording_id": recording_id} if recording_id else None
        resp = requests.post(f"{self.base_url}/api/v1/extensions/{extension_id}/test", params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def get_extension_contract(self) -> dict[str, Any]:
        resp = requests.get(f"{self.base_url}/api/v1/extensions/contract", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def get_insights(self, recording_id: str, refresh: bool = False) -> list[dict[str, Any]]:
        resp = requests.get(
            f"{self.base_url}/api/v1/extensions/insights/{recording_id}",
            params={"refresh": "true"} if refresh else None,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_export_url(self, recording_id: str, format_str: str) -> str:
        return f"{self.base_url}/api/v1/transcriptions/{recording_id}/export?format={format_str}"

    def get_stream_url(self, recording_id: str) -> str:
        return f"{self.base_url}/api/v1/media/{recording_id}/stream"
