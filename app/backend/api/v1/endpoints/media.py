"""Media Ingestion, Upload, and Audio Streaming Endpoints."""
from __future__ import annotations

import tempfile
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.api.v1.responses import error, json_example
from backend.database.models import RecordingRepository
from backend.services.audio_service import AudioService
from engine.schemas import MediaUploadResponse

router = APIRouter()

_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".ogg": "audio/ogg",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


@router.post(
    "/media/upload",
    response_model=MediaUploadResponse,
    tags=["Media"],
    responses={
        400: error(
            "Invalid upload: empty filename, unsupported format, too large or too long",
            "Unsupported file format '.txt'. Supported: .aac, .avi, .flac, .m4a, .mkv, .mov, .mp3, .mp4, .ogg, .wav, .webm",
        ),
        500: error("Unexpected processing failure", "Failed to process upload: <error>"),
    },
)
async def upload_media(file: UploadFile = File(...)) -> MediaUploadResponse:
    """Accept and validate a pre-recorded audio or video recording."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    # Save to temporary file to calculate hash and metadata safely
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        content = await file.read()
        tmp_path.write_bytes(content)

    try:
        record = AudioService.save_and_ingest(tmp_path, file.filename)
        return MediaUploadResponse(
            recording_id=record.recording_id,
            filename=record.filename,
            size_bytes=record.size_bytes,
            duration_seconds=record.duration,
            sha256=record.sha256,
            status="ingested",
            message=f"Media file successfully validated and ingested ({record.duration:.1f}s).",
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {exc}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@router.get(
    "/media",
    tags=["Media"],
    responses={
        200: json_example(
            "All ingested recordings, newest first",
            [
                {
                    "recording_id": "4d595f83-4dda-44f3-95c2-12f2d1228309",
                    "filename": "meeting.mp4",
                    "duration": 83.77,
                    "size_bytes": 25327338,
                    "sha256": "74702005cb3c8ae5327cfa89b691e996dc3db558899b7cad36acee68ac7bc885",
                    "created_at": "2026-11-13T18:22:41.512345+00:00",
                }
            ],
        )
    },
)
def list_recordings() -> list[dict[str, str | float | int]]:
    """List all ingested recordings."""
    records = RecordingRepository.list_all()
    return [
        {
            "recording_id": r.recording_id,
            "filename": r.filename,
            "duration": r.duration,
            "size_bytes": r.size_bytes,
            "sha256": r.sha256,
            "created_at": r.created_at,
        }
        for r in records
    ]


@router.get(
    "/media/{recording_id}/stream",
    tags=["Media"],
    response_class=FileResponse,
    responses={
        200: {
            "description": "The original uploaded file. Supports HTTP Range requests (206 Partial Content) for seeking.",
            "content": {
                media_type: {"schema": {"type": "string", "format": "binary"}}
                for media_type in sorted(set(_MEDIA_TYPES.values()) | {"application/octet-stream"})
            },
        },
        404: error("Unknown recording or media file missing", "Recording media not found."),
    },
)
def stream_media(recording_id: str) -> FileResponse:
    """Stream media file directly for audio playback in frontend."""
    record = RecordingRepository.get(recording_id)
    if not record or not Path(record.filepath).exists():
        raise HTTPException(status_code=404, detail="Recording media not found.")

    ext = Path(record.filepath).suffix.lower()
    media_type = _MEDIA_TYPES.get(ext, "application/octet-stream")
    return FileResponse(record.filepath, media_type=media_type, filename=record.filename)
