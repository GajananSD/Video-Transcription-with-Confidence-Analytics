"""Health Check and Engine Status Endpoint."""
from __future__ import annotations

from fastapi import APIRouter
from backend.api.v1.responses import json_example
from backend.config import settings

router = APIRouter()


def _effective_device(requested: str) -> str:
    """Report the device inference actually runs on (resolves 'auto' / missing GPU)."""
    req = (requested or "auto").strip().lower()
    if req == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    return "cuda" if has_cuda else "cpu"


@router.get(
    "/health",
    tags=["System"],
    responses={
        200: json_example(
            "Server is up",
            {
                "status": "ok",
                "app": "CS698B Video Transcription with Confidence Analytics",
                "version": "1.0.0",
                "mock_inference": False,
                "device": "cpu",
            },
        )
    },
)
def health_check() -> dict[str, str | bool]:
    """Return application server status and engine configuration."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "mock_inference": settings.mock_inference,
        "device": _effective_device(settings.device),
    }
