"""OpenAPI documentation helpers for responses that are not plain 200 JSON models.

These only describe responses in /docs and /openapi.json; they do not change
what the endpoints return.
"""
from __future__ import annotations

from typing import Any


def error(description: str, detail: str) -> dict[str, Any]:
    """Document an HTTPException response, which returns {"detail": "..."}."""
    return {
        "description": description,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "properties": {"detail": {"type": "string"}},
                    "required": ["detail"],
                },
                "example": {"detail": detail},
            }
        },
    }


def json_example(description: str, example: Any) -> dict[str, Any]:
    """Document an untyped JSON response with a representative example."""
    return {"description": description, "content": {"application/json": {"example": example}}}
