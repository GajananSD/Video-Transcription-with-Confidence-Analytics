"""Extensions: let external services add insight cards under a transcript.

An extension is any web service with a URL. When a transcript is ready (or edited),
the backend POSTs a TranscriptEvent to that URL. The service answers with an
InsightCard (plain data, validated here) and the dashboard draws it with its own
accessible components and the user's colour scheme.
"""
from .contract import EVENTS, InsightCard, TranscriptEvent, build_event, example_card
from .runner import ExtensionRepository, insights_for, notify_plugins, send_test_event

__all__ = [
    "EVENTS",
    "InsightCard",
    "TranscriptEvent",
    "build_event",
    "example_card",
    "ExtensionRepository",
    "insights_for",
    "notify_plugins",
    "send_test_event",
]
