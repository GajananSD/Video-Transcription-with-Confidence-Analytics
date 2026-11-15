"""Adaptive Feedback Loop: learns from user corrections.

Every correction a user makes to a transcript turn teaches the system two things:

1. **Domain vocabulary.** Words the user had to type in (because the ASR got them
   wrong) are collected and counted. The most frequent ones become a Whisper
   ``initial_prompt`` for future transcriptions, biasing the decoder toward them.
2. **Confidence labels.** Each original word is labelled correct (kept unchanged)
   or wrong (replaced/deleted). These ``(raw_confidence, label)`` pairs are used
   to re-fit the confidence calibrator (see ``adaptive.confidence``).
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any, Iterable, Optional, Sequence

# Common function words that carry no domain meaning and should never be "learned".
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "did", "do", "for", "from",
    "had", "has", "have", "he", "her", "him", "his", "how", "i", "if", "in", "is", "it", "its",
    "me", "my", "no", "not", "of", "oh", "ok", "okay", "on", "or", "our", "she", "so", "that",
    "the", "their", "them", "then", "there", "they", "this", "to", "uh", "um", "up", "was", "we",
    "were", "what", "when", "where", "who", "why", "will", "with", "yeah", "yes", "you", "your",
    "all", "just", "get", "got", "out", "now", "one", "two", "off", "too", "don't", "i'm", "it's",
}
MAX_PROMPT_TERMS = 30

_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def tokenize(text: str) -> list[str]:
    """Split text into words on whitespace (punctuation stays attached, like ASR word tokens)."""
    return (text or "").split()


def strip_punct(token: str) -> str:
    """Remove leading/trailing punctuation: 'Taser!' -> 'Taser', keeps 'Code-4' and "don't"."""
    return _EDGE_PUNCT_RE.sub("", token)


def _norm(token: str) -> str:
    return strip_punct(token).lower()


def diff_words(original: Sequence[str], corrected: Sequence[str]) -> list[tuple[str, int, int, int, int]]:
    """Word-level diff (difflib opcodes) between original and corrected token lists."""
    matcher = difflib.SequenceMatcher(
        a=[_norm(w) for w in original], b=[_norm(w) for w in corrected], autojunk=False
    )
    return matcher.get_opcodes()


def correctness_labels(original: Sequence[str], corrected: Sequence[str]) -> list[int]:
    """Label each original word 1 (user kept it) or 0 (user replaced or deleted it)."""
    labels = [0] * len(original)
    for tag, i1, i2, _j1, _j2 in diff_words(original, corrected):
        if tag == "equal":
            for i in range(i1, i2):
                labels[i] = 1
    return labels


def introduced_terms(original: Sequence[str], corrected: Sequence[str]) -> list[str]:
    """Words the user typed in that the ASR did not produce: candidates for domain vocabulary."""
    terms: list[str] = []
    for tag, _i1, _i2, j1, j2 in diff_words(original, corrected):
        if tag in ("replace", "insert"):
            for tok in corrected[j1:j2]:
                clean = strip_punct(tok)
                if len(clean) >= 3 and _norm(clean) not in STOPWORDS and not clean.isdigit():
                    terms.append(clean)
    return terms


class FeedbackLoopManager:
    """Accumulates learned vocabulary from corrections and builds a Whisper prompt from it."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._display: dict[str, str] = {}  # lowercase key -> spelling the user typed

    # ------------------------------------------------------------------ learning
    def reset(self) -> None:
        self._counts.clear()
        self._display.clear()

    def learn_terms(self, terms: Iterable[str]) -> None:
        for term in terms:
            key = term.lower()
            self._counts[key] += 1
            self._display[key] = term

    def rebuild(self, corrections: Iterable[tuple[str, str]]) -> None:
        """Rebuild vocabulary from scratch from stored (original_text, corrected_text) pairs."""
        self.reset()
        for original, corrected in corrections:
            self.learn_terms(introduced_terms(tokenize(original), tokenize(corrected)))

    def register_correction(
        self,
        original_text: str,
        corrected_text: str,
        original_speaker: Optional[str] = None,
        corrected_speaker: Optional[str] = None,
    ) -> dict[str, Any]:
        """Analyse a correction, classify the error type, and learn any new vocabulary."""
        is_speaker_change = (
            original_speaker is not None
            and corrected_speaker is not None
            and original_speaker != corrected_speaker
        )
        orig_tokens, corr_tokens = tokenize(original_text), tokenize(corrected_text)
        is_text_change = [_norm(t) for t in orig_tokens] != [_norm(t) for t in corr_tokens]

        if is_speaker_change and is_text_change:
            error_type = "compound"
        elif is_speaker_change:
            error_type = "diarization"
        elif is_text_change:
            error_type = "acoustic_lexical"
        else:
            error_type = "none"

        new_terms = introduced_terms(orig_tokens, corr_tokens)
        self.learn_terms(new_terms)
        labels = correctness_labels(orig_tokens, corr_tokens)

        return {
            "error_type": error_type,
            "extracted_terms": new_terms,
            "words_changed": labels.count(0),
            "words_total": len(labels),
            "total_verified_vocabulary": len(self._counts),
        }

    # ------------------------------------------------------------------ using what was learned
    def get_vocabulary(self) -> list[str]:
        """All learned terms, most frequently corrected first."""
        return [self._display[k] for k, _ in self._ranked()]

    def get_vocabulary_counts(self) -> list[dict[str, Any]]:
        return [{"term": self._display[k], "count": c} for k, c in self._ranked()]

    def get_whisper_prompt(self, base_prompt: Optional[str] = None) -> str:
        """Build a Faster-Whisper ``initial_prompt`` that boosts the most-corrected terms."""
        terms = self.get_vocabulary()[:MAX_PROMPT_TERMS]
        prefix = f"Domain terms: {', '.join(terms)}." if terms else ""
        return " ".join(p for p in (prefix, (base_prompt or "").strip()) if p)

    def _ranked(self) -> list[tuple[str, int]]:
        return sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))
