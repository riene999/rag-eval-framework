from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Lowercase text and collapse whitespace for robust rule-based matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def summarize_text(text: str, max_chars: int = 240) -> str:
    """Return a compact one-line text preview."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."
