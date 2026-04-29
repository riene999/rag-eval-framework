from __future__ import annotations

import re
from collections import Counter
from typing import Any

from agentic_eval.utils.text import normalize_text


def compute_generation_metrics(
    answer: str,
    expected_answer: str | None,
    expected_keywords: list[str],
    citations: list[Any],
    retrieved_chunks: list[Any],
    latency_ms: int | None,
    client_latency_ms: int,
) -> dict[str, Any]:
    """Compute generation quality and response metadata metrics."""
    normalized_answer = normalize_text(answer)
    matched_keywords = [
        keyword for keyword in expected_keywords if normalize_text(keyword) in normalized_answer
    ]
    keyword_recall = (
        len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0
    )

    return {
        "keyword_recall": keyword_recall,
        "matched_keywords": matched_keywords,
        **compute_answer_overlap_metrics(answer, expected_answer),
        "answer_length": len(answer),
        "citation_count": len(citations),
        "citation_consistency": citations_match_retrieved_chunks(citations, retrieved_chunks),
        "latency_ms": latency_ms,
        "client_latency_ms": client_latency_ms,
    }


def citations_match_retrieved_chunks(citations: list[Any], retrieved_chunks: list[Any]) -> bool:
    """Return whether every citation can be found in the retrieved chunk metadata or text."""
    if not citations:
        return True

    searchable_chunks: list[str] = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            searchable_chunks.append(
                normalize_text(
                    " ".join(
                        str(chunk.get(key, ""))
                        for key in ["chunk_id", "id", "source", "title", "text", "content"]
                    )
                )
            )
        else:
            searchable_chunks.append(normalize_text(str(chunk)))

    for citation in citations:
        normalized_citation = normalize_text(str(citation))
        if not any(normalized_citation and normalized_citation in chunk for chunk in searchable_chunks):
            return False
    return True


def compute_answer_overlap_metrics(answer: str, expected_answer: str | None) -> dict[str, float]:
    """Compute token-level answer precision, recall, and F1 against a reference answer."""
    if not expected_answer:
        return {
            "answer_precision": 1.0,
            "answer_recall": 1.0,
            "answer_f1": 1.0,
        }

    answer_tokens = tokenize_for_f1(answer)
    expected_tokens = tokenize_for_f1(expected_answer)
    if not answer_tokens or not expected_tokens:
        return {
            "answer_precision": 0.0,
            "answer_recall": 0.0,
            "answer_f1": 0.0,
        }

    answer_counts = Counter(answer_tokens)
    expected_counts = Counter(expected_tokens)
    overlap = sum((answer_counts & expected_counts).values())
    precision = overlap / len(answer_tokens) if answer_tokens else 0.0
    recall = overlap / len(expected_tokens) if expected_tokens else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "answer_precision": precision,
        "answer_recall": recall,
        "answer_f1": f1,
    }


def tokenize_for_f1(text: str) -> list[str]:
    """Tokenize text for lightweight, dependency-free answer F1."""
    return re.findall(r"[a-z0-9]+", normalize_text(text))
