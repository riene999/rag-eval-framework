from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any

from agentic_eval.utils.text import normalize_text


def compute_retrieval_metrics(gold_evidence: list[str], retrieved_chunks: list[Any], k: int = 5) -> dict[str, Any]:
    """Compute retrieval Recall@k, MRR, source coverage, and precision metrics."""
    top_chunks = retrieved_chunks[:k]
    hit_ranks: list[int] = []
    relevant_in_top_k = 0
    for index, chunk in enumerate(top_chunks, start=1):
        if chunk_matches_any_gold(chunk, gold_evidence):
            hit_ranks.append(index)
            relevant_in_top_k += 1

    matched_expected_sources = [
        gold for gold in gold_evidence if any(chunk_matches_gold(chunk, gold) for chunk in top_chunks)
    ]
    expected_source_count = len([gold for gold in gold_evidence if gold])
    matched_expected_source_count = len(matched_expected_sources)
    source_coverage_ratio = (
        matched_expected_source_count / expected_source_count if expected_source_count else 1.0
    )

    recall_at_k = 1.0 if hit_ranks else 0.0
    mrr = 1.0 / hit_ranks[0] if hit_ranks else 0.0
    precision_at_k = relevant_in_top_k / len(top_chunks) if top_chunks else 0.0
    return {
        "recall_at_k": recall_at_k,
        "mrr": mrr,
        "retrieved_count": len(retrieved_chunks),
        "matched_expected_sources": matched_expected_sources,
        "matched_expected_source_count": matched_expected_source_count,
        "expected_source_count": expected_source_count,
        "source_coverage_ratio": source_coverage_ratio,
        "all_expected_sources_hit": (
            matched_expected_source_count == expected_source_count if expected_source_count else True
        ),
        "precision_at_k": precision_at_k,
        "relevant_in_top_k": relevant_in_top_k,
        "avg_relevant_in_top_k": float(relevant_in_top_k),
    }


def normalize_chunk(chunk: Any) -> str:
    """Convert a retrieved chunk into searchable normalized text."""
    if isinstance(chunk, dict):
        values = [
            chunk.get("chunk_id", ""),
            chunk.get("id", ""),
            chunk.get("source", ""),
            chunk.get("title", ""),
            chunk.get("text", ""),
            chunk.get("content", ""),
        ]
        return normalize_text(" ".join(str(value) for value in values if value is not None))
    return normalize_text(str(chunk))


def chunk_matches_any_gold(chunk: Any, gold_evidence: list[str]) -> bool:
    """Return whether a chunk matches any expected evidence item."""
    return any(chunk_matches_gold(chunk, gold) for gold in gold_evidence if gold)


def chunk_matches_gold(chunk: Any, gold: str) -> bool:
    """Return whether a chunk matches one expected evidence item.

    Matching uses both plain normalized substring checks and a looser filename-style
    normalization so source paths like ``D:/data/Local_SGD.pdf`` can match expected
    names like ``Local SGD.pdf``.
    """
    chunk_text = normalize_chunk(chunk)
    normalized_gold = normalize_text(gold)
    if normalized_gold and normalized_gold in chunk_text:
        return True

    compact_gold = normalize_source_name(gold)
    compact_chunk = normalize_source_name(extract_chunk_source_text(chunk))
    return bool(compact_gold and compact_gold in compact_chunk)


def extract_chunk_source_text(chunk: Any) -> str:
    """Extract source-like metadata from a retrieved chunk."""
    if isinstance(chunk, dict):
        values = [
            chunk.get("chunk_id", ""),
            chunk.get("id", ""),
            chunk.get("source", ""),
            chunk.get("title", ""),
            chunk.get("filename", ""),
            chunk.get("file_name", ""),
            chunk.get("path", ""),
        ]
        return " ".join(str(value) for value in values if value is not None)
    return str(chunk)


def normalize_source_name(value: str) -> str:
    """Normalize source names for robust path and PDF filename matching."""
    normalized = normalize_text(value).replace("\\", "/")
    basename = PurePath(normalized).name or normalized.split("/")[-1]
    basename = re.sub(r"\.pdf$", "", basename)
    return re.sub(r"[^a-z0-9]+", "", basename)
