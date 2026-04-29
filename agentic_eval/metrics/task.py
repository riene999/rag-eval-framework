from __future__ import annotations

from typing import Any


def compute_passed(metrics: dict[str, Any]) -> bool:
    """Apply a simple default pass rule over retrieval and generation metrics."""
    expected_source_count = int(metrics.get("expected_source_count", 0))
    retrieval_passed = float(metrics.get("recall_at_k", 0.0)) > 0.0
    if expected_source_count > 1:
        retrieval_passed = float(metrics.get("source_coverage_ratio", 0.0)) >= 0.5

    return (
        float(metrics.get("keyword_recall", 0.0)) >= 0.5
        and retrieval_passed
        and bool(metrics.get("citation_consistency", True))
    )


def infer_error_type(metrics: dict[str, Any], passed: bool) -> str | None:
    """Infer the likely error type from metric values."""
    if passed:
        return None
    if float(metrics.get("recall_at_k", 0.0)) == 0.0 or float(metrics.get("source_coverage_ratio", 1.0)) == 0.0:
        return "retrieval"
    if not bool(metrics.get("citation_consistency", True)):
        return "citation"
    if float(metrics.get("keyword_recall", 0.0)) < 0.5:
        return "generation"
    return "unknown"
