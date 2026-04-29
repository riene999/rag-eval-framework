from __future__ import annotations

from typing import Any

from agentic_eval.schemas.result_schema import EvaluationResult


class DiagnosisAgent:

    def __init__(self, latency_threshold_ms: int = 5000) -> None:
        self.latency_threshold_ms = latency_threshold_ms

    def diagnose(self, result: EvaluationResult) -> dict[str, Any]:
        # 诊断单个失败用例
        metrics = result.metrics
        observed_latency = metrics.get("latency_ms") or metrics.get("client_latency_ms") or 0

        if bool(metrics.get("target_error", False)):
            diagnosis = {
                "failed_stage": "target_error",
                "root_cause": str(metrics.get("target_error_message", "Target call failed.")),
                "suggestion": "Check target service availability, timeout settings, and slow cases in the target RAG pipeline.",
                "priority": "high",
            }
        elif result.passed:
            diagnosis = {
                "failed_stage": "none",
                "root_cause": "Case passed the default evaluation rules.",
                "suggestion": "No immediate action required.",
                "priority": "low",
            }
        elif float(metrics.get("recall_at_k", 0.0)) == 0.0 or float(metrics.get("source_coverage_ratio", 1.0)) == 0.0:
            diagnosis = {
                "failed_stage": "retrieval",
                "root_cause": "Gold evidence was not found in the retrieved chunks.",
                "suggestion": "Improve indexing, chunking, query rewriting, embedding quality, or top_k retrieval settings.",
                "priority": "high",
            }
        elif float(metrics.get("source_coverage_ratio", 1.0)) < 0.5:
            diagnosis = {
                "failed_stage": "retrieval",
                "root_cause": "The retriever found some evidence but missed most expected sources for a multi-source case.",
                "suggestion": "Improve multi-hop query decomposition, source diversity, and reranking so all expected sources can be covered.",
                "priority": "high",
            }
        elif not bool(metrics.get("citation_consistency", True)):
            diagnosis = {
                "failed_stage": "citation",
                "root_cause": "At least one citation does not map to retrieved chunks.",
                "suggestion": "Ensure generated citations use stable retrieved chunk ids or source identifiers.",
                "priority": "medium",
            }
        elif float(metrics.get("keyword_recall", 0.0)) < 0.5:
            diagnosis = {
                "failed_stage": "generation",
                "root_cause": "Relevant evidence was retrieved, but the answer missed expected keywords.",
                "suggestion": "Improve answer synthesis prompts, context grounding, and instruction following.",
                "priority": "high",
            }
        elif int(observed_latency) > self.latency_threshold_ms:
            diagnosis = {
                "failed_stage": "latency",
                "root_cause": f"Observed latency exceeded {self.latency_threshold_ms} ms.",
                "suggestion": "Profile retrieval, model calls, reranking, and tool execution paths.",
                "priority": "medium",
            }
        else:
            diagnosis = {
                "failed_stage": "unknown",
                "root_cause": "The failure did not match a known rule.",
                "suggestion": "Inspect raw response, trace, retrieved chunks, and expected case definition.",
                "priority": "low",
            }

        result.diagnosis = diagnosis
        return diagnosis

    def diagnose_all(self, results: list[EvaluationResult]) -> list[EvaluationResult]:
        for result in results:
            self.diagnose(result)
        return results
