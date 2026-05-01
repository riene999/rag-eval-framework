from __future__ import annotations

import time
from typing import Any

from agentic_eval.clients.base import TargetAgentClient
from agentic_eval.metrics.generation import compute_generation_metrics
from agentic_eval.metrics.retrieval import compute_retrieval_metrics
from agentic_eval.metrics.task import compute_passed, infer_error_type
from agentic_eval.schemas.case_schema import EvaluationCase
from agentic_eval.schemas.result_schema import EvaluationResult, TargetAgentResponse


class EvaluationAgent:
    """进行测评"""

    def __init__(self, client: TargetAgentClient, top_k: int = 5) -> None:
        self.client = client
        self.top_k = top_k

    def evaluate_case(self, case: EvaluationCase, **kwargs: Any) -> EvaluationResult:
        """用于测评单个用例"""
        start = time.perf_counter()
        try:
            raw_response = self.client.ask(case.question, case_id=case.case_id, top_k=self.top_k, **kwargs)
        except Exception as exc:
            client_latency_ms = int((time.perf_counter() - start) * 1000)
            response = TargetAgentResponse(
                answer="",
                retrieved_chunks=[],
                citations=[],
                trace={"error": str(exc), "case_id": case.case_id},
                latency_ms=None,
                raw_response={"error": str(exc)},
                client_latency_ms=client_latency_ms,
            )
            metrics = {
                **compute_retrieval_metrics(case.gold_evidence, response.retrieved_chunks, k=self.top_k),
                **compute_generation_metrics(
                    response.answer,
                    case.expected_answer,
                    case.expected_keywords,
                    response.citations,
                    response.retrieved_chunks,
                    response.latency_ms,
                    response.client_latency_ms,
                ),
                "target_error": True,
                "target_error_message": str(exc),
            }
            return EvaluationResult(
                case=case,
                response=response,
                metrics=metrics,
                passed=False,
                error_type="target_error",
            )

        # 通过TargetAgentResponse的回复生成标准response->计算指标metrics->metrics被包含在EvaluationResult中返回
        response = TargetAgentResponse.from_dict(raw_response)
        metrics = {
            **compute_retrieval_metrics(case.gold_evidence, response.retrieved_chunks, k=self.top_k),
            **compute_generation_metrics(
                response.answer,
                case.expected_answer,
                case.expected_keywords,
                response.citations,
                response.retrieved_chunks,
                response.latency_ms,
                response.client_latency_ms,
            ),
            "target_error": False,
        }
        passed = compute_passed(metrics)
        return EvaluationResult(
            case=case,
            response=response,
            metrics=metrics,
            passed=passed,
            error_type=infer_error_type(metrics, passed),
        )

