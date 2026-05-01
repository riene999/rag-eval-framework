from __future__ import annotations

import json
import os
from typing import Any

import requests

from agentic_eval.schemas.result_schema import EvaluationResult


class DiagnosisAgent:

    def __init__(
        self,
        latency_threshold_ms: int = 5000,
        use_llm: bool = False,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        llm_timeout: int = 60,
    ) -> None:
        self.latency_threshold_ms = latency_threshold_ms
        self.use_llm = use_llm
        self.llm_model = llm_model or os.getenv("DIAGNOSIS_LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL") or "deepseek-v4-flash"
        self.llm_api_key = llm_api_key or os.getenv("DIAGNOSIS_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_base_url = (
            llm_base_url
            or os.getenv("DIAGNOSIS_LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.llm_timeout = llm_timeout

    def diagnose(self, result: EvaluationResult) -> dict[str, Any]:
        rule_diagnosis = self._diagnose_by_rules(result)
        if not self.use_llm:
            result.diagnosis = rule_diagnosis
            return rule_diagnosis

        if not self.llm_api_key:
            diagnosis = {
                **rule_diagnosis,
                "diagnosis_source": "rule_fallback",
                "llm_error": "LLM diagnosis is enabled, but no API key was provided.",
            }
            result.diagnosis = diagnosis
            return diagnosis

        try:
            diagnosis = self._diagnose_with_llm(result, rule_diagnosis)
        except Exception as exc:
            diagnosis = {
                **rule_diagnosis,
                "diagnosis_source": "rule_fallback",
                "llm_error": str(exc),
            }

        result.diagnosis = diagnosis
        return diagnosis

    def diagnose_with_rules(self, result: EvaluationResult) -> dict[str, Any]:
        diagnosis = self._diagnose_by_rules(result)
        result.diagnosis = diagnosis
        return diagnosis

    def _diagnose_by_rules(self, result: EvaluationResult) -> dict[str, Any]:
        metrics = result.metrics
        observed_latency = metrics.get("latency_ms") or metrics.get("client_latency_ms") or 0

        if bool(metrics.get("target_error", False)):
            return {
                "failed_stage": "target_error",
                "root_cause": str(metrics.get("target_error_message", "Target call failed.")),
                "suggestion": "Check target service availability, timeout settings, and slow cases in the target RAG pipeline.",
                "priority": "high",
                "diagnosis_source": "rule",
            }
        if result.passed:
            return {
                "failed_stage": "none",
                "root_cause": "Case passed the default evaluation rules.",
                "suggestion": "No immediate action required.",
                "priority": "low",
                "diagnosis_source": "rule",
            }
        if float(metrics.get("recall_at_k", 0.0)) == 0.0 or float(metrics.get("source_coverage_ratio", 1.0)) == 0.0:
            return {
                "failed_stage": "retrieval",
                "root_cause": "Gold evidence was not found in the retrieved chunks.",
                "suggestion": "Improve indexing, chunking, query rewriting, embedding quality, or top_k retrieval settings.",
                "priority": "high",
                "diagnosis_source": "rule",
            }
        if float(metrics.get("source_coverage_ratio", 1.0)) < 0.5:
            return {
                "failed_stage": "retrieval",
                "root_cause": "The retriever found some evidence but missed most expected sources for a multi-source case.",
                "suggestion": "Improve multi-hop query decomposition, source diversity, and reranking so all expected sources can be covered.",
                "priority": "high",
                "diagnosis_source": "rule",
            }
        if not bool(metrics.get("citation_consistency", True)):
            return {
                "failed_stage": "citation",
                "root_cause": "At least one citation does not map to retrieved chunks.",
                "suggestion": "Ensure generated citations use stable retrieved chunk ids or source identifiers.",
                "priority": "medium",
                "diagnosis_source": "rule",
            }
        if float(metrics.get("keyword_recall", 0.0)) < 0.5:
            return {
                "failed_stage": "generation",
                "root_cause": "Relevant evidence was retrieved, but the answer missed expected keywords.",
                "suggestion": "Improve answer synthesis prompts, context grounding, and instruction following.",
                "priority": "high",
                "diagnosis_source": "rule",
            }
        if int(observed_latency) > self.latency_threshold_ms:
            return {
                "failed_stage": "latency",
                "root_cause": f"Observed latency exceeded {self.latency_threshold_ms} ms.",
                "suggestion": "Profile retrieval, model calls, reranking, and tool execution paths.",
                "priority": "medium",
                "diagnosis_source": "rule",
            }
        return {
            "failed_stage": "unknown",
            "root_cause": "The failure did not match a known rule.",
            "suggestion": "Inspect raw response, trace, retrieved chunks, and expected case definition.",
            "priority": "low",
            "diagnosis_source": "rule",
        }

    def _diagnose_with_llm(self, result: EvaluationResult, rule_diagnosis: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior RAG evaluation diagnostician. "
                        "Analyze one evaluation result and return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instructions": {
                                "required_keys": [
                                    "failed_stage",
                                    "root_cause",
                                    "suggestion",
                                    "priority",
                                ],
                                "failed_stage_values": [
                                    "none",
                                    "target_error",
                                    "retrieval",
                                    "generation",
                                    "citation",
                                    "latency",
                                    "tool_use",
                                    "unknown",
                                ],
                                "priority_values": ["low", "medium", "high"],
                                "style": "Be concise and actionable. Prefer Chinese if the case question is Chinese.",
                            },
                            "rule_diagnosis": rule_diagnosis,
                            "evaluation_result": self._compact_result(result),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.2,
        }
        response = requests.post(
            f"{self.llm_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.llm_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.llm_timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        diagnosis = self._parse_llm_json(content)
        return self._normalize_llm_diagnosis(diagnosis, rule_diagnosis)

    def _compact_result(self, result: EvaluationResult) -> dict[str, Any]:
        response = result.response
        return {
            "case": {
                "case_id": result.case.case_id,
                "question": result.case.question,
                "case_type": result.case.case_type,
                "difficulty": result.case.difficulty,
                "expected_answer": self._truncate(result.case.expected_answer or "", 1000),
                "expected_keywords": result.case.expected_keywords,
                "gold_evidence": result.case.gold_evidence,
            },
            "target_response": {
                "answer": self._truncate(response.answer, 2000),
                "retrieved_chunks": self._truncate_json(response.retrieved_chunks, 3000),
                "citations": self._truncate_json(response.citations, 1500),
                "trace": self._truncate_json(response.trace, 2000),
                "latency_ms": response.latency_ms,
                "client_latency_ms": response.client_latency_ms,
            },
            "metrics": result.metrics,
            "passed": result.passed,
            "error_type": result.error_type,
        }

    def _normalize_llm_diagnosis(
        self,
        diagnosis: dict[str, Any],
        rule_diagnosis: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = {
            "failed_stage": str(diagnosis.get("failed_stage") or rule_diagnosis["failed_stage"]),
            "root_cause": str(diagnosis.get("root_cause") or rule_diagnosis["root_cause"]),
            "suggestion": str(diagnosis.get("suggestion") or rule_diagnosis["suggestion"]),
            "priority": str(diagnosis.get("priority") or rule_diagnosis["priority"]),
            "diagnosis_source": "llm",
            "rule_diagnosis": rule_diagnosis,
        }
        for key, value in diagnosis.items():
            if key not in normalized:
                normalized[key] = value
        return normalized

    def _parse_llm_json(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("LLM response did not contain a JSON object.")
            parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM diagnosis must be a JSON object.")
        return parsed

    def _truncate_json(self, value: Any, limit: int) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return self._truncate(text, limit)

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 15] + "...[truncated]"

