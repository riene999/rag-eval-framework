from __future__ import annotations

from typing import Any

from agentic_eval.clients.base import TargetAgentClient


class MockTargetAgentClient(TargetAgentClient):
    """模拟被测试的RAG系统"""

    def ask(self, question: str, **kwargs: Any) -> dict[str, Any]:
        case_id = str(kwargs.get("case_id", "mock_case"))
        lowered = question.lower()
        answer_parts = [f"Mock answer for: {question}"]

        if "api" in lowered or "endpoint" in lowered or "http" in lowered:
            answer_parts.append("The target API should expose POST /ask and return answer, retrieved_chunks, citations, trace, and latency_ms.")
        if "schema" in lowered or "fields" in lowered or "response" in lowered:
            answer_parts.append("The schema includes answer text, retrieved_chunks, citations, trace metadata, and raw response.")
        if "error" in lowered or "fail" in lowered:
            answer_parts.append("Failures should include clear status, URL, and response text.")
        if "citation" in lowered or "evidence" in lowered:
            answer_parts.append("Citations should point to retrieved chunk identifiers.")
        if "trace" in lowered or "tool" in lowered:
            answer_parts.append("Agent trace metadata may include tools, rewrite_query, and steps for tool-use diagnosis.")
        if "password" in lowered or "production database" in lowered:
            answer_parts.append("The sample docs contain no evidence for that secret, so it is unknown and not documented.")

        raw_response = {
            "answer": " ".join(answer_parts),
            "retrieved_chunks": [
                {
                    "chunk_id": "api_doc",
                    "text": "POST /ask accepts a question and returns answer, retrieved_chunks, citations, trace, and latency_ms.",
                    "score": 1.0,
                    "source": "examples/sample_docs/api_doc.md",
                },
                {
                    "chunk_id": "schema_doc",
                    "text": "A normalized target response contains answer, retrieved_chunks, citations, trace, latency_ms, raw_response, and client_latency_ms.",
                    "score": 0.9,
                    "source": "examples/sample_docs/schema_doc.md",
                },
                {
                    "chunk_id": "error_cases",
                    "text": "HTTP failures should expose status code, URL, and response text to make diagnosis clear.",
                    "score": 0.8,
                    "source": "examples/sample_docs/error_cases.md",
                },
            ],
            "citations": ["api_doc", "schema_doc"],
            "trace": {"mock": True, "case_id": case_id},
            "latency_ms": 0,
        }
        return {
            **raw_response,
            "raw_response": raw_response,
            "client_latency_ms": 0,
        }
