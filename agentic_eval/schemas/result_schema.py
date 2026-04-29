from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agentic_eval.schemas.case_schema import EvaluationCase


@dataclass
class TargetAgentResponse:
    """Normalized response returned by a target RAG or Agent system."""

    answer: str
    retrieved_chunks: list[Any] = field(default_factory=list)
    citations: list[Any] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    latency_ms: int | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    client_latency_ms: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TargetAgentResponse":
        """Create a normalized target response from a dictionary."""
        return cls(
            answer=str(data.get("answer", "")),
            retrieved_chunks=list(data.get("retrieved_chunks", [])),
            citations=list(data.get("citations", [])),
            trace=dict(data.get("trace", {})),
            latency_ms=data.get("latency_ms"),
            raw_response=dict(data.get("raw_response", data)),
            client_latency_ms=int(data.get("client_latency_ms", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the response into a JSON-compatible dictionary."""
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result for one case."""

    case: EvaluationCase
    response: TargetAgentResponse
    metrics: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    error_type: str | None = None
    diagnosis: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into a JSON-compatible dictionary."""
        return asdict(self)
