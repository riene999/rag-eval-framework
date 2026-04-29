from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvaluationCase:
    """A single test case for evaluating an external RAG or Agent system."""

    case_id: str
    question: str
    case_type: str
    expected_answer: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    gold_evidence: list[str] = field(default_factory=list)
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationCase":
        """Create an evaluation case from a dictionary."""
        return cls(
            case_id=str(data["case_id"]),
            question=str(data["question"]),
            case_type=str(data.get("case_type", "single_hop")),
            expected_answer=data.get("expected_answer"),
            expected_keywords=list(data.get("expected_keywords", [])),
            gold_evidence=list(data.get("gold_evidence", [])),
            difficulty=str(data.get("difficulty", "medium")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the case into a JSON-compatible dictionary."""
        return asdict(self)
