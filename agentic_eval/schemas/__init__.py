"""Dataclass schemas used by the evaluation framework."""

from agentic_eval.schemas.case_schema import EvaluationCase
from agentic_eval.schemas.result_schema import EvaluationResult, TargetAgentResponse

__all__ = ["EvaluationCase", "EvaluationResult", "TargetAgentResponse"]
