from __future__ import annotations

from pathlib import Path

from agentic_eval.schemas.case_schema import EvaluationCase
from agentic_eval.utils.io import read_jsonl, write_jsonl
from agentic_eval.utils.text import summarize_text


class CaseGeneratorAgent:

    def __init__(self, docs_dir: str | Path = "examples/sample_docs") -> None:
        self.docs_dir = Path(docs_dir)

    def load_cases(self, case_file: str | Path) -> list[EvaluationCase]:
        """从已有JSON文件加载测评用例"""
        return [EvaluationCase.from_dict(row) for row in read_jsonl(case_file)]

    def save_cases(self, cases: list[EvaluationCase], output_file: str | Path) -> None:
        """将测评用例保存在JSON"""
        write_jsonl(output_file, (case.to_dict() for case in cases))

    def generate_cases(self, max_cases: int = 5) -> list[EvaluationCase]:
        """基于提供的文档生成测评用例"""
        docs = self._load_docs()
        cases = self._template_cases(docs)
        return cases[:max_cases]

    def _load_docs(self) -> dict[str, str]:
        docs: dict[str, str] = {}
        for path in sorted(self.docs_dir.glob("*.md")):
            docs[path.stem] = path.read_text(encoding="utf-8")
        return docs

    def _template_cases(self, docs: dict[str, str]) -> list[EvaluationCase]:
        api_preview = summarize_text(docs.get("api_doc", ""), 180)
        schema_preview = summarize_text(docs.get("schema_doc", ""), 180)
        error_preview = summarize_text(docs.get("error_cases", ""), 180)
        return [
            EvaluationCase(
                case_id="case_single_hop_api",
                question="What HTTP endpoint should a target RAG system implement?",
                case_type="single_hop",
                expected_answer="The target should implement POST /ask.",
                expected_keywords=["POST", "/ask", "question"],
                gold_evidence=["api_doc", "POST /ask", api_preview],
                difficulty="easy",
            ),
            EvaluationCase(
                case_id="case_multi_hop_response",
                question="Which fields should the target response include for answer quality and retrieval diagnosis?",
                case_type="multi_hop",
                expected_answer="It should include answer, retrieved_chunks, citations, trace, and latency information.",
                expected_keywords=["answer", "retrieved_chunks", "citations", "trace", "latency_ms"],
                gold_evidence=["api_doc", "schema_doc", schema_preview],
                difficulty="medium",
            ),
            EvaluationCase(
                case_id="case_citation_consistency",
                question="How should citations relate to retrieved evidence chunks?",
                case_type="citation",
                expected_answer="Citations should reference retrieved chunk identifiers or sources.",
                expected_keywords=["citations", "retrieved", "chunk"],
                gold_evidence=["api_doc", "schema_doc"],
                difficulty="medium",
            ),
            EvaluationCase(
                case_id="case_tool_use_trace",
                question="What trace information can an Agent target return for tool-use diagnosis?",
                case_type="tool_use",
                expected_answer="The trace can include tools, query rewrite, and execution steps.",
                expected_keywords=["tools", "rewrite_query", "steps"],
                gold_evidence=["api_doc", "trace", "tools"],
                difficulty="hard",
            ),
            EvaluationCase(
                case_id="case_hallucination_unknown",
                question="What exact production database password is documented in the sample docs?",
                case_type="hallucination",
                expected_answer=None,
                expected_keywords=["not documented", "unknown", "no evidence"],
                gold_evidence=["error_cases", error_preview],
                difficulty="hard",
            ),
        ]
