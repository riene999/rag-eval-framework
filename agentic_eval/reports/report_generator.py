from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from agentic_eval.schemas.result_schema import EvaluationResult
from agentic_eval.utils.io import write_json, write_text
from agentic_eval.utils.text import summarize_text


class ReportGenerator:

    def __init__(self, output_dir: str | Path = "outputs") -> None:
        self.output_dir = Path(output_dir)

    def generate(self, results: list[EvaluationResult]) -> dict[str, str | None]:
        # 生成Markdown报告和JSON结果文件，并返回路径
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / "eval_results.json"
        md_path = self.output_dir / "eval_report.md"
        chart_path = self._generate_failure_chart(results)

        write_json(json_path, [result.to_dict() for result in results])
        write_text(md_path, self._render_markdown(results, chart_path))
        return {
            "markdown": str(md_path),
            "json": str(json_path),
            "failure_chart": str(chart_path) if chart_path else None,
        }

    def _render_markdown(self, results: list[EvaluationResult], chart_path: Path | None) -> str:
        total = len(results)
        passed = sum(1 for result in results if result.passed)
        pass_rate = passed / total if total else 0.0
        avg_recall = avg_metric(results, "recall_at_k")
        avg_mrr = avg_metric(results, "mrr")
        avg_source_coverage = avg_metric(results, "source_coverage_ratio")
        all_sources_hit_rate = avg_bool_metric(results, "all_expected_sources_hit")
        avg_precision_at_k = avg_metric(results, "precision_at_k")
        avg_relevant = avg_metric(results, "avg_relevant_in_top_k")
        avg_keyword = avg_metric(results, "keyword_recall")
        avg_answer_f1 = avg_metric(results, "answer_f1")
        avg_latency = avg_latency_metric(results)
        latency_p50 = latency_percentile(results, 50)
        latency_p95 = latency_percentile(results, 95)
        latency_p99 = latency_percentile(results, 99)
        bad_cases = [result for result in results if not result.passed]
        failure_counts = Counter(
            result.diagnosis.get("failed_stage", result.error_type or "unknown")
            for result in bad_cases
        )

        lines = [
            "# Agentic Evaluation Report",
            "",
            "## Summary",
            "",
            f"- Total cases: {total}",
            f"- Passed cases: {passed}",
            f"- Pass rate: {pass_rate:.2%}",
            f"- Average Recall@k: {avg_recall:.3f}",
            f"- Average MRR: {avg_mrr:.3f}",
            f"- Average source coverage: {avg_source_coverage:.3f}",
            f"- All expected sources hit rate: {all_sources_hit_rate:.2%}",
            f"- Average Precision@k: {avg_precision_at_k:.3f}",
            f"- Average relevant chunks in Top-K: {avg_relevant:.3f}",
            f"- Average keyword recall: {avg_keyword:.3f}",
            f"- Average answer F1: {avg_answer_f1:.3f}",
            f"- Average latency: {avg_latency:.1f} ms",
            f"- Latency p50: {latency_p50:.1f} ms",
            f"- Latency p95: {latency_p95:.1f} ms",
            f"- Latency p99: {latency_p99:.1f} ms",
            "",
        ]

        if chart_path:
            lines.extend(["## Failure Distribution", "", f"![Failure Distribution]({chart_path.name})", ""])

        lines.extend(["## Pass Rate By Case Type", "", "| Case type | Total | Passed | Pass rate |", "|---|---:|---:|---:|"])
        for case_type, group in grouped_by_case_type(results).items():
            group_passed = sum(1 for result in group if result.passed)
            lines.append(f"| {case_type} | {len(group)} | {group_passed} | {group_passed / len(group):.2%} |")

        lines.extend(["", "## Bad Case Classification", "", "| Failed stage | Count |", "|---|---:|"])
        if failure_counts:
            for stage, count in failure_counts.most_common():
                lines.append(f"| {stage} | {count} |")
        else:
            lines.append("| none | 0 |")

        lines.extend(["", "## Top Failed Cases", "", "| Case ID | Type | Stage | Recall@k | Source coverage | Keyword recall | Answer F1 | Question | Suggestion |", "|---|---|---|---:|---:|---:|---:|---|---|"])
        for result in bad_cases[:10]:
            diagnosis = result.diagnosis or {}
            lines.append(
                "| {case_id} | {case_type} | {stage} | {recall:.3f} | {coverage:.3f} | {keyword:.3f} | {answer_f1:.3f} | {question} | {suggestion} |".format(
                    case_id=result.case.case_id,
                    case_type=result.case.case_type,
                    stage=diagnosis.get("failed_stage", result.error_type or "unknown"),
                    recall=float(result.metrics.get("recall_at_k", 0.0)),
                    coverage=float(result.metrics.get("source_coverage_ratio", 0.0)),
                    keyword=float(result.metrics.get("keyword_recall", 0.0)),
                    answer_f1=float(result.metrics.get("answer_f1", 0.0)),
                    question=summarize_text(result.case.question, 120).replace("|", "\\|"),
                    suggestion=summarize_text(str(diagnosis.get("suggestion", "")), 160).replace("|", "\\|"),
                )
            )
        if not bad_cases:
            lines.append("| none | - | - | - | - | - | - | - | - |")

        lines.extend(["", "## Retrieval Detail By Case", "", "| Case ID | Type | Expected sources | Matched sources | Source coverage | All hit | Precision@k | MRR |", "|---|---|---:|---:|---:|---|---:|---:|"])
        for result in results:
            lines.append(
                "| {case_id} | {case_type} | {expected} | {matched} | {coverage:.3f} | {all_hit} | {precision:.3f} | {mrr:.3f} |".format(
                    case_id=result.case.case_id,
                    case_type=result.case.case_type,
                    expected=int(result.metrics.get("expected_source_count", 0)),
                    matched=int(result.metrics.get("matched_expected_source_count", 0)),
                    coverage=float(result.metrics.get("source_coverage_ratio", 0.0)),
                    all_hit="yes" if result.metrics.get("all_expected_sources_hit", False) else "no",
                    precision=float(result.metrics.get("precision_at_k", 0.0)),
                    mrr=float(result.metrics.get("mrr", 0.0)),
                )
            )

        lines.extend(["", "## Suggestions By Failure Type", "", "| Failed stage | Suggestion |", "|---|---|"])
        suggestions = suggestions_by_stage(bad_cases)
        if suggestions:
            for stage, suggestion in suggestions.items():
                escaped_suggestion = suggestion.replace("|", "\\|")
                lines.append(f"| {stage} | {escaped_suggestion} |")
        else:
            lines.append("| none | No failed cases. |")

        return "\n".join(lines) + "\n"

    def _generate_failure_chart(self, results: list[EvaluationResult]) -> Path | None:
        # 生成失败分布图表
        failures = [
            result.diagnosis.get("failed_stage", result.error_type or "unknown")
            for result in results
            if not result.passed
        ]
        if not failures:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return None

        counts = Counter(failures)
        chart_path = self.output_dir / "failure_distribution.png"
        try:
            plt.figure(figsize=(7, 4))
            plt.bar(list(counts.keys()), list(counts.values()), color="#4f8fba")
            plt.title("Failure Distribution")
            plt.xlabel("Failed stage")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(chart_path)
            plt.close()
            return chart_path
        except Exception:
            return None


def grouped_by_case_type(results: list[EvaluationResult]) -> dict[str, list[EvaluationResult]]:
    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    for result in results:
        grouped[result.case.case_type].append(result)
    return dict(grouped)


def avg_metric(results: list[EvaluationResult], metric: str) -> float:
    values = [float(result.metrics.get(metric, 0.0)) for result in results]
    return mean(values) if values else 0.0


def avg_bool_metric(results: list[EvaluationResult], metric: str) -> float:
    values = [1.0 if result.metrics.get(metric, False) else 0.0 for result in results]
    return mean(values) if values else 0.0


def avg_latency_metric(results: list[EvaluationResult]) -> float:
    values = latency_values(results)
    return mean(values) if values else 0.0


def latency_percentile(results: list[EvaluationResult], percentile: int) -> float:
    values = sorted(latency_values(results))
    if not values:
        return 0.0
    rank = max(1, round(percentile / 100 * len(values)))
    return values[min(rank, len(values)) - 1]


def latency_values(results: list[EvaluationResult]) -> list[float]:
    values: list[float] = []
    for result in results:
        latency = result.metrics.get("latency_ms")
        if latency is None:
            latency = result.metrics.get("client_latency_ms", 0)
        values.append(float(latency))
    return values


def suggestions_by_stage(results: list[EvaluationResult]) -> dict[str, str]:
    suggestions: dict[str, str] = {}
    for result in results:
        stage = str(result.diagnosis.get("failed_stage", result.error_type or "unknown"))
        suggestion = str(result.diagnosis.get("suggestion", "Inspect the failed case."))
        suggestions.setdefault(stage, suggestion)
    return suggestions
