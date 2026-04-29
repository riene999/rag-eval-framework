from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_eval.agents import CaseGeneratorAgent, DiagnosisAgent, EvaluationAgent
from agentic_eval.clients import HttpTargetAgentClient, MockTargetAgentClient
from agentic_eval.reports import ReportGenerator
from agentic_eval.schemas.result_schema import EvaluationResult

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import track
    from rich.table import Table

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic RAG/Agent evaluation pipeline.")
    parser.add_argument("--mock", action="store_true", help="Use the built-in mock target client.")
    parser.add_argument("--base-url", help="Base URL for the external target service.")
    parser.add_argument("--endpoint", default="/ask", help="Target ask endpoint. Defaults to /ask.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--case-file", help="JSONL file containing evaluation cases.")
    parser.add_argument("--docs-dir", default="examples/sample_docs", help="Directory of Markdown docs for generated cases.")
    parser.add_argument("--generated-case-file", default="outputs/generated_cases.jsonl", help="Where generated cases are saved.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for reports.")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval top_k for target calls and metrics.")
    parser.add_argument("--max-cases", type=int, default=5, help="Maximum generated cases when no case file is provided.")
    return parser.parse_args()


def main() -> int:
    # 确定client->确定case_generator->确定evaluator->诊断结果->生成报告
    args = parse_args()
    if args.mock:
        client = MockTargetAgentClient()
    elif args.base_url:
        client = HttpTargetAgentClient(args.base_url, endpoint=args.endpoint, timeout=args.timeout)
    else:
        print_error("Error: provide either --mock or --base-url.")
        return 2

    case_generator = CaseGeneratorAgent(args.docs_dir)
    # 如果提供了文件，则从文件加载测试用例，否则自行生成测试用例
    if args.case_file:
        cases = case_generator.load_cases(args.case_file)
    else:
        cases = case_generator.generate_cases(max_cases=args.max_cases)
        case_generator.save_cases(cases, args.generated_case_file)

    evaluator = EvaluationAgent(client, top_k=args.top_k)
    diagnosis = DiagnosisAgent()
    reporter = ReportGenerator(args.output_dir)

    try:
        results: list[EvaluationResult] = []
        case_iterable = (
            track(cases, description="Evaluating cases", transient=False)
            if RICH_AVAILABLE
            else cases
        )
        for case in case_iterable:
            results.append(evaluator.evaluate_case(case))
        diagnosis.diagnose_all(results)
        artifacts = reporter.generate(results)
    except Exception as exc:
        print_error(f"Evaluation failed: {exc}")
        return 1

    print_run_summary(results, artifacts)
    return 0


def print_error(message: str) -> None:
    if RICH_AVAILABLE and console is not None:
        console.print(f"[bold red]{message}[/bold red]")
    else:
        print(message, file=sys.stderr)


def print_run_summary(results: list[EvaluationResult], artifacts: dict[str, str | None]) -> None:
    # 根据EvaluationResult通过RICH_CLI创建结果表格
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    pass_rate = passed / total if total else 0.0
    failed = total - passed

    if RICH_AVAILABLE and console is not None:
        summary = Table(title="Evaluation Summary", show_header=True, header_style="bold cyan")
        summary.add_column("Metric")
        summary.add_column("Value", justify="right")
        summary.add_row("Total cases", str(total))
        summary.add_row("Passed", f"[green]{passed}[/green]")
        summary.add_row("Failed", f"[red]{failed}[/red]" if failed else "0")
        summary.add_row("Pass rate", f"{pass_rate:.2%}")
        summary.add_row("Average Recall@k", f"{avg_metric(results, 'recall_at_k'):.3f}")
        summary.add_row("Average MRR", f"{avg_metric(results, 'mrr'):.3f}")
        summary.add_row("Average source coverage", f"{avg_metric(results, 'source_coverage_ratio'):.3f}")
        summary.add_row("Average keyword recall", f"{avg_metric(results, 'keyword_recall'):.3f}")
        summary.add_row("Average answer F1", f"{avg_metric(results, 'answer_f1'):.3f}")
        summary.add_row("Latency p95", f"{latency_percentile(results, 95):.1f} ms")
        console.print(summary)

        failed_results = [result for result in results if not result.passed]
        if failed_results:
            table = Table(title="Top Failed Cases", show_header=True, header_style="bold red")
            table.add_column("Case ID")
            table.add_column("Stage")
            table.add_column("Recall@k", justify="right")
            table.add_column("Coverage", justify="right")
            table.add_column("Keyword", justify="right")
            table.add_column("Answer F1", justify="right")
            for result in failed_results[:8]:
                table.add_row(
                    result.case.case_id,
                    str(result.diagnosis.get("failed_stage", result.error_type or "unknown")),
                    f"{float(result.metrics.get('recall_at_k', 0.0)):.3f}",
                    f"{float(result.metrics.get('source_coverage_ratio', 0.0)):.3f}",
                    f"{float(result.metrics.get('keyword_recall', 0.0)):.3f}",
                    f"{float(result.metrics.get('answer_f1', 0.0)):.3f}",
                )
            console.print(table)

        paths = [
            f"[bold]Markdown report:[/bold] {artifacts['markdown']}",
            f"[bold]JSON report:[/bold] {artifacts['json']}",
        ]
        if artifacts.get("failure_chart"):
            paths.append(f"[bold]Failure chart:[/bold] {artifacts['failure_chart']}")
        console.print(Panel("\n".join(paths), title="Artifacts", border_style="blue"))
    else:
        print(f"Total cases: {total}")
        print(f"Passed: {passed}")
        print(f"Pass rate: {pass_rate:.2%}")
        print(f"Markdown report: {artifacts['markdown']}")
        print(f"JSON report: {artifacts['json']}")
        if artifacts.get("failure_chart"):
            print(f"Failure chart: {artifacts['failure_chart']}")


def avg_metric(results: list[EvaluationResult], metric: str) -> float:
    # 返回指定指标的默认值
    values = [float(result.metrics.get(metric, 0.0)) for result in results]
    return mean(values) if values else 0.0


def latency_percentile(results: list[EvaluationResult], percentile: int) -> float:
    # 返回延迟的指定百分位数
    values = sorted(latency_values(results))
    if not values:
        return 0.0
    rank = max(1, round(percentile / 100 * len(values)))
    return values[min(rank, len(values)) - 1]


def latency_values(results: list[EvaluationResult]) -> list[float]:
    # 收集每个案例的延迟，优先选择目标延迟而非客户端延迟
    values: list[float] = []
    for result in results:
        latency = result.metrics.get("latency_ms")
        if latency is None:
            latency = result.metrics.get("client_latency_ms", 0)
        values.append(float(latency))
    return values


if __name__ == "__main__":
    raise SystemExit(main())
