from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_eval.agents import CaseGeneratorAgent, DiagnosisAgent, EvaluationAgent
from agentic_eval.agents.pipeline import EvalPipeline
from agentic_eval.clients import HttpTargetAgentClient, MockTargetAgentClient
from agentic_eval.reports import ReportGenerator
from agentic_eval.schemas.result_schema import EvaluationResult

try:
    import yaml
except ImportError:
    yaml = None

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    console = None
    RICH_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agentic RAG/Agent evaluation pipeline.")
    parser.add_argument("--config", default="config.yaml", help="Path to the evaluation config YAML file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except Exception as exc:
        print_error(f"Error loading config: {exc}")
        return 2

    target_config = config["target"]
    if bool(target_config["mock"]):
        if target_config["base_url"]:
            print_info("target.mock=true, so target.base_url is ignored.")
        client = MockTargetAgentClient()
        target_mode = "mock"
    elif target_config["base_url"]:
        client = HttpTargetAgentClient(
            target_config["base_url"],
            endpoint=target_config["endpoint"],
            timeout=int(target_config["timeout"]),
        )
        target_mode = f"http {client.url}"
    else:
        print_error("Error: set target.mock=true or target.base_url in config.yaml.")
        return 2

    case_config = config["cases"]
    case_generator = CaseGeneratorAgent(case_config["docs_dir"])
    if case_config["case_file"]:
        cases = case_generator.load_cases(case_config["case_file"])
    else:
        cases = case_generator.generate_cases(max_cases=int(case_config["max_cases"]))
        case_generator.save_cases(cases, case_config["generated_case_file"])

    evaluation_config = config["evaluation"]
    diagnosis_config = config["diagnosis"]
    llm_config = diagnosis_config["llm"]

    evaluator = EvaluationAgent(client, top_k=int(evaluation_config["top_k"]))
    diagnosis = DiagnosisAgent(
        latency_threshold_ms=int(diagnosis_config["latency_threshold_ms"]),
        use_llm=bool(diagnosis_config["use_llm"]),
        llm_model=llm_config["model"],
        llm_api_key=resolve_api_key(llm_config),
        llm_base_url=llm_config["base_url"],
        llm_timeout=int(llm_config["timeout"]),
    )
    run_id = make_run_id()
    base_output_dir = Path(config["report"]["output_dir"])
    run_output_dir = base_output_dir / run_id
    reporter = ReportGenerator(run_output_dir)
    print_info(
        f"run_id={run_id}; loaded {len(cases)} cases; target={target_mode}; "
        f"diagnosis_llm={bool(diagnosis_config['use_llm'])}."
    )

    concurrency = int(evaluation_config.get("concurrency", 1))
    llm_failed_only = bool(diagnosis_config["llm_failed_only"])

    pipeline = EvalPipeline(
        evaluator=evaluator,
        diagnoser=diagnosis,
        eval_workers=concurrency,
        diagnosis_workers=max(1, concurrency // 2),
        llm_failed_only=llm_failed_only,
    )

    started_at = datetime.now()
    try:
        if RICH_AVAILABLE and console is not None:
            with Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as prog:
                eval_task = prog.add_task(f"Evaluating ×{concurrency}", total=len(cases))
                diag_task = prog.add_task("Diagnosing", total=len(cases))

                results: list[EvaluationResult] = pipeline.run_with_progress(
                    cases,
                    on_eval_done=lambda __: prog.advance(eval_task),
                    on_diagnosis_done=lambda __: prog.advance(diag_task),
                )
        else:
            results = pipeline.run(cases)

        artifacts = reporter.generate(results)
        finished_at = datetime.now()
        save_run_meta(
            run_output_dir, base_output_dir, run_id,
            started_at, finished_at, results, config, target_mode, concurrency,
        )
    except Exception as exc:
        print_error(f"Evaluation failed: {exc}")
        return 1

    print_run_summary(results, artifacts)
    return 0


def load_config(path: str | Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read config.yaml. Install dependencies with pip install -r requirements.txt.")

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist.")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Top-level config must be a YAML mapping.")
    return merge_dicts(default_config(), loaded)


def default_config() -> dict[str, Any]:
    return {
        "target": {
            "mock": True,
            "base_url": None,
            "endpoint": "/ask",
            "timeout": 60,
        },
        "cases": {
            "case_file": None,
            "docs_dir": "examples/sample_docs",
            "generated_case_file": "outputs/generated_cases.jsonl",
            "max_cases": 5,
        },
        "evaluation": {
            "top_k": 5,
            "concurrency": 1,
        },
        "diagnosis": {
            "use_llm": False,
            "llm_failed_only": True,
            "latency_threshold_ms": 5000,
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "base_url": "https://api.deepseek.com",
                "api_key": None,
                "api_key_env": "DS_API_KEY",
                "timeout": 60,
            },
        },
        "report": {
            "output_dir": "outputs",
        },
    }


def merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    snapshot = copy.deepcopy(config)
    llm = snapshot.get("diagnosis", {}).get("llm", {})
    if llm.get("api_key"):
        llm["api_key"] = "[redacted]"
    return snapshot


def save_run_meta(
    run_output_dir: Path,
    base_output_dir: Path,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    results: list[EvaluationResult],
    config: dict[str, Any],
    target_mode: str,
    concurrency: int,
) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    meta = {
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_s": round((finished_at - started_at).total_seconds(), 1),
        "target": target_mode,
        "concurrency": concurrency,
        "total_cases": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "config_snapshot": redact_config(config),
    }
    (run_output_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (base_output_dir / "latest_run_id.txt").write_text(run_id, encoding="utf-8")


def resolve_api_key(llm_config: dict[str, Any]) -> str | None:
    api_key = llm_config.get("api_key")
    if api_key:
        return str(api_key)
    api_key_env = llm_config.get("api_key_env")
    if api_key_env:
        return os.getenv(str(api_key_env))
    return None



def print_error(message: str) -> None:
    if RICH_AVAILABLE and console is not None:
        console.print(f"[bold red]{message}[/bold red]")
    else:
        print(message, file=sys.stderr)


def print_info(message: str) -> None:
    if RICH_AVAILABLE and console is not None:
        console.print(f"[cyan]{message}[/cyan]")
    else:
        print(message)


def print_run_summary(results: list[EvaluationResult], artifacts: dict[str, str | None]) -> None:
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
    values = [float(result.metrics.get(metric, 0.0)) for result in results]
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


if __name__ == "__main__":
    raise SystemExit(main())
