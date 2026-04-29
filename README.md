# agentic-eval-framework

`agentic-eval-framework` is a standalone evaluation framework for external RAG and Agent systems. It does not implement a RAG system itself. Instead, it generates evaluation cases, calls a target system through a unified HTTP API, computes retrieval and generation metrics, diagnoses failures, and produces reports.

## Architecture

```text
Sample Docs / Case File
        |
        v
CaseGeneratorAgent
        |
        v
EvaluationAgent ---> TargetAgentClient ---> External RAG/Agent HTTP API
        |
        v
DiagnosisAgent
        |
        v
ReportGenerator ---> Markdown + JSON + optional chart
```

The target system can be `paper-rag-agent` or any other RAG/Agent service, as long as it exposes the expected HTTP API. This project never imports or copies code from the target project.

## Agent Responsibilities

- `CaseGeneratorAgent`: loads existing JSONL cases or generates rule-based test cases from Markdown documents.
- `EvaluationAgent`: calls the target RAG/Agent system and computes retrieval, generation, and task-level metrics.
- `DiagnosisAgent`: analyzes failed cases and returns root cause, failed stage, priority, and optimization suggestions.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run With Mock Target

The mock client lets you run the full demo without any external RAG service:

```bash
python scripts/test_target_client.py --mock --question "What does the API return?"
python scripts/run_eval.py --mock
```

You can also use the prepared cases:

```bash
python scripts/run_eval.py --mock --case-file examples/sample_cases.jsonl
```

## Connect an External RAG Service

Start your target service, then run:

```bash
python scripts/test_target_client.py --base-url http://localhost:8000 --question "How should errors be returned?"
python scripts/run_eval.py --base-url http://localhost:8000 --case-file examples/sample_cases.jsonl --output-dir outputs
```

By default, the framework sends `POST /ask`. See [docs/target_agent_api.md](docs/target_agent_api.md) for the full contract.

For a Chinese step-by-step integration manual for the external RAG/Agent project, see [docs/external_rag_integration_guide.md](docs/external_rag_integration_guide.md).

## Example Output

After an evaluation run, check:

- `outputs/eval_report.md`: Markdown summary with pass rate, metrics, bad case table, and suggestions.
- `outputs/eval_results.json`: machine-readable results.
- `outputs/failure_distribution.png`: failure distribution chart, generated when `matplotlib` is installed and failures exist.

Console summary example:

```text
Total cases: 5
Passed: 3
Pass rate: 60.00%
Markdown report: outputs/eval_report.md
JSON report: outputs/eval_results.json
```

## Highlights

- Multi-agent evaluation pipeline with separated case generation, execution, diagnosis, and reporting.
- Decoupled target client interface, supporting both HTTP services and local mock runs.
- Retrieval quality metrics: Recall@k, MRR, retrieved count.
- Generation quality metrics: keyword recall, answer length, citation consistency, latency.
- Rule-based diagnosis for retrieval, generation, citation, and latency failures.
- Lightweight dependencies and plain Python modules.
- Rich-enhanced CLI with progress display, summary tables, failed-case tables, and pretty target responses.

## Extension Ideas

- Add LLM-based case generation and diagnosis.
- Add faithfulness, answer relevance, and context precision metrics.
- Support batch HTTP APIs and async target clients.
- Add dataset adapters for public RAG benchmarks.
- Add richer report artifacts such as HTML dashboards or trend comparisons.
