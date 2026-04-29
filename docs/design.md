# Design

`agentic-eval-framework` is organized as a decoupled multi-agent evaluation pipeline.

## Goals

- Evaluate external RAG and Agent systems without depending on their internal code.
- Cover retrieval quality, generation quality, task pass/fail, and failure diagnosis.
- Provide a local mock mode so the full workflow can run without a target service.
- Keep the implementation lightweight and easy to extend.

## Components

## CaseGeneratorAgent

Generates or loads `EvaluationCase` objects. The current implementation supports JSONL loading and rule-based case generation from Markdown sample docs. Future versions can add LLM-based generation or benchmark adapters.

## EvaluationAgent

Calls a `TargetAgentClient`, normalizes the returned response, computes metrics, and creates `EvaluationResult` objects.

## DiagnosisAgent

Uses explicit rules to classify bad cases into retrieval, generation, citation, latency, or unknown failures.

## TargetAgentClient

The target client boundary makes the framework independent from any specific RAG system.

- `HttpTargetAgentClient` calls external systems via `POST /ask`.
- `MockTargetAgentClient` returns deterministic demo responses.

## Metrics

Retrieval metrics include Recall@k, MRR, retrieved count, Precision@k, matched expected source count, source coverage ratio, all expected sources hit, and average relevant chunks in Top-K. The source coverage metrics are especially useful for multi-hop cases that require evidence from more than one paper or document.

Generation metrics include keyword recall, token-level answer precision/recall/F1 against the reference answer, answer length, citation count, citation consistency, target latency, and client latency.

Task pass/fail currently uses a simple rule: keyword recall must be at least 0.5, Recall@k must be greater than zero, and citation consistency must pass.

## Reports

`ReportGenerator` writes Markdown and JSON reports. If `matplotlib` is installed and failures exist, it also writes `failure_distribution.png`.
