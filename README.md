# agentic-eval-framework

`agentic-eval-framework` 是一个独立于被测系统的 RAG/Agent 自动化评测框架。它不实现任何 RAG 功能，而是通过 HTTP API 对外部 RAG/Agent 服务进行黑盒评测：生成测试用例、调用目标系统、计算检索与生成指标、归因失败原因、输出报告。

## 架构

框架由三个 Agent 和一个报告模块组成，通过 `EvalPipeline` 以队列化流水线方式协作运行。`EvaluationAgent` 完成单条用例后立刻将结果推入消息队列，`DiagnosisAgent` 无需等待全部评测完成即可并发消费，eval 与 diagnosis 在时间上重叠执行。

```text
用例文件 / 文档目录
        │
        ▼
CaseGeneratorAgent          生成或加载 JSONL 评测用例
        │
        ▼
EvalPipeline ──────────────────────────────────────────────
│                                                          │
│  eval_queue        diagnosis_queue       done_queue      │
│  ┌─────────┐      ┌──────────────┐      ┌──────────┐    │
│  │ eval×N  │─────▶│  diag×M     │─────▶│ collect  │    │
│  └─────────┘      └──────────────┘      └──────────┘    │
│       │                                                  │
│       ▼                                                  │
│  TargetAgentClient ──▶ 外部 RAG/Agent HTTP API           │
└──────────────────────────────────────────────────────────┘
        │
        ▼
ReportGenerator             输出 Markdown / JSON / 失败分布图
```

**各模块职责：**

- `CaseGeneratorAgent`：从 JSONL 文件加载用例，或从 Markdown 文档生成模板用例
- `EvaluationAgent`：调用目标系统，计算 Recall@k、MRR、多源覆盖率、关键词召回、答案 F1、引用一致性、p95 延迟等指标
- `DiagnosisAgent`：对失败用例进行规则归因（检索/生成/引用/延迟），可选接入 LLM 生成根因分析与优化建议
- `EvalPipeline`：以生产者-消费者队列将上述 Agent 串联为并发流水线，eval worker 与 diagnosis worker 独立线程运行

## 执行

```bash
python scripts/run_eval.py
```