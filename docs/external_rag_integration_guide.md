# 外部 RAG/Agent 项目接入使用手册

本文档面向被评测的外部 RAG/Agent 项目，例如 `paper-rag-agent`。`agentic-eval-framework` 不会 import 外部项目代码，也不会复制外部项目代码，只会通过 HTTP API 调用外部服务。

## 1. 接入目标

外部 RAG/Agent 项目只需要启动一个 HTTP 服务，并提供统一问答接口：

```text
POST /ask
```

推荐本地运行地址：

```text
http://localhost:8010
```

评测框架会请求：

```text
POST http://localhost:8010/ask
```

## 2. 请求格式

评测框架会发送 JSON 请求：

```json
{
  "question": "用户问题",
  "stream": false,
  "top_k": 5,
  "case_id": "case_single_hop_api"
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `question` | string | 是 | 当前评测 case 的问题 |
| `stream` | boolean | 是 | 当前框架默认使用非流式响应，值为 `false` |
| `top_k` | integer | 否 | 希望目标系统召回的文档数量 |
| `case_id` | string | 否 | 当前评测 case ID，便于日志追踪 |

外部项目可以忽略不支持的可选字段，但必须能处理 `question`。

## 3. 推荐响应格式

外部项目推荐返回：

```json
{
  "answer": "目标系统生成的回答",
  "retrieved_chunks": [
    {
      "chunk_id": "doc_001_chunk_003",
      "text": "被召回的证据文本",
      "score": 0.92,
      "source": "docs/api.md"
    }
  ],
  "citations": ["doc_001_chunk_003"],
  "trace": {
    "tools": [],
    "rewrite_query": "改写后的检索 query",
    "steps": []
  },
  "latency_ms": 1234
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `answer` | string | 是 | 最终回答 |
| `retrieved_chunks` | list | 强烈建议 | 检索召回的证据块，用于计算 Recall@k 和 MRR |
| `citations` | list | 建议 | 回答引用的 chunk id 或 source，用于检查引用一致性 |
| `trace` | object | 建议 | Agent 工具调用、query rewrite、中间步骤等诊断信息 |
| `latency_ms` | integer/null | 建议 | 外部服务自身耗时，单位毫秒 |

## 4. 兼容字段名

如果外部项目已有自己的返回格式，不一定必须完全改成推荐格式。评测框架会自动兼容一些常见字段名：

| 标准字段 | 可兼容字段 |
|---|---|
| `answer` | `answer` / `result` / `text` / `output` |
| `retrieved_chunks` | `retrieved_chunks` / `contexts` / `documents` |
| `citations` | `citations` / `sources` / `references` |
| `trace` | `trace` / `metadata` |
| `latency_ms` | `latency_ms` / `elapsed_ms` |

例如下面这种响应也可以被识别：

```json
{
  "result": "这是回答",
  "contexts": [
    {
      "id": "chunk_1",
      "content": "证据文本",
      "source": "paper.md"
    }
  ],
  "sources": ["chunk_1"],
  "metadata": {
    "rewrite_query": "..."
  },
  "elapsed_ms": 830
}
```

## 5. FastAPI 示例

外部项目可以用任何 Web 框架。下面是一个最小 FastAPI 示例：

```python
from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI()


class AskRequest(BaseModel):
    question: str
    stream: bool = False
    top_k: int = 5
    case_id: str | None = None


@app.post("/ask")
def ask(req: AskRequest):
    start = time.perf_counter()

    # 这里替换成你的真实 RAG/Agent 调用逻辑：
    # 1. query rewrite
    # 2. retrieval
    # 3. rerank
    # 4. generation
    # 5. citation extraction
    answer = f"Answer for: {req.question}"
    retrieved_chunks = [
        {
            "chunk_id": "demo_chunk_1",
            "text": "This is a retrieved evidence chunk.",
            "score": 0.9,
            "source": "demo.md",
        }
    ]

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "citations": ["demo_chunk_1"],
        "trace": {
            "tools": [],
            "rewrite_query": "",
            "steps": [],
            "case_id": req.case_id,
        },
        "latency_ms": latency_ms,
    }
```

启动示例：

```bash
uvicorn app:app --host 0.0.0.0 --port 8010
```

## 6. Flask 示例

如果外部项目使用 Flask，可以参考：

```python
from flask import Flask, jsonify, request
import time

app = Flask(__name__)


@app.post("/ask")
def ask():
    start = time.perf_counter()
    data = request.get_json(force=True)
    question = data["question"]

    answer = f"Answer for: {question}"
    retrieved_chunks = [
        {
            "chunk_id": "demo_chunk_1",
            "text": "This is a retrieved evidence chunk.",
            "score": 0.9,
            "source": "demo.md",
        }
    ]

    return jsonify(
        {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "citations": ["demo_chunk_1"],
            "trace": {
                "tools": [],
                "rewrite_query": "",
                "steps": [],
            },
            "latency_ms": int((time.perf_counter() - start) * 1000),
        }
    )
```

启动示例：

```bash
flask --app app run --host 0.0.0.0 --port 8010
```

## 7. 接入后如何自测

先确认外部 RAG/Agent 服务已经启动在 `8010`：

```bash
curl -X POST http://localhost:8010/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"测试问题\",\"stream\":false,\"top_k\":5}"
```

PowerShell 也可以使用：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:8010/ask `
  -ContentType "application/json" `
  -Body '{"question":"测试问题","stream":false,"top_k":5}'
```

然后在 `agentic-eval-framework` 项目中测试 client：

```bash
python scripts/test_target_client.py --base-url http://localhost:8010 --question "测试问题"
```

如果返回了标准 JSON，说明 HTTP 接入成功。

## 8. 运行正式评测

在 `agentic-eval-framework` 项目目录中运行：

```bash
python scripts/run_eval.py --base-url http://localhost:8010 --case-file examples/sample_cases.jsonl
```

如果外部项目接口不是 `/ask`，例如 `/api/ask`，使用：

```bash
python scripts/run_eval.py --base-url http://localhost:8010 --endpoint /api/ask --case-file examples/sample_cases.jsonl
```

评测完成后查看：

```text
outputs/eval_report.md
outputs/eval_results.json
```

## 9. 如何让评测结果更准

为了让归因分析更有价值，外部项目最好返回完整的检索和生成信息：

- `retrieved_chunks[].chunk_id`：稳定、唯一的 chunk ID。
- `retrieved_chunks[].text`：真实召回文本。
- `retrieved_chunks[].source`：文档来源，例如文件名、URL、论文 ID。
- `citations`：回答实际引用的 chunk ID 或 source。
- `trace.rewrite_query`：如果做了 query rewrite，记录改写后的 query。
- `trace.tools`：如果 Agent 调用了工具，记录工具名和结果摘要。
- `trace.steps`：关键中间步骤，便于失败归因。

如果缺少 `retrieved_chunks`，框架仍然能调用目标系统，但检索指标会偏低或无法准确判断。

## 10. 常见问题

### 评测框架报 HTTP 状态码错误

检查外部服务是否启动、端口是否是 `8010`、接口路径是否是 `/ask`。

### 返回不是 JSON

`/ask` 必须返回 JSON。HTML 错误页、纯文本、流式 SSE 响应都会导致解析失败。

### Recall@k 很低

通常说明 `retrieved_chunks` 中没有包含评测 case 的 `gold_evidence`。检查 chunk ID、source、text 是否返回完整。

### citation_consistency 为 false

说明 `citations` 中的引用没有出现在 `retrieved_chunks` 的 `chunk_id`、`id`、`source`、`title`、`text` 或 `content` 中。建议 citations 使用稳定 chunk ID。

### keyword_recall 很低

说明回答没有覆盖 case 中的期望关键词。可能是生成 prompt 没有充分利用上下文，或回答过于简略。

## 11. 推荐接入 checklist

- 外部服务监听 `http://localhost:8010`
- 实现 `POST /ask`
- 请求支持 `question`
- 响应返回 JSON
- 响应包含 `answer`
- 响应尽量包含 `retrieved_chunks`
- 响应尽量包含 `citations`
- 响应尽量包含 `trace`
- 响应尽量包含 `latency_ms`
- 使用 `scripts/test_target_client.py` 自测通过
- 使用 `scripts/run_eval.py` 生成报告
