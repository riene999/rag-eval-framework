# Target Agent HTTP API

Any external RAG or Agent system can be evaluated by this framework if it exposes a compatible HTTP endpoint.

The framework is independent from the target implementation. It does not import code from `paper-rag-agent`, does not copy target code, and can be pointed at any service that follows this API.

## Endpoint

```http
POST /ask
```

## Request

```json
{
  "question": "string",
  "stream": false,
  "top_k": 5
}
```

Additional fields may be sent by the evaluator, such as `case_id`.

## Recommended Response

```json
{
  "answer": "string",
  "retrieved_chunks": [
    {
      "chunk_id": "string",
      "text": "string",
      "score": 0.0,
      "source": "string"
    }
  ],
  "citations": ["chunk_id"],
  "trace": {
    "tools": [],
    "rewrite_query": "",
    "steps": []
  },
  "latency_ms": 1234
}
```

## Supported Field Aliases

The evaluator normalizes common response variants:

- `answer`: `answer`, `result`, `text`, or `output`
- `retrieved_chunks`: `retrieved_chunks`, `contexts`, or `documents`
- `citations`: `citations`, `sources`, or `references`
- `trace`: `trace` or `metadata`
- `latency_ms`: `latency_ms` or `elapsed_ms`

## Notes

- `paper-rag-agent` can be used as a demo target if it exposes this API.
- Other RAG or Agent systems can be evaluated by implementing the same endpoint.
- Citations should reference retrieved chunk ids or source fields so citation consistency can be checked.
