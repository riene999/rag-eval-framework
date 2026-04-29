# Target Agent API

The evaluated RAG or Agent system should expose an HTTP endpoint named `POST /ask`.

The request body contains a `question` string, a `stream` boolean, and optional settings such as `top_k`.

The recommended response includes:

- `answer`: generated answer text.
- `retrieved_chunks`: evidence chunks returned by retrieval.
- `citations`: identifiers or sources used by the answer.
- `trace`: optional Agent execution metadata, including `tools`, `rewrite_query`, and `steps`.
- `latency_ms`: server-side latency in milliseconds.

Citations should reference retrieved chunk identifiers or source fields so evaluators can verify grounding.
