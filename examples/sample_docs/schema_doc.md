# Normalized Response Schema

The framework normalizes target responses into a stable schema.

The normalized schema contains `answer`, `retrieved_chunks`, `citations`, `trace`, `latency_ms`, `raw_response`, and `client_latency_ms`.

Common target field aliases are supported. For example, answer text may come from `answer`, `result`, `text`, or `output`. Retrieved chunks may come from `retrieved_chunks`, `contexts`, or `documents`.
