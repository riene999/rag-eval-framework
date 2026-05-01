from __future__ import annotations

import time
from typing import Any
from urllib.parse import urljoin

import requests

from agentic_eval.clients.base import TargetAgentClient


class TargetAgentHTTPError(RuntimeError):
    """对于HTTP错误的自定义异常类"""


class HttpTargetAgentClient(TargetAgentClient):
    """外部RAG的HTTP接口"""

    def __init__(self, base_url: str, endpoint: str = "/ask", timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.endpoint = endpoint.lstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    @property
    def url(self) -> str:
        return urljoin(self.base_url, self.endpoint)

    def ask(self, question: str, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"question": question, "stream": False}
        payload.update(kwargs)

        start = time.perf_counter()
        try:
            response = self._session.post(self.url, json=payload, timeout=self.timeout)
            client_latency_ms = int((time.perf_counter() - start) * 1000)
        except requests.RequestException as exc:
            raise TargetAgentHTTPError(f"Request failed for {self.url}: {exc}") from exc

        if not response.ok:
            raise TargetAgentHTTPError(
                f"Target request failed: status={response.status_code}, "
                f"url={self.url}, response={response.text[:1000]}"
            )

        try:
            raw_response = response.json()
        except ValueError as exc:
            raise TargetAgentHTTPError(
                f"Target response is not JSON: url={self.url}, text={response.text[:1000]}"
            ) from exc

        if not isinstance(raw_response, dict):
            raise TargetAgentHTTPError(
                f"Target response JSON must be an object: url={self.url}, value={raw_response!r}"
            )

        return normalize_target_response(raw_response, client_latency_ms)


def normalize_target_response(raw_response: dict[str, Any], client_latency_ms: int) -> dict[str, Any]:
    # 从原始响应中提取标准字段
    answer = first_present(raw_response, ["answer", "result", "text", "output"], "")
    retrieved_chunks = first_present(
        raw_response,
        ["retrieved_chunks", "contexts", "documents"],
        [],
    )
    citations = first_present(raw_response, ["citations", "sources", "references"], [])
    trace = first_present(raw_response, ["trace", "metadata"], {})
    latency_ms = first_present(raw_response, ["latency_ms", "elapsed_ms"], None)

    return {
        "answer": str(answer) if answer is not None else "",
        "retrieved_chunks": retrieved_chunks if isinstance(retrieved_chunks, list) else [],
        "citations": citations if isinstance(citations, list) else [],
        "trace": trace if isinstance(trace, dict) else {},
        "latency_ms": int(latency_ms) if isinstance(latency_ms, (int, float)) else None,
        "raw_response": raw_response,
        "client_latency_ms": client_latency_ms,
    }


def first_present(data: dict[str, Any], keys: list[str], default: Any) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default
