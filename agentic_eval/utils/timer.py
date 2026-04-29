from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def elapsed_ms() -> Iterator[dict[str, int]]:
    """Measure elapsed milliseconds into a mutable dictionary."""
    holder = {"elapsed_ms": 0}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["elapsed_ms"] = int((time.perf_counter() - start) * 1000)
