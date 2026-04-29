from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_text(path: str | Path) -> str:
    """Read UTF-8 text from disk."""
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text to disk, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into dictionaries."""
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries as JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Any) -> None:
    """Write pretty JSON to disk."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
