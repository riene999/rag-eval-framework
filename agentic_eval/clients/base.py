from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TargetAgentClient(ABC):
    """抽象基类，用于定义目标RAG系统的接口"""

    @abstractmethod
    def ask(self, question: str, **kwargs: Any) -> dict[str, Any]:
        """抽象方法，用于向目标系统提问并返回标准化的响应"""
