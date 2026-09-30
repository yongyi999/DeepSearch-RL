# -*- coding: utf-8 -*-
"""工具基类与统一返回结构。"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .cache import PersistentToolCache
from .exceptions import ToolError


@dataclass
class ToolResult:
    """单次工具执行的统一结果。"""

    name: str
    argument: str
    content: str = ""               # 拼进 <observation> 的文本
    ok: bool = True
    error_type: Optional[str] = None
    from_cache: bool = False
    elapsed: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "argument": self.argument,
            "ok": self.ok,
            "error_type": self.error_type,
            "from_cache": self.from_cache,
            "elapsed": round(self.elapsed, 3),
        }


class BaseTool(abc.ABC):
    """所有工具的基类：缓存 + 计时 + 异常包装由基类统一处理。"""

    name: str = "base"

    def __init__(self, cache: Optional[PersistentToolCache] = None) -> None:
        self.cache = cache

    @abc.abstractmethod
    def _run(self, argument: str) -> str:
        """子类实现：真正调用外部服务，返回拼进 observation 的文本；失败抛 ToolError。"""

    def execute(self, argument: str) -> ToolResult:
        """执行一次工具调用，带缓存与统一异常包装。"""
        argument = (argument or "").strip()
        start = time.time()
        # 1. 命中缓存
        if self.cache is not None:
            cached = self.cache.get(self.name, argument)
            if cached is not None:
                return ToolResult(
                    name=self.name,
                    argument=argument,
                    content=str(cached),
                    ok=True,
                    from_cache=True,
                    elapsed=time.time() - start,
                )
        # 2. 真正执行
        try:
            content = self._run(argument)
        except ToolError as exc:
            return ToolResult(
                name=self.name,
                argument=argument,
                content="",
                ok=False,
                error_type=exc.error_type.value,
                elapsed=time.time() - start,
                meta=exc.to_dict(),
            )
        # 3. 写缓存
        if self.cache is not None and content:
            self.cache.set(self.name, argument, content)
        return ToolResult(
            name=self.name,
            argument=argument,
            content=content,
            ok=True,
            elapsed=time.time() - start,
        )

    async def aexecute(self, argument: str) -> ToolResult:
        """异步执行（默认在事件循环里把同步 execute 交给线程，避免阻塞事件循环）。"""
        import asyncio

        return await asyncio.to_thread(self.execute, argument)


__all__ = ["BaseTool", "ToolResult"]
