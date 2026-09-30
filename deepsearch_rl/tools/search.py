# -*- coding: utf-8 -*-
"""
Search 工具
===========

把「搜索引擎后端 + API-key 轮换 + 失败重试 + 持久缓存」组装成 SearchTool。

重试策略（与异常分类配合）：
- 限流/AUTH/配额耗尽：先冷却当前 key，轮换到下一个 key 重试；
- 超时/网络/5xx：指数退避后重试（同一个 key）；
- 参数错误/404：不重试，直接返回错误观测。
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..protocol import SEARCH_TAG
from .base import BaseTool
from .cache import PersistentToolCache
from .exceptions import ToolError, ToolErrorType
from .key_pool import KeyRotator
from .search_backend import SearchItem, build_provider


class SearchTool(BaseTool):
    """多跳搜索工具。"""

    name = SEARCH_TAG

    def __init__(
        self,
        backend: str = "serper",
        rotator: Optional[KeyRotator] = None,
        cache: Optional[PersistentToolCache] = None,
        *,
        num_results: int = 5,
        max_retries: int = 4,
        timeout: float = 15.0,
        backoff_base: float = 1.5,
    ) -> None:
        super().__init__(cache=cache)
        self.provider = build_provider(backend, timeout=timeout)
        self.backend = backend
        self.rotator = rotator or KeyRotator()
        self.num_results = num_results
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    # ------------------------------------------------------------------
    def _call_once(self, query: str, key: Optional[str]) -> List[SearchItem]:
        return self.provider.search(query, key=key, num=self.num_results)

    def _run(self, argument: str) -> str:
        if not argument:
            raise ToolError(ToolErrorType.BAD_REQUEST, "搜索查询为空")

        last_exc: Optional[ToolError] = None
        for attempt in range(self.max_retries + 1):
            key = self.rotator.get()
            # ddg 后端不需要 key
            if self.backend != "ddg" and key is None:
                # 所有 key 都在冷却，等待后继续
                time.sleep(min(self.backoff_base ** attempt, 8.0))
                continue
            try:
                items = self._call_once(argument, key)
                return self._format(argument, items)
            except ToolError as exc:
                last_exc = exc
                # 需要轮换 key：冷却当前 key
                if exc.should_rotate_key() and key is not None:
                    self.rotator.cooldown(key)
                # 不可重试：直接抛出
                if not exc.retryable:
                    raise
                # 可重试：指数退避
                if attempt < self.max_retries:
                    time.sleep(min(self.backoff_base ** attempt, 8.0))
                continue

        # 重试耗尽
        if last_exc is not None:
            raise last_exc
        raise ToolError(ToolErrorType.UNKNOWN, "搜索重试耗尽")

    # ------------------------------------------------------------------
    @staticmethod
    def _format(query: str, items: List[SearchItem]) -> str:
        if not items:
            return (
                f"对查询「{query}」没有找到任何搜索结果。"
                "请尝试更换关键词或拆分为更具体的子问题。"
            )
        lines = [f"以下是关于「{query}」的搜索结果："]
        for i, it in enumerate(items, 1):
            lines.append(f"[{i}] {it.title}\n    URL: {it.url}\n    摘要: {it.snippet}")
        lines.append(
            "如需查看某条结果的完整内容，请使用 <open>对应URL</open>。"
        )
        return "\n".join(lines)


__all__ = ["SearchTool"]
