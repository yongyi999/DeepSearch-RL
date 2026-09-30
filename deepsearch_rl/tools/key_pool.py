# -*- coding: utf-8 -*-
"""
API Key 轮换池
==============

搜索服务（SerpAPI / Bing / Serper / Brave / Tavily 等）通常需要 API key，且有 QPS /
配额限制。veRL 会用大量 worker 并发 rollout，单个 key 很快触发限流。本模块提供：

- 从环境变量 / 文件加载多个 key（逗号分隔，或每行一个）；
- 轮询（round-robin）取用；
- 对限流/失效 key 做「冷却」（cooldown），到期自动恢复；
- 线程安全（每个 worker 进程内）。

支持的环境变量（按工具后端读取）：
    SEARCH_API_KEYS   通用搜索 key（逗号分隔，可多个）
    SERPAPI_API_KEYS
    BING_API_KEYS
    SERPER_API_KEYS
    BRAVE_API_KEYS
    TAVILY_API_KEYS
也可用 key_file（每行一个 key）。
"""

from __future__ import annotations

import os
import threading
import time
from typing import List, Optional


class KeyRotator:
    """带冷却机制的 key 轮询池。"""

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        *,
        cooldown_seconds: float = 60.0,
        name: str = "search",
    ) -> None:
        self.name = name
        self.cooldown_seconds = cooldown_seconds
        # key -> 冷却到期的时间戳（0 表示可用）
        self._cooldown_until: dict = {}
        self._keys: List[str] = []
        self._idx = 0
        self._lock = threading.Lock()
        for k in keys or []:
            self.add(k)

    # ------------------------------------------------------------------
    def add(self, key: str) -> None:
        key = (key or "").strip()
        if key and key not in self._keys:
            self._keys.append(key)
            self._cooldown_until[key] = 0.0

    def load_from_env(self, *env_names: str) -> int:
        """从若干环境变量加载（逗号分隔），返回新增 key 数。"""
        before = len(self._keys)
        for env_name in env_names:
            raw = os.environ.get(env_name, "")
            for part in raw.split(","):
                self.add(part)
        return len(self._keys) - before

    def load_from_file(self, path: str) -> int:
        """从文件加载（每行一个 key，# 开头为注释）。"""
        before = len(self._keys)
        if not path or not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    self.add(line)
        return len(self._keys) - before

    # ------------------------------------------------------------------
    def get(self) -> Optional[str]:
        """取一个当前可用的 key；全部冷却时返回 None（调用方应退避等待）。"""
        with self._lock:
            if not self._keys:
                return None
            now = time.time()
            n = len(self._keys)
            for _ in range(n):
                key = self._keys[self._idx % n]
                self._idx += 1
                if self._cooldown_until.get(key, 0) <= now:
                    return key
            return None

    def cooldown(self, key: str, seconds: Optional[float] = None) -> None:
        """把某个 key 冷却一段时间（遇到限流/失效时调用）。"""
        if key is None:
            return
        with self._lock:
            self._cooldown_until[key] = time.time() + (
                seconds if seconds is not None else self.cooldown_seconds
            )

    def available_count(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for k in self._keys if self._cooldown_until.get(k, 0) <= now)

    def __len__(self) -> int:
        return len(self._keys)


def build_key_rotator(
    env_names: tuple = ("SEARCH_API_KEYS",),
    key_file: Optional[str] = None,
    cooldown_seconds: float = 60.0,
    name: str = "search",
) -> KeyRotator:
    """构造一个已从环境变量/文件加载好的轮换池。"""
    rotator = KeyRotator(cooldown_seconds=cooldown_seconds, name=name)
    rotator.load_from_env(*env_names)
    if key_file:
        rotator.load_from_file(key_file)
    return rotator


__all__ = ["KeyRotator", "build_key_rotator"]
