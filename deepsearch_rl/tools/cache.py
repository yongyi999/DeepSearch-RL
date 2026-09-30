# -*- coding: utf-8 -*-
"""
持久化工具缓存
==============

Agentic RL 的 rollout 会在同一批题目上反复采样，相同的查询/网页会被请求成百上千次。
使用 SQLite 持久缓存（WAL 模式，支持多进程/多 worker 并发读写），叠加进程内 LRU，
可以：
- 显著降低搜索 API 调用量与费用；
- 让重复实验结果可复现；
- 缩短 rollout 等待时间。

缓存两类对象：
    search : <namespace>:<归一化查询>   -> 搜索结果 JSON
    open   : <namespace>:<归一化URL>    -> 网页正文

注意：命中缓存的调用不计为「重复工具调用」（重复调用指同一轨迹内对相同参数的反复
请求），缓存层对 agent loop 透明。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

DEFAULT_DB_PATH = os.path.expanduser("~/.cache/deepsearch_rl/tool_cache.db")
_DEFAULT_TTL = 60 * 60 * 24 * 30  # 30 天


def _normalize_query(query: str) -> str:
    """归一化搜索查询：去多余空白、转小写、去首尾标点。"""
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    return q


def _normalize_url(url: str) -> str:
    """归一化 URL：去 fragment、常见追踪参数、末尾斜杠。"""
    u = (url or "").strip()
    u = re.sub(r"#.*$", "", u)
    u = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid|spm)=[^&]*", "", u)
    u = re.sub(r"[?&]$", "", u)
    return u.rstrip("/")


def _hash_key(namespace: str, raw: str) -> str:
    h = hashlib.sha256(f"{namespace}:{raw}".encode("utf-8")).hexdigest()
    return f"{namespace}:{h[:32]}"


class _LRU:
    """进程内 LRU（线程安全），降低 SQLite 访问频率。"""

    def __init__(self, capacity: int = 2048) -> None:
        self.capacity = capacity
        self._data: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            if len(self._data) > self.capacity:
                self._data.popitem(last=False)


class PersistentToolCache:
    """SQLite 持久缓存 + 进程内 LRU。"""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        *,
        ttl_seconds: int = _DEFAULT_TTL,
        lru_capacity: int = 2048,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self._lru = _LRU(lru_capacity)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self.db_path = db_path
        if enabled:
            self._init_db()

    def _init_db(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(directory, exist_ok=True)
        # check_same_thread=False：配合 self._lock 跨线程使用
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_cache (
                cache_key  TEXT PRIMARY KEY,
                namespace  TEXT NOT NULL,
                payload    TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ns ON tool_cache(namespace);"
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # key 构造
    # ------------------------------------------------------------------
    @staticmethod
    def make_key(namespace: str, argument: str) -> str:
        if namespace == "search":
            argument = _normalize_query(argument)
        elif namespace == "open":
            argument = _normalize_url(argument)
        return _hash_key(namespace, argument)

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def get(self, namespace: str, argument: str) -> Optional[Any]:
        if not self.enabled:
            return None
        key = self.make_key(namespace, argument)
        hit = self._lru.get(key)
        if hit is not None:
            return hit
        with self._lock:
            assert self._conn is not None
            cur = self._conn.execute(
                "SELECT payload, created_at FROM tool_cache WHERE cache_key=?",
                (key,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        payload, created_at = row
        if self.ttl_seconds > 0 and (time.time() - created_at) > self.ttl_seconds:
            return None
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return None
        self._lru.set(key, value)
        return value

    def set(self, namespace: str, argument: str, value: Any) -> None:
        if not self.enabled:
            return
        key = self.make_key(namespace, argument)
        try:
            payload = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return
        self._lru.set(key, value)
        with self._lock:
            assert self._conn is not None
            self._conn.execute(
                """
                INSERT OR REPLACE INTO tool_cache(cache_key, namespace, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, namespace, payload, time.time()),
            )
            self._conn.commit()

    def stats(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        with self._lock:
            assert self._conn is not None
            n = self._conn.execute("SELECT COUNT(*) FROM tool_cache").fetchone()[0]
        return {"enabled": True, "entries": n, "db_path": self.db_path}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


__all__ = [
    "PersistentToolCache",
    "DEFAULT_DB_PATH",
]
