# -*- coding: utf-8 -*-
"""
工具工厂（tool_factory）
========================

从一个普通 dict（YAML 配置 / 环境变量）构建真实的 SearchTool / OpenTool / 共享缓存。

设计目标：
- 检索服务（retrieval_server）与 veRL wrapper 的「本地兜底」分支共用同一套构建逻辑，
  保证两边行为一致（同一缓存、同一 key 轮换、同一重试退避）。
- 不在构建阶段发起任何网络请求；只有真正执行搜索/打开网页时才联网。
- 检索服务进程内：先 build_shared_cache 得到一个缓存实例，放进 config["cache"]，
  再分别 build_search_tool / build_open_tool，即可让两个工具共享同一个 SQLite 缓存。

支持的配置键（均为可选，缺省见各函数内）：
    backend        搜索后端：serper/serpapi/bing/brave/tavily/ddg（默认 serper）
    num_results    每次搜索默认返回条数（默认 5）
    max_retries    失败重试次数（search 默认 4，open 默认 3）
    timeout        单次网络超时秒数（search 默认 15，open 默认 20）
    cache_db_path  SQLite 缓存路径（缺省用 PersistentToolCache 默认路径）
    cache_enabled  是否启用缓存（默认 True）
    key_file       API key 文件（每行一个，# 开头注释）
    cooldown       key 冷却秒数（默认 60）
    max_chars      open 网页正文截断长度（默认 6000）
    cache          外部传入的共享 PersistentToolCache（传入则不再新建）

API key 来源（按后端读取对应环境变量，逗号分隔，可多个）：
    serper   <- SERPER_API_KEYS  (+ 通用 SEARCH_API_KEYS)
    serpapi  <- SERPAPI_API_KEYS (+ 通用 SEARCH_API_KEYS)
    bing     <- BING_API_KEYS    (+ 通用 SEARCH_API_KEYS)
    brave    <- BRAVE_API_KEYS   (+ 通用 SEARCH_API_KEYS)
    tavily   <- TAVILY_API_KEYS  (+ 通用 SEARCH_API_KEYS)
    ddg      <- 无需 key
外加 key_file（每行一个 key）始终加载。
"""

from __future__ import annotations

import os
from typing import Optional

from .cache import DEFAULT_DB_PATH, PersistentToolCache
from .key_pool import KeyRotator, build_key_rotator
from .open_page import OpenTool
from .search import SearchTool

# 后端 -> 对应 API key 环境变量（通用 SEARCH_API_KEYS 作为兜底始终参与）
_BACKEND_ENV_VARS = {
    "serper": ("SERPER_API_KEYS", "SEARCH_API_KEYS"),
    "serpapi": ("SERPAPI_API_KEYS", "SEARCH_API_KEYS"),
    "bing": ("BING_API_KEYS", "SEARCH_API_KEYS"),
    "brave": ("BRAVE_API_KEYS", "SEARCH_API_KEYS"),
    "tavily": ("TAVILY_API_KEYS", "SEARCH_API_KEYS"),
    # ddg 免费，不需要 key
    "ddg": (),
}


def build_key_rotator_for_backend(
    backend: str,
    key_file: Optional[str] = None,
    cooldown_seconds: float = 60.0,
) -> KeyRotator:
    """按后端选择对应的环境变量，构造一个已加载好 key 的轮换池。

    这样可以避免把 Bing 的 key 错误地喂给 Serper。
    """
    env_names = _BACKEND_ENV_VARS.get((backend or "ddg").lower(), ())
    return build_key_rotator(
        env_names=env_names,
        key_file=key_file,
        cooldown_seconds=cooldown_seconds,
        name=(backend or "ddg").lower(),
    )


def build_shared_cache(config: dict) -> PersistentToolCache:
    """构建持久化缓存（SQLite + 进程内 LRU）。

    多 worker 部署时，各 worker 进程通过同一个 SQLite 文件（WAL 模式）共享缓存，
    因此检索服务可以安全地用 uvicorn 多 worker 启动。
    """
    db_path = config.get("cache_db_path") or os.environ.get(
        "CACHE_DB_PATH", DEFAULT_DB_PATH
    )
    return PersistentToolCache(
        db_path=db_path,
        ttl_seconds=config.get("cache_ttl_seconds", 60 * 60 * 24 * 30),
        lru_capacity=config.get("cache_lru_capacity", 2048),
        enabled=config.get("cache_enabled", True),
    )


def build_search_tool(config: dict) -> SearchTool:
    """从配置构建真实 SearchTool（含缓存 / key 轮换 / 重试）。"""
    backend = (config.get("backend") or "serper").lower()
    # 允许外部传入共享缓存；否则内部新建一个
    cache = config.get("cache") or build_shared_cache(config)
    rotator = build_key_rotator_for_backend(
        backend=backend,
        key_file=config.get("key_file") or os.environ.get("KEY_FILE"),
        cooldown_seconds=float(config.get("cooldown", 60.0)),
    )
    return SearchTool(
        backend=backend,
        rotator=rotator,
        cache=cache,
        num_results=int(config.get("num_results", 5)),
        max_retries=int(config.get("max_retries", 4)),
        timeout=float(config.get("timeout", 15.0)),
    )


def build_open_tool(config: dict) -> OpenTool:
    """从配置构建真实 OpenTool（含缓存 / 重试）。"""
    cache = config.get("cache") or build_shared_cache(config)
    return OpenTool(
        cache=cache,
        max_retries=int(config.get("open_max_retries", config.get("max_retries", 3))),
        timeout=float(config.get("open_timeout", config.get("timeout", 20.0))),
        max_chars=int(config.get("max_chars", 6000)),
    )


__all__ = [
    "build_shared_cache",
    "build_search_tool",
    "build_open_tool",
    "build_key_rotator_for_backend",
]
