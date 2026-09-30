# -*- coding: utf-8 -*-
"""
检索服务（独立 FastAPI 进程）
=============================

对外暴露对齐 veRL 官方 search_tool_example 的协议（见 ENGINEERING_SPEC 4.3）：

    POST /retrieve   入 {"queries":[...], "topk":5, "return_scores":true}
                     出 {"result": [[{"document":{"title","url","contents"},"score"}]]}
    POST /open       入 {"url":"..."}
                     出 {"url":..., "contents":..., "ok":true}
    GET  /health     出 {"status":"ok","cache":{...},"keys_available":n}

设计要点：
- 内部用 tool_factory 构建真实 SearchTool / OpenTool（其 provider 已返回结构化
  SearchItem 列表）。为了把「给模型的文本」转成 /retrieve 需要的 documents 结构，
  这里直接复用 search_tool.provider + rotator + 与 search.py 完全一致的重试/退避逻辑，
  但拿到的是 List[SearchItem]，再转成 document；结构化结果单独缓存到 namespace
  "search_items"（与 SearchTool 给模型看的文本缓存 namespace "search" 隔离，避免格式冲突）。
- 缓存是 SQLite（WAL），天然支持 uvicorn 多 worker 共享。
- 并发由 asyncio.Semaphore 限制每实例在飞请求数（默认 120，环境变量 RETRIEVAL_CONCURRENCY）。
- 无任何 key 时自动从 serper 等付费后端回退到免费的 ddg，保证服务可启动。

启动：uvicorn deepsearch_rl.retrieval.retrieval_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from ..tools.exceptions import ToolError, ToolErrorType
from ..tools.tool_factory import (
    build_key_rotator_for_backend,
    build_open_tool,
    build_search_tool,
    build_shared_cache,
)

logger = logging.getLogger("deepsearch_rl.retrieval")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------
class RetrieveRequest(BaseModel):
    """POST /retrieve 入参。"""

    queries: List[str]
    topk: int = 5
    return_scores: bool = True


class OpenRequest(BaseModel):
    """POST /open 入参。"""

    url: str


# ---------------------------------------------------------------------------
# 运行时上下文（lifespan 构建，关闭时释放缓存）
# ---------------------------------------------------------------------------
class _Context:
    """持有本进程内共享的缓存、搜索/打开工具与并发信号量。"""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        # 先建一个共享缓存，并塞进 config，让 search/open 工具复用同一个 SQLite
        self.cache = build_shared_cache(cfg)
        cfg["cache"] = self.cache
        self.search_tool = build_search_tool(cfg)
        self.open_tool = build_open_tool(cfg)
        self.backend = self.search_tool.backend
        self.sem = asyncio.Semaphore(int(cfg.get("concurrency", 120)))


def _load_config_from_env() -> dict:
    """从环境变量读取服务配置（缺省值见下）。"""
    return {
        # 搜索后端：serper/serpapi/bing/brave/tavily/ddg；无 key 时自动回退 ddg
        "backend": os.environ.get("SEARCH_BACKEND", "serper"),
        "num_results": int(os.environ.get("RETRIEVE_TOPK", "5")),
        "max_retries": int(os.environ.get("SEARCH_MAX_RETRIES", "4")),
        "timeout": float(os.environ.get("SEARCH_TIMEOUT", "15")),
        "cache_db_path": os.environ.get("CACHE_DB_PATH") or None,
        "key_file": os.environ.get("KEY_FILE") or None,
        "cooldown": float(os.environ.get("KEY_COOLDOWN", "60")),
        # 每实例并发上限
        "concurrency": int(os.environ.get("RETRIEVAL_CONCURRENCY", "120")),
    }


def _resolve_backend(cfg: dict) -> str:
    """若选定的付费后端没有任何可用 key，则回退到免费的 ddg。"""
    backend = (cfg.get("backend") or "serper").lower()
    if backend == "ddg":
        return backend
    rotator = build_key_rotator_for_backend(backend, cfg.get("key_file"),
                                            float(cfg.get("cooldown", 60.0)))
    if len(rotator) == 0:
        logger.warning(
            "后端 %s 未配置任何 API key（环境变量/key_file），自动回退到免费 ddg。",
            backend,
        )
        return "ddg"
    return backend


def _build_context(cfg: dict) -> _Context:
    cfg["backend"] = _resolve_backend(cfg)
    ctx = _Context(cfg)
    logger.info(
        "检索服务就绪：backend=%s, keys_available=%d, concurrency=%d, cache=%s",
        ctx.backend,
        ctx.search_tool.rotator.available_count(),
        ctx.cfg.get("concurrency"),
        ctx.cache.stats(),
    )
    return ctx


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时构建工具与缓存，关闭时释放 SQLite 连接。"""
    app.state.ctx = _build_context(_load_config_from_env())
    try:
        yield
    finally:
        app.state.ctx.cache.close()


app = FastAPI(title="DeepSearch-RL Retrieval Service", lifespan=lifespan)


def _get_ctx() -> _Context:
    """获取运行时上下文；若未经过 lifespan（如裸 TestClient 测试）则惰性构建。"""
    ctx = getattr(app.state, "ctx", None)
    if ctx is None:
        ctx = _build_context(_load_config_from_env())
        app.state.ctx = ctx
    return ctx


# ---------------------------------------------------------------------------
# 结构化搜索：复用 SearchTool 的 provider/rotator/重试，但返回 List[SearchItem]
# ---------------------------------------------------------------------------
def _search_items_with_retry(search_tool, query: str, topk: int):
    """与 search.SearchTool._run 完全一致的 key 轮换 + 指数退避策略，

    区别仅在于不格式化为给模型的文本，而是直接返回 provider 给出的 List[SearchItem]。
    （provider.search 是同步 urllib 调用，本函数也同步，由调用方放进线程池。）
    """
    last_exc: Optional[ToolError] = None
    for attempt in range(search_tool.max_retries + 1):
        key = search_tool.rotator.get()
        # ddg 不需要 key；其它后端在所有 key 都冷却时退避等待
        if search_tool.backend != "ddg" and key is None:
            time.sleep(min(search_tool.backoff_base ** attempt, 8.0))
            continue
        try:
            return search_tool.provider.search(query, key=key, num=topk)
        except ToolError as exc:
            last_exc = exc
            # 限流/鉴权/配额：冷却当前 key，轮换下一个
            if exc.should_rotate_key() and key is not None:
                search_tool.rotator.cooldown(key)
            # 不可重试错误：直接抛出
            if not exc.retryable:
                raise
            # 可重试：指数退避
            if attempt < search_tool.max_retries:
                time.sleep(min(search_tool.backoff_base ** attempt, 8.0))
            continue
    if last_exc is not None:
        raise last_exc
    raise ToolError(ToolErrorType.UNKNOWN, "搜索重试耗尽")


def _retrieve_one_sync(ctx: _Context, query: str, topk: int, return_scores: bool) -> list:
    """单个 query 的结构化检索（同步）：先查结构化缓存，未命中再联网。"""
    normalized = re.sub(r"\s+", " ", (query or "").strip().lower())
    if not normalized:
        return []
    # 缓存 key 带上 topk，避免不同条数互相污染；namespace 与给模型的文本缓存隔离
    cache_arg = f"{topk}::{normalized}"
    cached = ctx.cache.get("search_items", cache_arg)
    if cached is not None:
        items = cached  # 已是 list[dict]
    else:
        try:
            search_items = _search_items_with_retry(ctx.search_tool, query, topk)
            items = [it.to_dict() for it in search_items]
            ctx.cache.set("search_items", cache_arg, items)
        except ToolError as exc:
            logger.warning("检索失败 query=%r error_type=%s", query, exc.error_type.value)
            # 单个 query 失败不拖垮整批，返回空结果
            return []

    docs = []
    n = len(items)
    for idx, it in enumerate(items):
        doc = {
            "title": it.get("title", ""),
            "url": it.get("url", ""),
            # 搜索结果摘要即文档 contents；正文请用 /open 拉取
            "contents": it.get("snippet", ""),
        }
        entry = {"document": doc}
        if return_scores:
            # 上游搜索引擎不返回相关度分数，这里用排名构造一个单调递减的伪分数
            entry["score"] = round((n - idx) / max(n, 1), 4)
        docs.append(entry)
    return docs


async def _retrieve_one(ctx: _Context, query: str, topk: int, return_scores: bool) -> list:
    """异步包装：用信号量限流，把同步检索丢进线程池。"""
    async with ctx.sem:
        return await asyncio.to_thread(_retrieve_one_sync, ctx, query, topk, return_scores)


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    """批量检索：对每个 query 返回一组 document。"""
    ctx = _get_ctx()
    if not req.queries:
        return {"result": []}
    results = await asyncio.gather(
        *[_retrieve_one(ctx, q, req.topk, req.return_scores) for q in req.queries]
    )
    return {"result": list(results)}


def _unwrap_open(url: str, raw: str) -> str:
    """去掉 OpenTool 拼在正文前的引导行「网页 <url> 的正文：」，保留纯正文。"""
    prefix = f"网页 {url} 的正文："
    if raw.startswith(prefix):
        return raw[len(prefix):].lstrip("\n")
    return raw


@app.post("/open")
async def open_page(req: OpenRequest):
    """打开一个 URL，返回抽取到的正文。"""
    ctx = _get_ctx()
    async with ctx.sem:
        result = await asyncio.to_thread(ctx.open_tool.execute, req.url)
    if result.ok:
        return {"url": req.url, "contents": _unwrap_open(req.url, result.content), "ok": True}
    # 失败：返回明确的错误信息，但不抛异常（HTTP 层 200，由 ok 字段表达成败）
    return {
        "url": req.url,
        "contents": f"[工具失败] {result.error_type}: {result.meta.get('message', '')}",
        "ok": False,
    }


@app.get("/health")
async def health():
    """健康检查：返回缓存统计与当前可用 key 数。"""
    ctx = _get_ctx()
    return {
        "status": "ok",
        "backend": ctx.backend,
        "cache": ctx.cache.stats(),
        "keys_available": ctx.search_tool.rotator.available_count(),
    }


__all__ = ["app"]
