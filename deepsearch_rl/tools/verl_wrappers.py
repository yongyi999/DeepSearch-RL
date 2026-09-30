# -*- coding: utf-8 -*-
"""
veRL 工具封装（verl_wrappers）
==============================

把真实的搜索/打开网页能力包装成 veRL 多轮训练可挂载的工具（继承
``verl.tools.base_tool.BaseTool``）。

两种执行模式（由 config["retrieval_service_url"] 决定）：
1. HTTP 模式（默认，对齐官方 search_tool_example）：用 aiohttp 调用独立检索服务
   /retrieve 或 /open；
2. 本地兜底模式：未配置 retrieval_service_url 时，直接用 tool_factory 构建真实
   SearchTool / OpenTool，在线程池里同步执行（便于单机调试、省去起服务）。

设计约束：
- 本文件顶部对 verl 采用 try/except 导入：本地没装 verl/torch 时给出清晰占位基类，
  保证 ``py_compile`` 与单测可跑；真正训练环境里再用 verl 的真实实现。
- ``__init__`` 不发起任何网络请求；网络调用只发生在 execute 里。
- 任何失败都不抛异常中断 rollout，而是返回 ``ToolResponse(text="[工具失败] ...")``
  并在 metrics 里带上 error_type；step_reward 恒为 0.0（奖励统一由 reward 函数计算）。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# verl 依赖的容错导入：本地没装 verl 时用占位类顶替，保证可导入 / 可单测
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 训练环境才有 verl
    from verl.tools.base_tool import BaseTool as _VerlBaseTool
except Exception:  # noqa: BLE001 - 本地占位
    class _VerlBaseTool:  # type: ignore[no-redef]
        """verl.tools.base_tool.BaseTool 的本地占位（仅用于无 verl 环境）。"""

        async def create(self, instance_id: Optional[str] = None, **kwargs: Any) -> None:
            return None

        async def release(self, instance_id: Optional[str] = None, **kwargs: Any) -> None:
            return None


try:  # pragma: no cover
    from verl.tools.schemas import ToolResponse as _VerlToolResponse
except Exception:  # noqa: BLE001
    class _VerlToolResponse:  # type: ignore[no-redef]
        """verl ToolResponse 的本地占位：text/image/video 三个字段。"""

        def __init__(
            self,
            text: Optional[str] = None,
            image: Optional[list] = None,
            video: Optional[list] = None,
        ) -> None:
            self.text = text
            self.image = image
            self.video = video


def _make_response(text: Optional[str]) -> _VerlToolResponse:
    """统一构造 ToolResponse（兼容 verl pydantic 实现与本地占位）。"""
    return _VerlToolResponse(text=text)


def _normalize_endpoint(base_url: str, path: str) -> str:
    """把配置里的 base_url 规范成完整 endpoint。

    允许 config 写成 ``http://host:port`` 或 ``http://host:port/retrieve``；
    这里统一补/剥路径，拼出正确的 /retrieve 或 /open。
    """
    base = (base_url or "").rstrip("/")
    if base.endswith(path):
        return base
    for p in ("/retrieve", "/open"):
        if base.endswith(p):
            base = base[: -len(p)]
            break
    return base + path


def _format_docs_to_text(query: str, docs: list) -> str:
    """把 /retrieve 返回的 documents 渲染成给模型看的文本（与 SearchTool._format 对齐）。"""
    if not docs:
        return (
            f"对查询「{query}」没有找到任何搜索结果。"
            "请尝试更换关键词或拆分为更具体的子问题。"
        )
    lines = [f"以下是关于「{query}」的搜索结果："]
    for i, entry in enumerate(docs, 1):
        d = entry.get("document", {}) if isinstance(entry, dict) else {}
        lines.append(
            f"[{i}] {d.get('title', '')}\n"
            f"    URL: {d.get('url', '')}\n"
            f"    摘要: {d.get('contents', '')}"
        )
    lines.append("如需查看某条结果的完整内容，请使用 <open>对应URL</open>。")
    return "\n".join(lines)


class _BaseWrapper(_VerlBaseTool):
    """两个 wrapper 的公共逻辑：配置保存、HTTP/本地分支、失败兜底。"""

    # 子类覆盖：期望从 parameters 里取的参数名（query / url）
    param_key: str = ""

    def __init__(self, config: Optional[dict] = None, tool_schema: Any = None) -> None:
        # verl 不同小版本基类构造参数略有差异：优先带参，失败回退无参
        try:
            super().__init__(config=config, tool_schema=tool_schema)
        except TypeError:
            try:
                super().__init__()
            except TypeError:
                pass
        self.config: dict = config or {}
        self.tool_schema = tool_schema
        self.base_url: Optional[str] = self.config.get("retrieval_service_url")
        self.timeout: float = float(self.config.get("timeout", 30.0))
        # 本地模式下惰性构建的真实工具（__init__ 不联网）
        self._local_tool: Any = None

    # -- 子类实现 ---------------------------------------------------------
    def _build_local_tool(self):
        raise NotImplementedError

    async def _call_http(self, value: str) -> str:
        """HTTP 模式：调用检索服务，返回给模型的文本。子类实现。"""
        raise NotImplementedError

    # -- 主流程 ------------------------------------------------------------
    async def execute(
        self,
        instance_id: Optional[str] = None,
        parameters: Optional[dict] = None,
        **kwargs: Any,
    ):
        """执行一次工具调用，返回 (ToolResponse, step_reward=0.0, metrics)。"""
        parameters = parameters or {}
        value = parameters.get(self.param_key, "")
        metrics: Dict[str, Any] = {"source": "http" if self.base_url else "local"}
        start = time.time()

        if not value:
            resp = _make_response(f"[工具失败] 缺少参数 {self.param_key}")
            return resp, 0.0, {"error_type": "bad_request", "elapsed": 0.0}

        try:
            if self.base_url:
                text = await self._call_http(value)
            else:
                tool = self._build_local_tool()
                result = await asyncio.to_thread(tool.execute, value)
                metrics["cache_hit"] = getattr(result, "from_cache", False)
                metrics["error_type"] = getattr(result, "error_type", None)
                if not getattr(result, "ok", True):
                    raise RuntimeError(getattr(result, "error_type", "unknown"))
                text = result.content
            metrics["elapsed"] = round(time.time() - start, 3)
            return _make_response(text), 0.0, metrics
        except Exception as exc:  # noqa: BLE001 - 工具失败绝不中断 rollout
            err_type = getattr(exc, "error_type", None) or type(exc).__name__
            metrics.update(
                {"error_type": err_type, "elapsed": round(time.time() - start, 3)}
            )
            return _make_response(f"[工具失败] {err_type}: {exc}"), 0.0, metrics


class SearchToolWrapper(_BaseWrapper):
    """搜索工具：parameters["query"] -> 搜索结果文本。"""

    param_key = "query"

    def _build_local_tool(self):
        if self._local_tool is None:
            from .tool_factory import build_search_tool

            self._local_tool = build_search_tool(self.config)
        return self._local_tool

    async def _call_http(self, query: str) -> str:
        # aiohttp 延迟导入：本地纯模式/单测不强制依赖
        import aiohttp

        url = _normalize_endpoint(self.base_url, "/retrieve")
        payload = {
            "queries": [query],
            "topk": int(self.config.get("num_results", 5)),
            "return_scores": True,
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
        docs = (data.get("result") or [[]])[0]
        return _format_docs_to_text(query, docs)


class OpenToolWrapper(_BaseWrapper):
    """打开网页工具：parameters["url"] -> 网页正文文本。"""

    param_key = "url"

    def _build_local_tool(self):
        if self._local_tool is None:
            from .tool_factory import build_open_tool

            self._local_tool = build_open_tool(self.config)
        return self._local_tool

    async def _call_http(self, url: str) -> str:
        import aiohttp

        endpoint = _normalize_endpoint(self.base_url, "/open")
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, json={"url": url}) as resp:
                resp.raise_for_status()
                data = await resp.json()
        if not data.get("ok", False):
            raise RuntimeError(data.get("contents", "open 失败"))
        return data.get("contents", "")


__all__ = ["SearchToolWrapper", "OpenToolWrapper"]
