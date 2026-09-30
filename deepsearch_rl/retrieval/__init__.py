# -*- coding: utf-8 -*-
"""检索服务（独立 FastAPI 进程）：/retrieve  /open  /health。

封装真实 SearchTool / OpenTool，对外暴露对齐 veRL 官方 search_tool_example 的协议。
"""

from .retrieval_server import app  # noqa: F401  (uvicorn 入口：deepsearch_rl.retrieval.retrieval_server:app)

__all__ = ["app"]
