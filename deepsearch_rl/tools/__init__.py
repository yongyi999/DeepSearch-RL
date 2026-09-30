# -*- coding: utf-8 -*-
"""Search/Open 工具链路：持久缓存、API-key 轮换、失败重试、异常分类。"""

from .base import BaseTool, ToolResult
from .cache import PersistentToolCache
from .exceptions import (
    ToolError,
    ToolErrorType,
    classify_exception,
    classify_http_status,
)
from .key_pool import KeyRotator, build_key_rotator

__all__ = [
    "BaseTool",
    "ToolResult",
    "PersistentToolCache",
    "ToolError",
    "ToolErrorType",
    "classify_exception",
    "classify_http_status",
    "KeyRotator",
    "build_key_rotator",
]
