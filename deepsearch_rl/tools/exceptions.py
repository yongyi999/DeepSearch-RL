# -*- coding: utf-8 -*-
"""
工具异常分类
============

长时间在线 rollout 中，搜索/打开网页会遇到各式各样的失败。这里把异常显式分类，
配合 search.py 中的重试策略与「异常分类日志」，决定：
- 哪些错误值得重试（超时、限流、网络抖动）；
- 哪些错误应直接返回给模型（参数错误、页面不存在）；
- 哪些错误需要切换 API key（401/403/配额耗尽）。

分类标签会进入工具观测与 SwanLab 监控（tool_error_rate by type）。
"""

from __future__ import annotations

import enum
from typing import Optional


class ToolErrorType(str, enum.Enum):
    # 可重试类
    RATE_LIMIT = "rate_limit"            # 429 / 限流：轮换 key 或退避后重试
    TIMEOUT = "timeout"                  # 请求超时
    NETWORK = "network"                  # 连接重置 / DNS 失败等网络问题
    SERVER = "server"                    # 5xx 服务端错误
    # 不可重试 / 直接反馈模型类
    AUTH = "auth"                        # 401/403：key 失效，需轮换
    QUOTA_EXHAUSTED = "quota_exhausted"  # 配额/余额耗尽：轮换 key
    NOT_FOUND = "not_found"              # 404 页面/资源不存在
    BAD_REQUEST = "bad_request"          # 400 参数错误
    PARSE = "parse"                      # 页面已拿到但解析失败
    BLOCKED = "blocked"                  # 反爬、验证码、地区限制
    UNSUPPORTED = "unsupported"          # 非 http/https、文件类型不支持
    # 其它
    UNKNOWN = "unknown"


# 哪些错误类型允许重试
RETRYABLE_TYPES = frozenset(
    {
        ToolErrorType.RATE_LIMIT,
        ToolErrorType.TIMEOUT,
        ToolErrorType.NETWORK,
        ToolErrorType.SERVER,
    }
)

# 哪些错误类型应当触发 API key 轮换
KEY_ROTATION_TYPES = frozenset(
    {
        ToolErrorType.AUTH,
        ToolErrorType.QUOTA_EXHAUSTED,
        ToolErrorType.RATE_LIMIT,
    }
)


class ToolError(Exception):
    """工具链路统一异常。"""

    def __init__(
        self,
        error_type: ToolErrorType,
        message: str = "",
        *,
        status_code: Optional[int] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        self.error_type = error_type
        self.status_code = status_code
        # 显式指定优先，否则按类型表判断
        self.retryable = (
            retryable if retryable is not None else error_type in RETRYABLE_TYPES
        )
        super().__init__(message or error_type.value)

    def should_rotate_key(self) -> bool:
        return self.error_type in KEY_ROTATION_TYPES

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type.value,
            "message": str(self),
            "status_code": self.status_code,
            "retryable": self.retryable,
        }


def classify_http_status(status_code: int, message: str = "") -> ToolError:
    """根据 HTTP 状态码构造对应类型的 ToolError。"""
    if status_code == 429:
        etype = ToolErrorType.RATE_LIMIT
    elif status_code in (401, 403):
        etype = ToolErrorType.AUTH
    elif status_code == 404:
        etype = ToolErrorType.NOT_FOUND
    elif status_code == 400:
        etype = ToolErrorType.BAD_REQUEST
    elif 500 <= status_code < 600:
        etype = ToolErrorType.SERVER
    else:
        etype = ToolErrorType.UNKNOWN
    return ToolError(etype, message or f"HTTP {status_code}", status_code=status_code)


def classify_exception(exc: BaseException) -> ToolError:
    """把底层第三方/网络异常归类为 ToolError（保留原始信息）。"""
    # 已经是 ToolError 直接返回
    if isinstance(exc, ToolError):
        return exc

    name = type(exc).__name__.lower()
    text = str(exc).lower()

    if "timeout" in name or "timed out" in text or "timeout" in text:
        return ToolError(ToolErrorType.TIMEOUT, str(exc))
    if "connection" in name or "connection" in text or "dns" in text or "reset" in text:
        return ToolError(ToolErrorType.NETWORK, str(exc))
    if "auth" in name or "key" in text:
        return ToolError(ToolErrorType.AUTH, str(exc))
    if "quota" in text or "balance" in text or "insufficient" in text:
        return ToolError(ToolErrorType.QUOTA_EXHAUSTED, str(exc))
    if "parse" in name or "decode" in text:
        return ToolError(ToolErrorType.PARSE, str(exc))
    return ToolError(ToolErrorType.UNKNOWN, f"{type(exc).__name__}: {exc}")


__all__ = [
    "ToolErrorType",
    "ToolError",
    "RETRYABLE_TYPES",
    "KEY_ROTATION_TYPES",
    "classify_http_status",
    "classify_exception",
]
