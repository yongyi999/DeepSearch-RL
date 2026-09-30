# -*- coding: utf-8 -*-
"""
统一日志
========

全工程用同一个格式打日志，级别由环境变量 ``LOG_LEVEL`` 控制
（DEBUG / INFO / WARNING / ERROR，默认 INFO）。
"""

from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def _configure_root() -> None:
    """只配置一次 root logger，避免重复 handler。"""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT))
    root = logging.getLogger()
    root.setLevel(level)
    # 清掉已有 handler，防止 veRL/Hydra 等库重复输出
    root.handlers.clear()
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """获取一个带统一格式的 logger。"""
    _configure_root()
    return logging.getLogger(name)


__all__ = ["get_logger"]
