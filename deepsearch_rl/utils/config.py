# -*- coding: utf-8 -*-
"""
配置加载与合并（OmegaConf 轻封装）
==================================

训练入口用 Hydra/OmegaConf 管理配置。这里提供几个最常用的小工具，
并在未安装 omegaconf 时退化为普通 dict，保证纯逻辑模块本地可跑。

- load_yaml(path)：读 yaml 配置文件。
- merge_config(*cfgs)：按顺序合并（后者覆盖前者）。
- dotlist(items)：把 ["a=1", "b.c=2"] 这样的命令行覆写列表转成配置。
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Union

# OmegaConf 是训练期重依赖，本地单测环境可能没有；缺失时用 dict 兜底。
try:  # pragma: no cover - 取决于运行环境
    from omegaconf import OmegaConf  # type: ignore

    _HAS_OMEGACONF = True
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore
    _HAS_OMEGACONF = False


ConfigLike = Union[dict, "Any"]  # OmegaConf DictConfig 或 dict


def load_yaml(path: Optional[str]) -> ConfigLike:
    """读取 yaml 配置文件；path 为空时返回空配置。"""
    if not path:
        return _empty()
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    if _HAS_OMEGACONF:
        return OmegaConf.load(path)
    # 退化路径：用 PyYAML（若有）
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except ImportError:
        # 连 PyYAML 都没有时，只能读成空 dict，避免训练期崩溃
        data = {}
    return data


def merge_config(*cfgs: Optional[ConfigLike]) -> ConfigLike:
    """按顺序合并多个配置（后者覆盖前者）。"""
    cfgs = [c for c in cfgs if c is not None]
    if not cfgs:
        return _empty()
    if _HAS_OMEGACONF:
        merged = OmegaConf.create()
        for c in cfgs:
            OmegaConf.merge(merged, c)
        return merged
    # 退化路径：浅合并（训练主路径走 OmegaConf，这里只保证不崩）
    out: dict = {}
    for c in cfgs:
        if isinstance(c, dict):
            out.update(c)
    return out


def dotlist(items: Optional[Iterable[str]]) -> ConfigLike:
    """把命令行覆写列表（如 ["trainer.n_gpus=4", "rollout.temperature=1.0"]）转成配置。"""
    if not items:
        return _empty()
    if _HAS_OMEGACONF:
        return OmegaConf.from_dotlist(list(items))
    out: dict = {}
    for item in items:
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        # 逐层写入嵌套 dict：a.b.c = v
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _maybe_scalar(value)
    return out


def _empty() -> ConfigLike:
    if _HAS_OMEGACONF:
        return OmegaConf.create()
    return {}


def _maybe_scalar(v: str):
    """粗略把字符串转成 int/float/bool。"""
    low = v.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


__all__ = ["load_yaml", "merge_config", "dotlist"]
