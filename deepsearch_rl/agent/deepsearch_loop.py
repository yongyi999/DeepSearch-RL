# -*- coding: utf-8 -*-
"""
DeepSearch Agent Loop
=====================

注册名 ``deepsearch_agent``，对应配置 ``agent.default_agent_loop=deepsearch_agent``。
在 veRL 官方 ``ToolAgentLoop`` 基础上只做一件事：
- ``init_class`` 调完父类后把 ``cls.tool_schemas`` 置空——
  我们的格式由自定义 XML 系统提示主导，不需要再往 prompt 里渲染 OpenAI function schema 块。

其余状态机 / TITO token 拼接 / 工具返回 mask 全部复用父类。

veRL 缺失时给出占位，保证本模块可独立 py_compile。
"""

from __future__ import annotations

from typing import Any

# 导入本模块即触发 search_xml_parser 的 @ToolParser.register("search_xml")，
# 否则 veRL 按 multi_turn.format=search_xml 查找解析器时会报「未注册」。
from . import search_xml_parser  # noqa: F401

# veRL 基类导入（缺失时占位）
try:  # pragma: no cover - 训练环境才有 verl
    from verl.experimental.agent_loop.agent_loop import register  # type: ignore
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop  # type: ignore
except Exception:  # pragma: no cover - 本地/单测环境
    def register(name: str):  # type: ignore
        def _deco(cls):
            return cls

        return _deco

    class ToolAgentLoop:  # type: ignore
        """最小占位基类，仅为了本地导入/编译通过。"""

        tool_schemas: list = []

        @classmethod
        def init_class(cls, *args: Any, **kwargs: Any) -> None:
            return None


@register("deepsearch_agent")
class DeepSearchAgentLoop(ToolAgentLoop):
    """自定义多轮 AgentLoop：关闭 OpenAI schema 渲染，其余复用父类。"""

    @classmethod
    def init_class(cls, config: Any, tokenizer: Any, processor: Any, **kwargs: Any) -> None:
        # 先让父类完成全部初始化（工具挂载、解析器构造等）
        super().init_class(config, tokenizer, processor, **kwargs)
        # 不渲染 OpenAI function schema 块——格式由 XML 系统提示主导
        cls.tool_schemas = []


__all__ = ["DeepSearchAgentLoop"]
