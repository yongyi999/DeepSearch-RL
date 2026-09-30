# -*- coding: utf-8 -*-
"""
search_xml 工具调用解析器
========================

把 veRL 的多轮状态机接到我们的纯文本标签协议上：
- 注册名 ``search_xml``，对应 ``multi_turn.format=search_xml``。
- ``extract_tool_calls(response_ids)``：先把 token id decode 成文本，
  再用 ``protocol.parse_tool_calls`` 解析；
  - 若出现 <answer>，返回 (text, []) 表示终止本轮（不再执行工具）；
  - 否则把每个可执行的 search/open 调用映射成 veRL 的 FunctionCall。

veRL 是训练期重依赖。这里顶部 try/except 导入其 ToolParser/FunctionCall，
本地未安装 veRL 时给出同名最小占位实现，保证本模块可独立 py_compile 与单测。
"""

from __future__ import annotations

import json
from typing import Any, List, Tuple

from .. import protocol
from ..protocol import SEARCH_TAG, OPEN_TAG

# ---------------------------------------------------------------------------
# veRL 基类导入（缺失时用最小占位，保证独立可跑）
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 训练环境才有 verl
    from verl.experimental.agent_loop.tool_parser import (  # type: ignore
        FunctionCall,
        ToolParser,
    )
except Exception:  # pragma: no cover - 本地/单测环境
    class FunctionCall:  # type: ignore
        """与 verl.tools.schemas.FunctionCall 对齐的最小结构。"""

        def __init__(self, name: str = "", arguments: str = "", **kwargs: Any) -> None:
            self.name = name
            self.arguments = arguments
            # 兼容可能存在的其它字段
            for k, v in kwargs.items():
                setattr(self, k, v)

        def __repr__(self) -> str:
            return f"FunctionCall(name={self.name!r}, arguments={self.arguments!r})"

    class ToolParser:  # type: ignore
        """最小 ToolParser 基类：带 register/get_tool_parser 类方法注册中心。"""

        _registry: dict = {}

        def __init__(self, tokenizer: Any = None) -> None:
            self.tokenizer = tokenizer

        @classmethod
        def register(cls, name: str):
            """装饰器：把子类注册到 name 下。"""

            def _deco(sub):
                cls._registry[name] = sub
                return sub

            return _deco

        @classmethod
        def get_tool_parser(cls, name: str, tokenizer: Any = None) -> "ToolParser":
            if name not in cls._registry:
                raise KeyError(f"未注册的 tool parser: {name}，已有: {list(cls._registry)}")
            return cls._registry[name](tokenizer=tokenizer)

        async def extract_tool_calls(self, response_ids: Any) -> Tuple[str, list]:
            raise NotImplementedError


# ---------------------------------------------------------------------------
# 文本解码：response_ids 可能是 token id 列表 / tensor / numpy / 纯文本
# ---------------------------------------------------------------------------
def _decode_ids(tokenizer: Any, response_ids: Any) -> str:
    """把模型输出的 token id 序列 decode 成文本；已是文本则原样返回。"""
    if isinstance(response_ids, str):
        return response_ids

    # list / tuple 的 int
    if isinstance(response_ids, (list, tuple)):
        if response_ids and isinstance(response_ids[0], int):
            if tokenizer is None:
                raise ValueError("response_ids 是 token id 列表，但未提供 tokenizer")
            return tokenizer.decode(response_ids)
        # 嵌套（如 batch）：取第一条
        if response_ids and isinstance(response_ids[0], (list, tuple)):
            return _decode_ids(tokenizer, response_ids[0])

    # torch tensor
    if hasattr(response_ids, "tolist"):
        try:
            return _decode_ids(tokenizer, response_ids.tolist())
        except Exception:
            pass

    # 其它类型：尽力转 str
    return str(response_ids)


@ToolParser.register("search_xml")
class SearchXmlParser(ToolParser):
    """基于 <search>/<open>/<answer> 标签的工具解析器。"""

    async def extract_tool_calls(self, response_ids: Any) -> Tuple[str, List[FunctionCall]]:
        # 1. decode 成文本
        text = _decode_ids(self.tokenizer, response_ids)

        # 2. 用共享协议解析
        result = protocol.parse_tool_calls(text)

        # 3. 出现 answer 即终止（不再派发工具调用）
        if result.has_answer:
            return text, []

        # 4. 把可执行调用映射成 FunctionCall
        calls: List[FunctionCall] = []
        for tc in result.executable_calls:
            if tc.name == SEARCH_TAG:
                calls.append(
                    FunctionCall(
                        name=SEARCH_TAG,
                        arguments=json.dumps({"query": tc.argument}, ensure_ascii=False),
                    )
                )
            elif tc.name == OPEN_TAG:
                calls.append(
                    FunctionCall(
                        name=OPEN_TAG,
                        arguments=json.dumps({"url": tc.argument}, ensure_ascii=False),
                    )
                )
        return text, calls


__all__ = ["SearchXmlParser", "FunctionCall", "ToolParser"]
