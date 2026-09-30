# -*- coding: utf-8 -*-
"""
独立推理 Agent（不依赖 veRL）
=============================

用于离线评估 / 调试：给定一个 question，按协议驱动 search/open 工具多轮检索，
直到模型给出 <answer> 或达到 max_turns。完全不走 veRL/SGLang rollout，
方便直接对接任意 OpenAI 兼容端点（SGLang / vLLM / 本地模型）。

- ``StandaloneAgent``：基类，``generate(messages)`` 是抽象方法，由后端注入。
- ``OpenAICompatAgent``：用 ``openai.AsyncOpenAI`` 调 chat completions 的现成实现。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import protocol
from ..protocol import (
    OPEN_TAG,
    SEARCH_TAG,
    SYSTEM_PROMPT,
    build_error_observation,
    build_observation,
    build_user_prompt,
)
from .trajectory import TrajectoryAnalyzer

Message = Dict[str, str]


@dataclass
class StandaloneResult:
    """一次独立推理的完整结果。"""

    question: str
    answer: Optional[str]
    messages: List[Message] = field(default_factory=list)
    tool_calls: List[str] = field(default_factory=list)   # 已执行的调用描述串
    num_search: int = 0
    num_open: int = 0
    num_duplicate: int = 0
    full_text: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "num_search": self.num_search,
            "num_open": self.num_open,
            "num_duplicate": self.num_duplicate,
            "num_tool_calls": self.num_search + self.num_open,
            "tool_calls": list(self.tool_calls),
            "messages": self.messages,
        }


class StandaloneAgent:
    """异步多轮搜索 agent（基类）。

    Args:
        search_tool: 搜索工具，需有 ``async aexecute(query) -> ToolResult``。
        open_tool: 网页打开工具，接口同上。
        tokenizer: 可选 tokenizer（本类一般用不到，仅为对齐接口保留）。
        max_turns: 最多多少轮 assistant 生成；到顶仍无 answer 则强制结束。
        max_parallel_calls: 单轮最多并行执行多少个 search/open。
        obs_max_chars: 每条观测拼回对话时的最大字符数。
    """

    def __init__(
        self,
        search_tool: Any,
        open_tool: Any,
        tokenizer: Any = None,
        *,
        max_turns: int = 6,
        max_parallel_calls: int = 3,
        obs_max_chars: int = 2000,
    ) -> None:
        self.search_tool = search_tool
        self.open_tool = open_tool
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.max_parallel_calls = max_parallel_calls
        self.obs_max_chars = obs_max_chars

    # ------------------------------------------------------------------
    # 后端注入点
    # ------------------------------------------------------------------
    async def generate(self, messages: List[Message]) -> str:
        """根据对话历史生成下一段 assistant 文本。子类必须实现。"""
        raise NotImplementedError(
            "请由 SGLang / vLLM / OpenAI 等后端注入 generate(messages) -> str"
        )

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def run(self, question: str) -> StandaloneResult:
        messages: List[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question)},
        ]
        full_text_parts: List[str] = []
        executed: List[str] = []
        num_search = 0
        num_open = 0
        answer: Optional[str] = None

        for _ in range(self.max_turns):
            # 1. 生成
            assistant_text = await self.generate(messages)
            messages.append({"role": "assistant", "content": assistant_text})
            full_text_parts.append(assistant_text)

            # 2. 解析
            parsed = protocol.parse_tool_calls(assistant_text)
            if parsed.has_answer:
                answer = parsed.final_answer
                break

            calls = parsed.executable_calls[: self.max_parallel_calls]
            if not calls:
                # 模型既不 answer 也不发工具：给一句错误观测，逼它按协议重试
                obs = build_error_observation(
                    "no_executable_call",
                    "未检测到合法的 <search>/<open>/<answer>。请按协议重试。",
                )
                messages.append({"role": "user", "content": obs})
                full_text_parts.append(obs)
                continue

            # 3. 并行执行工具调用
            obs_blocks = await asyncio.gather(*[self._exec_one(c) for c in calls])
            for c in calls:
                executed.append(f"{c.name}: {c.argument}")
                if c.name == SEARCH_TAG:
                    num_search += 1
                elif c.name == OPEN_TAG:
                    num_open += 1

            obs_text = "\n\n".join(obs_blocks)
            messages.append({"role": "user", "content": obs_text})
            full_text_parts.append(obs_text)

        # 4. 统计重复调用（复用与缓存一致的归一化规则）
        full_text = "\n".join(full_text_parts)
        num_duplicate = TrajectoryAnalyzer.from_solution(full_text).num_duplicate

        return StandaloneResult(
            question=question,
            answer=answer,
            messages=messages,
            tool_calls=executed,
            num_search=num_search,
            num_open=num_open,
            num_duplicate=num_duplicate,
            full_text=full_text,
        )

    async def _exec_one(self, call: protocol.ToolCall) -> str:
        """执行单个工具调用，返回拼回对话的 observation 文本。"""
        tool = self.search_tool if call.name == SEARCH_TAG else self.open_tool
        try:
            result = await tool.aexecute(call.argument)
        except Exception as exc:  # 工具层未捕获的异常
            return build_error_observation("exception", f"{type(exc).__name__}: {exc}")
        if not getattr(result, "ok", True):
            return build_error_observation(
                getattr(result, "error_type", None) or "unknown",
                getattr(result, "content", "") or "工具执行失败",
            )
        return build_observation(result.content, max_chars=self.obs_max_chars)


# ---------------------------------------------------------------------------
# 现成后端：OpenAI 兼容端点（SGLang / vLLM）
# ---------------------------------------------------------------------------
class OpenAICompatAgent(StandaloneAgent):
    """走 ``openai.AsyncOpenAI`` chat completions 的独立 agent。

    base_url 指向 SGLang/vLLM 的 OpenAI 端点（如 http://127.0.0.1:30000/v1）。
    """

    def __init__(
        self,
        search_tool: Any,
        open_tool: Any,
        *,
        model: str,
        base_url: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: Optional[int] = 4096,
        tokenizer: Any = None,
        max_turns: int = 6,
        max_parallel_calls: int = 3,
        obs_max_chars: int = 2000,
        client: Any = None,
    ) -> None:
        super().__init__(
            search_tool,
            open_tool,
            tokenizer=tokenizer,
            max_turns=max_turns,
            max_parallel_calls=max_parallel_calls,
            obs_max_chars=obs_max_chars,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client
        self._client_kwargs = {"base_url": base_url, "api_key": api_key}

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI  # type: ignore  # 重依赖延迟导入

            self._client = AsyncOpenAI(**self._client_kwargs)
        return self._client

    async def generate(self, messages: List[Message]) -> str:
        client = self._get_client()
        resp = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


__all__ = ["StandaloneAgent", "StandaloneResult", "OpenAICompatAgent"]
