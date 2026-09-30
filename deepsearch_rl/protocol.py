# -*- coding: utf-8 -*-
"""
工具调用协议（Tool-Call Protocol）
=================================

DeepSearch-RL 采用「纯文本工具标签」与环境交互（不依赖模型原生 function-calling，
兼容 Qwen3-8B，也便于在 SGLang rollout 中以正则解析）。支持三种动作：

    <search>查询词</search>          调用搜索引擎，返回若干条结果（标题 + URL + 摘要）
    <open>URL 或 doc_id</open>       打开某条搜索结果，返回网页正文（截断）
    <answer>最终答案</answer>        给出最终答案，终止本轮交互

一个生成片段里可以出现多个 <search>/<open>（并行执行）；<answer> 出现即视为结束。
观测结果以 <observation> ... </observation> 包裹后拼回对话，进入下一轮。

本模块是整个工程的「共享契约」：agent loop、工具、奖励、评估都依赖这里的解析函数，
请勿在各模块中私自改写正则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ---------------------------------------------------------------------------
# 标签常量
# ---------------------------------------------------------------------------
SEARCH_TAG = "search"
OPEN_TAG = "open"
ANSWER_TAG = "answer"
OBSERVATION_TAG = "observation"

# 工具最大并行数（单条轨迹一个 step 内最多执行多少个工具调用）
MAX_PARALLEL_TOOL_CALLS = 4

# 每次工具观测在拼回对话时的最大字符数（防止超长网页撑爆上下文）
DEFAULT_OBS_MAX_CHARS = 4000


class ToolName(str, Enum):
    SEARCH = SEARCH_TAG
    OPEN = OPEN_TAG
    ANSWER = ANSWER_TAG


@dataclass
class ToolCall:
    """解析出的一次工具调用。"""

    name: str                      # search / open / answer
    argument: str                  # 查询词 / URL / 最终答案
    raw: str = ""                  # 原始标签片段
    span: tuple = (0, 0)           # 在生成文本中的 (start, end)
    error: Optional[str] = None    # 解析阶段发现的错误（如空参数）

    @property
    def is_answer(self) -> bool:
        return self.name == ANSWER_TAG


@dataclass
class ParseResult:
    """对一段模型生成文本的解析结果。"""

    text: str                                  # 原始生成文本
    tool_calls: List[ToolCall] = field(default_factory=list)
    # 无法被识别为合法标签的「疑似工具调用」文本（如只写了 <search> 没闭合），用于格式奖励
    malformed: List[str] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        return any(tc.is_answer for tc in self.tool_calls)

    @property
    def executable_calls(self) -> List[ToolCall]:
        """可执行的工具调用（search/open，且无解析错误），answer 不在这里执行。"""
        return [
            tc
            for tc in self.tool_calls
            if tc.name in (SEARCH_TAG, OPEN_TAG) and tc.error is None
        ]

    @property
    def final_answer(self) -> Optional[str]:
        for tc in self.tool_calls:
            if tc.is_answer:
                return tc.argument.strip()
        return None


# ---------------------------------------------------------------------------
# 正则
# ---------------------------------------------------------------------------
# 匹配成对的工具标签，参数允许跨行
_PAIR_RE = re.compile(
    r"<(?P<tag>search|open|answer)>(?P<arg>.*?)</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)

# 匹配「开了头但没正确闭合」的疑似标签，用于识别格式错误
_MALFORMED_RE = re.compile(
    r"<\s*(?P<tag>/?\s*(?:search|open|answer))[^>]*>(?:(?![\s\S]*?</\s*(?:search|open|answer)\s*>)[\s\S]*)?",
    re.IGNORECASE,
)

# 形如 <search> 的孤立开标签（更容易命中的兜底）
_ORPHAN_OPEN_RE = re.compile(
    r"<\s*(search|open|answer)\s*>(?![\s\S]*?</\s*\1\s*>)",
    re.IGNORECASE,
)
# 形如 </search> 的孤立闭标签
_ORPHAN_CLOSE_RE = re.compile(
    r"(?<!<)"  # 占位，保持可读；实际用下面的简单形式
    r"</\s*(search|open|answer)\s*>",
    re.IGNORECASE,
)


def parse_tool_calls(text: str, max_parallel: int = MAX_PARALLEL_TOOL_CALLS) -> ParseResult:
    """从模型生成文本中解析全部工具调用。

    Args:
        text: 单步生成的文本。
        max_parallel: 单步最多执行的 search/open 调用数；超出部分截断并标记，
                      避免模型一次刷几十个搜索。
    """
    result = ParseResult(text=text)
    if not text:
        return result

    covered_spans: List[tuple] = []
    n_exec = 0
    for m in _PAIR_RE.finditer(text):
        tag = m.group("tag").lower()
        arg = m.group("arg").strip()
        call = ToolCall(
            name=tag,
            argument=arg,
            raw=m.group(0),
            span=(m.start(), m.end()),
        )
        if tag in (SEARCH_TAG, OPEN_TAG) and not arg:
            call.error = "empty_argument"
        if tag in (SEARCH_TAG, OPEN_TAG):
            if n_exec >= max_parallel:
                call.error = "exceed_parallel_limit"
            elif call.error is None:
                n_exec += 1
        result.tool_calls.append(call)
        covered_spans.append((m.start(), m.end()))

    # 收集未被成对标签覆盖的孤立标签，视为格式错误
    def _covered(pos: int) -> bool:
        return any(s <= pos < e for s, e in covered_spans)

    for m in _ORPHAN_OPEN_RE.finditer(text):
        if not _covered(m.start()):
            result.malformed.append(m.group(0))
    for m in _ORPHAN_CLOSE_RE.finditer(text):
        if not _covered(m.start()):
            result.malformed.append(m.group(0))

    return result


# ---------------------------------------------------------------------------
# 观测拼接
# ---------------------------------------------------------------------------
def truncate_observation(content: str, max_chars: int = DEFAULT_OBS_MAX_CHARS) -> str:
    """把工具返回内容截断到 max_chars，并给出提示。"""
    if content is None:
        return ""
    if len(content) <= max_chars:
        return content
    omitted = len(content) - max_chars
    return (
        content[:max_chars]
        + f"\n...[内容过长，已省略 {omitted} 字符；可使用 <open> 查看其他结果]"
    )


def build_observation(content: str, max_chars: int = DEFAULT_OBS_MAX_CHARS) -> str:
    """把一次工具返回包装成 <observation> 块。"""
    body = truncate_observation(content, max_chars)
    return f"<{OBSERVATION_TAG}>\n{body}\n</{OBSERVATION_TAG}>"


def build_error_observation(error_type: str, message: str) -> str:
    """工具执行失败时返回给模型的观测（让模型学会从失败中恢复，而不是奖励空转）。"""
    return (
        f"<{OBSERVATION_TAG}>\n"
        f"[工具调用失败] 错误类型: {error_type}，信息: {message}\n"
        f"请检查参数后重试，或换一种方式继续。\n"
        f"</{OBSERVATION_TAG}>"
    )


def build_turn_prefix(step_idx: int) -> str:
    """每一轮环境回合前的前缀（可选，用于多轮拼接的可读性）。"""
    return f"\n[第 {step_idx} 轮工具返回]\n"


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个擅长多跳搜索的智能研究助手。你可以通过工具在互联网上检索资料，经过多轮搜索与网页阅读后给出准确、简洁、有证据支撑的最终答案。

你可以使用以下工具（请严格使用给定的 XML 标签，不要使用其他格式）：

1. 搜索：
<search>查询词</search>
返回若干搜索结果（标题、链接、摘要）。查询词应当简洁明确，像真实搜索引擎的关键词。

2. 打开网页：
<open>搜索结果中的链接</open>
返回该网页的正文内容。当搜索摘要不足以回答问题时，再打开网页查看细节。

3. 给出最终答案：
<answer>你的最终答案</answer>
给出最终答案后任务结束。

工作要求：
- 复杂的多跳问题需要先搜索、再根据中间结果继续搜索，必要时用 <open> 阅读关键网页。
- 不要在证据不足时草率作答（避免 under-search）；也不要重复搜索相同的内容、或在已经拿到充分证据后继续冗余调用（避免 over-search）。
- 每次可以发起一个或多个 <search>/<open> 调用，但不要无意义地重复。
- 最终答案应当简洁、直接回应问题；实体、日期、数字等关键信息必须来自检索到的证据。
- 除工具标签和必要的简短说明外，不要输出与任务无关的内容。
"""


def build_user_prompt(question: str) -> str:
    """构造单条样本的用户提问。"""
    return (
        f"问题：{question}\n\n"
        "请通过多轮 <search> / <open> 检索证据，最后用 <answer> 给出最终答案。"
    )


__all__ = [
    "ToolName",
    "ToolCall",
    "ParseResult",
    "parse_tool_calls",
    "truncate_observation",
    "build_observation",
    "build_error_observation",
    "build_turn_prefix",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "SEARCH_TAG",
    "OPEN_TAG",
    "ANSWER_TAG",
    "OBSERVATION_TAG",
    "MAX_PARALLEL_TOOL_CALLS",
    "DEFAULT_OBS_MAX_CHARS",
]
