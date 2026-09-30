# -*- coding: utf-8 -*-
"""
轨迹分析器（TrajectoryAnalyzer）
================================

对一条完整 rollout 文本（solution_str，含多轮模型生成 + <observation> 观测块）做结构化解析，
供奖励函数、SwanLab 日志、评测统计使用。

核心解析原则（SPEC 3.2）：
1. 先用正则（DOTALL）把所有 ``<observation>...</observation>`` 观测块切出来，
   观测块内部的文本**不参与**工具调用解析——避免观测正文里出现的 ``<search>`` 等字样
   被误算成模型的调用。
2. 对剩余「非观测文本」按片段逐段调用 ``protocol.parse_tool_calls``，收集全部
   search/open/answer 调用与 malformed。逐段解析是因为 parse_tool_calls 自带
   「单轮并行调用上限」逻辑，跨轮拼接会误判超限。

只读属性全部按 SPEC 3.2 暴露。重复调用的归一化规则与工具缓存完全一致
（查询：小写+压缩空白；URL：去 fragment/追踪参数）。
"""

from __future__ import annotations

import re
from typing import List, Optional

from .. import protocol
from ..protocol import ANSWER_TAG, OPEN_TAG, OBSERVATION_TAG, SEARCH_TAG, ToolCall

# 与缓存相同的归一化规则（tools.cache 仅依赖标准库，导入安全）
try:
    from ..tools.cache import _normalize_query, _normalize_url
except Exception:  # pragma: no cover - 兜底，正常不会走到
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", (query or "").strip().lower())

    def _normalize_url(url: str) -> str:
        u = (url or "").strip()
        u = re.sub(r"#.*$", "", u)
        u = re.sub(r"[?&](utm_[^=&]+|fbclid|gclid|spm)=[^&]*", "", u)
        u = re.sub(r"[?&]$", "", u)
        return u.rstrip("/")


# 匹配 <observation>...</observation>（允许大小写、跨行）
_OBS_RE = re.compile(
    rf"<{OBSERVATION_TAG}[^>]*>(?P<body>.*?)</{OBSERVATION_TAG}\s*>",
    re.DOTALL | re.IGNORECASE,
)


class TrajectoryAnalyzer:
    """一条轨迹的结构化解析结果（只读）。"""

    def __init__(
        self,
        tool_calls: List[ToolCall],
        malformed: List[str],
        observations: List[str],
        evidence_text: str,
        raw: str,
    ) -> None:
        # 内部存储（外部请只读访问属性，不要直接改这些字段）
        self._raw = raw
        self._tool_calls = tool_calls
        self._malformed = malformed
        self._observations = observations
        self._evidence_text = evidence_text
        self._duplicate_calls = self._find_duplicates(tool_calls)

    # ------------------------------------------------------------------
    # 构造入口
    # ------------------------------------------------------------------
    @classmethod
    def from_solution(cls, solution_str: str) -> "TrajectoryAnalyzer":
        """从完整 rollout 文本构造分析器。"""
        solution_str = solution_str or ""

        # 1. 切出所有 observation 块，分离「观测正文」与「非观测文本」
        observations: List[str] = []
        non_obs_parts: List[str] = []
        cursor = 0
        for m in _OBS_RE.finditer(solution_str):
            non_obs_parts.append(solution_str[cursor : m.start()])
            observations.append(m.group("body").strip())
            cursor = m.end()
        non_obs_parts.append(solution_str[cursor:])

        # 2. 对每段非观测文本逐轮解析（避免观测内标签误算）
        all_calls: List[ToolCall] = []
        malformed: List[str] = []
        for chunk in non_obs_parts:
            if not chunk.strip():
                continue
            parsed = protocol.parse_tool_calls(chunk)
            all_calls.extend(parsed.tool_calls)
            malformed.extend(parsed.malformed)

        # 3. evidence_text：各 observation 内部内容拼接
        evidence_text = "\n\n".join(observations).strip()

        return cls(
            tool_calls=all_calls,
            malformed=malformed,
            observations=observations,
            evidence_text=evidence_text,
            raw=solution_str,
        )

    # ------------------------------------------------------------------
    # 内部：重复调用检测
    # ------------------------------------------------------------------
    @staticmethod
    def _find_duplicates(tool_calls: List[ToolCall]) -> List[ToolCall]:
        """归一化参数后，同轨迹内第 2 次及以后出现的重复调用。"""
        seen: set = set()
        dups: List[ToolCall] = []
        for tc in tool_calls:
            if tc.name == SEARCH_TAG:
                key = (SEARCH_TAG, _normalize_query(tc.argument))
            elif tc.name == OPEN_TAG:
                key = (OPEN_TAG, _normalize_url(tc.argument))
            else:
                # answer 不参与重复统计
                continue
            if not key[1]:
                # 空参数不算有效调用，也不计重复
                continue
            if key in seen:
                dups.append(tc)
            else:
                seen.add(key)
        return dups

    # ------------------------------------------------------------------
    # 只读属性（SPEC 3.2）
    # ------------------------------------------------------------------
    @property
    def final_answer(self) -> Optional[str]:
        """最后一个 <answer> 的内容；没有则 None。"""
        for tc in reversed(self._tool_calls):
            if tc.is_answer:
                return tc.argument.strip()
        return None

    @property
    def has_answer(self) -> bool:
        return self.final_answer is not None

    @property
    def search_queries(self) -> List[str]:
        """原始（未归一）的 search 查询列表。"""
        return [tc.argument for tc in self._tool_calls if tc.name == SEARCH_TAG]

    @property
    def open_urls(self) -> List[str]:
        """原始（未归一）的 open URL 列表。"""
        return [tc.argument for tc in self._tool_calls if tc.name == OPEN_TAG]

    @property
    def num_search(self) -> int:
        return len(self.search_queries)

    @property
    def num_open(self) -> int:
        return len(self.open_urls)

    @property
    def num_tool_calls(self) -> int:
        """search + open 的总次数（answer 不计入工具调用）。"""
        return self.num_search + self.num_open

    @property
    def duplicate_calls(self) -> List[ToolCall]:
        """同轨迹内归一化参数重复的第 2 次及以后调用。"""
        return list(self._duplicate_calls)

    @property
    def num_duplicate(self) -> int:
        return len(self._duplicate_calls)

    @property
    def malformed(self) -> List[str]:
        """格式错误片段（未闭合/孤立标签等）。"""
        return list(self._malformed)

    @property
    def evidence_text(self) -> str:
        """全部 observation 内部内容拼接（供 Evidence Judge）。"""
        return self._evidence_text

    @property
    def observations(self) -> List[str]:
        """每个 observation 块的内部内容。"""
        return list(self._observations)

    @property
    def raw(self) -> str:
        """原始 solution_str。"""
        return self._raw

    # ------------------------------------------------------------------
    # 方法
    # ------------------------------------------------------------------
    def duplicate_rate(self) -> float:
        """重复调用占比 = num_duplicate / max(num_tool_calls, 1)。"""
        return self.num_duplicate / max(self.num_tool_calls, 1)

    # 便于调试
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TrajectoryAnalyzer(has_answer={self.has_answer}, "
            f"num_search={self.num_search}, num_open={self.num_open}, "
            f"num_duplicate={self.num_duplicate}, "
            f"malformed={len(self.malformed)}, obs={len(self.observations)})"
        )


__all__ = ["TrajectoryAnalyzer"]
