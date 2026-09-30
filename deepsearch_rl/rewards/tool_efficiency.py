# -*- coding: utf-8 -*-
"""
工具调用效率奖励（ENGINEERING_SPEC 3.7 的 ``R_tool``）
======================================================

设计目标：抑制「重复搜索同一内容」与「证据已足够仍继续 over-search」，同时
**奖励用尽量少的有效调用拿到充分证据**。同时刻意保护「证据充分前的探索」——
在证据还不充分时，不对 distinct（非重复）调用数量收任何冗余费。

规则（默认权重见 :class:`ToolEfficiencyConfig`）：
1. 重复调用：每出现 1 次重复（``analyzer.num_duplicate``）扣 ``dup_penalty``，
   累计封顶 ``dup_cap``。重复在任何阶段都是浪费，故无论证据是否充分都扣。
2. 证据充分（``evidence_score >= sufficiency_threshold``）后：
   distinct 调用数（``num_tool_calls - num_duplicate``）超过 ``efficient_budget``
   的部分，每次扣 ``redundant_penalty``，累计封顶 ``redundant_cap``。
3. 同时满足「无重复 + 无超额 + 证据充分」时，奖励 ``efficient_bonus``。

返回 ``(reward, breakdown)``，``reward`` 量级约 [-0.5, +0.1]。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

__all__ = ["ToolEfficiencyConfig", "compute_tool_reward"]


@dataclass
class ToolEfficiencyConfig:
    """工具效率奖励权重（默认值对齐 SPEC 3.7）。"""

    #: 每次重复调用的惩罚（负值幅度）
    dup_penalty: float = 0.15
    #: 重复惩罚的累计封顶
    dup_cap: float = 0.3
    #: 证据充分后，每次「超额 distinct 调用」的惩罚
    redundant_penalty: float = 0.1
    #: 冗余惩罚的累计封顶
    redundant_cap: float = 0.2
    #: 高效（无重复/无超额/证据充分）奖励
    efficient_bonus: float = 0.1
    #: 高效预算：证据充分时允许的 distinct 调用数
    efficient_budget: int = 4
    #: 证据充分判定阈值（与 Evidence Judge 的 0.6 一致）
    sufficiency_threshold: float = 0.6


def compute_tool_reward(
    analyzer,
    evidence_score: float,
    cfg: ToolEfficiencyConfig,
) -> Tuple[float, dict]:
    """根据轨迹的工具调用情况计算 ``R_tool``。

    Args:
        analyzer: :class:`deepsearch_rl.agent.trajectory.TrajectoryAnalyzer`。
        evidence_score: 当前 ``R_evidence``（0..1），用于判断证据是否充分。
        cfg: 效率奖励权重配置。

    Returns:
        ``(reward, breakdown)``：``reward`` 为带符号的工具奖励；
        ``breakdown`` 为各项明细，便于日志 / SwanLab。
    """
    num_dup = int(analyzer.num_duplicate)
    num_tool = int(analyzer.num_tool_calls)
    # distinct 调用 = 总工具调用中剔除重复的部分
    distinct = num_tool - num_dup
    sufficient = evidence_score >= cfg.sufficiency_threshold

    # 1) 重复惩罚（始终生效，封顶）
    dup_pen = min(num_dup * cfg.dup_penalty, cfg.dup_cap)

    # 2) 冗余惩罚：仅在证据充分后，才对「超过预算的 distinct 调用」收费
    #    —— 证据充分前的有效探索不收费。
    redundant_pen = 0.0
    excess = 0
    if sufficient:
        excess = distinct - cfg.efficient_budget
        if excess > 0:
            redundant_pen = min(excess * cfg.redundant_penalty, cfg.redundant_cap)

    # 3) 高效奖励：无重复、无超额、且证据充分
    bonus = 0.0
    if sufficient and num_dup == 0 and excess <= 0:
        bonus = cfg.efficient_bonus

    reward = bonus - dup_pen - redundant_pen

    breakdown = {
        "num_tool_calls": num_tool,
        "num_duplicate": num_dup,
        "distinct_calls": distinct,
        "evidence_sufficient": sufficient,
        "excess_calls": max(excess, 0),
        "dup_penalty": -dup_pen,
        "redundant_penalty": -redundant_pen,
        "efficient_bonus": bonus,
    }
    return reward, breakdown
