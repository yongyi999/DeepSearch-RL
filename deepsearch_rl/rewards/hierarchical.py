# -*- coding: utf-8 -*-
"""
分层奖励函数（ENGINEERING_SPEC 3.7）
====================================

veRL 自定义奖励入口：``custom_reward_function.path = deepsearch_rl/rewards/hierarchical.py``，
``custom_reward_function.name = compute_score``。

签名（veRL 约定）::

    async def compute_score(data_source, solution_str, ground_truth, extra_info) -> dict

奖励构成（权重集中在顶部 :class:`RewardWeights`，可调）::

    R_format   ∈ [0, 0.2]  有 answer 且无 malformed=0.2；有 answer 但有 malformed=0.1；无 answer=0
    R_answer   ∈ [0, 1]     EM / best_f1(>=0.9) 命中=1；否则 Answer Judge：correct=1 / partial=0.5 / wrong=0
    R_evidence ∈ [0, 1]     Evidence Judge 连续分
    R_tool     ∈ [-0.5,0.1] 重复/冗余惩罚 + 高效奖励（见 tool_efficiency.py）

    R_total = R_format
            + R_answer * (0.2 + 0.8 * R_evidence)   # 证据充分度门控答案奖励
            + 0.3 * R_evidence * R_answer            # 正确且证据充分的联合奖励
            + R_tool

量级参考：正确+充分≈1.6；正确但无证据(猜)≈0.4；错误≈0.2；无答案=0。

**同步降级（必须）**：Judge 客户端构建失败（本地缺 openai/tenacity）或调用失败
（服务不可达 / 返回非法 JSON）时，一律捕获异常并退回规则分——
答案用 EM/F1，证据用「evidence_text 命中 gold 答案或 supporting_titles」的代理
（命中给 0.6，否则 0）。任何情况下都不得抛异常中断训练。

关于同步 / 异步：veRL 会自动识别 ``compute_score`` 是 coroutine 函数（async def）
还是普通函数，并分别用 ``await`` 或直接调用，因此本文件**同时**提供
``compute_score``（async，训练主路径）与 :func:`compute_score_sync`
（用 ``asyncio.run`` 兜底的同步包装，供 pytest / 离线脚本使用）。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .answer_metrics import best_f1, em_match, normalize_answer
from .tool_efficiency import ToolEfficiencyConfig, compute_tool_reward
from ..agent.trajectory import TrajectoryAnalyzer

__all__ = ["RewardWeights", "compute_score", "compute_score_sync"]


# ---------------------------------------------------------------------------
# 权重（默认值即 SPEC 3.7 冻结值，集中在此便于调参）
# ---------------------------------------------------------------------------
@dataclass
class RewardWeights:
    """分层奖励权重。"""

    #: 有 answer 且格式完全正确
    format_full: float = 0.2
    #: 有 answer 但存在 malformed 片段
    format_partial: float = 0.1
    #: 答案门控下限（无论证据如何都给的基础分）
    answer_gate_floor: float = 0.2
    #: 答案门控中证据充分度的系数
    answer_gate_evidence: float = 0.8
    #: 「正确且证据充分」的联合奖励系数
    joint_bonus: float = 0.3
    #: Answer Judge 判为 partial 时的得分
    partial_answer: float = 0.5


# 模块级默认配置（训练时通常不构造新实例，直接用默认值）
_DEFAULT_WEIGHTS = RewardWeights()
_TOOL_CFG = ToolEfficiencyConfig()


# ---------------------------------------------------------------------------
# 入参容错 / 提取
# ---------------------------------------------------------------------------
def _coerce_gold(ground_truth: Any) -> list:
    """把 ground_truth 容错地归一化为 gold 答案列表。

    兼容三种输入：
    - dict：取 ``ground_truth["target"]``（可能是 str / list / None）；
    - list / tuple：直接当 gold 列表；
    - None / 其它：空列表。
    """
    if ground_truth is None:
        return []
    if isinstance(ground_truth, dict):
        target = ground_truth.get("target", [])
    elif isinstance(ground_truth, (list, tuple)):
        target = ground_truth
    else:
        target = [ground_truth]

    if target is None:
        return []
    if isinstance(target, str):
        return [target] if target.strip() else []
    return [str(g) for g in target if str(g).strip()]


def _extract_question(extra_info: Any, ground_truth: Any) -> str:
    """尽量从 extra_info / ground_truth 里取出原始问题文本（供 Judge 使用）。

    veRL 的 compute_score 签名里没有独立的 question 形参，问题通常藏在
    extra_info（或 ground_truth）里；找不到就返回空串，不影响 EM/F1 主路径。
    """
    for src in (extra_info, ground_truth):
        if isinstance(src, dict):
            for key in ("question", "query", "problem", "prompt"):
                val = src.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""


# ---------------------------------------------------------------------------
# Judge 懒加载 + 降级
# ---------------------------------------------------------------------------
def _build_judge():
    """懒加载构建 JudgeClient；任何失败（缺 openai/tenacity 等）都返回 None。

    刻意在函数内 import，保证本地无 openai 的环境也能 import 本模块、跑离线单测。
    """
    try:
        from ..judge.judge_client import build_judge_client_from_env

        return build_judge_client_from_env()
    except Exception:  # noqa: BLE001 - 构建失败即走规则降级
        return None


def _is_degraded_reason(reason: str) -> bool:
    """判断 Judge 返回的 reason 是否表明「服务不可达 / 非法 JSON」的降级结果。

    judge_client 在故障时不抛异常，而是返回最低分 verdict，并在 reason 里写明
    "unreachable" / "invalid JSON" / "fallback"。据此识别后，我们改用规则分，
    避免把「裁判挂了」误判成「答案错误」。
    """
    r = (reason or "").lower()
    return ("unreachable" in r) or ("invalid json" in r) or ("fallback" in r)


def _fallback_evidence_score(
    analyzer: TrajectoryAnalyzer,
    gold: Sequence[str],
    extra_info: Any,
) -> float:
    """Judge 不可达时的证据充分度代理分。

    规则：evidence_text 中能命中任一 gold 答案、或 extra_info 里的
    ``supporting_titles`` 任一标题，即认为证据充分（0.6）；否则为 0。
    """
    text = (analyzer.evidence_text or "").lower()
    if not text:
        return 0.0

    # 1) gold 答案命中 evidence_text
    for g in gold:
        g_norm = normalize_answer(g)
        if g_norm and g_norm in text:
            return 0.6

    # 2) supporting_titles 命中 evidence_text
    titles: list = []
    if isinstance(extra_info, dict):
        st = extra_info.get("supporting_titles")
        if isinstance(st, (list, tuple)):
            titles = list(st)
        elif isinstance(st, str):
            titles = [st]
    for t in titles:
        t_norm = normalize_answer(t)
        if t_norm and t_norm in text:
            return 0.6

    return 0.0


# ---------------------------------------------------------------------------
# 主入口（async）
# ---------------------------------------------------------------------------
async def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> dict:
    """分层奖励主函数（veRL 异步入口）。"""
    w = _DEFAULT_WEIGHTS
    gold = _coerce_gold(ground_truth)
    analyzer = TrajectoryAnalyzer.from_solution(solution_str)
    pred = analyzer.final_answer or ""

    # ---- R_format：格式奖励 ---------------------------------------------
    if not analyzer.has_answer:
        r_format = 0.0
    elif analyzer.malformed:
        r_format = w.format_partial
    else:
        r_format = w.format_full

    # ---- R_answer：先规则（EM / 高 F1），未命中再走 Judge ---------------
    hit_em = em_match(pred, gold)
    f1 = best_f1(pred, gold)
    # 规则分：EM 命中=1；best_f1>=0.9 视为 1；否则取 best_f1 本身
    rule_answer = 1.0 if hit_em else (1.0 if f1 >= 0.9 else f1)

    question = _extract_question(extra_info, ground_truth)
    judge = _build_judge()
    judge_ok = judge is not None

    r_answer = rule_answer
    r_evidence = 0.0
    try:
        if not (hit_em or f1 >= 0.9):
            # 规则没直接命中，才麻烦 Judge 判答案
            if judge_ok:
                av = await judge.judge_answer(question, pred, gold or None)
                if _is_degraded_reason(av.reason):
                    r_answer = rule_answer  # Judge 故障，退回规则分
                elif av.correct:
                    r_answer = 1.0
                else:
                    r_answer = w.partial_answer if av.score >= 0.5 else 0.0
            else:
                r_answer = rule_answer

        # ---- R_evidence：Evidence Judge / 代理降级 ----------------------
        if judge_ok:
            ev = await judge.judge_evidence(
                question, pred, analyzer.evidence_text, gold or None
            )
            if _is_degraded_reason(ev.reason):
                r_evidence = _fallback_evidence_score(analyzer, gold, extra_info)
            else:
                r_evidence = float(ev.score)
        else:
            r_evidence = _fallback_evidence_score(analyzer, gold, extra_info)
    except Exception:  # noqa: BLE001 - 兜底：任何意外都退回规则分
        r_answer = rule_answer
        r_evidence = _fallback_evidence_score(analyzer, gold, extra_info)
    finally:
        if judge_ok:
            try:
                await judge.close()
            except Exception:  # noqa: BLE001
                pass

    # ---- R_tool：工具效率奖励 -------------------------------------------
    r_tool, _tool_bd = compute_tool_reward(analyzer, r_evidence, _TOOL_CFG)

    # ---- R_total：合成 ---------------------------------------------------
    r_total = (
        r_format
        + r_answer * (w.answer_gate_floor + w.answer_gate_evidence * r_evidence)
        + w.joint_bonus * r_evidence * r_answer
        + r_tool
    )

    return {
        "score": r_total,
        "r_format": r_format,
        "r_answer": r_answer,
        "r_evidence": r_evidence,
        "r_tool": r_tool,
        "num_search": analyzer.num_search,
        "num_open": analyzer.num_open,
        "num_duplicate": analyzer.num_duplicate,
        "duplicate_rate": analyzer.duplicate_rate(),
    }


# ---------------------------------------------------------------------------
# 同步包装（供 pytest / 离线调试；训练走上面的 async 版本）
# ---------------------------------------------------------------------------
def compute_score_sync(
    data_source: Any,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
) -> dict:
    """``compute_score`` 的同步包装：用 ``asyncio.run`` 兜底。

    veRL 训练时直接调用 async 版 ``compute_score``；本包装仅在需要同步调用的
    场景（离线单测、本地脚本）使用。
    """
    try:
        return asyncio.run(
            compute_score(data_source, solution_str, ground_truth, extra_info)
        )
    except RuntimeError:
        # 极端情况：当前线程已有运行中的 event loop（asyncio.run 会报错），
        # 手动新建一个独立 loop 跑。
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                compute_score(data_source, solution_str, ground_truth, extra_info)
            )
        finally:
            loop.close()
