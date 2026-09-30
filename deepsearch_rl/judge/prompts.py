# -*- coding: utf-8 -*-
"""
Judge 提示词模板与 message 构造
=================================

两类裁判共用同一套「只输出 JSON」的约束。所有模板里的占位符都用 ``str.format``
填充，因此模板正文中的字面大括号需要写成 ``{{`` / ``}}``。

- :data:`ANSWER_JUDGE_PROMPT`：含 ``{question}`` / ``{gold_answers}`` / ``{pred_answer}``
- :data:`EVIDENCE_JUDGE_PROMPT`：含 ``{question}`` / ``{pred_answer}`` / ``{evidence}`` /
  ``{gold_answers}``

构造函数返回标准 OpenAI chat messages（``[{"role": ..., "content": ...}]``），
可直接喂给 ``openai.AsyncOpenAI.chat.completions.create``。
"""

from __future__ import annotations

from typing import List, Optional

__all__ = [
    "ANSWER_JUDGE_SYSTEM",
    "ANSWER_JUDGE_PROMPT",
    "EVIDENCE_JUDGE_SYSTEM",
    "EVIDENCE_JUDGE_PROMPT",
    "build_answer_judge_messages",
    "build_evidence_judge_messages",
]

# ---------------------------------------------------------------------------
# Answer Judge
# ---------------------------------------------------------------------------

ANSWER_JUDGE_SYSTEM = (
    "你是一名严格、公正的问答裁判（Answer Judge）。"
    "你的任务是判断【预测答案】是否正确回答了【问题】。"
    "你必须只输出一个 JSON 对象，不要输出任何解释性文字、Markdown 代码块或前后缀。"
)

ANSWER_JUDGE_PROMPT = """请判断下面的【预测答案】是否正确回答了【问题】。

【问题】
{question}

【标准答案 gold answers】
{gold_answers}

【预测答案】
{pred_answer}

判分规则：
1. 实体、日期、数字、人名、机构名、地点等关键信息必须与 gold 一致；
   表述不同但语义等价（含别名、缩写、大小写差异）应判 correct。
2. 仅部分命中、只答出一部分、关键实体或数字错误、答非所问，判 partial。
3. 完全错误、与 gold 冲突、或没有给出有效答案，判 wrong。
4. 若【标准答案】为空，则没有 gold 可对照：请仅凭客观事实判断预测答案是否正确；
   你确信正确判 correct，存疑判 partial，明显错误判 wrong。

只输出一个 JSON 对象，格式严格如下（不要加任何额外文字）：
{{"verdict": "correct 或 partial 或 wrong", "score": 0.0到1.0之间的小数, "reason": "一句话理由"}}
"""

# ---------------------------------------------------------------------------
# Evidence Judge
# ---------------------------------------------------------------------------

EVIDENCE_JUDGE_SYSTEM = (
    "你是一名严格、公正的证据充分性裁判（Evidence Judge）。"
    "你的任务是判断【收集到的证据】是否足以支撑【预测答案】。"
    "你必须只输出一个 JSON 对象，不要输出任何解释性文字、Markdown 代码块或前后缀。"
)

EVIDENCE_JUDGE_PROMPT = """请判断下面【收集到的证据】是否足以支撑【预测答案】。

【问题】
{question}

【预测答案】
{pred_answer}

【标准答案 gold answers】（仅供你判断证据是否覆盖了关键事实，可为空）
{gold_answers}

【收集到的证据】（多轮搜索摘要 / 网页正文拼接，可能含噪声）
{evidence}

判分规则：
1. 证据是否**直接包含**支撑预测答案所需的关键事实（实体、日期、数字、因果关系等）。
2. 多跳问题需要每一跳的事实都在证据中出现，且链路完整、能推出最终答案；
   只覆盖其中一跳、或链路断裂，应给低分。
3. 证据里只有「结论性表述」但没有来源 / 具体事实支撑，应给低分；
   证据为空、与问题无关、或明显误导，score 给 0。
4. 证据充分、关键事实齐备、可直接推出预测答案，score 接近 1。

只输出一个 JSON 对象，格式严格如下（不要加任何额外文字）：
{{"sufficient": true 或 false, "score": 0.0到1.0之间的小数, "reason": "一句话理由"}}
"""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _format_gold(gold_answers: Optional[List[str]]) -> str:
    """把 gold 答案列表格式化为可读文本；为空时给出明确提示。"""
    if not gold_answers:
        return "（无标准答案，按客观事实判断）"
    lines = [f"- {g}" for g in gold_answers]
    return "\n".join(lines)


def build_answer_judge_messages(
    question: str,
    pred_answer: str,
    gold_answers: Optional[List[str]] = None,
) -> List[dict]:
    """构造 Answer Judge 的 OpenAI chat messages。

    Args:
        question: 原始问题。
        pred_answer: 模型给出的预测答案。
        gold_answers: 标准答案列表（可为 None / 空，表示无 gold）。

    Returns:
        ``[{"role": "system", ...}, {"role": "user", ...}]``，可直接用于
        ``AsyncOpenAI.chat.completions.create``。
    """
    user = ANSWER_JUDGE_PROMPT.format(
        question=question or "",
        gold_answers=_format_gold(gold_answers),
        pred_answer=pred_answer or "",
    )
    return [
        {"role": "system", "content": ANSWER_JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_evidence_judge_messages(
    question: str,
    pred_answer: str,
    evidence: str,
    gold_answers: Optional[List[str]] = None,
) -> List[dict]:
    """构造 Evidence Judge 的 OpenAI chat messages。

    Args:
        question: 原始问题。
        pred_answer: 模型给出的预测答案。
        evidence: 多轮搜索 / 网页正文拼接后的证据文本（可能很长）。
        gold_answers: 标准答案列表（可为 None / 空，仅供参考）。

    Returns:
        ``[{"role": "system", ...}, {"role": "user", ...}]``。
    """
    user = EVIDENCE_JUDGE_PROMPT.format(
        question=question or "",
        pred_answer=pred_answer or "",
        evidence=evidence or "（无任何证据）",
        gold_answers=_format_gold(gold_answers),
    )
    return [
        {"role": "system", "content": EVIDENCE_JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
