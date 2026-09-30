# -*- coding: utf-8 -*-
"""
远程 vLLM Judge（裁判）子模块
=================================

奖励函数依赖两类裁判，均由同一个独立的 vLLM OpenAI 兼容服务提供：

- **Answer Judge**：判断「预测答案是否正确回答了问题」，对照 gold answers
  （若有），输出 correct / partial / wrong + 分数 + 理由。
- **Evidence Judge**：判断「收集到的证据（多轮搜索 / 网页返回拼接）是否足以
  支撑预测答案」，输出 0..1 连续分 + 是否充分 + 理由。

典型用法::

    from deepsearch_rl.judge import build_judge_client_from_env

    client = build_judge_client_from_env()
    try:
        verdict = await client.judge_answer(question, pred_answer, gold)
        evi     = await client.judge_evidence(question, pred_answer, evidence)
    finally:
        await client.close()

注意：本模块的 ``judge_answer`` / ``judge_evidence`` 在裁判不可达、超时、或模型
返回非法 JSON 时**不会抛异常**，而是返回默认（最低分）verdict，保证训练主流程
不被裁判故障中断（对齐 ENGINEERING_SPEC 3.7 的「同步降级」要求）。
"""

from .prompts import (
    ANSWER_JUDGE_PROMPT,
    EVIDENCE_JUDGE_PROMPT,
    build_answer_judge_messages,
    build_evidence_judge_messages,
)
from .judge_client import (
    AnswerVerdict,
    EvidenceVerdict,
    JudgeClient,
    build_judge_client_from_env,
    extract_json_object,
)

__all__ = [
    "ANSWER_JUDGE_PROMPT",
    "EVIDENCE_JUDGE_PROMPT",
    "build_answer_judge_messages",
    "build_evidence_judge_messages",
    "AnswerVerdict",
    "EvidenceVerdict",
    "JudgeClient",
    "build_judge_client_from_env",
    "extract_json_object",
]
