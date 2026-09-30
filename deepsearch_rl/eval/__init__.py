# -*- coding: utf-8 -*-
"""
500 题冻结评测子模块
======================

对齐 ENGINEERING_SPEC 4.2 / 第 7 节，对被评模型（SGLang/vLLM OpenAI 端点）在
``data/eval_hard_500.jsonl``（HotpotQA 150 + 2Wiki 125 + MuSiQue 100 + Bamboogle 125）
上做离线多轮检索评测，输出与简历口径一致的 5 个核心指标：

- Accuracy             = mean(answer_correct)
- Evidence Sufficiency = mean(evidence_sufficiency)
- Correct & Sufficient = mean(correct_and_sufficient)
- Duplicate Call Rate  = mean(每题 duplicate_rate)
- Avg Search/query     = mean(num_search)

入口：``python -m deepsearch_rl.eval.evaluate``（见 evaluate.py 的 argparse）。
另提供 ``compare`` 函数对比两份 metrics JSON（baseline vs trained，含 +pp 差值）。

注意：本包顶层不 import openai / torch / verl；工具与 Judge 客户端在函数内懒加载，
Judge 不可达时自动降级为 EM/F1 + supporting-title 命中代理，绝不中断评测。
"""

from .evaluate import (
    aggregate_metrics,
    build_markdown_table,
    compare_metrics,
    evaluate_main,
)

__all__ = [
    "aggregate_metrics",
    "build_markdown_table",
    "compare_metrics",
    "evaluate_main",
]
