# -*- coding: utf-8 -*-
"""
分层奖励子模块
==============

按 ENGINEERING_SPEC 3.7 / 3.8 实现：

- :mod:`answer_metrics`：答案归一化 + EM / token-F1（对齐 Search-R1 ``qa_em``）。
- :mod:`tool_efficiency`：工具调用效率奖励（重复惩罚 / 冗余惩罚 / 高效奖励）。
- :mod:`hierarchical`：分层加权奖励 ``compute_score``（veRL 自定义奖励入口）。

注意：本包为纯逻辑模块，**顶层不 import openai / verl / torch**；Judge 客户端在
``hierarchical.py`` 内懒加载，裁判不可达时自动降级为规则分，保证训练不中断。
"""

from .answer_metrics import (
    best_f1,
    em_match,
    normalize_answer,
    token_f1,
)
from .tool_efficiency import (
    ToolEfficiencyConfig,
    compute_tool_reward,
)
from .hierarchical import (
    RewardWeights,
    compute_score,
    compute_score_sync,
)

__all__ = [
    # answer_metrics
    "normalize_answer",
    "em_match",
    "token_f1",
    "best_f1",
    # tool_efficiency
    "ToolEfficiencyConfig",
    "compute_tool_reward",
    # hierarchical
    "RewardWeights",
    "compute_score",
    "compute_score_sync",
]
