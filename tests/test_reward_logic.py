# -*- coding: utf-8 -*-
"""
分层奖励的离线逻辑测试（不连接 Judge）
======================================

本测试在「本地无 openai/tenacity、无 Judge 服务」的环境下运行，
验证 :func:`compute_score` 的 **同步降级路径** 与门控行为：

1. 答案指标（normalize / em / f1）基本正确；
2. 构造一条「含重复 search + 观测命中 gold + answer 命中 gold」的轨迹：
   - Judge 构建失败 -> 自动降级；
   - R_format=0.2、R_answer=1.0（EM）、R_evidence=0.6（代理命中）；
   - 重复 1 次 -> R_tool 为负（-0.15）；
3. 构造一条「答案正确但无任何观测（猜的）」的轨迹：
   - R_evidence=0、R_tool=0；
   - 门控使其总分显著低于「正确 + 证据充分」。

直接运行：``py -3.13 tests/test_reward_logic.py``
"""

from __future__ import annotations

import os
import sys

# 把工程根目录加入 sys.path，保证离线运行也能 import deepsearch_rl
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepsearch_rl.protocol import build_observation  # noqa: E402
from deepsearch_rl.rewards.hierarchical import compute_score_sync  # noqa: E402
from deepsearch_rl.rewards.answer_metrics import (  # noqa: E402
    best_f1,
    em_match,
    normalize_answer,
    token_f1,
)


def _obs(body: str) -> str:
    """构造一个 <observation> 块（复用协议，保证标签正确）。"""
    return build_observation(body)


def test_answer_metrics() -> None:
    # 归一化：小写 + 去标点 + 去冠词 + 压缩空白
    assert normalize_answer("The U.S. Capital!") == "us capital"
    assert normalize_answer("  A  An  THE  ") == ""  # 只剩冠词 -> 空串

    # EM
    assert em_match("Paris", ["Paris", "paris, france"]) is True
    assert em_match("London", ["Paris"]) is False
    assert em_match("", ["Paris"]) is False          # 空预测不命中
    assert em_match("Paris", []) is False            # 空 gold 不命中

    # token F1
    assert token_f1("the cat sat", "cat sat on mat") > 0.0
    assert token_f1("", "") == 1.0                  # 两侧都空
    assert token_f1("abc", "") == 0.0               # 一侧空
    assert best_f1("paris france", ["paris", "london"]) == token_f1("paris france", "paris")
    print("[ok] test_answer_metrics")


def test_correct_with_duplicate_and_evidence() -> dict:
    """重复 search + 观测命中 gold + answer 命中 gold 的降级路径。"""
    solution = (
        "我先查一下。\n"
        "<search>法国首都</search>\n"
        + _obs("巴黎是法国的首都及最大城市。") + "\n"
        "再确认一次。\n"
        "<search>法国首都</search>\n"
        + _obs("法国首都为巴黎。") + "\n"
        "<answer>巴黎</answer>"
    )
    ground_truth = {"target": ["巴黎"]}
    extra_info = {"supporting_titles": ["巴黎"]}

    res = compute_score_sync("nq", solution, ground_truth, extra_info)

    print("  result:", res)
    assert res["r_format"] == 0.2, f"R_format 应为 0.2，实际 {res['r_format']}"
    assert res["r_answer"] == 1.0, f"R_answer 应为 1.0(EM)，实际 {res['r_answer']}"
    assert res["r_evidence"] == 0.6, f"R_evidence 代理命中应为 0.6，实际 {res['r_evidence']}"
    assert res["num_duplicate"] == 1, f"应有 1 次重复调用，实际 {res['num_duplicate']}"
    assert res["r_tool"] < 0, f"重复调用应使 R_tool 为负，实际 {res['r_tool']}"
    # R_tool = 0 - min(1*0.15,0.3) - 0 = -0.15
    assert abs(res["r_tool"] - (-0.15)) < 1e-6, f"R_tool 应为 -0.15，实际 {res['r_tool']}"
    assert res["duplicate_rate"] > 0
    print("[ok] test_correct_with_duplicate_and_evidence")
    return res


def test_correct_without_evidence() -> dict:
    """答案正确但无任何观测（猜测）：门控应使分数显著偏低。"""
    solution = "我根据已有知识直接回答。\n<answer>巴黎</answer>"
    ground_truth = {"target": ["巴黎"]}
    extra_info = {}

    res = compute_score_sync("nq", solution, ground_truth, extra_info)

    print("  result:", res)
    assert res["r_format"] == 0.2
    assert res["r_answer"] == 1.0
    assert res["r_evidence"] == 0.0, f"无观测时代理应 0，实际 {res['r_evidence']}"
    assert res["r_tool"] == 0.0, f"无工具调用时 R_tool 应为 0，实际 {res['r_tool']}"
    # 0.2 + 1.0*(0.2+0) + 0 + 0 = 0.4
    assert abs(res["score"] - 0.4) < 1e-6, f"总分应为 0.4，实际 {res['score']}"
    print("[ok] test_correct_without_evidence")
    return res


def test_gate_gap() -> None:
    """门控：正确+充分 应显著高于 正确+无证据。"""
    r_evi = test_correct_with_duplicate_and_evidence()
    r_noevi = test_correct_without_evidence()
    gap = r_evi["score"] - r_noevi["score"]
    print(f"  门控分差 = {r_evi['score']:.3f} - {r_noevi['score']:.3f} = {gap:.3f}")
    assert gap > 0.4, f"门控应使两者分差 > 0.4，实际 {gap:.3f}"
    print("[ok] test_gate_gap")


if __name__ == "__main__":
    test_answer_metrics()
    test_gate_gap()
    print("\n全部离线逻辑测试通过 ✅")
