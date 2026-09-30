# -*- coding: utf-8 -*-
"""
答案归一化与匹配指标（对齐 Search-R1 的 ``qa_em`` / token-F1）
================================================================

约定（与 Search-R1 一致）：
1. 转小写；
2. 删除标点（标点替换为空格，避免 ``U.S.`` 被粘连成 ``us``）；
3. 删除英文冠词 ``a / an / the``；
4. 压缩连续空白为单个空格并 strip。

对外提供：
- :func:`normalize_answer`：归一化。
- :func:`em_match`：预测与任一 gold 归一化后**精确相等**即命中。
- :func:`token_f1`：预测与单个 gold 之间的 token（多集合）P/R/F1。
- :func:`best_f1`：预测与 gold_list 中**最大**的 token-F1。

边界：空串 / 空 gold_list 均安全处理，不抛异常。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional, Sequence

__all__ = ["normalize_answer", "em_match", "token_f1", "best_f1"]

# 英文冠词，归一化时丢弃
_ARTICLES = {"a", "an", "the"}

# 标点字符（SQuAD / Search-R1 口径：直接删除标点，使 "U.S." -> "us"）
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_answer(s: Optional[str]) -> str:
    """把答案归一化为可比较的 token 串。

    步骤：转小写 -> 删除标点（对齐 SQuAD ``qa_em``，故 ``U.S.`` -> ``us``）
    -> 去冠词 -> 压缩空白。``None`` / 非字符串输入一律按空串处理。
    """
    if s is None:
        return ""
    text = str(s).lower()
    text = _PUNCT_RE.sub("", text)
    tokens = [tok for tok in text.split() if tok not in _ARTICLES]
    return " ".join(tokens).strip()


def _tokens(s: Optional[str]) -> list[str]:
    """归一化后切 token；空答案返回空列表。"""
    return normalize_answer(s).split()


def em_match(pred: Optional[str], gold_list: Optional[Sequence[str]]) -> bool:
    """预测答案是否与 gold_list 中**任意一个**归一化后完全相等。

    - 预测为空串 -> False（空答案不算命中）。
    - gold_list 为空 / None -> False。
    """
    pred_norm = normalize_answer(pred)
    if not pred_norm:
        return False
    for gold in gold_list or []:
        if normalize_answer(gold) == pred_norm:
            return True
    return False


def token_f1(pred: Optional[str], gold: Optional[str]) -> float:
    """预测与单个 gold 的 token 级 F1（基于多集合 / Counter 的 SQuAD 口径）。

    - 任一侧为空：两侧都空才算完全重合（F1=1），否则 F1=0。
    - 无公共 token -> 0.0。
    """
    pred_tokens = _tokens(pred)
    gold_tokens = _tokens(gold)

    if not pred_tokens or not gold_tokens:
        return 1.0 if (not pred_tokens and not gold_tokens) else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def best_f1(pred: Optional[str], gold_list: Optional[Sequence[str]]) -> float:
    """预测与 gold_list 之间的最大 token-F1。

    gold_list 为空 / None 时返回 0.0。
    """
    gold_list = list(gold_list or [])
    if not gold_list:
        return 0.0
    return max(token_f1(pred, gold) for gold in gold_list)
