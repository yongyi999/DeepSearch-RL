# -*- coding: utf-8 -*-
"""
离线单测：验证 prepare_train.record_to_parquet_row 的转换逻辑
============================================================

不触网、不依赖 verl/torch：
  1. 手工构造几条统一中间记录（覆盖 nq / hotpotqa / musique）；
  2. 直接调用 prepare_train.record_to_parquet_row 转成 parquet 行；
  3. 用 pyarrow 写到内存（BytesIO）再读回，校验：
       - prompt 是 list[dict]，含 system + user 两条；
       - system.content == SYSTEM_PROMPT，user.content 含问题；
       - reward_model.ground_truth.target 是 list[str]；
       - data_source / ability / agent_name / extra_info 字段齐全。
"""

from __future__ import annotations

import io
import os
import sys

# 让测试能 import 工程里的 data.prepare_train
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# data/ 不是包，直接按文件路径导入
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "data"))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from prepare_train import record_to_parquet_row, SYSTEM_PROMPT  # noqa: E402


def _sample_records():
    return [
        {
            "id": "nq_abc123",
            "source": "nq",
            "question": "When was the Eiffel Tower built?",
            "gold_answers": ["1889", "31 March 1889"],
            "type": "single-hop",
            "level": "easy",
            "num_hops": 1,
            "supporting_titles": [],
        },
        {
            "id": "hotpot_deadbeef",
            "source": "hotpotqa",
            "question": "What language is the most spoken in the country where the Colosseum is located?",
            "gold_answers": ["Italian"],
            "type": "bridge",
            "level": "hard",
            "num_hops": 2,
            "supporting_titles": ["Colosseum", "Italy"],
        },
        {
            "id": "musique_999",
            "source": "musique",
            "question": "Who wrote the novel that the 2003 film 'Monster' was based on?",
            "gold_answers": ["Charlotte Gore", "another alias"],
            "type": "compositional",
            "level": "hard",
            "num_hops": 3,
            "supporting_titles": ["Monster (2003 film)", "…"],
            "answerable": True,
        },
    ]


def main() -> int:
    records = _sample_records()
    rows = [
        record_to_parquet_row(r, split="train", index=i)
        for i, r in enumerate(records)
    ]

    # 1) 基础结构断言
    for i, row in enumerate(rows):
        assert row["data_source"] == records[i]["source"], f"row{i} data_source 错"
        assert row["ability"] == "multi-hop-search"
        assert row["agent_name"] == "deepsearch_agent"
        assert row["extra_info"]["split"] == "train"
        assert row["extra_info"]["index"] == i
        assert row["extra_info"]["num_hops"] == records[i]["num_hops"]

        prompt = row["prompt"]
        assert isinstance(prompt, list) and len(prompt) == 2, f"row{i} prompt 不是 2 条"
        assert prompt[0]["role"] == "system"
        assert prompt[0]["content"] == SYSTEM_PROMPT
        assert prompt[1]["role"] == "user"
        assert records[i]["question"] in prompt[1]["content"], f"row{i} user prompt 未含问题"

        gt = row["reward_model"]["ground_truth"]["target"]
        assert isinstance(gt, list)
        assert gt == records[i]["gold_answers"]
        assert row["reward_model"]["style"] == "rule"

    # 2) 写 parquet 到内存再读回（验证 pyarrow 能序列化 list[dict] 结构）
    buf = io.BytesIO()
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, buf)

    buf.seek(0)
    rt = pq.read_table(buf)
    back = rt.to_pylist()
    assert len(back) == len(rows), "parquet 读回行数不一致"

    for orig, got in zip(rows, back):
        assert got["data_source"] == orig["data_source"]
        assert got["prompt"][0]["role"] == "system"
        assert got["prompt"][0]["content"] == SYSTEM_PROMPT
        assert got["prompt"][1]["role"] == "user"
        assert got["reward_model"]["ground_truth"]["target"] == \
            orig["reward_model"]["ground_truth"]["target"]
        assert got["extra_info"]["num_hops"] == orig["extra_info"]["num_hops"]

    print("[PASS] record_to_parquet_row 结构断言全部通过：")
    for r in rows:
        q = r["prompt"][1]["content"]
        print(f"  - source={r['data_source']:<9} hops={r['extra_info']['num_hops']} "
              f"target={r['reward_model']['ground_truth']['target']}")
    print("[PASS] pyarrow 内存写读回一致，prompt 为 list[dict]、ground_truth.target 为 list[str]。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
