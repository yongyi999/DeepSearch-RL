# -*- coding: utf-8 -*-
"""
TrajectoryAnalyzer 与 SearchXmlParser 的逻辑自测。

运行（工程根目录下）：
    py -3.13 tests/test_agent_logic.py
"""

import asyncio
import json
import sys
from pathlib import Path

# 保证从工程根目录导入 deepsearch_rl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deepsearch_rl.agent.trajectory import TrajectoryAnalyzer
from deepsearch_rl.agent.search_xml_parser import SearchXmlParser


def build_demo_solution() -> str:
    """构造一条含 3 个 search（1 个重复）、2 个 observation、1 个 answer 的轨迹。"""
    return (
        "<think>我先搜一下北京天气。</think>\n"
        "<search>北京天气</search>\n"
        "<observation>\n北京今天晴，25 度，微风。</observation>\n"
        # 归一化后与第一条重复（首尾/中间多余空白，collapse 后相同）
        "<search>  北京天气  </search>\n"
        "<search>上海常住人口</search>\n"
        "<observation>\n"
        # 观测正文里伪造一个 <search> 标签，必须被忽略，不能算成模型调用
        "上海常住人口约 2487 万。（这段正文里出现的 <search>不应被解析</search> 是噪声）\n"
        "</observation>\n"
        "<answer>北京今天晴 25 度；上海常住人口约 2487 万。</answer>"
    )


def test_trajectory() -> None:
    sol = build_demo_solution()
    a = TrajectoryAnalyzer.from_solution(sol)

    assert a.num_search == 3, f"num_search 期望 3，实际 {a.num_search}"
    assert a.num_open == 0, f"num_open 期望 0，实际 {a.num_open}"
    assert a.num_tool_calls == 3, f"num_tool_calls 期望 3，实际 {a.num_tool_calls}"
    assert a.num_duplicate == 1, f"num_duplicate 期望 1，实际 {a.num_duplicate}"
    assert len(a.duplicate_calls) == 1
    assert a.duplicate_rate() == 1 / 3, f"duplicate_rate 期望 1/3，实际 {a.duplicate_rate()}"

    assert a.has_answer is True
    assert a.final_answer == "北京今天晴 25 度；上海常住人口约 2487 万。", (
        f"final_answer 不符: {a.final_answer!r}"
    )

    # 观测切分：2 个 observation
    assert len(a.observations) == 2, f"observations 数量期望 2，实际 {len(a.observations)}"
    # evidence_text 拼接了两块观测正文，且不含 observation 标签本身
    assert "<observation>" not in a.evidence_text, "evidence_text 不应残留 observation 标签"
    assert "北京今天晴" in a.evidence_text
    assert "2487 万" in a.evidence_text
    # 观测内部伪造的 search 标签不应出现在 search_queries 里
    assert "不应被解析" not in a.search_queries, "观测块内的标签被误算成工具调用！"
    assert len(a.search_queries) == 3
    # 原始查询未归一化大小写/内部空白；注意 parse_tool_calls 已做首尾 strip
    assert a.search_queries == ["北京天气", "北京天气", "上海常住人口"], (
        f"search_queries 不符: {a.search_queries}"
    )

    print("[PASS] TrajectoryAnalyzer 全部断言通过")
    print(f"       num_search={a.num_search}, num_duplicate={a.num_duplicate}, "
          f"duplicate_rate={a.duplicate_rate():.3f}, obs={len(a.observations)}")
    print(f"       final_answer={a.final_answer}")


def test_parser_answer_terminates() -> None:
    """纯 <answer> 文本：返回空调用（终止）。"""
    parser = SearchXmlParser(tokenizer=None)
    text = "我整理一下结果。\n<answer>42</answer>"
    out_text, calls = asyncio.run(parser.extract_tool_calls(text))
    assert out_text == text
    assert calls == [], f"含 answer 时应返回空调用，实际 {calls}"
    print("[PASS] SearchXmlParser：纯 <answer> 文本返回 (text, [])")


def test_parser_search_call() -> None:
    """含 <search> 的文本：映射成 FunctionCall(name=search, arguments={query})。"""
    parser = SearchXmlParser(tokenizer=None)
    text = "<search>Qwen3 参数量</search>"
    out_text, calls = asyncio.run(parser.extract_tool_calls(text))
    assert len(calls) == 1, f"期望 1 个调用，实际 {len(calls)}"
    fc = calls[0]
    assert fc.name == "search"
    args = json.loads(fc.arguments)
    assert args == {"query": "Qwen3 参数量"}, f"arguments 不符: {args}"
    print("[PASS] SearchXmlParser：<search> 映射为 FunctionCall(search, {query})")


def test_parser_open_call_and_registry() -> None:
    """含 <open> 的文本映射；并验证 get_tool_parser 注册中心可用。"""
    parser = SearchXmlParser.get_tool_parser("search_xml")
    text = "<open>https://example.com/a?utm_source=x#frag</open>"
    out_text, calls = asyncio.run(parser.extract_tool_calls(text))
    assert len(calls) == 1
    assert calls[0].name == "open"
    args = json.loads(calls[0].arguments)
    assert args["url"].startswith("https://example.com/a")
    print("[PASS] SearchXmlParser：注册中心取到 search_xml，<open> 映射正确")


if __name__ == "__main__":
    test_trajectory()
    test_parser_answer_terminates()
    test_parser_search_call()
    test_parser_open_call_and_registry()
    print("\n全部逻辑验证通过 ✅")
