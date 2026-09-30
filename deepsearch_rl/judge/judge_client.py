# -*- coding: utf-8 -*-
"""
远程 vLLM Judge 客户端
=================================

通过 OpenAI 兼容协议（``/v1/chat/completions``）调用独立部署的 vLLM Judge 服务。

对齐 ENGINEERING_SPEC 3.3 / 3.7：

- 环境变量：``JUDGE_BASE_URL``（默认 ``http://127.0.0.1:8001/v1``）、
  ``JUDGE_MODEL``（默认 ``Qwen/Qwen3-8B``）、``JUDGE_API_KEY``（默认 ``EMPTY``，
  vLLM 本地服务不需要真实 key）。
- 信号量限流（默认 64 并发）；tenacity 对网络 / 超时 / 5xx / 429 做指数退避重试，
  参数类错误（400/422）不重试。
- **容错约定**：连不上服务、重试耗尽、或模型返回非法 JSON 时，方法都不抛异常，
  而是返回默认（最低分）verdict，保证奖励计算 / 训练主流程不中断。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from .prompts import build_answer_judge_messages, build_evidence_judge_messages

__all__ = [
    "AnswerVerdict",
    "EvidenceVerdict",
    "JudgeClient",
    "build_judge_client_from_env",
    "extract_json_object",
]

# Evidence Judge：score >= 该阈值才认为证据充分（对齐 SPEC 3.7，默认 0.6）
DEFAULT_SUFFICIENCY_THRESHOLD = 0.6

# 允许重试的异常：连接失败 / 超时 / 限流(429) / 服务端错误(5xx)。
# 注意：BadRequestError(400/422，参数错误) 不在此列，不重试。
_RETRYABLE_EXC = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class AnswerVerdict:
    """Answer Judge 结果。

    Attributes:
        correct: 预测答案是否正确（verdict == correct 时为 True）。
        score: 归一化分数，correct=1.0 / partial=0.5 / wrong=0.0。
        reason: 裁判给出的一句话理由（含降级原因）。
    """

    correct: bool
    score: float
    reason: str


@dataclass
class EvidenceVerdict:
    """Evidence Judge 结果。

    Attributes:
        sufficient: 证据是否充分（score >= sufficiency_threshold 或模型显式给出 true）。
        score: 证据充分度连续分，0..1。
        reason: 裁判给出的一句话理由（含降级原因）。
    """

    sufficient: bool
    score: float
    reason: str


# ---------------------------------------------------------------------------
# 纯函数：JSON 容错抽取（无第三方依赖，便于单测）
# ---------------------------------------------------------------------------

def extract_json_object(text: Optional[str]) -> Optional[dict]:
    """从模型自由输出中容错地抽取首个 JSON 对象。

    容错策略：
    1. 先匹配 ```json ... ``` / ``` ... ``` 代码块；
    2. 否则用正则截取第一个 ``{`` 到最后一个 ``}`` 的片段；
    3. ``json.loads`` 失败时，尝试截断到最后一个 ``}`` 再解析一次（应对尾部被
       max_tokens 截断的情况）。

    Returns:
        解析出的 dict；若完全无法解析则返回 None。
    """
    if not text:
        return None

    # 1) 剥离 markdown 代码块
    fenced = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate: Optional[str] = fenced.group(1) if fenced else None

    # 2) 截取首个 { ... } 片段
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end >= start:
            candidate = text[start:end + 1]

    if candidate is None:
        return None

    # 3) 直接解析
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    # 4) 截断到最后一个 } 再试一次（处理尾部截断）
    last_brace = candidate.rfind("}")
    if last_brace != -1:
        try:
            obj = json.loads(candidate[: last_brace + 1])
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _to_float(value, default: float = 0.0) -> float:
    """安全地把模型输出的数值字段转成 float，夹到 [0, 1]。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# JudgeClient
# ---------------------------------------------------------------------------

class JudgeClient:
    """远程 vLLM Judge 的异步客户端。

    Args:
        base_url: OpenAI 兼容服务地址，如 ``http://127.0.0.1:8001/v1``。
        model: 模型名（须与服务端 ``--served-model-name`` 一致）。
        timeout: 单次请求超时（秒）。
        max_retries: 可重试异常的最大尝试次数（含首次）。
        max_concurrency: 最大并发请求数（信号量限流）。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_concurrency: int = 64,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.sufficiency_threshold = DEFAULT_SUFFICIENCY_THRESHOLD

        # 信号量：限制同时在飞的请求数，避免打挂 vLLM。
        self._sem = asyncio.Semaphore(max_concurrency)
        # api_key：vLLM 本地 OpenAI 兼容端点不校验，给占位符即可。
        # 关闭 openai 客户端自身重试，统一由 tenacity 控制。
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ.get("JUDGE_API_KEY", "EMPTY"),
            timeout=self.timeout,
            max_retries=0,
        )

    # -- 内部：带重试的 chat completion -------------------------------------

    async def _chat(self, messages: List[dict]) -> str:
        """请求一次 chat completion，返回文本内容；重试耗尽后抛异常。"""
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
            retry=retry_if_exception_type(_RETRYABLE_EXC),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                # 信号量只包住真正的网络调用，退避等待期间不占用并发槽。
                async with self._sem:
                    resp = await self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=512,
                    )
                return resp.choices[0].message.content or ""
        # 理论上不会走到这里（AsyncRetrying 耗尽会抛异常）
        raise RuntimeError("unreachable: retry loop exited without result")

    # -- 对外接口 ------------------------------------------------------------

    async def judge_answer(
        self,
        question: str,
        pred_answer: str,
        gold_answers: Optional[List[str]] = None,
    ) -> AnswerVerdict:
        """判断预测答案是否正确。

        任何失败（连不上 / 超时 / 非法 JSON）都返回
        ``AnswerVerdict(correct=False, score=0.0, reason=...)``，不抛异常。
        """
        messages = build_answer_judge_messages(question, pred_answer, gold_answers)
        try:
            raw = await self._chat(messages)
        except Exception as exc:  # noqa: BLE001 - 降级：绝不中断训练
            return AnswerVerdict(
                correct=False,
                score=0.0,
                reason=f"judge unreachable or error, fallback to wrong: {exc!r}",
            )

        obj = extract_json_object(raw)
        if obj is None:
            return AnswerVerdict(
                correct=False,
                score=0.0,
                reason=f"judge returned invalid JSON, fallback to wrong: {raw[:200]!r}",
            )

        verdict = str(obj.get("verdict", "")).strip().lower()
        reason = str(obj.get("reason", "")).strip()
        if verdict == "correct":
            return AnswerVerdict(correct=True, score=1.0, reason=reason)
        if verdict == "partial":
            return AnswerVerdict(correct=False, score=0.5, reason=reason)
        # wrong / 无法识别的 verdict 一律按错误处理
        return AnswerVerdict(correct=False, score=0.0, reason=reason)

    async def judge_evidence(
        self,
        question: str,
        pred_answer: str,
        evidence: str,
        gold_answers: Optional[List[str]] = None,
    ) -> EvidenceVerdict:
        """判断证据是否足以支撑预测答案。

        任何失败都返回 ``EvidenceVerdict(sufficient=False, score=0.0, reason=...)``，
        不抛异常。模型未显式给出 sufficient 时，按 ``score >= 0.6`` 判定。
        """
        messages = build_evidence_judge_messages(
            question, pred_answer, evidence, gold_answers
        )
        try:
            raw = await self._chat(messages)
        except Exception as exc:  # noqa: BLE001 - 降级：绝不中断训练
            return EvidenceVerdict(
                sufficient=False,
                score=0.0,
                reason=f"judge unreachable or error, fallback to insufficient: {exc!r}",
            )

        obj = extract_json_object(raw)
        if obj is None:
            return EvidenceVerdict(
                sufficient=False,
                score=0.0,
                reason=f"judge returned invalid JSON, fallback to insufficient: {raw[:200]!r}",
            )

        score = _to_float(obj.get("score"), 0.0)
        reason = str(obj.get("reason", "")).strip()
        sufficient_raw = obj.get("sufficient")
        if isinstance(sufficient_raw, bool):
            sufficient = sufficient_raw
        else:
            # 模型没给布尔值时，用阈值兜底
            sufficient = score >= self.sufficiency_threshold
        return EvidenceVerdict(sufficient=sufficient, score=score, reason=reason)

    async def close(self) -> None:
        """关闭底层 HTTP 连接池。"""
        await self._client.close()


# ---------------------------------------------------------------------------
# 便捷工厂
# ---------------------------------------------------------------------------

def build_judge_client_from_env() -> JudgeClient:
    """从环境变量构造 :class:`JudgeClient`。

    - ``JUDGE_BASE_URL``：默认 ``http://127.0.0.1:8001/v1``
    - ``JUDGE_MODEL``：默认 ``Qwen/Qwen3-8B``
    - ``JUDGE_MAX_CONCURRENCY``：默认 64
    - ``JUDGE_TIMEOUT``：默认 60（秒）
    """
    base_url = os.environ.get("JUDGE_BASE_URL", "http://127.0.0.1:8001/v1")
    model = os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-8B")
    max_concurrency = int(os.environ.get("JUDGE_MAX_CONCURRENCY", "64"))
    timeout = float(os.environ.get("JUDGE_TIMEOUT", "60"))
    return JudgeClient(
        base_url=base_url,
        model=model,
        timeout=timeout,
        max_concurrency=max_concurrency,
    )
