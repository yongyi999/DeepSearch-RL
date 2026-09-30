# -*- coding: utf-8 -*-
"""
SwanLab 实验追踪封装
====================

【结论先行：veRL v0.6.0 已自带 SwanLab 追踪】
------------------------------------------------
核实 ``verl/utils/tracking.py``（v0.6.0 源码）：
``Tracking.supported_backend`` 列表里已经包含 ``"swanlab"``，并且对该后端有完整实现：

* 读取环境变量 ``SWANLAB_API_KEY`` —— 若存在则 ``swanlab.login(SWANLAB_API_KEY)``；
* 读取 ``SWANLAB_LOG_DIR``（默认 ``swanlog``）、``SWANLAB_MODE``（默认 ``cloud``）；
* 调用 ``swanlab.init(project=project_name, experiment_name=experiment_name,
  config={"FRAMEWORK":"verl", **config}, logdir=..., mode=...)``；
* 把 ``swanlab`` 模块本体挂到 ``self.logger["swanlab"]``，训练中所有
  ``actor/*``、``reward/*``、``tool/*``、``rollout/*``、``timing/*`` 指标都由
  veRL 的 ``Tracking.log(data, step)`` 统一打到 swanlab；
* ``__del__`` 中调用 ``swanlab.finish()``。

因此我们的 yaml 里只要写 ``trainer.logger: [console, swanlab]``（已写好），veRL
就会自动完成 swanlab 的初始化与全量指标打点，**无需我们再实现一遍全量 logger**。

【本模块提供什么】
------------------------------------------------
1. :func:`init_swanlab(config)` —— 一个轻量封装：在「我们自己」需要主动初始化
   swanlab（例如独立评估脚本 standalone_agent / eval 也要往同一个 project 写东西）
   时，统一做 ``swanlab.login(SWANLAB_API_KEY)`` + ``swanlab.init(project, name, config,
   mode=SWANLAB_MODE)``。offline 时 ``mode="local"``。
2. :class:`SwanlabLogger` —— 一个最小可用的独立 logger（init / log_metrics /
   log_trajectories / close），主要用于「swanlab.Text 记录完整轨迹」这类 veRL
   原生 Tracking 不覆盖的内容；veRL 训练本身不依赖它。
3. 顶部对 ``swanlab`` 做 try/except 导入：若未安装，则把 ``SwanlabLogger`` 替换成
   :class:`NoOpLogger`（所有方法空操作），并打印一次中文提示，避免训练因为缺
   swanlab 直接崩掉。

环境变量约定（与 veRL 原生 tracking.py 对齐）：
- ``SWANLAB_API_KEY``：云端模式必填；
- ``SWANLAB_MODE``：``cloud``（默认）或 ``offline``（offline 时 swanlab 写本地，
  对应 swanlab.init(mode="local")）；
- ``SWANLAB_LOG_DIR``：本地落盘目录，默认 ``swanlog``。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# swanlab 容错导入：未安装时给出 NoOpLogger，保证训练/评估不因此崩溃
# ---------------------------------------------------------------------------
try:  # pragma: no cover - 训练环境才真正装 swanlab
    import swanlab  # type: ignore

    _SWANLAB_AVAILABLE = True
    _SWANLAB_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # noqa: BLE001 - 本地/最小环境
    swanlab = None  # type: ignore[assignment]
    _SWANLAB_AVAILABLE = False
    _SWANLAB_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def _mode_from_env() -> str:
    """读取 SWANLAB_MODE，返回 swanlab.init 的 mode 参数（cloud -> "cloud"，offline -> "local"）。"""
    mode = os.environ.get("SWANLAB_MODE", "cloud").strip().lower()
    if mode in ("offline", "local", "disabled"):
        return "local"
    return "cloud"


# ---------------------------------------------------------------------------
# NoOp 兜底：swanlab 未安装时使用，所有方法空操作
# ---------------------------------------------------------------------------
class NoOpLogger:
    """swanlab 未安装时的空实现，保证训练流程不中断。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _SWANLAB_AVAILABLE:
            print(
                "[swanlab_callback] 警告：未安装 swanlab（原因："
                f"{_SWANLAB_IMPORT_ERROR}），本次使用 NoOpLogger，"
                "指标不会被上传。训练本身不受影响；如需云端追踪请执行 "
                "`pip install swanlab==0.9.0` 并设置 SWANLAB_API_KEY。"
            )

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:  # noqa: D401
        """空实现。"""

    def log_trajectories(self, text: str, step: Optional[int] = None) -> None:
        """空实现。"""

    def close(self) -> None:
        """空实现。"""


def init_swanlab(config: Optional[dict] = None) -> Any:
    """轻量封装：统一做 swanlab.login + swanlab.init。

    在 veRL 训练主流程里**不需要**主动调用本函数——veRL 的 Tracking 看到
    ``trainer.logger`` 含 ``swanlab`` 时会自己初始化。本函数仅用于：
    - 独立评估脚本（``deepsearch_rl/eval/evaluate.py``）单独往同一个 project 写；
    - 本地调试 / standalone_agent 想看曲线时。

    参数 config 会被原样传给 ``swanlab.init(config=...)``，建议传入拍平后的超参 dict。
    返回初始化后的 swanlab 模块（或 swanlab 不可用时返回 None）。
    """
    if not _SWANLAB_AVAILABLE:
        print(
            f"[swanlab_callback] 未安装 swanlab（{_SWANLAB_IMPORT_ERROR}），"
            "init_swanlab 直接返回 None，不上传任何指标。"
        )
        return None

    cfg = config or {}
    project = os.environ.get("SWANLAB_PROJECT", "deepsearch-rl")
    experiment_name = os.environ.get("SWANLAB_EXPERIMENT_NAME", "qwen3-8b-grpo-search")
    log_dir = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
    mode = _mode_from_env()

    api_key = os.environ.get("SWANLAB_API_KEY")
    # 云端模式必须有 key；offline/local 模式不需要
    if mode == "cloud" and api_key:
        try:
            swanlab.login(api_key)
        except Exception as exc:  # noqa: BLE001
            print(f"[swanlab_callback] swanlab.login 失败（{exc}），继续尝试 init ...")

    swanlab.init(
        project=project,
        experiment_name=experiment_name,
        config={"FRAMEWORK": "deepsearch-rl", **cfg},
        logdir=log_dir,
        mode=mode,
    )
    return swanlab


class SwanlabLogger:
    """独立、最小可用的 SwanLab logger（与 veRL 原生 Tracking 互补，不冲突）。

    主要用途：周期性把「完整多轮轨迹」（含搜索 query / open URL / observation /
    最终答案）用 ``swanlab.Text`` 落盘到 swanlab，便于人工回看。
    veRL 原生 Tracking 负责所有标量指标（actor/reward/tool/rollout/timing），
    这里只额外打文本轨迹与少量自定义标量。

    指标分组（对齐 ENGINEERING_SPEC 第 6 节，键名直接用 swanlab 习惯的 ``组/名`` 形式）：
    - actor/grad_norm, actor/policy_loss, actor/kl, actor/entropy, actor/lr
    - reward/score_mean, reward/score_max, reward/score_min,
      reward/format, reward/answer, reward/evidence, reward/tool
    - tool/num_search_mean, tool/num_open_mean, tool/duplicate_rate, tool/error_rate
    - rollout/num_turns_mean, rollout/traj_len_mean
    - timing/step_sec, timing/rollout_sec
    """

    def __init__(
        self,
        project: str = "deepsearch-rl",
        experiment_name: str = "qwen3-8b-grpo-search",
        config: Optional[dict] = None,
    ) -> None:
        if not _SWANLAB_AVAILABLE:
            # 退化成 NoOp，保证调用方代码不用判空
            self._noop = NoOpLogger()
            self._swanlab = None
            return
        self._noop = None
        self._swanlab = swanlab

        api_key = os.environ.get("SWANLAB_API_KEY")
        mode = _mode_from_env()
        if mode == "cloud" and api_key:
            try:
                swanlab.login(api_key)
            except Exception as exc:  # noqa: BLE001
                print(f"[SwanlabLogger] login 失败：{exc}")
        swanlab.init(
            project=project,
            experiment_name=experiment_name,
            config={"FRAMEWORK": "deepsearch-rl", **(config or {})},
            logdir=os.environ.get("SWANLAB_LOG_DIR", "swanlog"),
            mode=mode,
        )

    def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """打一组标量指标；metrics 的 key 建议已带 ``组/名`` 前缀。"""
        if self._noop is not None:
            return self._noop.log_metrics(step, metrics)
        # swanlab.log 接受 {key: value}，step 用参数指定
        try:
            self._swanlab.log(metrics, step=step)
        except Exception as exc:  # noqa: BLE001
            print(f"[SwanlabLogger] log_metrics 失败：{exc}")

    def log_trajectories(self, text: str, step: Optional[int] = None) -> None:
        """把一段完整轨迹文本用 swanlab.Text 记录（人工回看用）。"""
        if self._noop is not None:
            return self._noop.log_trajectories(text, step)
        try:
            payload = {"trajectories": self._swanlab.Text(text)}
            if step is not None:
                self._swanlab.log(payload, step=step)
            else:
                self._swanlab.log(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[SwanlabLogger] log_trajectories 失败：{exc}")

    def close(self) -> None:
        """结束本次 run（swanlab.finish）。"""
        if self._noop is not None:
            return self._noop.close()
        try:
            self._swanlab.finish()
        except Exception as exc:  # noqa: BLE001
            print(f"[SwanlabLogger] close 失败：{exc}")


__all__ = ["init_swanlab", "SwanlabLogger", "NoOpLogger", "_SWANLAB_AVAILABLE"]
