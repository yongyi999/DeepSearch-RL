# -*- coding: utf-8 -*-
"""
DeepSearch-RL 训练入口包
=========================

对外暴露：
- :mod:`deepsearch_rl.train.train_grpo`   —— 主训练入口（拍平 yaml → Hydra dotlist → run_ppo）
- :mod:`deepsearch_rl.train.swanlab_callback` —— SwanLab 轻量封装（veRL 已自带 swanlab 追踪，
  本模块只做 login/init 的薄封装与「swanlab 未安装」时的 NoOp 兜底）

注意：本包在顶层不强制导入 verl / torch / hydra，保证 ``py_compile`` 与 dry-run 在
无 verl 环境下也能跑；重依赖一律在函数内惰性导入。
"""

from __future__ import annotations

__all__ = ["train_grpo", "swanlab_callback"]
