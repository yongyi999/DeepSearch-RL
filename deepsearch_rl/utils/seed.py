# -*- coding: utf-8 -*-
"""
随机种子
========

统一设置 random / numpy / torch 的种子，保证实验可复现。
torch / numpy 都是重依赖，在函数内导入，未安装时不影响纯逻辑模块。
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, *, deterministic: bool = False) -> None:
    """设置随机种子。

    Args:
        seed: 随机种子。
        deterministic: 是否强制 cuDNN 确定性（会变慢，复现实验时开启）。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # numpy：函数内导入
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    # torch：函数内导入
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


__all__ = ["set_seed"]
