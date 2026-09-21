"""WeSpeaker 归一化余弦相似度（0~1）。

单独成模块：`speaker.py` 与 `owner.py` 都要用它，抽出来避免循环导入。
"""

from __future__ import annotations

import math
from typing import Sequence


def cosine_score(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    # WeSpeaker uses this normalized cosine range for recognition.
    return max(0.0, min(1.0, (dot / (left_norm * right_norm) + 1.0) / 2.0))
