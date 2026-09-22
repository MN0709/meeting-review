"""R-P2.1-7：待跟进跨会议跟踪（指纹 + upsert 的纯函数部分）。

红线：待跟进只在**所属项目内**合并，跨项目绝不合并；状态只能由人改（AI 只建议，
本阶段 M2 不实现「AI 建议已解决」）。

指纹采用与内容级去重同一套归一化（`app/conclusions.normalize_text`）：
去空白、去中英标点、去引导词，再转小写。归一化后为空时回退到原文字符的 sha256 前缀，
避免「全是标点」的条目互相误合并。
"""

from __future__ import annotations

import hashlib

from app.conclusions import normalize_text


def fingerprint(text: str) -> str:
    normalized = normalize_text(text)
    if normalized:
        return normalized
    return "sha256:" + hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]
