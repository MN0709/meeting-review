"""R-P1.5-9（阶段 10-B）：术语热词。

做两件事：
1. 把「手动维护的热词 + 已确认成员姓名」合成一份词表（成员姓名**不落表**，
   改名后自动生效，不需要同步任务）；
2. 把词表拼成一句注入转写的提示（`initial_prompt`），并设长度上限。

原则（PRD 1.2 §3 R-P1.5-9）：
- 热词**只影响转写质量**，不改变引文校验规则（引文仍必须是转写原文的完整子串）；
- **未配置或关闭时返回 None**，转写调用与现状完全一致（不传该参数）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

MAX_TERM_CHARS = 40
DEFAULT_PROMPT_MAX_CHARS = 200
PROMPT_PREFIX = "以下是本次中文会议的录音，可能涉及以下专有名词，请按这些写法转写："


def normalize_term(value: Any) -> str:
    return str(value or "").strip()


def collect_terms(db, team_id: int, *, members: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
    """合并手动热词与已确认成员姓名（成员来源标记为 `member`，不可删除）。"""
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in db.list_terms(team_id):
        term = normalize_term(row.get("term"))
        if term and term not in seen:
            seen.add(term)
            items.append({
                "id": row.get("id"), "term": term, "note": row.get("note") or "",
                "source": "manual", "updated_at": row.get("updated_at"),
            })
    member_names = members
    if member_names is None:
        member_names = [item.name for item in db.list_members(team_id)]
    for name in member_names:
        term = normalize_term(name)
        if term and term not in seen:
            seen.add(term)
            items.append({"id": None, "term": term, "note": "已确认成员姓名", "source": "member",
                          "updated_at": None})
    return items


def build_prompt(terms: Sequence[str], max_chars: int = DEFAULT_PROMPT_MAX_CHARS) -> Optional[str]:
    """把词表拼成注入转写的提示；没有词或全部被截断时返回 None。"""
    cleaned = [normalize_term(term) for term in terms]
    cleaned = [term for term in cleaned if term]
    if not cleaned:
        return None
    budget = max(0, int(max_chars) - len(PROMPT_PREFIX) - 2)
    kept: List[str] = []
    used = 0
    for term in cleaned:
        cost = len(term) + 1  # 顿号或句号
        if used + cost > budget:
            break
        kept.append(term[:MAX_TERM_CHARS])
        used += cost
    if not kept:
        return None
    return "{} {}。".format(PROMPT_PREFIX, "、".join(kept))


def build_team_prompt(db, team_id: Optional[int], *, enabled: bool, max_chars: int) -> Optional[str]:
    """按团队组装转写提示；关闭或无团队时恒为 None（行为与现状一致）。"""
    if not enabled or not team_id:
        return None
    terms = [item["term"] for item in collect_terms(db, team_id)]
    return build_prompt(terms, max_chars)
