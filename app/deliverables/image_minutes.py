"""R-P1.5-1：图片纪要（四板块卡片长图）+ 导出 PDF。

设计约束（对齐阶段 9 文档）：
- **四板块顺序固定**：这次会议的核心 / 紧急事项 / 待办 / 我答应的任务；
- 内容**只来自**报告与行动项，**不新增任何句子**（不编造）；
- 空板块显示明确空态文案（如「本次未识别到紧急事项」），**不许空白**；
- 「我答应的任务」在未指定「我」时显示「未指定你自己」，**绝不推断**；
- 模板与数据分离：版式在 `templates/card_v1.html`，文案取值在本模块；
- 所有模型/数据文本经 `html.escape` 写入，避免注入。
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_TEMPLATE = "card_v1"

# 四板块顺序固定（PRD 1.2 §3 R-P1.5-1）
PART_KEYS = ("core", "urgent", "todo", "mine")
PART_TITLES = {
    "core": "① 这次会议的核心是什么",
    "urgent": "② 紧急事项是什么",
    "todo": "③ 待办是什么",
    "mine": "④ 我答应的任务是什么",
}
PART_SUBTITLES = {
    "core": "会议总览、要点与带时间戳的关键决策",
    "urgent": "会上明确要求「尽快 / 今天就 / 上线前」处理的事项",
    "todo": "本场尚未完成的行动项",
    "mine": "只列负责人是你自己的任务",
}
EMPTY_NOTES = {
    "core": "本次未生成会议总览。",
    "urgent": "本次未识别到紧急事项。",
    "todo": "本次没有未完成的待办。",
    "mine": "未指定你自己。",
}
# 已完成/已取消不算待办（沿用 PRODUCT_SPEC_V1 的「未完成」口径）
OPEN_STATUSES = ("待确认", "进行中")

PART_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>@@TITLE@@</title>
<style>
  @page { size: A4; margin: 12mm; }
  :root {
    --ink: #16241d; --muted: #5d6b62; --line: #dfe6e0;
    --c1: #e8f3ff; --b1: #3dadff;
    --c2: #fff0e6; --b2: #ff8a3d;
    --c3: #eef7ee; --b3: #4caf50;
    --c4: #f1ebff; --b4: #874fff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 28px 32px; background: #fff; color: var(--ink);
         font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
  header { margin-bottom: 22px; }
  header h1 { margin: 0 0 8px; font-size: 26px; line-height: 1.35; }
  header .meta { color: var(--muted); font-size: 13px; line-height: 1.7; white-space: pre-line; }
  .card { border: 1px solid var(--line); border-radius: 16px; padding: 16px 18px; margin-bottom: 16px;
          border-left: 6px solid var(--line); page-break-inside: avoid; }
  .card h2 { margin: 0 0 4px; font-size: 18px; }
  .card .sub { color: var(--muted); font-size: 12px; margin: 0 0 10px; }
  .card.c-core { background: var(--c1); border-left-color: var(--b1); }
  .card.c-urgent { background: var(--c2); border-left-color: var(--b2); }
  .card.c-todo { background: var(--c3); border-left-color: var(--b3); }
  .card.c-mine { background: var(--c4); border-left-color: var(--b4); }
  ul { margin: 0; padding-left: 20px; }
  li { margin: 7px 0; line-height: 1.7; font-size: 14px; }
  .ts { display: inline-block; margin-right: 6px; padding: 1px 6px; border-radius: 6px;
        background: #ffffff; border: 1px solid var(--line); color: var(--muted);
        font-size: 12px; font-variant-numeric: tabular-nums; }
  .meta-line { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; }
  .empty { color: var(--muted); font-size: 13px; margin: 0; }
  footer { margin-top: 18px; color: var(--muted); font-size: 11px; line-height: 1.6; white-space: pre-line; }
</style>
</head>
<body>
<header>
  <h1>@@TITLE@@</h1>
  <div class="meta">@@META@@</div>
</header>
@@BODY@@
<footer>@@FOOTER@@</footer>
</body>
</html>
"""

_TEMPLATE_SLOTS = ("@@TITLE@@", "@@META@@", "@@BODY@@", "@@FOOTER@@")


def _clean(value: Any) -> str:
    return html.escape(str(value if value is not None else "").strip())


def _status_by_index(action_rows: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    statuses: Dict[int, str] = {}
    for index, row in enumerate(action_rows):
        raw_index = row.get("item_index")
        key = int(raw_index) if isinstance(raw_index, int) else index
        statuses[key] = str(row.get("status") or "待确认")
    return statuses


def build_parts(
    *, report: Any, action_rows: Sequence[Dict[str, Any]] = (),
    self_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """把报告与行动项渲染成四个板块（纯数据，便于测试与复用）。"""
    statuses = _status_by_index(action_rows)

    core_items: List[Dict[str, Any]] = []
    overview = str(getattr(report, "overview", "") or "").strip()
    if overview:
        core_items.append({"text": overview, "kind": "overview"})
    for point in getattr(report, "meeting_points", []) or []:
        text = str(point).strip()
        if text:
            core_items.append({"text": text, "kind": "point"})
    # 关键决策补进「核心」板块：决策自带原话与时间戳，这样第一块也能回到原文
    # （总览与要点是模型归纳，本身没有时间戳）。
    for decision in getattr(report, "decisions", []) or []:
        text = str(getattr(decision, "content", "") or "").strip()
        if not text:
            continue
        evidence = getattr(decision, "evidence", None)
        core_items.append({
            "text": text,
            "kind": "decision",
            "timestamp": evidence.timestamp if evidence else None,
        })

    urgent_items = [
        {
            "text": str(item.content).strip(),
            "timestamp": item.evidence.timestamp if item.evidence else None,
            "meta": None,
        }
        for item in (getattr(report, "urgent_items", []) or [])
        if str(getattr(item, "content", "") or "").strip()
    ]

    todo_items: List[Dict[str, Any]] = []
    mine_items: List[Dict[str, Any]] = []
    for index, item in enumerate(getattr(report, "action_items", []) or []):
        status = statuses.get(index, "待确认")
        if status not in OPEN_STATUSES:
            continue
        entry = {
            "text": str(item.task).strip(),
            "timestamp": item.evidence.timestamp if getattr(item, "evidence", None) else None,
            "meta": "负责人：{} ｜ 截止：{} ｜ 状态：{}".format(
                item.owner or "未明确", item.deadline or "未明确", status,
            ),
            "owner": str(item.owner or "").strip(),
            "status": status,
        }
        if not entry["text"]:
            continue
        todo_items.append(entry)
        if self_name and entry["owner"] == self_name:
            mine_items.append(entry)

    parts: List[Dict[str, Any]] = []
    for key in PART_KEYS:
        if key == "core":
            items = core_items
        elif key == "urgent":
            items = urgent_items
        elif key == "todo":
            items = todo_items
        else:
            items = mine_items
        parts.append({
            "key": key,
            "title": PART_TITLES[key],
            "subtitle": PART_SUBTITLES[key],
            "items": items,
            "empty_note": EMPTY_NOTES[key] if not items else None,
        })
    return parts


def _render_items(items: Iterable[Dict[str, Any]]) -> str:
    rows = []
    for item in items:
        timestamp = item.get("timestamp")
        badge = '<span class="ts">{}</span>'.format(_clean(timestamp)) if timestamp else ""
        meta = item.get("meta")
        meta_html = '<span class="meta-line">{}</span>'.format(_clean(meta)) if meta else ""
        rows.append("<li>{}{}{}</li>".format(badge, _clean(item.get("text")), meta_html))
    return "<ul>{}</ul>".format("".join(rows))


def render_html(
    *, report: Any, action_rows: Sequence[Dict[str, Any]] = (),
    self_name: Optional[str] = None, title: str = "会议图片纪要",
    meta: str = "", footer: str = "", template: str = DEFAULT_TEMPLATE,
) -> str:
    """渲染四板块卡片式长图的 HTML（PDF 由 render.py 基于它导出）。"""
    template_path = TEMPLATE_DIR / "{}.html".format(template)
    tpl = template_path.read_text(encoding="utf-8") if template_path.exists() else PART_TEMPLATE
    cards = []
    for part in build_parts(report=report, action_rows=action_rows, self_name=self_name):
        if part["items"]:
            inner = _render_items(part["items"])
        else:
            inner = '<p class="empty">{}</p>'.format(_clean(part["empty_note"] or "暂无内容"))
        cards.append(
            '<section class="card c-{key}"><h2>{title}</h2><p class="sub">{sub}</p>{inner}</section>'.format(
                key=part["key"], title=_clean(part["title"]), sub=_clean(part["subtitle"]), inner=inner,
            )
        )
    rendered = tpl
    for token, value in zip(_TEMPLATE_SLOTS, (
        _clean(title), _clean(meta), "".join(cards), _clean(footer),
    )):
        rendered = rendered.replace(token, value)
    return rendered
