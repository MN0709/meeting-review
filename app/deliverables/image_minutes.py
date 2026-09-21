"""R-P1.5-1 / R-P2-8：图片纪要（四板块卡片长图）+ 导出 PDF。

设计约束（对齐 PRD 2.0 §3 R-P2-8）：
- **四板块顺序固定**：① 这次会议的核心 / ② 待办（含紧急）/ ③ 我答应的任务 / ④ 关键决策；
- 「紧急事项」不再独立成板块，而是并入「待办」，紧急条目带「紧急」标签并排在最前；
- **去重**：同一内容（文本一致或指向同一引文）不得同时出现在两个板块；
- 「我答应的任务」受门控：**未识别到「我」时整块隐藏**（不再是「未指定」占位）；
- 内容**只来自**报告与行动项，**不新增任何句子**（不编造）；
- 空板块显示明确空态文案，**不许空白**；
- 模板与数据分离：版式在 `templates/card_v1.html`，文案取值在本模块；
- 所有模型/数据文本经 `html.escape` 写入，避免注入。
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_TEMPLATE = "card_v1"

# 四板块顺序固定（PRD 2.0 R-P2-8）
PART_KEYS = ("core", "todo", "mine", "decisions")
PART_TITLES = {
    "core": "① 这次会议的核心",
    "todo": "② 待办（含紧急）",
    "mine": "③ 我答应的任务",
    "decisions": "④ 关键决策",
}
PART_SUBTITLES = {
    "core": "会议总览与要点",
    "todo": "本场未完成的行动项；要求尽快处理的带「紧急」标签并排在最前",
    "mine": "只列负责人是你自己的任务",
    "decisions": "带时间戳的关键决策，可回到原话",
}
EMPTY_NOTES = {
    "core": "本次未生成会议总览。",
    "todo": "本次没有待办事项。",
    "mine": "本场没有你负责的待办。",
    "decisions": "本次未识别到关键决策。",
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
  .card.c-todo { background: var(--c2); border-left-color: var(--b2); }
  .card.c-mine { background: var(--c3); border-left-color: var(--b3); }
  .card.c-decisions { background: var(--c4); border-left-color: var(--b4); }
  ul { margin: 0; padding-left: 20px; }
  li { margin: 7px 0; line-height: 1.7; font-size: 14px; }
  .ts { display: inline-block; margin-right: 6px; padding: 1px 6px; border-radius: 6px;
        background: #ffffff; border: 1px solid var(--line); color: var(--muted);
        font-size: 12px; font-variant-numeric: tabular-nums; }
  .urgent { display: inline-block; margin-right: 6px; padding: 1px 6px; border-radius: 6px;
            background: #ffe0cc; border: 1px solid var(--b2); color: #b3490a;
            font-size: 12px; font-weight: 700; }
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


def _normalize_key(value: Any) -> str:
    """去空白后的比较键，用于判断「同一内容」。"""
    return "".join(str(value or "").split())


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
    """把报告与行动项渲染成四个板块（纯数据，便于测试与复用）。

    R-P2-8：紧急事项并入待办、同一内容去重、未识别到「我」时隐藏「我答应的任务」。
    """
    statuses = _status_by_index(action_rows)
    items: Dict[str, List[Dict[str, Any]]] = {key: [] for key in PART_KEYS}

    # ① 核心：会议总览 + 要点（决策已单独成板块，避免同一条重复出现）
    overview = str(getattr(report, "overview", "") or "").strip()
    if overview:
        items["core"].append({"text": overview, "kind": "overview"})
    for point in getattr(report, "meeting_points", []) or []:
        text = str(point).strip()
        if text:
            items["core"].append({"text": text, "kind": "point"})

    # ④ 关键决策：每条带时间戳，可回到原话
    for decision in getattr(report, "decisions", []) or []:
        text = str(getattr(decision, "content", "") or "").strip()
        if not text:
            continue
        evidence = getattr(decision, "evidence", None)
        items["decisions"].append({
            "text": text,
            "timestamp": evidence.timestamp if evidence else None,
            "meta": "决策人：{}".format(str(getattr(decision, "decision_maker", "") or "未明确")),
        })

    # ② 待办：未完成行动项；同时收集「我答应的任务」（同一对象引用，后续标记紧急会同步）
    todo: List[Dict[str, Any]] = []
    mine: List[Dict[str, Any]] = []
    for index, item in enumerate(getattr(report, "action_items", []) or []):
        status = statuses.get(index, "待确认")
        text = str(item.task).strip()
        if not text or status not in OPEN_STATUSES:
            continue
        evidence = getattr(item, "evidence", None)
        entry = {
            "text": text,
            "timestamp": evidence.timestamp if evidence else None,
            "meta": "负责人：{} ｜ 截止：{} ｜ 状态：{}".format(
                item.owner or "未明确", item.deadline or "未明确", status,
            ),
            "owner": str(item.owner or "").strip(),
            "status": status,
            "urgent": False,
            "_text_key": _normalize_key(text),
            "_quote_key": _normalize_key(getattr(evidence, "quote", "")) if evidence else "",
        }
        todo.append(entry)
        if self_name and entry["owner"] == self_name:
            mine.append(entry)

    # 紧急事项并入待办：与已有行动项「同一内容」时只标紧急，不重复出现
    for urgent in getattr(report, "urgent_items", []) or []:
        content = str(getattr(urgent, "content", "") or "").strip()
        if not content:
            continue
        evidence = getattr(urgent, "evidence", None)
        text_key = _normalize_key(content)
        quote_key = _normalize_key(getattr(evidence, "quote", "")) if evidence else ""
        merged = next(
            (
                entry for entry in todo
                if (text_key and entry["_text_key"] == text_key)
                or (quote_key and entry["_quote_key"] == quote_key)
            ),
            None,
        )
        if merged is not None:
            merged["urgent"] = True
            continue
        todo.append({
            "text": content,
            "timestamp": evidence.timestamp if evidence else None,
            "meta": None,
            "owner": "",
            "status": "紧急",
            "urgent": True,
            "_text_key": text_key,
            "_quote_key": quote_key,
        })

    # 紧急排最前（稳定排序，组内保持原顺序）
    todo.sort(key=lambda entry: 0 if entry.get("urgent") else 1)
    items["todo"] = todo
    items["mine"] = mine

    parts: List[Dict[str, Any]] = []
    for key in PART_KEYS:
        # R-P2-8 ④ / R-P2-3 ⑤：未识别到「我」时整块隐藏，不显示占位。
        if key == "mine" and not self_name:
            continue
        part_items = items[key]
        parts.append({
            "key": key,
            "title": PART_TITLES[key],
            "subtitle": PART_SUBTITLES[key],
            "items": part_items,
            "empty_note": EMPTY_NOTES[key] if not part_items else None,
        })
    return parts


def _render_items(items: Iterable[Dict[str, Any]]) -> str:
    rows = []
    for item in items:
        badges = ""
        if item.get("urgent"):
            badges += '<span class="urgent">紧急</span>'
        timestamp = item.get("timestamp")
        if timestamp:
            badges += '<span class="ts">{}</span>'.format(_clean(timestamp))
        meta = item.get("meta")
        meta_html = '<span class="meta-line">{}</span>'.format(_clean(meta)) if meta else ""
        rows.append("<li>{}{}{}</li>".format(badges, _clean(item.get("text")), meta_html))
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
