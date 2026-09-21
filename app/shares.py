"""R-P1.5-3（阶段 11／M4）：分享链接。

三件事：
1. 生成/校验令牌（只把哈希落库）；
2. 有效期判定（默认 72 小时，过期返回 410 语义）；
3. **按勾选裁剪 payload** —— 未勾选的字段**绝不进入响应体**（红线 6），
   且分享视图里不出现团队成员身份库、其它会议与成本数据（红线 7）。

分享页 HTML 由本模块生成，**不复用团队单页**，避免把团队功能面暴露给外链。
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

SCOPE_IMAGE_MINUTES = "image_minutes"
SCOPE_REPORT = "report"
SCOPE_TRANSCRIPT = "transcript"
SCOPE_VOICE = "voice"
SCOPE_TASKS = "tasks"
ALL_SCOPES = (SCOPE_IMAGE_MINUTES, SCOPE_REPORT, SCOPE_TRANSCRIPT, SCOPE_VOICE, SCOPE_TASKS)
SCOPE_LABELS = {
    SCOPE_IMAGE_MINUTES: "图片纪要（四板块）",
    SCOPE_REPORT: "精简纪要（文字报告）",
    SCOPE_TRANSCRIPT: "逐字稿",
    SCOPE_VOICE: "代表语音片段",
    SCOPE_TASKS: "任务单（行动项 + 紧急事项）",
}
SHARE_PRIVACY_NOTE = (
    "本页由会议作者主动分享，链接 3 天后自动失效，可随时被作者撤销。"
    "完整录音已按隐私策略删除，此处最多只含每位说话人 3 段、每段不超过 12 秒的代表性片段。"
    "页面只包含作者勾选的内容。"
)


def new_token() -> str:
    """32 字节随机令牌（URL 安全）。"""
    return secrets.token_urlsafe(32)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def expires_at(ttl_hours: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=max(1, int(ttl_hours)))).isoformat()


def is_expired(record: Dict[str, Any], *, now: Optional[datetime] = None) -> bool:
    try:
        deadline = datetime.fromisoformat(str(record.get("expires_at")))
    except (TypeError, ValueError):
        return True
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) >= deadline


def invalid_scopes(requested: Sequence[Any]) -> List[str]:
    return [str(item) for item in requested if str(item) not in ALL_SCOPES]


def normalize_scopes(requested: Sequence[Any]) -> List[str]:
    """去重并保持固定顺序，便于前端与测试断言。"""
    wanted = {str(item) for item in requested}
    return [scope for scope in ALL_SCOPES if scope in wanted]


def build_payload(
    *, record: Dict[str, Any], history: Any, image_minutes: Optional[Dict[str, Any]],
    my_tasks: Optional[Dict[str, Any]], clips: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """**只输出被勾选的内容**。未勾选的键根本不出现（不是置空）。"""
    scopes = set(record.get("scopes") or [])
    payload: Dict[str, Any] = {
        "title": history.title,
        "meeting_date": history.created_at[:10],
        "duration_seconds": history.duration_seconds,
        "scopes": [scope for scope in ALL_SCOPES if scope in scopes],
        "expires_at": record.get("expires_at"),
        "privacy_note": SHARE_PRIVACY_NOTE,
    }
    if SCOPE_REPORT in scopes:
        report = history.report
        payload["report"] = {
            "overview": report.overview,
            "meeting_points": list(report.meeting_points),
            "decisions": [item.model_dump() for item in report.decisions],
            "unresolved_issues": [item.model_dump() for item in report.unresolved_issues],
        }
    if SCOPE_TASKS in scopes:
        payload["tasks"] = {
            "action_items": [item.model_dump() for item in history.report.action_items],
            "urgent_items": [item.model_dump() for item in history.report.urgent_items],
        }
    if SCOPE_TRANSCRIPT in scopes:
        payload["transcript"] = [
            {"start": item.start, "end": item.end, "speaker_label": item.speaker_label,
             "text": item.text}
            for item in history.transcript
        ]
    if SCOPE_VOICE in scopes:
        payload["voice"] = list(clips)
    if SCOPE_IMAGE_MINUTES in scopes and image_minutes is not None:
        payload["image_minutes"] = image_minutes.get("parts", [])
    return payload


def render_page() -> str:
    """最小只读分享页（自带渲染，不复用团队单页）。"""
    return """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>共享的会议纪要</title>
<style>
  :root{--ink:#16241d;--muted:#5d6b62;--line:#dfe6e0;--card:#fff}
  *{box-sizing:border-box}
  body{margin:0;padding:28px 18px 60px;background:#f6f8f6;color:var(--ink);
       font-family:-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
  main{max-width:880px;margin:0 auto}
  h1{font-size:24px;margin:0 0 6px}
  .meta{color:var(--muted);font-size:13px;margin:0 0 14px;line-height:1.7}
  .note{border:1px solid var(--line);background:#fffdf2;border-radius:14px;padding:12px 14px;
        font-size:13px;color:#6b5b2c;line-height:1.7;margin-bottom:18px}
  section{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px;margin-bottom:14px}
  section h2{margin:0 0 10px;font-size:17px}
  ul{margin:0;padding-left:20px}li{margin:6px 0;line-height:1.7;font-size:14px}
  .ts{display:inline-block;margin-right:6px;padding:1px 6px;border-radius:6px;background:#f2f4f2;
      border:1px solid var(--line);color:var(--muted);font-size:12px}
  .empty{color:var(--muted);font-size:13px}
  .clip{margin:6px 0}
  audio{height:32px;vertical-align:middle}
  pre{white-space:pre-wrap;word-break:break-word;font-size:13px;line-height:1.7;margin:0;
      font-family:inherit;color:#31403a}
  footer{color:var(--muted);font-size:12px;text-align:center;margin-top:22px}
</style>
</head>
<body>
<main>
  <h1 id="title">加载中…</h1>
  <p class="meta" id="meta"></p>
  <div class="note" id="note"></div>
  <div id="content"></div>
  <footer>由「会脉 · 团队会议记忆」按作者勾选范围生成</footer>
</main>
<script>
const token=location.pathname.split('/').filter(Boolean).pop();
const el=(tag,text,cls)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node};
function section(title,items,emptyText){const box=el('section');box.append(el('h2',title));if(!items.length){box.append(el('p',emptyText,'empty'));return box}const list=el('ul');items.forEach(node=>list.append(node));box.append(list);return box}
function line(text,timestamp){const li=el('li');if(timestamp)li.append(el('span',timestamp,'ts'));li.append(document.createTextNode(text));return li}
function group(text,timestamp,meta){const li=el('li');if(timestamp)li.append(el('span',timestamp,'ts'));li.append(document.createTextNode(text));if(meta)li.append(el('div',meta,'empty'));return li}
async function load(){
  const response=await fetch('/api/shares/'+encodeURIComponent(token),{cache:'no-store'});
  const body=await response.json();
  if(!response.ok){
    document.getElementById('title').textContent='这个分享链接不可用';
    document.getElementById('note').textContent=(body.error&&body.error.message)||'链接可能已过期或被撤销。';
    return;
  }
  document.getElementById('title').textContent=body.title||'会议纪要';
  document.getElementById('meta').textContent=`${body.meeting_date||''} ｜ 时长 ${Math.round((body.duration_seconds||0)/60)} 分钟 ｜ 有效期至 ${(body.expires_at||'').slice(0,10)}`;
  document.getElementById('note').textContent=body.privacy_note||'';
  const content=document.getElementById('content');
  if(body.image_minutes){
    (body.image_minutes||[]).forEach(part=>{
      content.append(section(part.title,(part.items||[]).map(item=>group(item.text,item.timestamp,item.meta)),part.empty_note||'本板块无内容。'));
    });
  }
  if(body.report){
    const report=body.report;
    content.append(section('会议总览',[line(report.overview||'（无）')],'（无）'));
    content.append(section('会议要点',(report.meeting_points||[]).map(point=>line(point)),'（无）'));
    content.append(section('决策清单',(report.decisions||[]).map(item=>group(item.content,item.evidence&&item.evidence.timestamp,item.decision_maker?('决策人：'+item.decision_maker):null)),'（无）'));
    content.append(section('遗留问题',(report.unresolved_issues||[]).map(item=>group(item.content,item.evidence&&item.evidence.timestamp)),'（无）'));
  }
  if(body.tasks){
    content.append(section('待办（行动项）',(body.tasks.action_items||[]).map(item=>group(item.task,item.evidence&&item.evidence.timestamp,`负责人：${item.owner||'未明确'} ｜ 截止：${item.deadline||'未明确'}`)),'（无）'));
    content.append(section('紧急事项',(body.tasks.urgent_items||[]).map(item=>group(item.content,item.evidence&&item.evidence.timestamp)),'（无）'));
  }
  if(body.voice){
    const clips=el('section');clips.append(el('h2','代表语音片段'));
    if(!(body.voice||[]).length){clips.append(el('p','（无）','empty'))}
    (body.voice||[]).forEach(clip=>{const row=el('div','','clip');row.append(el('span',clip.timestamp||'','ts'),el('span',clip.speaker_label||'','clip-speaker'));const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src='/api/shares/'+encodeURIComponent(token)+'/clips/'+clip.id;row.append(audio);clips.append(row)});
    content.append(clips);
  }
  if(body.transcript){
    const box=el('section');box.append(el('h2','逐字稿'));
    const lines=(body.transcript||[]).map(item=>{const li=el('li');li.append(el('span',item.speaker_label||'','ts'));li.append(document.createTextNode(item.text));return li});
    if(!lines.length){box.append(el('p','（无）','empty'))}else{const list=el('ul');lines.forEach(node=>list.append(node));box.append(list)}
    content.append(box);
  }
}
load().catch(()=>{document.getElementById('title').textContent='加载失败';document.getElementById('note').textContent='请稍后重试。'});
</script>
</body>
</html>
"""
