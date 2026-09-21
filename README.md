# 会脉 · 团队会议记忆

> 当前 10 天产品化迭代的状态、风险和每日验收入口见 [PROJECT_CONTROL.md](PROJECT_CONTROL.md)，详细路线见 [docs/ROADMAP_10_DAYS.md](docs/ROADMAP_10_DAYS.md)。

`meeting-review` 面向需要沉淀连续内部会议的团队，不限定团队职能、固定成员或参会人数。服务在本机完成转写、说话人分离与声纹匹配，再生成带原话证据的团队报告。用户首次把“说话人 1”确认为真实姓名并授权保存声纹后，后续会议会自动尝试识别；低置信度或多个候选过于接近时必须回退为“待确认”。

页面使用团队口令登录。口令保存在浏览器 `localStorage`，全部业务请求通过 `X-Access-Token` 发送，不使用 Cookie。上传后页面每 2 秒查询任务状态；会议完成后也可从团队历史页重新打开报告。

## 页面与团队报告

- 上传：可填写会议标题，支持 MP3/M4A/WAV，默认最大 300 MB、4 小时；2-4 小时录音建议使用 M4A/MP3。**必须先勾选同意**（「我已知晓：音频将上传至本服务用于本次转写，其中包含他人声音」）才能开始；未勾选时前端按钮禁用，绕过前端直接调接口返回 422 且不建会议、不入队、不留临时文件（R-P1.5-6）。同意后写入 `consent_records` 留证：协议版本号（`CONSENT_VERSION`，默认 `v1`）+ 同意时间 + 来源 IP；改文案升版本时旧记录不被覆盖。
- 会后整理：报告生成后自动打开“整理本次会议”，用户可确认/编辑 AI 建议标题，并根据原话片段确认说话人；允许稍后处理。**若标题仍是“未命名会议”（用户从未填写或修改），报告生成后会自动采用 AI 建议标题（D-027）；用户自己填过或改过的标题不会被覆盖。**
- 导航：登录后左侧有“首页 / 全部会议 / 项目 / 搜索”；主页专注上传，不再同屏堆叠历史和项目记忆。
- 跨会议搜索（R-P1.5-4）：“搜索”页输入一句话，在**当前团队全部历史会议**的转写原文里查找；结果带会议名、项目、时间戳、说话人与命中片段，点击直接跳到那场会议并定位。底层复用 FTS5 转写索引（中文 ≥ 3 字走 FTS5，更短或含特殊字符自动回退 `LIKE`），不占上传限频与每日名额。
- 逐字稿常驻侧栏（R-P1.5-2）：报告页右侧常驻逐字稿（桌面左右分栏，≤ 900 px 折叠为底部抽屉）。侧栏文本与 `GET /api/meetings/{id}` 的 `transcript` 逐字一致；自带关键词过滤与上一个/下一个跳转；点报告里任一时间戳 → 侧栏滚到那段并高亮；长会议按 200 段分批渲染，滚动到底自动续。
- 项目：“项目”页的每个文件夹就是一个项目，用户界面不展示二级目录；点击文件夹进入独立项目页。新上传先选择项目，标题仍可不填。
- 会议归档调整：历史页允许用户手动选择另一个项目，确认后移动；弹窗会标记并置灰当前项目，转写稿和报告不变，AI 不会自动执行移动。
- 删除项目时必须选择：保留会议并移入“未分类”，或连同会议、转写稿和报告一起删除。永久删除前会二次警告且不可恢复；含处理中会议的项目禁止连带删除。
- 处理进度：排队中、转写中、AI 分析中、完成或失败；超过 30 分钟的已知音频会提示用户可以关闭页面，任务仍在后台继续，完成后从历史记录查看。
- 会议历史：“全部会议”页按时间列出团队历史；项目详情页只列出该项目的会议。数据库保留的旧二级数据会自动归并到所属项目，不单独展示。
- 项目连续回顾：选择项目后聚合最近 3 场已完成会议的决策、行动项和遗留问题；每项标明来源会议，行动项可人工设置状态。
- 说话人确认：报告中展示本场说话人、可点击回听的代表片段、时长、自动匹配置信度和状态。同一声音被过度切成多个标签时会先尝试合并（阈值 `SPEAKER_INTRA_MERGE_THRESHOLD`，默认 0.85），**但两个簇各自命中不同的已知成员时一律不合并**（身份优先于相似度，2026-09-21 真实录音校准后新增），最终身份仍由用户确认。
- 声音授权：不再要求每位说话人分别勾选同意；点击“完成整理”时统一显示参会者名单并确认授权。
- 团队声纹身份库：身份跨项目共享；姓名、角色和“关键决策人”由用户维护，声纹可单独删除。
- 报告：①会议总览；②会议要点；③决策清单；④行动项；**④-2 我答应的任务（R-P1.5-5）**；⑤遗留问题；⑥说话人确认；⑦识别状态。点击引文时间戳可在**右侧逐字稿侧栏**定位高亮，也可点「看上下文」展开带说话人标签的转写上下文。**决策、行动项、遗留问题都带原话证据（D-028）：引文旁标「原话」，点「看上下文」展开原文核对；旧报告无行动项原话时明确显示「无法核对」，不伪造。**
- 逐项勾选分享（R-P1.5-3）：报告页右上「分享」→ 勾选要外发的内容（图片纪要 / 精简纪要 / 逐字稿 / 代表语音片段 / 任务单，**默认全不勾**）→ 生成只读链接。**未勾选的字段后端根本不会出现在响应里**（不是前端隐藏）；链接**固定 3 天有效**（`SHARE_TTL_HOURS=72`）、可随时撤销、每次访问写 `share_audit`（令牌哈希 / 时间 / IP / 结果码）。数据库**只存令牌的 sha256 哈希**（原文只在创建时返回一次），因此链接列表用 `share_id` 撤销。分享读取是独立受限出口：只有 `GET /api/shares/**` 免团队口令，其余 `/api/*` 仍然必须鉴权；分享页（`/s/{token}`）不复用团队单页，也看不到团队成员身份库、其它会议与成本数据。代表语音走 `/api/shares/{token}/clips/{id}`，**未勾选语音时读片段返回 403**。分享页会明确标注有效期与「完整录音已删除，仅含代表性片段」。
- AI 项目建议（R-P1.5-8）：**未归类**会议的报告顶部会出现一张带「**AI 建议**」角标的卡片，给出建议归入的已有项目（优先复用，不编造）或建议的新项目名，并附一句依据。三个动作都由人决定：**加入该项目 / 新建「建议名」并加入 / 用新名字新建并加入 / 不加入**；选「不加入」保持未分类、清除建议、**不留副作用**。**AI 永不自动移动会议**（有专门测试断言）；建议里出现团队不存在的项目 id 时**整条丢弃**；旧报告没有该字段时界面不显示卡片。团队还没有项目时不传项目列表，分析调用与之前完全一致。
- 交付物状态与「待核对」（R-P1.5-7）：报告页顶部有「交付物状态」条——**逐字稿 / 文字报告 / 任务单 / 图片纪要+PDF** 各自独立成/败。某一项失败只标那一项「待核对」并给出「重试」按钮，**其它交付物照常可看**；重试只重跑失败的那一项（图片纪要与任务单的重试**不调用 AI**；文字报告的重试会调用 AI，界面先弹确认框）。转写需要原始录音、而录音已按隐私策略删除，因此不可重试，会明确提示重新上传。升级前创建的旧会议没有状态记录，会按「有逐字稿/有报告」推断为正常，不显示成「待生成」。
- 图片纪要（R-P1.5-1）：报告底部「⑩ 图片纪要」把会议渲染成**四板块卡片长图**——① 这次会议的核心（总览 + 要点 + 带时间戳的关键决策）、② 紧急事项（会上明确要求「尽快 / 今天就 / 上线前」处理的事项，带原话时间戳）、③ 待办（未完成的行动项）、④ 我答应的任务；带时间戳的条目点击可跳到右侧逐字稿。点「导出 PDF」得到可外发的 PDF（HTML → PDF；渲染器优先 Playwright，缺失时自动回退本机 Chrome，都不通时返回 503 并把该交付物标「待核对」，不影响文字报告与逐字稿）。版式模板在 `app/deliverables/templates/card_v1.html`，与数据分离。
- 术语热词表（R-P1.5-9）：「项目」页底部可维护团队热词（人名 / 术语），转写时作为 `initial_prompt` 注入以提高专名准确率。**已确认的成员姓名自动生效**（不落表，改名后自动跟随，界面标注来源且不可删）。热词**只影响转写**，不改变引文校验规则（引文仍必须是转写原文的完整一致子串）；`TEAM_TERMS_ENABLED=false` 或词表为空时，转写仍走**单参数调用**，行为与未引入热词时逐字节一致；提示长度上限 `TERM_PROMPT_MAX_CHARS`（默认 200 字）。实测：同一段真实音频对拍，热词写成「胡董」后 `古董` 10 次 → 0 次、`胡董` 0 → 10 次；但热词写成发音不符的「胡泊」时无效果（只对发音对得上的词起作用）。
- 我答应的任务（R-P1.5-5）：在「整理本次会议」里选「本场哪个说话人是我」（可选已命名成员，也可选未命名说话人），报告「④-2」就只列**负责人完全等于你**的行动项。**未指定时一律显示「未指定你自己」，绝不推断**；同时提示本场有多少条负责人为「未明确」、无法归属到任何人。

旧的个人表现模型仍在 `models.py` 中标记为 legacy，以保留已有代码契约与测试；团队 UI 不呈现个人评分、口头禅或个人表现模块。

## 文件结构

```text
meeting-review/
├── app/
│   ├── config.py          # TEAM_TOKENS、长会限制和服务配置
│   ├── db.py              # SQLite 建表、团队同步和持久化查询
│   ├── llm.py             # 团队/legacy 分块分析、校验与降级
│   ├── main.py            # FastAPI、团队鉴权、上传和历史接口
│   ├── models.py          # 团队报告、任务与 legacy 数据契约
│   ├── pipeline.py        # 团队与 legacy 报告编排
│   ├── security.py        # IP 小时限频与上海自然日额度
│   ├── speaker.py         # WeSpeaker 分离、embedding 和安全身份匹配
│   ├── stats.py           # legacy 个人精确统计
│   ├── tasks.py           # 单 worker FIFO 队列、超时与清理
│   └── transcription.py   # PyAV 预检和 faster-whisper 转写
├── static/index.html      # 登录、上传、进度、历史和团队报告
├── tests/                 # 无真实 Whisper/LLM 依赖的单元测试
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── run.py
```

## 架构

```text
浏览器单页
  |-- GET /api/auth/check（团队口令）
  |-- GET/POST/PATCH /api/projects（团队文件夹）
  |-- GET /api/projects/{id}/memory（最近 3 场连续回顾）
  |-- PATCH /api/action-items/{id}（人工更新状态）
  |-- POST /api/review（标题 + 音频）
  |-- GET /api/tasks/{id}（仅处理中/短期任务）
  |-- GET /api/meetings[/{id}]（当前团队历史）
  |-- POST /api/meetings/{id}/speakers/{label}/confirm
  |-- POST /api/meetings/{id}/finalize（统一授权与声纹落库）
  |-- GET /api/speaker-clips/{id}（团队隔离的代表音频）
  `-- GET/PATCH /api/members + DELETE /api/members/{id}/voiceprint
                         |
鉴权 -> IP 限频 -> 每日上限 -> 队列上限 -> 磁盘检查
                         |
                UUID 临时音频 + PyAV 时长预检
                         |
                   FIFO 单 worker
                         |
              faster-whisper 带时间戳转写
              >30 分钟音频按 30 分钟解码分块，控制峰值内存
                         |
              WeSpeaker 本地说话人分离
              同场高相似标签合并
              团队声纹匹配（阈值 + 候选差值）
                         |-- 提取每人最多 3 段代表音频
                         |-- 然后删除原始音频
                         |-- segments 写入 SQLite
                         `-- 代码层二次时长检查
                                  |
                 OpenAI-compatible LLM
                 <=6000 字直接分析；更长文本并发分块再归并
                 JSON Schema / JSON Object 降级 + Pydantic
                 决策/遗留问题引文与时间戳严格校验（±5 秒）
                                  |
                 团队报告写入 SQLite，任务标记完成
```

## 技术选型

- **FastAPI + Pydantic**：接口与模型共享严格契约。
- **SQLite**：MVP 单机部署无需额外服务，会议、转写和报告可跨重启保留；所有资源查询都带 `team_id`。
- **PyAV + faster-whisper**：先读容器元数据秒拒超长音频，再在本机完成中文转写；超过 30 分钟的录音按 30 分钟解码分块并还原整场时间轴，避免 4 小时音频一次性解码占用近 1 GB 内存；转写后的真实时长仍会二次校验。
- **WeSpeaker**：选用 Apache-2.0 官方项目的中文模型，上传的 MP3/M4A/WAV 会先在本地流式转为单声道 16 kHz WAV，再做说话人分离和 embedding 提取；只在阈值与区分度同时达标时自动识别。
- **OpenAI 兼容协议**：Key、Base URL、模型名全部由环境变量提供；不支持 JSON Schema 时降级 JSON Object 并注入字段契约。
- **严格证据校验**：决策和遗留问题的引文必须是某个原始 segment 的完整子串，时间戳须在该 segment 范围内（允许 5 秒容差），失败会触发修复重试。
- **单进程 FIFO**：同一时刻只处理一个音频，避免本地 Whisper 在小服务器上并发挤爆内存。
- **安全文本渲染**：模型内容全部用 DOM `textContent` 写入，不通过 `innerHTML` 注入。

## 本地运行

建议 Python 3.9-3.12：

```bash
cd meeting-review
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env：填写 TEAM_TOKENS 和 OPENAI_API_KEY
# TEAM_TOKENS 的口令有强度要求（启动时校验，不合规会直接拒绝启动）：
#   · 长度 ≥ 16 位
#   · 不能是纯数字，也不能是纯英文
#   生成示例：python3 -c "import secrets;print('mr-'+secrets.token_urlsafe(16))"
python run.py
```

> 依赖区分两份清单：`requirements.txt` 是运行所需的完整集合（含 WeSpeaker 等本地模型依赖）；
> `requirements-ci.txt` 是 CI 用的轻量集合（不装 torch / faster-whisper / wespeaker，
> 因为它们都是延迟导入、测试里被 mock）。`requirements.lock` 是当前环境的完整快照。

打开 <http://127.0.0.1:8000>。健康检查无需团队口令：

### 日常怎么用（自己操作）

```bash
# 启动（后台运行，日志写到 /tmp/meeting-review.log）
cd meeting-review && nohup .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 >/tmp/meeting-review.log 2>&1 &

# 看是否在跑
curl -s http://127.0.0.1:8000/health          # {"status":"ok"}

# 停止
pkill -f "uvicorn app.main:app --host 127.0.0.1 --port 8000"
```

| 事项 | 位置 / 做法 |
| --- | --- |
| 数据库（会议、报告、交付物状态、热词、同意留证、分享链接） | `data/meeting-review.db` |
| 备份（说话人过度合并缺陷修复前的整库快照） | `data/meeting-review.db.bak-20260921-152829` |
| 恢复备份 | 先停服务，再 `cp data/meeting-review.db.bak-20260921-152829 data/meeting-review.db` |
| 运行日志 | `/tmp/meeting-review.log`（出问题时先看这里的最后几行） |
| 上传的原始录音 | **转写完成后即删除**（由隐私策略决定，不保留） |
| 团队口令 / API Key | 只写在 `.env`（已被 `.gitignore` 排除，不会进仓库） |

> 分享链接默认按请求地址生成（本机就是 `127.0.0.1:8000`）。要给**别人**打开，需要先做内网穿透或公网地址，再把 `.env` 里的 `SHARE_BASE_URL` 填成那个地址。

不要直接双击 `static/index.html` 使用 `file://` 地址；该页面无法连接 FastAPI 后端。若误开，本地页面会显示原因并提供正确服务地址。

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

## API

- `GET /health`：免鉴权健康检查。
- `GET /api/auth/check`：校验团队口令并返回团队 ID，错误为 403。
- `GET /api/agent/tools`：Agent 工具调试端点，枚举注册的 12 个工具（8 只读 + 1 规划 + 3 能力），并导出动态系统提示词。**默认关闭**：`AGENT_MODE=pipeline` 时该路由不存在（404）；仅 `shadow`/`agent` 模式且 team 内可访问。
- `GET /api/usage`：按 `stage` / 会议聚合 LLM 用量与估算成本，严格限定在本团队内。
  参数：`meeting_id?`、`project_id?`、`from?`、`to?`（ISO8601）；跨团队 403，非法日期 422，表为空返回空数组。
  金额需配置 `LLM_PRICE_PROMPT_PER_1K` / `LLM_PRICE_COMPLETION_PER_1K`；未配置时 `cost` 为 `null`。
  这是**新增接口**，错误结构为 `{"error": {"code", "message"}}`；老接口仍为 `{"detail": ...}`（契约不动）。
- `GET /api/search`：**跨会议搜索转写片段（R-P1.5-4，阶段 8／M1）**。
  参数：`q`（必填，1–50 字）、`project_id?`、`from?`、`to?`（`YYYY-MM-DD` 按上海自然日解释，或完整 ISO8601）、`limit?`（默认 20，上限 50）。
  响应：`{"query", "count", "hits": [{meeting_id, meeting_title, project_id, project_name, start, end, timestamp, speaker_label, text_snippet}]}`；空结果返回 200 + `count=0`。
  同一场会议最多返回 5 条，避免单场会议淹没结果；严格 `team_id` 隔离（跨团队 403 且不返回数据）；不计入限频与每日名额；错误结构 `{"error": {"code", "message"}}`（`invalid_args` 422 / `not_found` 404 / `team_forbidden` 403）。
- `GET /api/meetings/{id}/my-tasks`：**「我答应的任务」切面（R-P1.5-5）**。只返回 `owner` 与「我」完全一致的行动项；未指定时 `self_speaker_set=false` + 空列表，另给 `owner_unknown`（负责人为「未明确」的条数）。跨团队 403。
- `POST /api/meetings/{id}/self-speaker`：指定本场「我」对应哪个说话人（`{"local_label": "说话人 1"}` 或 `{"member_id": 3}`；传空对象表示清除）。非法说话人/非本场成员 422；无报告 404；跨团队 403。
- `GET /api/projects`：列出当前团队项目文件夹及会议数量。
- `GET /api/projects/{id}/memory`：聚合当前团队当前项目最近 3 场已完成会议的决策、行动项和遗留问题；包含来源会议。
- `POST /api/projects`：创建项目文件夹，可传 `parent_id` 创建第二级；第三层会被拒绝。`PATCH /api/projects/{id}` 修改名称。
- `DELETE /api/projects/{id}`：默认把会议移入未分类；显式传入 `delete_meetings=true` 才连同终态会议、转写稿和报告删除。
- `POST /api/review`：multipart 字段 `file`、可选 `title` 和可选 `project_id`，成功返回 HTTP 202 与任务 ID；新版页面要求先选文件夹，API 保留未分类兼容能力。
- `GET /api/tasks/{task_id}`：先查内存中的处理中或 30 分钟内终态任务；内存未命中时回落 SQLite（重启后仍可读）；跨团队访问返回 403。
- `GET /api/meetings`：当前团队会议列表，可用 `project_id` 或 `unclassified=true` 过滤；一级目录查询可传 `include_children=true` 汇总二级目录。
- `GET /api/meetings/{id}`：从 SQLite 读取当前团队历史报告与转写片段，用于时间戳上下文；跨团队访问返回 403。
- `POST /api/meetings/{id}/image-minutes`：**生成图片纪要四板块（R-P1.5-1）**。幂等、只读渲染，不写任何业务数据；返回 `{meeting_id, title, meta, parts:[{key,title,subtitle,items,empty_note}]}`，`key` 固定为 `core/urgent/todo/mine`。无报告 404，跨团队 403。
- `GET /api/meetings/{id}/image-minutes.pdf`：**导出图片纪要 PDF**。响应 `application/pdf` + 中文文件名（RFC 5987）；渲染器不可用时返回 503 `renderer_unavailable`（可读中文原因），其它交付物不受影响。
- `POST /api/meetings/{id}/share`：**生成分享链接（R-P1.5-3）**。`{scopes:[...]}`，`scopes` ∈ `image_minutes/report/transcript/voice/tasks`；返回一次性可见的 `token` 与 `url`；空勾选或未知项 422、无报告 404、无口令 403。
- `GET /api/meetings/{id}/shares`：列出本场链接（`share_id` / 勾选范围 / 有效期 / 是否生效 / 访问次数 / 令牌前缀；**不回显令牌原文**）。`DELETE /api/meetings/{id}/shares/{share_id 或 token}` 撤销，立即失效，重复撤销 404。
- `GET /api/shares/{token}`：**免团队口令**读取分享内容；**只返回被勾选的键**；过期 410 `share_expired`、已撤销 410 `share_revoked`、不存在或篡改 404。`GET /api/shares/{token}/clips/{id}` 提供分享范围内的代表语音（未勾语音 403）。`GET /s/{token}` 是最小只读分享页。
- `POST /api/meetings/{id}/suggested-project`：**处置 AI 项目建议（R-P1.5-8）**。`{action:"accept", project_id}` 归入指定项目；`{action:"rename", name}` 新建项目后归入；`{action:"dismiss"}` 保持未分类并清除建议。三者都要求人触发，**不存在自动移动**；未知 action 422、缺参 422、项目不存在 404、别人的项目/会议 403。
- `GET /api/terms`：**术语热词表（R-P1.5-9）**。返回 `{items:[{id,term,note,source,updated_at}], prompt}`；`source` 为 `manual`（可删）或 `member`（来自已确认成员姓名，自动、不可删）。
- `POST /api/terms`：新增/更新热词（`{term, note?}`）；空词或超过 40 字 → 422 `invalid_args`。`DELETE /api/terms/{id}`：删除手动词，未知 id 404；成员来源的词删不掉。
- `GET /api/meetings/{id}/deliverables`：**四类交付物状态（R-P1.5-7）**。返回 `{meeting_id, needs_review, items:[{kind,label,status,error_code,message,retryable,updated_at}]}`；`kind` ∈ `transcript/report/tasks/image_minutes`，`status` ∈ `pending/ok/failed/needs_review`。跨团队 403，会议不存在 404。
- `POST /api/meetings/{id}/retry?kind=`：**按交付物重试，不重跑已成功的部分**。`image_minutes` 会真的再渲染一次 PDF；`tasks` 按现有报告重建；`report` 会**重新调用 AI**（失败写 `analysis_failed`）；`transcript` 返回 409 `not_retryable`；未知 kind 422。
- `GET /api/meetings/{id}/agent-trace`：**只读**返回这场会议里 Agent 的每一步工具调用（会话 / 步 / 工具 / 判定 / 结果 / 耗时），按 `team_id` 隔离，跨团队 403。它是页面第 ⑨ 节「Agent 步骤（只读）」的数据源，也是 M1 的产品验收入口。
- `PATCH /api/meetings/{id}/title`：用户接受或编辑 AI 建议标题后更新会议标题；跨团队访问返回 403。
- `PATCH /api/meetings/{id}/project`：经用户确认后移动到当前团队的另一文件夹，保留转写稿和报告。
- `PATCH /api/action-items/{id}`：把行动项状态更新为待确认、进行中、已完成或已取消；跨团队访问返回 403。
- `POST /api/meetings/{id}/speakers/{label}/confirm`：确认本场说话人；`remember_voice=true` 只标记待保存，不在此接口写入长期声纹。
- `POST /api/meetings/{id}/finalize`：完成整理并统一确认参会者授权；确认后才把待保存声纹写入团队身份库。
- `GET /api/speaker-clips/{id}`：读取历史报告中的代表性短音频；按团队隔离，跨团队返回 403。
- `GET /api/members`：列出当前团队身份库；`PATCH /api/members/{id}` 修改姓名、角色和关键决策人标记。
- `DELETE /api/members/{id}/voiceprint`：只删除该成员声纹，保留姓名与历史会议。
- `POST /api/members/{id}/merge`：把重复身份合并进同团队的目标成员，并同步明确匹配的历史责任人名称。

业务 API 都要带 `X-Access-Token`。每个响应包含 `X-Request-ID`。只有 `POST /api/review` 计入限频和每日名额；健康检查、鉴权检查、任务轮询和历史查询不计数。门禁顺序为：鉴权 → 限频 → 每日上限 → 队列上限 → 至少 1 GB 空闲磁盘 → 保存文件 → 入队，拒绝时不会留下上传文件。

## 环境变量

| 变量 | 必需 | 默认值 | 用途 |
|---|---:|---|---|
| `TEAM_TOKENS` | **是** | 无 | `团队名:口令,团队名:口令`；启动时解析并 upsert 到 `teams`，本地也必须配置。**口令强度：≥ 16 位，且不能是纯数字或纯英文；不合规服务直接拒绝启动**（见 R-P0-3） |
| `OPENAI_API_KEY` | **是** | 无 | OpenAI 兼容 LLM 密钥 |
| `OPENAI_BASE_URL` | 否 | SDK 默认 | 兼容服务 `/v1` 地址 |
| `OPENAI_MODEL` | 否 | `gpt-4o-mini` | 分析模型 |
| `WHISPER_MODEL` | 否 | `small` | faster-whisper 模型或本地路径；4 GB 内存服务器建议 `base` |
| `WHISPER_DEVICE` | 否 | `cpu` | `cpu` / `cuda` |
| `WHISPER_COMPUTE_TYPE` | 否 | `int8` | CPU 常用 `int8`，CUDA 可选 `float16` |
| `SPEAKER_RECOGNITION_ENABLED` | 否 | `true` | 是否执行本地说话人分离与声纹匹配 |
| `SPEAKER_MODEL` | 否 | `chinese` | WeSpeaker 中文模型名或本地模型目录 |
| `SPEAKER_MATCH_THRESHOLD` | 否 | `0.72` | 自动身份匹配的最低置信度，需真实录音校准 |
| `SPEAKER_MATCH_MARGIN` | 否 | `0.05` | 第一与第二候选的最小分差，防止相似声音误认 |
| `SPEAKER_INTRA_MERGE_THRESHOLD` | 否 | `0.85` | 同场被过度切分的说话人标签合并阈值（2026-09-21 用真实录音由 0.78 上调）；**两簇各自命中不同成员时一律不合并** |
| `CONSENT_VERSION` | 否 | `v1` | 上传同意协议版本号；改动同意文案时递增，旧留证记录不被覆盖（R-P1.5-6） |
| `PDF_RENDERER` | 否 | `auto` | 图片纪要导出 PDF 的渲染器：`auto`（Playwright 优先，回退本机 Chrome）/ `playwright` / `chrome` / `none` |
| `IMAGE_MINUTES_TEMPLATE` | 否 | `card_v1` | 图片纪要版式模板名（模板在 `app/deliverables/templates/`） |
| `DATABASE_PATH` | 否 | `data/meeting-review.db` | SQLite 路径；Compose 使用 `/data/meeting-review.db` |
| `MAX_UPLOAD_MB` | 否 | `300` | 上传大小上限（MB） |
| `MAX_AUDIO_MINUTES` | 否 | `240` | PyAV 与转写后双重时长上限（默认 4 小时） |
| `PROCESSING_TIMEOUT_SECONDS` | 否 | `21600` | 单任务开始处理后的超时（默认 6 小时） |
| `QUEUE_MAX` | 否 | `5` | 等待队列上限，不含正在处理的任务 |
| `TASK_RETENTION_MINUTES` | 否 | `30` | 终态任务状态的内存保留时间，不删除历史报告 |
| `RATE_LIMIT_PER_HOUR` | 否 | `10` | 单 IP 每小时创建任务上限 |
| `DAILY_TASK_LIMIT` | 否 | `30` | 按 `Asia/Shanghai` 自然日计算的全局任务上限 |
| `TRANSCRIPT_CHUNK_CHARS` | 否 | `6000` | 长文本分块阈值 |
| `LLM_MAX_RETRIES` | 否 | `2` | 结构或证据校验失败后的重试次数 |
| `LLM_MAX_CONCURRENCY` | 否 | `4` | 分块分析的并发上限，避免长会议无条件并发数十次调用 |
| `LLM_PRICE_PROMPT_PER_1K` | 否 | 空 | 输入 token 单价（每 1000）；留空时只记 token、`cost` 返回 `null`，价格不硬编码 |
| `LLM_PRICE_COMPLETION_PER_1K` | 否 | 空 | 输出 token 单价（每 1000）；同上 |
| `APP_HOST` / `APP_PORT` | 否 | `127.0.0.1` / `8000` | 监听地址与端口 |
| `FORWARDED_ALLOW_IPS` | 否 | `127.0.0.1` | Uvicorn 信任的直接代理 IP/网段 |
| `AGENT_MODE` | 否 | `pipeline` | 会议处理走哪条路径：`pipeline`（默认，零行为变化）\| `shadow`（并行对拍不落库）\| `agent`（模型自主编排）。P1-A 起生效 |
| `AGENT_MAX_STEPS` | 否 | `20` | 单会话最大步数，超限安全终止（阶段 5 使用） |
| `AGENT_CONTEXT_BUDGET_TOKENS` | 否 | `60000` | 上下文预算，超限触发压缩（阶段 7 使用） |
| `AGENT_TOOLS_ENABLED` | 否 | `readonly` | 允许模型调用的工具级别（阶段 5 使用） |
| `AGENT_WRITE_TOOLS_ENABLED` | 否 | `false` | 写工具总开关；P1 固定 false |
| `AGENT_AUDIT_ENABLED` | 否 | `true` | 是否写入 `agent_audit` 审计表 |
| `AGENT_STEP_TIMEOUT_SECONDS` | 否 | `120` | Agent 单步超时（一次模型调用或一次工具执行） |
| `AGENT_SESSION_TIMEOUT_SECONDS` | 否 | `1800` | Agent 单会话总超时 |
| `AGENT_GOAL_JUDGE` | 否 | `rule` | 完成判定评估器：`rule`（零成本可复现）\| `model`（按 `stage=goal_judge` 单独计费） |

## Agent 层（P1）

- `AGENT_MODE=pipeline`（默认）：完全走原有链路，行为不变。
- `AGENT_MODE=agent`：由模型自主决定读哪段转写、是否回读原文、何时停止；报告产出后仍由确定性代码做引文校验并落库。
- 执行前先用 `todo_write` 写计划；停手前由**独立完成判定**（默认规则评估器）确认是否真的完成，未达成则带缺失项继续，仍不达标则交还人（`needs_human`）。
- `AGENT_MODE=shadow`：pipeline 正常出结果并落库；Agent 也跑一遍但**不落库**，只记录关键差异，用于灰度对拍。
- 权限：Agent 只能调用只读工具；删除会议/项目/声纹、合并成员、写长期声纹属于 host-owned，**Agent 永远不可调用**。
- 审计：每次工具调用写入 `agent_audit`，只存参数摘要（非原文），可按 `session_id` 回放。
- 上下文：超预算时自动压缩较早的工具结果（`AGENT_CONTEXT_BUDGET_TOKENS`），保留最近结果与消息结构。
- 任务：处理状态同时写入 SQLite，服务重启后 `GET /api/tasks/{id}` 仍可读；临时音频还在时会排队续跑。

## 服务器部署（Ubuntu + Docker）

在干净 Ubuntu 上安装 Docker：

```bash
sudo apt update && sudo apt install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

部署项目：

```bash
git clone <YOUR_REPOSITORY_URL> meeting-review
cd meeting-review
cp .env.example .env
chmod 600 .env
nano .env   # 填写 TEAM_TOKENS、OPENAI_API_KEY 等
sudo docker compose up -d --build
sudo docker compose ps
sudo docker compose logs -f meeting-review
```

Compose 使用 `meeting-data` 保存 SQLite，使用 `whisper-models` 缓存模型权重。4 GB 内存服务器建议在 `.env` 设置 `WHISPER_MODEL=base`；`small` 处理 2-4 小时音频可能内存紧张。4 小时是接收上限，不是完成时间承诺；CPU 服务器的实际耗时需用真实录音压测。必须保持一个 Uvicorn worker/一个应用副本，内存队列和限频暂不支持横向扩容。公网前应在 Caddy/Nginx 配 HTTPS，并正确设置 `FORWARDED_ALLOW_IPS`，不要直接信任访客伪造的转发头。

## 隐私与已知限制

- 完整音频在转写和本地声纹特征提取完成后即删除；转写失败、分析失败、超时和优雅停机也会清理临时音频。文件以任务 ID 随机命名，原文件名不落盘。
- 用户选择保留代表性声音：每位说话人最多 3 段、每段最长 12 秒，与报告一起长期保存在本服务器 SQLite，按团队隔离，删除会议时级联删除。“完成整理”时仅统一授权一次。
- 经用户确认和授权的声纹 embedding 仅保存在本服务器 SQLite，按团队隔离，可在“团队声纹身份库”单独删除。未确认的本场候选特征会随会议数据保存，用于原音频删除后仍能完成人工确认；删除会议时一并删除。
- 转写文本与报告仅保存在本服务器 SQLite，按团队隔离，不会用于其他用途；分析时文本会发送给运营方配置的 AI 服务，其数据政策取决于服务商。
- 日志不记录原文件名、转写文本、模型引文、团队口令或 API Key，仅保留请求 ID、任务 ID、阶段、错误类型和 Token 用量。
- 声纹阈值 `0.72/0.05` 是安全初值，不是已校准的准确率承诺；设备、距离、噪声、重叠说话和太短片段都会影响识别，上线前必须用同人跨会议真实录音校准。
- 参会人数没有人为上限，但更多人、重叠发言、远场收音和噪声会降低转写与责任人判断质量；上线前需用不同人数的真实录音验证，不能把“人数不限”理解为准确率不受影响。
- Whisper 噪声会传导到要点、行动项和引用，例如专有名词识别错误；后续方向是术语词表注入与可追溯纠错 pass。
- 团队口令适合固定小团队 MVP，不等价于完整账号、成员权限和口令自助轮换体系。

## Roadmap

以下均为 MVP 后迭代项：

1. **AI 角色追问（persona follow-up）**：报告生成后选择严厉面试官、行业前辈等 AI 角色继续追问；作为付费增值层，复用现有转写稿与报告上下文增量实现。（原名「数字人二次讨论」，2026-09-18 因名实不符更名。）
2. **待办到期提醒**：利用任务/负责人/截止时间结构化数据接入邮件或企业微信机器人；依赖账号体系和通知通道，因此排在 MVP 验证之后。

## 测试与成本占位

```bash
# 统一用 `python -m` 调用，避免依赖 venv 里脚本的绝对路径
python -m pytest -q            # 本地完整环境：385 passed
python -m ruff check app tests # lint（E501 已按项目理由关闭，见 pyproject.toml）
python -m compileall -q app    # 语法编译
```

**CI（R-P0-1）**：`.github/workflows/ci.yml` 在 Python **3.9 / 3.11 / 3.12** 三版本上跑「语法编译 + pytest + ruff + 密钥扫描」。
CI 使用 `requirements-ci.txt` 轻量集合，跳过的那 1 项由测试自身的 `pytest.importorskip("torch")` 标为可选；完整 385 项在本地验证。

测试使用模拟转写和假 LLM，覆盖 legacy 契约、团队登录/upsert、团队隔离、历史查询、时长双检、队列、音频全分支清理、转写入库与日志脱敏；P0 新增口令强度、启动拒绝、密钥不入日志、认证路径健壮性（非 ASCII 口令返回 403 而非 500）、成本落库与归因、`json_schema` 能力缓存（避免重复的必然失败请求）。
LLM 每次调用记录 `stage/model/prompt_tokens/completion_tokens/total_tokens/duration_ms` **并写入 `llm_usage` 表**，经 `GET /api/usage` 按 stage / 会议 / 项目归因；真实会议成本待真实录音后回填：

声纹真实录音验收按 [docs/VOICEPRINT_TEST_SCRIPT.md](docs/VOICEPRINT_TEST_SCRIPT.md) 录制两场会议：第一场确认身份，第二场验证同人自动识别和新人待确认。

| 场景 | 输入 Token | 输出 Token | 单次成本 |
|---|---:|---:|---:|
| 30 分钟会议 | 待测 | 待测 | 待测 |
| 60 分钟会议 | 待测 | 待测 | 待测 |
| 2 小时会议 | 待测 | 待测 | 待测 |
| 4 小时会议 | 待测 | 待测 | 待测 |
