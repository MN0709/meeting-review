# 团队长会复盘器

> 当前 10 天产品化迭代的状态、风险和每日验收入口见 [PROJECT_CONTROL.md](PROJECT_CONTROL.md)，详细路线见 [docs/ROADMAP_10_DAYS.md](docs/ROADMAP_10_DAYS.md)。

`meeting-review` 面向需要沉淀连续内部会议的团队，不限定团队职能或参会人数，同一系列不同会议可以有不同参会者。团队上传 MP3、M4A 或 WAV 录音，服务在本机完成带时间戳转写，再生成会议要点、带原话证据的决策清单和结构化行动项。转写稿、报告与会议元数据按团队隔离并长期保存在 SQLite；原始音频在转写完成或处理失败后删除。

页面使用团队口令登录。口令保存在浏览器 `localStorage`，全部业务请求通过 `X-Access-Token` 发送，不使用 Cookie。上传后页面每 2 秒查询任务状态；会议完成后也可从团队历史页重新打开报告。

## 页面与团队报告

- 上传：可填写会议标题，支持 MP3/M4A/WAV，默认最大 300 MB、60 分钟。
- 项目文件夹：支持最多两级（如“客户项目 / 2026年度”）；新上传先选择人工创建的文件夹，标题仍可不填；历史会议可按文件夹或未分类筛选。
- 会议归档调整：历史页允许用户手动选择另一个文件夹，确认后移动；转写稿和报告不变，AI 不会自动执行移动。
- 删除文件夹时必须选择：保留会议并移入“未分类”，或连同会议、转写稿和报告一起删除。永久删除前会二次警告且不可恢复；有二级文件夹的上级目录及含处理中会议的目录禁止连带删除。
- 处理进度：排队中、转写中、AI 分析中、完成或失败；超过 30 分钟的已知音频会提示用户可以关闭页面，完成后从历史记录查看。
- 会议历史：仅列出当前团队的会议标题、日期、时长，可点击读取历史报告。
- 报告：①会议要点；②决策清单（内容、决策人、逐字引文、时间戳）；③行动项（任务、负责人、截止时间）；④发言统计占位“说话人识别将于下一版本支持”。本版本不生成任何说话人统计数字。

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
  |-- POST /api/review（标题 + 音频）
  |-- GET /api/tasks/{id}（仅处理中/短期任务）
  `-- GET /api/meetings[/{id}]（当前团队历史）
                         |
鉴权 -> IP 限频 -> 每日上限 -> 队列上限 -> 磁盘检查
                         |
                UUID 临时音频 + PyAV 时长预检
                         |
                   FIFO 单 worker
                         |
              faster-whisper 带时间戳转写
                         |-- 立即删除原始音频
                         |-- segments 写入 SQLite
                         `-- 代码层二次时长检查
                                  |
                 OpenAI-compatible LLM
                 <=6000 字直接分析；更长文本并发分块再归并
                 JSON Schema / JSON Object 降级 + Pydantic
                 决策引文子串和时间戳严格校验（±5 秒）
                                  |
                 团队报告写入 SQLite，任务标记完成
```

## 技术选型

- **FastAPI + Pydantic**：接口与模型共享严格契约。
- **SQLite**：MVP 单机部署无需额外服务，会议、转写和报告可跨重启保留；所有资源查询都带 `team_id`。
- **PyAV + faster-whisper**：先读容器元数据秒拒超长音频，再在本机完成中文转写；转写后的真实时长仍会二次校验。
- **OpenAI 兼容协议**：Key、Base URL、模型名全部由环境变量提供；不支持 JSON Schema 时降级 JSON Object 并注入字段契约。
- **严格证据校验**：决策引文必须是某个原始 segment 的完整子串，时间戳须在该 segment 范围内（允许 5 秒容差），失败会触发修复重试。
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
# 编辑 .env，至少填写 TEAM_TOKENS 和 OPENAI_API_KEY
python run.py
```

打开 <http://127.0.0.1:8000>。健康检查无需团队口令：

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

## API

- `GET /health`：免鉴权健康检查。
- `GET /api/auth/check`：校验团队口令并返回团队 ID，错误为 403。
- `GET /api/projects`：列出当前团队项目文件夹及会议数量。
- `POST /api/projects`：创建项目文件夹，可传 `parent_id` 创建第二级；第三层会被拒绝。`PATCH /api/projects/{id}` 修改名称。
- `DELETE /api/projects/{id}`：默认把会议移入未分类；显式传入 `delete_meetings=true` 才连同终态会议、转写稿和报告删除。
- `POST /api/review`：multipart 字段 `file`、可选 `title` 和可选 `project_id`，成功返回 HTTP 202 与任务 ID；新版页面要求先选文件夹，API 保留未分类兼容能力。
- `GET /api/tasks/{task_id}`：只查询内存中的处理中或 30 分钟内终态任务；跨团队访问返回 403。
- `GET /api/meetings`：当前团队会议列表，可用 `project_id` 或 `unclassified=true` 过滤。
- `GET /api/meetings/{id}`：从 SQLite 读取当前团队历史报告；跨团队访问返回 403。
- `PATCH /api/meetings/{id}/project`：经用户确认后移动到当前团队的另一文件夹，保留转写稿和报告。

业务 API 都要带 `X-Access-Token`。每个响应包含 `X-Request-ID`。只有 `POST /api/review` 计入限频和每日名额；健康检查、鉴权检查、任务轮询和历史查询不计数。门禁顺序为：鉴权 → 限频 → 每日上限 → 队列上限 → 至少 1 GB 空闲磁盘 → 保存文件 → 入队，拒绝时不会留下上传文件。

## 环境变量

| 变量 | 必需 | 默认值 | 用途 |
|---|---:|---|---|
| `TEAM_TOKENS` | **是** | 无 | `团队名:口令,团队名:口令`；启动时解析并 upsert 到 `teams`，本地也必须配置 |
| `OPENAI_API_KEY` | **是** | 无 | OpenAI 兼容 LLM 密钥 |
| `OPENAI_BASE_URL` | 否 | SDK 默认 | 兼容服务 `/v1` 地址 |
| `OPENAI_MODEL` | 否 | `gpt-4o-mini` | 分析模型 |
| `WHISPER_MODEL` | 否 | `small` | faster-whisper 模型或本地路径；4 GB 内存服务器建议 `base` |
| `WHISPER_DEVICE` | 否 | `cpu` | `cpu` / `cuda` |
| `WHISPER_COMPUTE_TYPE` | 否 | `int8` | CPU 常用 `int8`，CUDA 可选 `float16` |
| `DATABASE_PATH` | 否 | `data/meeting-review.db` | SQLite 路径；Compose 使用 `/data/meeting-review.db` |
| `MAX_UPLOAD_MB` | 否 | `300` | 上传大小上限（MB） |
| `MAX_AUDIO_MINUTES` | 否 | `60` | PyAV 与转写后双重时长上限 |
| `PROCESSING_TIMEOUT_SECONDS` | 否 | `5400` | 单任务开始处理后的超时（秒） |
| `QUEUE_MAX` | 否 | `5` | 等待队列上限，不含正在处理的任务 |
| `TASK_RETENTION_MINUTES` | 否 | `30` | 终态任务状态的内存保留时间，不删除历史报告 |
| `RATE_LIMIT_PER_HOUR` | 否 | `10` | 单 IP 每小时创建任务上限 |
| `DAILY_TASK_LIMIT` | 否 | `30` | 按 `Asia/Shanghai` 自然日计算的全局任务上限 |
| `TRANSCRIPT_CHUNK_CHARS` | 否 | `6000` | 长文本分块阈值 |
| `LLM_MAX_RETRIES` | 否 | `2` | 结构或证据校验失败后的重试次数 |
| `APP_HOST` / `APP_PORT` | 否 | `127.0.0.1` / `8000` | 监听地址与端口 |
| `FORWARDED_ALLOW_IPS` | 否 | `127.0.0.1` | Uvicorn 信任的直接代理 IP/网段 |

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

Compose 使用 `meeting-data` 保存 SQLite，使用 `whisper-models` 缓存模型权重。4 GB 内存服务器建议在 `.env` 设置 `WHISPER_MODEL=base`；`small` 处理长音频可能内存紧张。必须保持一个 Uvicorn worker/一个应用副本，内存队列和限频暂不支持横向扩容。公网前应在 Caddy/Nginx 配 HTTPS，并正确设置 `FORWARDED_ALLOW_IPS`，不要直接信任访客伪造的转发头。

## 隐私与已知限制

- 音频转写完成后即删除；转写失败、分析失败、超时和优雅停机也会清理临时音频。文件以任务 ID 随机命名，原文件名不落盘。
- 转写文本与报告仅保存在本服务器 SQLite，按团队隔离，不会用于其他用途；分析时文本会发送给运营方配置的 AI 服务，其数据政策取决于服务商。
- 日志不记录原文件名、转写文本、模型引文、团队口令或 API Key，仅保留请求 ID、任务 ID、阶段、错误类型和 Token 用量。
- 本轮 `speaker_label` 存 `NULL`，不做说话人识别或发言统计；“与上次会议待办衔接”待说话人分离上线后再做。
- 参会人数没有人为上限，但更多人、重叠发言、远场收音和噪声会降低转写与责任人判断质量；上线前需用不同人数的真实录音验证，不能把“人数不限”理解为准确率不受影响。
- Whisper 噪声会传导到要点、行动项和引用，例如专有名词识别错误；后续方向是术语词表注入与可追溯纠错 pass。
- 团队口令适合固定小团队 MVP，不等价于完整账号、成员权限和口令自助轮换体系。

## Roadmap

以下均为 MVP 后迭代项：

1. **数字人二次讨论**：报告生成后选择严厉面试官、行业前辈等 AI 角色继续追问；作为付费增值层，复用现有转写稿与报告上下文增量实现。
2. **待办到期提醒**：利用任务/负责人/截止时间结构化数据接入邮件或企业微信机器人；依赖账号体系和通知通道，因此排在 MVP 验证之后。

## 测试与成本占位

```bash
pytest -q
docker build -t meeting-review .
docker compose config
```

测试使用模拟转写和假 LLM，覆盖 legacy 契约、团队登录/upsert、团队隔离、历史查询、时长双检、队列、音频全分支清理、转写入库与日志脱敏。LLM 每次调用会记录 `stage/model/prompt_tokens/completion_tokens/total_tokens`，后续用真实会议填写成本：

| 场景 | 输入 Token | 输出 Token | 单次成本 |
|---|---:|---:|---:|
| 30 分钟会议 | 待测 | 待测 | 待测 |
| 60 分钟会议 | 待测 | 待测 | 待测 |
