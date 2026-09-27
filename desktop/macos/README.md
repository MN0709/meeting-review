# 会脉 macOS 朋友内测版

这个原生 macOS 外壳会启动 App 内置的 FastAPI 后端，并在 WebKit 窗口中打开会脉。分发包内含 Python 运行时、后端代码、Whisper 与 WeSpeaker 模型，接收者不需安装 Python 或拿到项目源码。

## 构建

```bash
desktop/macos/build_backend.sh
desktop/macos/build_dmg.sh
```

输出：`dist/会脉-0.2.0-beta-arm64.dmg`。

## 接收者使用

1. 打开 DMG，把「会脉」拖到「Applications」。
2. 首次在 Finder 里右键「会脉」，选「打开」。
3. 先进入主界面浏览；需要复盘时，在顶部「AI 服务状态」栏填写使用者自己的 DeepSeek API Key。
4. 密钥保存在 macOS 钥匙串；会议数据保存在 `~/Library/Application Support/会脉/`。

## 边界

- 支持 Apple Silicon（arm64）和 macOS 13+。
- 当前是 ad-hoc 签名的内测包，没有 Apple 公证；顺滑双击打开需要 Developer ID 证书与公证。
- 每位使用者自备 DeepSeek API Key，模型费用由各自账号承担。
- 原始音频处理后删除；转写文本和报告会发给 DeepSeek；会议数据和代表性说话人片段保存在本机。
- 本次内测包不含 PDF 导出和自动更新。
