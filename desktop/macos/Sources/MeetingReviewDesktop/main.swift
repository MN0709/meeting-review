import AppKit
import Foundation
import Security
import UniformTypeIdentifiers
import WebKit

private let serviceURL = URL(string: "http://127.0.0.1:8000/")!
private let healthURL = URL(string: "http://127.0.0.1:8000/health")!
private let keychainService = "com.huimai.meetingreview.deepseek"
private let keychainAccount = "api-key"

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var backendProcess: Process?
    private var pollTimer: Timer?
    private var remainingHealthChecks = 450

    func applicationDidFinishLaunching(_ notification: Notification) {
        configureMenu()
        configureWindow()
        showLoadingPage(message: "正在启动会脉…")
        checkServiceAndStartIfNeeded()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        pollTimer?.invalidate()
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: "apiKey")
        if let process = backendProcess, process.isRunning {
            process.terminate()
        }
    }

    private func configureMenu() {
        let menu = NSMenu()
        let appItem = NSMenuItem()
        menu.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "关于会脉", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        let apiKeyItem = appMenu.addItem(withTitle: "设置 DeepSeek API Key…", action: #selector(configureAPIKey), keyEquivalent: ",")
        apiKeyItem.target = self
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "退出会脉", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        NSApp.mainMenu = menu
    }

    @objc private func configureAPIKey() {
        webView.evaluateJavaScript(
            "document.getElementById('apiKeyBar')?.scrollIntoView({behavior:'smooth',block:'center'});"
                + "document.getElementById('apiKeyInput')?.focus();"
        )
    }

    private func loadAPIKey() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: keychainAccount,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func saveAPIKey(_ value: String) -> Bool {
        let identity: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: keychainService,
            kSecAttrAccount as String: keychainAccount,
        ]
        let attributes: [String: Any] = [kSecValueData as String: Data(value.utf8)]
        let status = SecItemUpdate(identity as CFDictionary, attributes as CFDictionary)
        if status == errSecSuccess { return true }
        guard status == errSecItemNotFound else { return false }
        var item = identity
        item[kSecValueData as String] = Data(value.utf8)
        return SecItemAdd(item as CFDictionary, nil) == errSecSuccess
    }

    private func configureWindow() {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .default()
        configuration.userContentController.add(self, name: "apiKey")
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = self
        webView.uiDelegate = self
        webView.allowsMagnification = true

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 820),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "会脉 · 我的会议记忆"
        window.titlebarAppearsTransparent = true
        window.minSize = NSSize(width: 960, height: 680)
        window.contentView = webView
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.name == "apiKey",
              let payload = message.body as? [String: Any],
              payload["action"] as? String == "save" else { return }
        let value = (payload["value"] as? String ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else {
            notifyAPIKeySave(success: false, message: "API Key 不能为空")
            return
        }
        validateAndSaveAPIKey(value)
    }

    private func validateAndSaveAPIKey(_ value: String) {
        var request = URLRequest(url: URL(string: "https://api.deepseek.com/models")!)
        request.timeoutInterval = 20
        request.setValue("Bearer \(value)", forHTTPHeaderField: "Authorization")
        URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            Task { @MainActor in
                guard let self else { return }
                if error != nil {
                    self.notifyAPIKeySave(success: false, message: "无法连接 DeepSeek，请检查网络后重试")
                    return
                }
                let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
                guard statusCode == 200 else {
                    let message = statusCode == 401 || statusCode == 403
                        ? "API Key 无效或已失效，请重新检查"
                        : "DeepSeek 验证失败（\(statusCode)），请检查账户权限与余额"
                    self.notifyAPIKeySave(success: false, message: message)
                    return
                }
                guard self.saveAPIKey(value) else {
                    self.notifyAPIKeySave(success: false, message: "保存失败，请检查系统钥匙串权限")
                    return
                }
                self.notifyAPIKeySave(success: true, message: "Key 验证通过，正在重启本地服务…")
                self.restartBackend()
            }
        }.resume()
    }

    private func notifyAPIKeySave(success: Bool, message: String) {
        guard let data = try? JSONSerialization.data(
            withJSONObject: ["success": success, "message": message]
        ), let json = String(data: data, encoding: .utf8) else { return }
        webView.evaluateJavaScript("window.huimaiApiKeySaveResult(\(json))")
    }

    private func checkServiceAndStartIfNeeded() {
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 1
        URLSession.shared.dataTask(with: request) { [weak self] _, response, _ in
            let isHealthy = (response as? HTTPURLResponse)?.statusCode == 200
            Task { @MainActor in
                guard let self else { return }
                if isHealthy {
                    self.loadApp()
                } else {
                    self.startBackend()
                }
            }
        }.resume()
    }

    func webView(
        _ webView: WKWebView,
        runOpenPanelWith parameters: WKOpenPanelParameters,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable ([URL]?) -> Void
    ) {
        let panel = NSOpenPanel()
        panel.title = parameters.allowsMultipleSelection ? "选择会议录音" : "选择录音文件"
        panel.prompt = "选择"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = parameters.allowsMultipleSelection
        panel.allowedContentTypes = [.audio]
        panel.beginSheetModal(for: window) { response in
            completionHandler(response == .OK ? panel.urls : nil)
        }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptAlertPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable () -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = "会脉提示"
        alert.informativeText = message
        alert.addButton(withTitle: "知道了")
        alert.beginSheetModal(for: window) { _ in
            completionHandler()
        }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptConfirmPanelWithMessage message: String,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable (Bool) -> Void
    ) {
        let alert = NSAlert()
        alert.messageText = "请确认"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: "继续")
        alert.addButton(withTitle: "取消")
        alert.beginSheetModal(for: window) { response in
            completionHandler(response == .alertFirstButtonReturn)
        }
    }

    func webView(
        _ webView: WKWebView,
        runJavaScriptTextInputPanelWithPrompt prompt: String,
        defaultText: String?,
        initiatedByFrame frame: WKFrameInfo,
        completionHandler: @escaping @MainActor @Sendable (String?) -> Void
    ) {
        let alert = NSAlert()
        let input = NSTextField(string: defaultText ?? "")
        input.frame = NSRect(x: 0, y: 0, width: 320, height: 24)
        alert.messageText = "请输入"
        alert.informativeText = prompt
        alert.accessoryView = input
        alert.addButton(withTitle: "确定")
        alert.addButton(withTitle: "取消")
        alert.beginSheetModal(for: window) { response in
            completionHandler(response == .alertFirstButtonReturn ? input.stringValue : nil)
        }
    }

    private func startBackend() {
        guard backendProcess == nil else { return }
        guard let resourcesURL = Bundle.main.resourceURL else {
            showFailurePage(title: "找不到会脉资源", detail: "请重新安装会脉。")
            return
        }
        let backendURL = resourcesURL.appendingPathComponent("Backend/meeting-review-backend")
        let whisperModelURL = resourcesURL.appendingPathComponent("Models/faster-whisper-small")
        let speakerModelURL = resourcesURL.appendingPathComponent("Models/wespeaker-chinese")
        guard FileManager.default.isExecutableFile(atPath: backendURL.path),
              FileManager.default.fileExists(atPath: whisperModelURL.path),
              FileManager.default.fileExists(atPath: speakerModelURL.path) else {
            showFailurePage(
                title: "找不到会脉运行环境",
                detail: "安装包不完整，请重新下载并把“会脉”拖到应用程序文件夹。"
            )
            return
        }

        let fileManager = FileManager.default
        guard let supportRoot = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            showFailurePage(title: "无法创建数据目录", detail: "系统没有返回应用支持目录。")
            return
        }
        let appSupportURL = supportRoot.appendingPathComponent("会脉", isDirectory: true)
        let dataURL = appSupportURL.appendingPathComponent("data", isDirectory: true)
        let cacheURL = appSupportURL.appendingPathComponent("cache", isDirectory: true)
        do {
            try fileManager.createDirectory(at: dataURL, withIntermediateDirectories: true)
            try fileManager.createDirectory(at: cacheURL, withIntermediateDirectories: true)
        } catch {
            showFailurePage(title: "无法创建数据目录", detail: error.localizedDescription)
            return
        }

        let process = Process()
        process.executableURL = backendURL
        process.currentDirectoryURL = appSupportURL
        var environment = ProcessInfo.processInfo.environment
        if let apiKey = loadAPIKey(), !apiKey.isEmpty {
            environment["OPENAI_API_KEY"] = apiKey
        } else {
            environment.removeValue(forKey: "OPENAI_API_KEY")
        }
        environment["OPENAI_BASE_URL"] = "https://api.deepseek.com"
        environment["OPENAI_MODEL"] = "deepseek-flash"
        environment["APP_HOST"] = "127.0.0.1"
        environment["APP_PORT"] = "8000"
        environment["AUTH_ENABLED"] = "false"
        environment["DATABASE_PATH"] = dataURL.appendingPathComponent("meeting-review.db").path
        environment["WHISPER_MODEL"] = whisperModelURL.path
        environment["SPEAKER_MODEL"] = speakerModelURL.path
        environment["HF_HOME"] = cacheURL.appendingPathComponent("huggingface", isDirectory: true).path
        environment["WESPEAKER_HOME"] = cacheURL.appendingPathComponent("wespeaker", isDirectory: true).path
        environment["PDF_RENDERER"] = "none"
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        process.terminationHandler = { [weak self] process in
            guard process.terminationStatus != 0 else { return }
            Task { @MainActor in
                self?.showFailurePage(title: "本地服务未能启动", detail: "请回到项目目录检查运行环境，或把此页面截图发给开发者。")
            }
        }

        do {
            try process.run()
            backendProcess = process
            pollForService()
        } catch {
            showFailurePage(title: "本地服务未能启动", detail: error.localizedDescription)
        }
    }

    private func restartBackend() {
        pollTimer?.invalidate()
        pollTimer = nil
        remainingHealthChecks = 450
        showLoadingPage(message: "正在应用 API Key…")
        guard let process = backendProcess, process.isRunning else {
            backendProcess = nil
            startBackend()
            return
        }
        backendProcess = nil
        process.terminationHandler = { [weak self] _ in
            Task { @MainActor in
                self?.startBackend()
            }
        }
        process.terminate()
    }

    private func pollForService() {
        pollTimer?.invalidate()
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { [weak self] timer in
            guard let self else {
                timer.invalidate()
                return
            }
            Task { @MainActor in
                self.runHealthCheck()
            }
        }
    }

    private func runHealthCheck() {
        guard remainingHealthChecks > 0 else {
            pollTimer?.invalidate()
            showFailurePage(title: "启动超时", detail: "本地服务 3 分钟内没有就绪。请退出会脉后重试。")
            return
        }
        remainingHealthChecks -= 1
        var request = URLRequest(url: healthURL)
        request.timeoutInterval = 1
        URLSession.shared.dataTask(with: request) { [weak self] _, response, _ in
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { return }
            Task { @MainActor in
                self?.loadApp()
            }
        }.resume()
    }

    private func loadApp() {
        pollTimer?.invalidate()
        pollTimer = nil
        webView.load(URLRequest(url: serviceURL, cachePolicy: .reloadIgnoringLocalCacheData))
    }

    private func showLoadingPage(message: String) {
        let html = shellPage(title: message, detail: "会议数据只保存在这台 Mac 上。", isError: false)
        webView.loadHTMLString(html, baseURL: nil)
    }

    private func showFailurePage(title: String, detail: String) {
        pollTimer?.invalidate()
        let html = shellPage(title: title, detail: detail, isError: true)
        webView.loadHTMLString(html, baseURL: nil)
    }

    private func shellPage(title: String, detail: String, isError: Bool) -> String {
        let safeTitle = title.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;")
        let safeDetail = detail.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: "\n", with: "<br>")
        let accent = isError ? "#b4473d" : "#222428"
        return """
        <!doctype html><meta charset="utf-8"><style>
        body{margin:0;background:#f2f3f4;color:#17191d;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;display:grid;place-items:center;min-height:100vh}
        main{width:min(520px,calc(100% - 48px));padding:46px;border-radius:30px;background:#fff;box-shadow:0 24px 70px rgba(20,24,30,.08);text-align:center}
        i{display:grid;width:66px;height:66px;margin:0 auto 24px;place-items:center;border-radius:22px;background:\(accent);color:white;font-style:normal;font-weight:800;font-size:24px}
        h1{margin:0;font-size:25px}p{margin:12px 0 0;color:#737981;line-height:1.7}
        </style><main><i>脉</i><h1>\(safeTitle)</h1><p>\(safeDetail)</p></main>
        """
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.run()
