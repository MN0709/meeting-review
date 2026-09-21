"""R-P1.5-1：HTML → PDF 渲染。

优先级（`PDF_RENDERER`）：
- `playwright`：Playwright Chromium（需 `playwright install chromium`）；
- `chrome`：本机已安装的 Chrome / Chromium / Edge 的 headless CLI（无需额外下载）；
- `auto`（默认）：先 Playwright，不可用则回退本机 Chrome；
- `none`：关闭 PDF 导出（接口返回明确错误，交付物标记「待核对」）。

任何渲染失败都抛 `RendererUnavailable`，由上层转成可读错误 + `needs_review`，
**绝不让整场会议显示失败**（R-P1.5-7）。
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)
RENDER_TIMEOUT_SECONDS = 60.0


class RendererUnavailable(RuntimeError):
    """没有可用的渲染器，或渲染过程失败。"""


def chrome_binary() -> Optional[str]:
    """找本机可用的 Chromium 内核浏览器（可用 CHROME_PATH 覆盖）。"""
    import os

    override = os.environ.get("CHROME_PATH")
    candidates: List[str] = ([override] if override else []) + list(CHROME_CANDIDATES)
    candidates += [shutil.which(name) or "" for name in ("google-chrome", "chromium", "chromium-browser")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


async def _render_with_playwright(html_text: str) -> bytes:
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:  # 未安装可选依赖
        raise RendererUnavailable("未安装 Playwright，无法导出 PDF") from exc
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch()
            try:
                page = await browser.new_page()
                await page.set_content(html_text, wait_until="load")
                return await page.pdf(format="A4", print_background=True)
            finally:
                await browser.close()
    except RendererUnavailable:
        raise
    except Exception as exc:
        raise RendererUnavailable("Playwright 渲染失败（{}）".format(type(exc).__name__)) from exc


async def _render_with_chrome(html_text: str) -> bytes:
    binary = chrome_binary()
    if binary is None:
        raise RendererUnavailable("本机没有可用的 Chrome / Chromium，无法导出 PDF")
    with tempfile.TemporaryDirectory(prefix="meeting-review-pdf-") as workdir:
        source = Path(workdir) / "minutes.html"
        target = Path(workdir) / "minutes.pdf"
        source.write_text(html_text, encoding="utf-8")
        command = [
            binary, "--headless=new", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer", "--print-to-pdf={}".format(target),
            source.as_uri(),
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise RendererUnavailable("无法启动 Chrome（{}）".format(type(exc).__name__)) from exc
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=RENDER_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            process.kill()
            raise RendererUnavailable("Chrome 导出 PDF 超时") from exc
        if process.returncode != 0 or not target.exists():
            detail = (stderr or b"").decode("utf-8", "ignore").strip().splitlines()
            raise RendererUnavailable(
                "Chrome 导出 PDF 失败{}".format("：" + detail[-1][:120] if detail else "")
            )
        return target.read_bytes()


async def render_pdf(html_text: str, *, renderer: str = "auto") -> bytes:
    """把 HTML 渲染成 PDF 字节；失败抛 `RendererUnavailable`（带可读中文原因）。"""
    if renderer == "none":
        raise RendererUnavailable("PDF 导出已关闭（PDF_RENDERER=none）")
    order = ["playwright", "chrome"] if renderer == "auto" else [renderer]
    problems: List[str] = []
    for name in order:
        try:
            if name == "playwright":
                return await _render_with_playwright(html_text)
            if name == "chrome":
                return await _render_with_chrome(html_text)
            problems.append("未知渲染器：{}".format(name))
        except RendererUnavailable as exc:
            problems.append(str(exc))
    raise RendererUnavailable("；".join(problems) or "没有可用的 PDF 渲染器")
