from __future__ import annotations
import ctypes
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import webview

APP_TITLE = "智骨云析｜AI 辅助骨科 CT 影像教学与三维实训平台"
APP_URL = "http://127.0.0.1:8000/"
HEALTH_URL = "http://127.0.0.1:8000/api/health"
PROJECT_ROOT = Path(os.environ.get("ZHIGU_PROJECT_ROOT", r"D:\国创项目"))
PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
_backend: subprocess.Popen | None = None
_owned_backend = False


def health_ready(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
            return payload.get("status") == "ok"
    except Exception:
        return False


def start_backend() -> None:
    global _backend, _owned_backend
    if health_ready():
        return
    if not PYTHON.exists():
        raise RuntimeError(f"找不到项目 Python 环境：{PYTHON}")
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    _backend = subprocess.Popen(
        [str(PYTHON), "-m", "uvicorn", "web.backend.app:app",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=flags,
    )
    _owned_backend = True
    deadline = time.time() + 45
    while time.time() < deadline:
        if _backend.poll() is not None:
            raise RuntimeError("Web 后端启动失败，进程已提前退出。")
        if health_ready():
            return
        time.sleep(0.35)
    raise RuntimeError("Web 后端启动超时，请检查项目运行环境。")


def stop_backend() -> None:
    global _backend
    if not _owned_backend or _backend is None or _backend.poll() is not None:
        return
    _backend.terminate()
    try:
        _backend.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _backend.kill()
        _backend.wait(timeout=3)


def show_error(message: str) -> None:
    ctypes.windll.user32.MessageBoxW(0, message, APP_TITLE, 0x10)


def main() -> int:
    try:
        start_backend()
    except Exception as exc:
        show_error(f"智骨云析启动失败：\n\n{exc}")
        stop_backend()
        return 1

    window = webview.create_window(
        APP_TITLE,
        APP_URL,
        width=1600,
        height=1000,
        min_size=(1100, 700),
        resizable=True,
        fullscreen=False,
        frameless=False,
        background_color="#07111f",
        text_select=True,
    )
    window.events.closed += stop_backend

    def maximize_window() -> None:
        time.sleep(0.15)
        window.maximize()

    try:
        webview.start(maximize_window, gui="edgechromium", debug=False)
    except Exception as exc:
        show_error(f"桌面窗口启动失败：\n\n{exc}")
        stop_backend()
        return 2
    finally:
        stop_backend()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())