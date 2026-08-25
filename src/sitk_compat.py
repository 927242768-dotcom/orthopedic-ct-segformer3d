"""SimpleITK 在 Windows 非 ASCII 路径下的轻量兼容层。

当前项目位于 ``D:\\国创项目``。实测 SimpleITK 2.3.1 的 Windows 构建对包含中文的
**绝对路径**可能报 ``Unable to open``，而在项目根目录运行时使用等价的 ASCII 相对路径
（例如 ``data/processed/...``）可以正常读写。

本模块只做路径字符串选择，不复制/移动医学数据，也不改变 SimpleITK 图像内容。
"""

from __future__ import annotations

import os
from pathlib import Path


def _is_ascii(value: str) -> bool:
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def sitk_io_path(path: str | Path, *, cwd: str | Path | None = None) -> str:
    """返回更适合当前 Windows SimpleITK 构建的路径字符串。

    规则：
    1. 调用方本来就传相对路径时保持相对路径；
    2. 绝对路径本身全 ASCII 时直接使用；
    3. 绝对路径含非 ASCII 时，尝试相对于当前工作目录转换；只有得到的相对路径
       全 ASCII 才使用它；
    4. 无法得到安全相对路径时回退原绝对路径，让 SimpleITK 自己报出真实 I/O 错误。

    因此项目脚本应继续从项目根目录启动（``web/run_web.ps1`` 已保证这一点）。
    """
    value = Path(path)
    if not value.is_absolute():
        return os.fspath(value)

    absolute = os.fspath(value)
    if _is_ascii(absolute):
        return absolute

    base = Path.cwd() if cwd is None else Path(cwd)
    try:
        relative = os.path.relpath(value, start=base)
    except (OSError, ValueError):
        return absolute
    return relative if _is_ascii(relative) else absolute


def sitk_io_paths(paths: list[str | Path] | tuple[str | Path, ...]) -> list[str]:
    """批量版本，适用于 ``ImageSeriesReader.SetFileNames``。"""
    return [sitk_io_path(path) for path in paths]
