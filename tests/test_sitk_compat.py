from pathlib import Path

from src.sitk_compat import sitk_io_path, sitk_io_paths


def test_relative_path_is_preserved() -> None:
    assert sitk_io_path(Path("data") / "case" / "label.nii.gz") == str(
        Path("data") / "case" / "label.nii.gz"
    )


def test_non_ascii_absolute_path_uses_ascii_relative_path_when_possible(tmp_path: Path) -> None:
    # 构造与当前项目相同的“中文父目录 + ASCII 子路径”情形。
    project = tmp_path / "中文项目"
    target = project / "data" / "case" / "label.nii.gz"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"x")

    result = sitk_io_path(target.resolve(), cwd=project.resolve())

    assert result == str(Path("data") / "case" / "label.nii.gz")
    result.encode("ascii")


def test_ascii_absolute_path_is_not_rewritten(tmp_path: Path) -> None:
    target = (tmp_path / "data" / "label.nii.gz").resolve()
    # pytest tmp path 在当前工作站为 ASCII；若运行环境改变，仍只验证函数返回可用字符串。
    result = sitk_io_path(target, cwd=tmp_path)
    if str(target).isascii():
        assert result == str(target)


def test_batch_helper_keeps_order() -> None:
    paths = [Path("a.dcm"), Path("b.dcm")]
    assert sitk_io_paths(paths) == ["a.dcm", "b.dcm"]
