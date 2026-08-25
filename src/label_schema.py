"""椎体标签 schema 的轻量读取与显示工具。

当前默认 schema 只用于工程 QC / Web / preflight 的可读显示，不决定正式论文任务
是 binary、multi-class semantic 还是 instance segmentation。
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs" / "label_schemas" / "ctspine1k_verse.json"


@lru_cache(maxsize=8)
def load_label_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, object]:
    schema_path = Path(path)
    if not schema_path.is_absolute():
        schema_path = PROJECT_ROOT / schema_path
    if not schema_path.exists():
        raise FileNotFoundError(schema_path)
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError("label schema 缺少 labels 映射")
    normalized: dict[str, str] = {}
    for key, value in labels.items():
        label_id = int(key)
        if label_id <= 0:
            raise ValueError("前景 label id 必须为正整数")
        name = str(value).strip()
        if not name:
            raise ValueError(f"label {label_id} 名称为空")
        normalized[str(label_id)] = name
    result = dict(payload)
    result["labels"] = normalized
    return result


def label_name(label_id: int, *, schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> str | None:
    label_id = int(label_id)
    if label_id == 0:
        return "background"
    labels = load_label_schema(schema_path)["labels"]
    assert isinstance(labels, dict)
    value = labels.get(str(label_id))
    return None if value is None else str(value)


def label_display(label_id: int, *, schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> str:
    name = label_name(label_id, schema_path=schema_path)
    if name is None:
        return f"label {int(label_id)}"
    if int(label_id) == 0:
        return "background (0)"
    return f"{name} ({int(label_id)})"


def label_items(
    values: Iterable[int],
    *,
    include_background: bool = False,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for value in sorted({int(v) for v in values}):
        if value == 0 and not include_background:
            continue
        items.append(
            {
                "value": value,
                "name": label_name(value, schema_path=schema_path),
                "display": label_display(value, schema_path=schema_path),
            }
        )
    return items
