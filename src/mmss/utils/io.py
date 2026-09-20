"""JSON read/write helpers for chunks in data/processed/ and other pipeline intermediates."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_jsonable(obj: Any) -> Any:
    return asdict(obj) if is_dataclass(obj) and not isinstance(obj, type) else obj


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_json_list(objs: list[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([_to_jsonable(o) for o in objs], f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
