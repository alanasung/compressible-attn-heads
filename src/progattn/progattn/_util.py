"""Stage helpers shared inside the nested domain package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            if part not in cur:
                return default
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)
    return default if cur is None else cur


def git_sha_safe() -> str:
    try:
        from ..utils.git import git_sha

        return str(git_sha())
    except Exception:
        return "unknown"


def stage_result(
    *, task: str, seed: int, n: int, metrics: dict[str, Any], **extra: Any
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "task": task,
        "seed": int(seed),
        "git_sha": git_sha_safe(),
        "n": int(n),
        "metrics": metrics,
    }
    out.update(extra)
    return out


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
