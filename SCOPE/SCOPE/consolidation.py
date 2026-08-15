"""Runtime experience consolidation shared by pipeline prompts.

The learned guidance is kept in an experiment artifact instead of being written
back into prompt source files. This makes full-vs-ablation runs reproducible:
the full model loads and appends this state; the wo-consolidation variant leaves
it disabled.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_LOCK = threading.Lock()
_ENABLED = False
_PATH: Optional[Path] = None
_MAX_ITEMS = 24
_ITEMS: List[Dict[str, Any]] = []
_REVISION = 0


def configure(
    enabled: bool,
    path: Optional[str] = None,
    max_items: int = 24,
    reset: bool = False,
) -> None:
    """Enable/disable consolidation and load existing experience if present."""
    global _ENABLED, _PATH, _MAX_ITEMS, _ITEMS, _REVISION
    with _LOCK:
        _ENABLED = bool(enabled)
        _PATH = Path(path) if path else None
        _MAX_ITEMS = max(1, int(max_items or 24))
        if reset:
            _ITEMS = []
            _REVISION = 0
        elif _ENABLED and _PATH and _PATH.exists():
            try:
                obj = json.loads(_PATH.read_text(encoding="utf-8"))
                _ITEMS = _normalize_items(obj.get("items") or [])
                _REVISION = int(obj.get("revision") or len(_ITEMS))
            except Exception:
                _ITEMS = []
                _REVISION = 0


def enabled() -> bool:
    return _ENABLED


def snapshot() -> Dict[str, Any]:
    with _LOCK:
        return {
            "enabled": _ENABLED,
            "path": str(_PATH) if _PATH else None,
            "revision": _REVISION,
            "items": list(_ITEMS),
        }


def replace_items(items: List[Dict[str, Any]], source: str = "llm") -> Dict[str, Any]:
    """Replace the active guidance with normalized model-produced lessons."""
    global _ITEMS, _REVISION
    with _LOCK:
        _ITEMS = _normalize_items(items)[-_MAX_ITEMS:]
        _REVISION += 1
        state = {
            "enabled": _ENABLED,
            "revision": _REVISION,
            "updated_at": time.time(),
            "source": source,
            "items": list(_ITEMS),
        }
        if _PATH:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            _PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        return state


def render(stage: Optional[str] = None, max_items: int = 8) -> str:
    """Return a compact prompt block for a stage: routing, semantic, planning."""
    if not _ENABLED:
        return ""
    stage_norm = (stage or "").strip().lower()
    with _LOCK:
        items = list(_ITEMS)
        revision = _REVISION
    if not items:
        return ""

    selected: List[Dict[str, Any]] = []
    for item in reversed(items):
        item_stage = (item.get("stage") or "general").strip().lower()
        if item_stage in {"general", stage_norm}:
            selected.append(item)
        if len(selected) >= max_items:
            break
    if not selected:
        return ""

    lines = [
        "Learned experience from earlier questions in this run.",
        f"Revision: {revision}. Use as soft guidance; if it conflicts with the current question or source evidence, follow the evidence.",
    ]
    for i, item in enumerate(selected, start=1):
        stage_label = item.get("stage") or "general"
        lesson = item.get("lesson") or item.get("guidance") or ""
        when = item.get("when") or ""
        action = item.get("action") or ""
        evidence = item.get("evidence") or ""
        parts = [f"{i}. [{stage_label}]"]
        if when:
            parts.append(f"When {when}")
        if action:
            parts.append(f"prefer {action}")
        if lesson:
            parts.append(str(lesson))
        if evidence:
            parts.append(f"Evidence: {evidence}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _normalize_items(items: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        stage = str(raw.get("stage") or "general").strip().lower()
        if stage not in {"general", "routing", "semantic", "planning", "execution"}:
            stage = "general"
        lesson = str(raw.get("lesson") or raw.get("guidance") or "").strip()
        action = str(raw.get("action") or "").strip()
        when = str(raw.get("when") or "").strip()
        evidence = str(raw.get("evidence") or "").strip()
        if not (lesson or action):
            continue
        out.append({
            "stage": stage,
            "when": when[:240],
            "action": action[:240],
            "lesson": lesson[:500],
            "evidence": evidence[:360],
        })
    return out
