"""轻量 trace 记录器：捕获每次 LLM 调用 + 每个阶段的执行细节，按 idx_<N> 分目录落盘。

启用方式：
    1) 设置环境变量 DEEPSIEVE_TRACE_DIR=/path/to/traces 或在 runner 里 set_output_dir
    2) 主驱动按阶段调用 set_idx / set_stage / record_meta / dump_stage

无侵入式日志：call_openai_chat 内部会自动调用 record_llm()。
"""

import json
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

try:
    if "/root/autodl-tmp" not in sys.path:
        sys.path.insert(0, "/root/autodl-tmp")
    from _common.cost_counter import bump_llm as _bump_llm
except Exception:  # pragma: no cover
    def _bump_llm(*_a, **_kw): pass


class TraceRecorder:
    _instance: Optional["TraceRecorder"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._output_dir: Optional[str] = None
        self._tls = threading.local()
        self._buffer: Dict[str, Dict[str, Any]] = {}
        self._buffer_lock = threading.Lock()
        self._enabled = False

    @property
    def _cur_idx(self) -> Optional[int]:
        return getattr(self._tls, "idx", None)

    @_cur_idx.setter
    def _cur_idx(self, value: Optional[int]) -> None:
        self._tls.idx = value

    @property
    def _cur_stage(self) -> Optional[str]:
        return getattr(self._tls, "stage", None)

    @_cur_stage.setter
    def _cur_stage(self, value: Optional[str]) -> None:
        self._tls.stage = value

    @classmethod
    def instance(cls) -> "TraceRecorder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = TraceRecorder()
                    env_dir = os.environ.get("DEEPSIEVE_TRACE_DIR", "").strip()
                    if env_dir:
                        cls._instance.set_output_dir(env_dir)
        return cls._instance

    def set_output_dir(self, path: str) -> None:
        if not path:
            self._enabled = False
            self._output_dir = None
            return
        os.makedirs(path, exist_ok=True)
        self._output_dir = os.path.abspath(path)
        self._enabled = True

    def set_idx(self, idx: int) -> None:
        self._cur_idx = idx
        self._cur_stage = None

    def set_stage(self, stage: str) -> None:
        self._cur_stage = stage
        key = self._key(self._cur_idx, stage)
        with self._buffer_lock:
            if key not in self._buffer:
                self._buffer[key] = {"llm_calls": [], "meta": {}}

    def enabled(self) -> bool:
        return bool(self._enabled and self._output_dir is not None)

    def _key(self, idx: Optional[int], stage: Optional[str]) -> str:
        return f"{idx}::{stage}"

    def record_llm(
        self,
        prompt: str,
        response: str,
        model: str,
        max_tokens: int = 0,
        finish_reason: str = "stop",
        temperature: float = 0.0,
        elapsed: Optional[float] = None,
    ) -> None:
        _bump_llm(stage=self._cur_stage)
        if not self.enabled() or self._cur_idx is None or self._cur_stage is None:
            return
        key = self._key(self._cur_idx, self._cur_stage)
        with self._buffer_lock:
            if key not in self._buffer:
                self._buffer[key] = {"llm_calls": [], "meta": {}}
            self._buffer[key]["llm_calls"].append(
                {
                    "stage": self._cur_stage,
                    "ts": time.time(),
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "finish_reason": finish_reason,
                    "elapsed": elapsed,
                    "prompt": prompt,
                    "response": response,
                }
            )

    def record_meta(self, key: str, value: Any) -> None:
        if not self.enabled() or self._cur_idx is None or self._cur_stage is None:
            return
        bk = self._key(self._cur_idx, self._cur_stage)
        with self._buffer_lock:
            if bk not in self._buffer:
                self._buffer[bk] = {"llm_calls": [], "meta": {}}
            self._buffer[bk]["meta"][key] = value

    def merge_meta(self, mapping: Dict[str, Any]) -> None:
        if not self.enabled() or self._cur_idx is None or self._cur_stage is None:
            return
        for k, v in (mapping or {}).items():
            self.record_meta(k, v)

    def dump_stage(self, filename: str, extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """落盘当前 (idx, stage) 缓存到 traces/idx_<N>/<filename>，并清掉对应 buffer。"""
        if not self.enabled() or self._cur_idx is None or self._cur_stage is None:
            return None
        key = self._key(self._cur_idx, self._cur_stage)
        with self._buffer_lock:
            bucket = self._buffer.get(key, {"llm_calls": [], "meta": {}})

        idx_dir = os.path.join(self._output_dir, f"idx_{self._cur_idx}")
        os.makedirs(idx_dir, exist_ok=True)
        out_path = os.path.join(idx_dir, filename)

        payload: Dict[str, Any] = {}
        meta = bucket.get("meta") or {}
        payload.update(meta)
        if extra:
            payload.update(extra)
        payload["llm_calls"] = bucket.get("llm_calls", [])

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        with self._buffer_lock:
            self._buffer.pop(key, None)
        return out_path

    def reset_idx(self) -> None:
        if self._cur_idx is None:
            return
        prefix = f"{self._cur_idx}::"
        with self._buffer_lock:
            for k in list(self._buffer.keys()):
                if k.startswith(prefix):
                    self._buffer.pop(k, None)
        self._cur_idx = None
        self._cur_stage = None


def get_recorder() -> TraceRecorder:
    return TraceRecorder.instance()
