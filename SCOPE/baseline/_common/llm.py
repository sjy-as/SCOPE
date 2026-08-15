"""OpenAI-compatible LLM client.

Mirrors baseline/HydraRAG/hydra_baseline/llm.py so the new baselines look
and feel identical to the existing ones (thread-safe call counter,
per-thread trace, retries, no env proxy).
"""
from __future__ import annotations

import sys
from pathlib import Path
import threading
import time
from typing import Dict, List, Tuple

import requests

try:
    code_root = str(Path(__file__).resolve().parents[2])
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from _common.cost_counter import bump_llm as _bump_llm
except Exception:  # pragma: no cover
    def _bump_llm(*_a, **_kw): pass


class LLMClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 3,
        timeout: int = 300,
        verbose: bool = False,
    ):
        if not api_key:
            raise ValueError("LLMClient requires api_key (--api-key or env LLM_API_KEY)")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.verbose = verbose

        self._lock = threading.Lock()
        self.call_count = 0
        self.token_in = 0
        self.token_out = 0
        self._tls = threading.local()

        self._session = requests.Session()
        self._session.trust_env = False

    def start_trace(self) -> None:
        self._tls.calls = []

    def pop_trace(self) -> List[dict]:
        calls = getattr(self._tls, "calls", [])
        self._tls.calls = []
        return calls

    def query_gpt4o(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stage: str = "",
    ) -> Tuple[str, Dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "think": False,
        }

        last_err = ""
        content, usage = "", {}
        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers, json=payload, timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {}) or {}
                break
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                if self.verbose:
                    print(f"[LLM] attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 * (attempt + 1))

        with self._lock:
            self.call_count += 1
            self.token_in += int(usage.get("prompt_tokens", 0) or 0)
            self.token_out += int(usage.get("completion_tokens", 0) or 0)
        _bump_llm(stage=stage)

        calls = getattr(self._tls, "calls", None)
        if calls is not None:
            calls.append({
                "stage": stage,
                "prompt": prompt,
                "response": content,
                "error": last_err if not content else "",
            })
        return content, usage
