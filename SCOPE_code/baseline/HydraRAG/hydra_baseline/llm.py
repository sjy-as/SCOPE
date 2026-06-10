"""OpenAI 兼容的 LLM 客户端，带重试与线程安全的调用计数。

方法名 `query_gpt4o` 与拷贝自 new_model 的 kg_retriever.py 保持一致，
这样 KGRetriever 内部的关系映射调用无需改动即可工作。
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Dict, List, Tuple

import requests

try:
    if "/root/autodl-tmp" not in sys.path:
        sys.path.insert(0, "/root/autodl-tmp")
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
            raise ValueError("LLMClient 需要 api_key（用 --api-key 或环境变量 LLM_API_KEY 提供）")
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
        # 线程局部的调用日志：每个问题处理前后由 pipeline snapshot
        self._tls = threading.local()

        # AutoDL 等环境会设 *_proxy 环境变量（如 127.0.0.1:7897），
        # 该代理对部分 LLM 站点 SSL 握手失败 —— trust_env=False 绕过代理直连。
        self._session = requests.Session()
        self._session.trust_env = False

    # ------------------------------------------------------------------ #
    #  per-question trace                                                #
    # ------------------------------------------------------------------ #
    def start_trace(self) -> None:
        """开始记录当前线程的 LLM 调用（每个问题开头调用）。"""
        self._tls.calls = []

    def pop_trace(self) -> List[dict]:
        """取出并清空当前线程的调用记录。"""
        calls = getattr(self._tls, "calls", [])
        self._tls.calls = []
        return calls

    # ------------------------------------------------------------------ #
    #  main entry                                                        #
    # ------------------------------------------------------------------ #
    def query_gpt4o(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stage: str = "",
    ) -> Tuple[str, Dict]:
        """调用 LLM，返回 (生成文本, usage)。失败重试 max_retries 次后返回 ("", {})。"""
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

        # per-question trace（线程局部，无需锁）
        calls = getattr(self._tls, "calls", None)
        if calls is not None:
            calls.append({
                "stage": stage,
                "prompt": prompt,
                "response": content,
                "error": last_err if not content else "",
            })
        return content, usage
