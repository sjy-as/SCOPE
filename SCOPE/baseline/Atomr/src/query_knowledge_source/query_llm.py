"""LLM 调用入口。

旧实现走本地 Flask 中转 (openai_service.py:9998)；新实现直接用 OpenAI SDK 连
任意 OpenAI-compatible 网关（chatanywhere / deepseek / sophnet ...）。

线程安全：cache 字典 + 计数器 + cache 文件追加都用锁保护，可在 ThreadPoolExecutor
里多线程调用同一个 OpenAICaller。

接口 query_deepseek(prompt, model, temperature, max_tokens, n, use_cache) 保持向后兼容。
"""

from typing import Optional
import json
import os
import re
import threading
import time

try:
    from openai import OpenAI
    import httpx as _httpx
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore
    _httpx = None  # type: ignore

try:
    # 让 OpenAICaller 不必显式依赖 trace_recorder：失败时静默
    import sys as _sys
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _src_dir not in _sys.path:
        _sys.path.append(_src_dir)
    from trace_recorder import get_recorder as _get_trace_recorder  # type: ignore
except Exception:
    _get_trace_recorder = None


def count_prompt_tokens(prompt: str) -> int:
    """轻量 prompt token 估算，用于成本统计；不追求精确。"""
    if not prompt:
        return 0
    return len(re.findall(r"\S+", str(prompt)))


class OpenAICaller():
    """OpenAI-compatible 网关客户端。

    构造参数（任一为 None 时回退到环境变量）：
      base_url   ─ OPENAI_BASE_URL
      api_key    ─ OPENAI_API_KEY
      model      ─ ATOMR_LLM_MODEL  (默认 deepseek-chat)
      cache_path ─ ATOMR_LLM_CACHE_PATH
      use_cache  ─ ATOMR_USE_LLM_CACHE=1 才启用
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_path: Optional[str] = "../../openai_service/llm_cache/cache.jsonl",
        # 旧字段兼容：之前的代码用 api_url 指向本地 Flask
        api_url: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ):
        # 解析连接参数
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "")).strip() or None
        self.api_key = (api_key or os.environ.get("OPENAI_API_KEY", "")).strip() or None
        self.default_model = (model or os.environ.get("ATOMR_LLM_MODEL", "deepseek-chat")).strip()

        # 兼容老接口：调用方仍可能传 api_url=http://127.0.0.1:9998/...，此时仅做提示并忽略
        if api_url and not self.base_url:
            print(f"[OpenAICaller] 注意：旧 api_url={api_url} 已忽略；请用 --llm-url 或 OPENAI_BASE_URL 指定 base_url。")

        if OpenAI is None:
            raise RuntimeError("openai SDK 未安装：pip install openai")
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "OpenAICaller 缺少 base_url 或 api_key。请通过 CLI --llm-url/--api-key "
                "或环境变量 OPENAI_BASE_URL/OPENAI_API_KEY 提供。"
            )

        # 默认绕开环境代理 (HTTP_PROXY/HTTPS_PROXY)；如果你确实要走系统代理，
        # 设置 ATOMR_LLM_TRUST_ENV=1 即可
        trust_env = os.environ.get("ATOMR_LLM_TRUST_ENV", "0") == "1"
        if _httpx is not None and not trust_env:
            http_client = _httpx.Client(trust_env=False, timeout=_httpx.Timeout(300.0, connect=15.0))
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=http_client)
        else:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # 缓存
        if use_cache is None:
            use_cache = os.getenv("ATOMR_USE_LLM_CACHE", "0") == "1"
        self.use_cache = bool(use_cache)
        self.cache_path = cache_path or os.environ.get("ATOMR_LLM_CACHE_PATH", "../../openai_service/llm_cache/cache.jsonl")
        self.cache = {}
        self._cache_lock = threading.Lock()
        self._cache_file_lock = threading.Lock()
        if self.use_cache:
            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            except Exception:
                pass
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            self.cache[tuple(entry["input"])] = entry["response"]
                        except Exception:
                            continue

        # 全局统计
        self.total_prompt_tokens = 0
        self.total_llm_calls = 0
        self._stat_lock = threading.Lock()

    def query_deepseek(self, prompt, model=None, temperature=0, max_tokens=10000, n=1, use_cache=None):
        """与旧实现保持同名同参数（默认模型回退到 self.default_model）。"""
        effective_use_cache = self.use_cache if use_cache is None else bool(use_cache)
        model_name = (model or "").strip() or self.default_model

        with self._stat_lock:
            self.total_llm_calls += 1
            self.total_prompt_tokens += count_prompt_tokens(prompt)

        cache_key = (prompt, model_name, max_tokens)
        t0 = time.time()

        if effective_use_cache and temperature == 0:
            with self._cache_lock:
                cached = self.cache.get(cache_key)
            if cached is not None:
                response_text = cached[0]['message']['content']
                finish_reason = cached[0]['finish_reason']
                self._trace_record(prompt, response_text, model_name, max_tokens,
                                   finish_reason, temperature, time.time() - t0, from_cache=True)
                return response_text, finish_reason

        # 直连上游
        try:
            resp = self._client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
        except Exception as e:
            raise Exception(f"[OpenAI Call Error]: {e}")

        try:
            resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else resp
            choice0 = resp_dict["choices"][0]
            finish_reason = choice0.get("finish_reason", "stop")
            response_text = choice0["message"]["content"]
        except Exception as e:
            raise Exception(f"[OpenAI Response Parse Error]: {e}; raw={resp}")

        if effective_use_cache and temperature == 0:
            with self._cache_lock:
                if cache_key not in self.cache:
                    self.cache[cache_key] = [choice0]
                    if self.cache_path:
                        try:
                            with self._cache_file_lock:
                                with open(self.cache_path, "a") as f:
                                    f.write("%s\n" % json.dumps({"input": list(cache_key), "response": [choice0]}))
                        except Exception:
                            pass

        self._trace_record(prompt, response_text, model_name, max_tokens,
                           finish_reason, temperature, time.time() - t0, from_cache=False)
        return response_text, finish_reason

    def _trace_record(self, prompt, response_text, model, max_tokens, finish_reason, temperature, elapsed, from_cache=False):
        if _get_trace_recorder is None:
            return
        try:
            rec = _get_trace_recorder()
            if not rec.enabled():
                return
            rec.record_llm(
                prompt=prompt,
                response=response_text,
                model=model,
                max_tokens=max_tokens,
                finish_reason=finish_reason,
                temperature=temperature,
                elapsed=elapsed,
            )
        except Exception:
            pass


if __name__ == "__main__":
    s = time.time()
    caller = OpenAICaller(
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model=os.environ.get("ATOMR_LLM_MODEL", "deepseek-chat"),
    )
    response, finish_reason = caller.query_deepseek(prompt="What is 3+2? Explain briefly.", max_tokens=10000, temperature=0)
    print("response:", response)
    print("finish_reason:", finish_reason)
    print("Time:", time.time() - s)
