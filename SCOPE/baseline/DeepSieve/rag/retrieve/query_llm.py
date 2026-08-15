"""Querying LLM via local proxy or OpenAI-compatible endpoint.

Hardened for concurrent (e.g. ThreadPoolExecutor) calls behind a flaky local
proxy:
- bypass HTTP_PROXY/HTTPS_PROXY by default (DEEPSIEVE_LLM_TRUST_ENV=1 to opt-in),
- application-level exponential-backoff retry for SSL/Connection/Timeout errors,
- accept either OPENAI_API_BASE or OPENAI_BASE_URL for the gateway URL.
"""
from typing import Optional, Dict, Any, Tuple
import json
import os
import time

import requests


_TRANSIENT_EXC = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.Timeout,
)


def _trust_env_enabled() -> bool:
    return os.environ.get("DEEPSIEVE_LLM_TRUST_ENV", "0").strip() == "1"


def _post_with_retry(url: str, headers: Dict[str, str], payload: Dict[str, Any],
                     timeout: int = 120, max_retries: int = 5) -> requests.Response:
    """POST with proxy bypass + exponential backoff on transient errors."""
    trust_env = _trust_env_enabled()
    proxies = None if trust_env else {"http": None, "https": None}

    last_err = None
    for attempt in range(max_retries):
        session = requests.Session()
        session.trust_env = trust_env
        try:
            resp = session.post(url, headers=headers, json=payload,
                                timeout=timeout, proxies=proxies)
            return resp
        except _TRANSIENT_EXC as e:
            last_err = e
            backoff = min(2 ** attempt, 30)
            print(f"🟠 [query_llm] transient err ({type(e).__name__}) attempt {attempt + 1}/{max_retries} → retry in {backoff}s")
            time.sleep(backoff)
            continue
        finally:
            session.close()

    raise Exception(f"[query_llm all retries failed] last_err={type(last_err).__name__}: {last_err}")


class OpenAICaller:
    def __init__(
        self,
        api_url: Optional[str] = None,
        cache_path: Optional[str] = "../../openai_service/llm_cache/cache.jsonl",
    ):
        self.cache = {}
        self.cache_path = cache_path
        self.cache_enabled = str(os.environ.get("ATOMR_LLM_DISABLE_CACHE", "0")).strip().lower() not in {"1", "true", "yes", "on"}

        if self.cache_enabled:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            self.cache[tuple(entry["input"])] = entry["response"]
                        except Exception:
                            continue

        # Legacy local proxy endpoint (project historical behavior)
        self.proxy_api_url = api_url or os.environ.get("ATOMR_LLM_API_URL", "").strip()

        # OpenAI-compatible settings: accept either OPENAI_API_BASE or OPENAI_BASE_URL.
        env_base = (os.environ.get("OPENAI_API_BASE", "").strip()
                    or os.environ.get("OPENAI_BASE_URL", "").strip())
        env_key = os.environ.get("OPENAI_API_KEY", "").strip()
        env_model = (os.environ.get("OPENAI_MODEL", "").strip()
                     or os.environ.get("ATOMR_KG_PARSER_MODEL", "").strip()
                     or "deepseek-chat")

        # Be robust to accidental swap: KEY contains URL and BASE contains token.
        if env_key.startswith("http") and env_base and (not env_base.startswith("http")):
            env_key, env_base = env_base, env_key

        self.openai_api_base = env_base
        self.openai_api_key = env_key
        self.default_model = env_model

    def _use_openai_compatible(self) -> bool:
        return bool(self.openai_api_base and self.openai_api_key)

    @staticmethod
    def _build_chat_completions_url(base: str) -> str:
        b = base.rstrip("/")
        if b.endswith("/chat/completions"):
            return b
        if b.endswith("/v1"):
            return f"{b}/chat/completions"
        return f"{b}/v1/chat/completions"

    def _call_openai_compatible(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        n: int,
    ) -> Tuple[str, str]:
        url = self._build_chat_completions_url(self.openai_api_base)
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n,
        }

        response = _post_with_retry(url, headers=headers, payload=payload, timeout=300)
        if response.status_code != 200:
            raise Exception(f"[OpenAI-Compatible Response Error]: {response.status_code} {response.text}")

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise Exception(f"[OpenAI-Compatible Parse Error]: choices empty, response={data}")

        first = choices[0]
        finish_reason = first.get("finish_reason", "")
        message = first.get("message") or {}
        response_text = message.get("content", "")
        return response_text, finish_reason

    def _call_legacy_proxy(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        n: int,
    ) -> Tuple[str, str]:
        if not self.proxy_api_url:
            raise Exception(
                "[OpenAI Call Error]: no ATOMR_LLM_API_URL and no OPENAI_API_BASE/OPENAI_API_KEY configured"
            )

        response = _post_with_retry(
            self.proxy_api_url,
            headers={"Content-Type": "application/json"},
            payload={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "n": n,
            },
            timeout=300,
        )
        if response.status_code != 200:
            raise Exception(f"[OpenAI Response Error]: {response.status_code} {response.text}")

        choices = response.json().get("choices") or []
        if not choices:
            raise Exception(f"[OpenAI Call Error]: choices empty, response={response.text}")

        finish_reason = choices[0].get("finish_reason", "")
        response_text = (choices[0].get("message") or {}).get("content", "")
        return response_text, finish_reason

    def query_deepseek(
        self,
        prompt,
        model="deepseek-chat",
        temperature=0,
        max_tokens=128,
        n=1,
        use_cache=True,
    ):
        model = (model or self.default_model or "deepseek-chat").strip()
        key = (prompt, model, max_tokens)

        if self.cache_enabled and use_cache and temperature == 0 and key in self.cache:
            cache_response = self.cache[key]
            response_text = cache_response[0]["message"]["content"]
            finish_reason = cache_response[0].get("finish_reason", "")
            return response_text, finish_reason

        try:
            if self._use_openai_compatible():
                response_text, finish_reason = self._call_openai_compatible(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
            else:
                response_text, finish_reason = self._call_legacy_proxy(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
            cache_item = {
                "message": {"content": response_text},
                "finish_reason": finish_reason,
            }
        except Exception as e:
            raise Exception("[OpenAI Call Error]:", e)

        if self.cache_enabled and temperature == 0 and key not in self.cache:
            self.cache[key] = [cache_item]
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write("%s\n" % json.dumps({"input": key, "response": [cache_item]}, ensure_ascii=False))

        return response_text, finish_reason


if __name__ == "__main__":
    s = time.time()

    openai_caller = OpenAICaller()
    print("Testing LLM caller...")
    response, finish_reason = openai_caller.query_deepseek(
        prompt="What is 3+2? Explain your answer.",
        max_tokens=200,
        temperature=0.7,
        n=1,
        use_cache=False,
    )
    print("response:", response)
    print("finish_reason:", finish_reason)
    print("Time:", time.time() - s)
