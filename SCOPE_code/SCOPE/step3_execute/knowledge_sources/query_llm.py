# step3_execute/knowledge_sources/query_llm.py
import os
import time
import requests


class LLMClient:
    """OpenAI-compatible chat client.

    Defaults to the chatanywhere proxy endpoint. The local HTTP proxy
    (http(s)_proxy env vars) breaks the TLS handshake to this host, so every
    request explicitly bypasses the proxy with proxies={"http": None, ...}.
    """

    DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"
    DEFAULT_API_KEY = "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"

    def __init__(self, api_key=None, base_url=None, model="deepseek-chat",
                 timeout: int = 60, max_retries: int = 4):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or self.DEFAULT_API_KEY
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("LLM API Key must be provided.")

    def query_gpt4o(self, prompt: str, max_tokens: int = 10000) -> tuple[str, dict]:
        """统一的 LLM 调用接口。

        返回: (生成文本, token 使用统计)。失败时返回 ("", {})。
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,  # 过滤和推理任务通常需要低温度保证确定性
        }

        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    # 绕过本地代理：代理会掐断到 API 主机的 TLS 连接。
                    proxies={"http": None, "https": None},
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                usage = result.get("usage", {})
                return content, usage
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(2)

        print(f"[LLM Error] 请求失败 (after {self.max_retries} attempts): {last_err}")
        return "", {}
