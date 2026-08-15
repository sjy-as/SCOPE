"""
utils/llm_call.py

Call OpenAI-compatible chat completions with:
- proxy bypass by default (env vars HTTP_PROXY/HTTPS_PROXY are ignored;
  set DEEPSIEVE_LLM_TRUST_ENV=1 to opt back in),
- application-level exponential-backoff retries for SSL / connection /
  read-timeout errors that survive urllib3's transport-level retry,
- automatic trace_recorder integration for prompt/response capture.
"""

import os
import time
import requests

try:
    from utils.trace_recorder import get_recorder as _get_trace_recorder
except Exception:
    try:
        from trace_recorder import get_recorder as _get_trace_recorder  # type: ignore
    except Exception:
        _get_trace_recorder = None


def _trust_env_enabled() -> bool:
    """Default: ignore HTTP(S)_PROXY env vars. Set DEEPSIEVE_LLM_TRUST_ENV=1 to opt-in."""
    return os.environ.get("DEEPSIEVE_LLM_TRUST_ENV", "0").strip() == "1"


def _trace_record_llm(prompt, response_text, model, finish_reason, elapsed):
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
            max_tokens=0,
            finish_reason=finish_reason,
            temperature=0.0,
            elapsed=elapsed,
        )
    except Exception:
        pass


_TRANSIENT_EXC = (
    requests.exceptions.SSLError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ReadTimeout,
    requests.exceptions.ConnectTimeout,
    requests.exceptions.Timeout,
)


def call_openai_chat(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_retries: int = 5,
    timeout: int = 300,
) -> str:
    """Call OpenAI-compatible chat completions endpoint.

    Returns the assistant message content on success, or ``""`` after exhausting
    retries. The caller can inspect the print log for ``[ALL RETRIES FAILED]``
    to disambiguate "LLM returned non-JSON" from "network died".
    """

    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    trust_env = _trust_env_enabled()

    last_err = None
    t_start = time.time()

    for attempt in range(max_retries):
        session = requests.Session()
        # Bypass system proxy by default — autodl/containers often inject
        # HTTP_PROXY/HTTPS_PROXY which break TLS to chatanywhere under concurrency.
        session.trust_env = trust_env
        proxies = None if trust_env else {"http": None, "https": None}

        try:
            t0 = time.time()
            response = session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                proxies=proxies,
            )
            response.raise_for_status()
            resp_json = response.json()
            choice0 = resp_json["choices"][0]
            content = choice0["message"]["content"]
            finish_reason = choice0.get("finish_reason", "stop")
            _trace_record_llm(prompt, content, model, finish_reason, time.time() - t0)
            return content
        except _TRANSIENT_EXC as e:
            last_err = e
            backoff = min(2 ** attempt, 30)
            print(f"🟠 transient network err (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e} → retry in {backoff}s")
            time.sleep(backoff)
            continue
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status in (408, 429, 500, 502, 503, 504):
                last_err = e
                backoff = min(2 ** attempt, 30)
                print(f"🟠 HTTP {status} (attempt {attempt + 1}/{max_retries}) → retry in {backoff}s")
                time.sleep(backoff)
                continue
            print(f"🔴 HTTP error (no retry): {e}")
            _trace_record_llm(prompt, "", model, f"http_{status}", time.time() - t_start)
            return ""
        except Exception as e:
            print(f"🔴 unexpected error: {type(e).__name__}: {e}")
            _trace_record_llm(prompt, "", model, "error", time.time() - t_start)
            return ""
        finally:
            session.close()

    print(f"🔴 [ALL RETRIES FAILED] after {max_retries} attempts; last_err={type(last_err).__name__ if last_err else 'None'}: {last_err}")
    _trace_record_llm(prompt, "", model, "all_retries_failed", time.time() - t_start)
    return ""
