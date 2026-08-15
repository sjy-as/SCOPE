'''
python /root/autodl-tmp/CCC_de_pro/new_model/step2_decompose/sq.py \
  --input /root/autodl-tmp/CCC_de_pro/new_model/qa_bench/kg-table-160.jsonl \
  --output /root/autodl-tmp/CCC_de_pro/new_model/step2_decompose/output/sq-kg-table-160.jsonl \
  --base-url https://api.chatanywhere.tech/v1 \
  --model deepseek-chat \
  --api-key "$LLM_API_KEY" \
  --verbose
'''


import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# =========================
# IO / LLM utilities
# =========================

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def call_llm(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 10000,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "n": 1,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=120, proxies={"http": None, "https": None})
    if resp.status_code >= 400:
        print(f"\n[!!! 触发 API HTTP {resp.status_code} 错误 !!!]")
        print(f"请求的模型名: {model}")
        print(f"服务器返回的具体原因: {resp.text}\n")

    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            return {}
    return {}


def call_and_parse_with_retry(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    required_keys: Optional[List[str]] = None,
    max_retries: int = 3,
) -> Tuple[Dict[str, Any], str]:
    last_error_msg = ""
    for attempt in range(max_retries):
        try:
            raw = call_llm(prompt, model=model, base_url=base_url, api_key=api_key)
            parsed = parse_json_block(raw)

            if not parsed:
                last_error_msg = "Empty or invalid JSON."
                print(f"[Warning] Attempt {attempt + 1}/{max_retries} failed: {last_error_msg}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue

            if required_keys:
                missing = [k for k in required_keys if k not in parsed]
                if missing:
                    last_error_msg = f"JSON missing critical keys {missing}."
                    print(f"[Warning] Attempt {attempt + 1}/{max_retries} failed: {last_error_msg}")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                    continue

            return parsed, raw

        except Exception as e:
            last_error_msg = f"Network or API Error: {str(e)}"
            print(f"[Warning] Attempt {attempt + 1}/{max_retries} failed with error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    print(f"[Error] All {max_retries} attempts failed. Returning fallback.")
    return {}, f"Failed after {max_retries} attempts. Last error: {last_error_msg}"


# =========================
# Decomposition
# =========================

def decompose_question(question: str, model: str, base_url: str, api_key: str) -> Dict[str, Any]:
    prompt = f"""You are a query decomposition assistant.

        Your job is to decide whether ONE natural-language semantic query needs
        decomposition and, only when useful, split it into the minimum number of
        executable sub-questions.

        Requirements:
        1. The input is already a natural semantic query. Do NOT rewrite it into a different intent.
        2. Decide adaptively whether decomposition is needed and how many sub-questions are necessary.
        3. If the query can be executed directly, return exactly one sub-question containing the original query.
        4. If decomposition is needed, return the minimum sufficient number of sub-questions (q1, q2, ...).
        5. Order sub-questions so every dependency is produced before it is consumed.
        6. A dependent sub-question must use [k] to refer to the answer of qk and must list qk in depends_on.
        7. Each sub-question must be independently executable once its declared dependencies are available.
        8. Do not answer the query. Do not do event judgment, schema routing, source selection, or query rewriting.
        9. Keep every sub-question faithful to the original meaning and avoid redundant steps.
        10. Return valid JSON only.

        Return format:
        {
        "should_decompose": true or false,
        "reasoning": "brief explanation of the decomposition",
        "subqueries": [
            {
            "id": "q1",
            "query": "...",
            "depends_on": [],
            "variables": []
            },
            {
            "id": "q2",
            "query": "... [1] ...",
            "depends_on": ["q1"],
            "variables": [{"name": "ref_1", "source_query": "q1"}]
            }
        ]
        }

        The second item above only illustrates a decomposed query; add zero or
        more dependent items as needed. For a direct query, return one item q1,
        set should_decompose to false, and copy the original query exactly into
        q1.query.

        Question: {question}

    """

    parsed, raw = call_and_parse_with_retry(prompt, model, base_url, api_key, required_keys=["subqueries"])
    subqueries = parsed.get("subqueries") if isinstance(parsed, dict) else None

    if not isinstance(subqueries, list) or not subqueries:
        return {
            "reasoning": parsed.get("reasoning") if isinstance(parsed, dict) else "",
            "raw": raw,
            "should_decompose": False,
            "subqueries": [{
                "id": "q1",
                "query": question,
                "depends_on": [],
                "variables": [],
            }],
        }

    normalized: List[Dict[str, Any]] = []
    for raw_sq in subqueries:
        if not isinstance(raw_sq, dict):
            continue
        query = str(raw_sq.get("query") or "").strip()
        if not query:
            continue
        sq_id = f"q{len(normalized) + 1}"
        allowed_ids = {f"q{j}" for j in range(1, len(normalized) + 1)}
        dependencies = raw_sq.get("depends_on") or []
        if not isinstance(dependencies, list):
            dependencies = []
        dependencies = [str(dep) for dep in dependencies if str(dep) in allowed_ids]
        variables = raw_sq.get("variables") or []
        if not isinstance(variables, list):
            variables = []
        normalized.append({
            "id": sq_id,
            "query": query,
            "depends_on": dependencies,
            "variables": variables,
        })

    if not normalized:
        normalized = [{"id": "q1", "query": question, "depends_on": [], "variables": []}]

    return {
        "reasoning": parsed.get("reasoning", "") if isinstance(parsed, dict) else "",
        "raw": raw,
        "should_decompose": len(normalized) > 1,
        "subqueries": normalized,
    }


# =========================
# Main transform
# =========================

def process_item(
    item: Dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    question = (item.get("question") or "").strip()
    if verbose:
        print(f"\n[idx={item.get('index')}] Original: {question}")

    decomposition = decompose_question(question, model=model, base_url=base_url, api_key=api_key)
    subqueries = decomposition.get("subqueries") or []
    q1 = ((subqueries[0] or {}).get("query") if len(subqueries) > 0 else question) or question
    q2 = ((subqueries[1] or {}).get("query") if len(subqueries) > 1 else "") or ""
    reasoning = decomposition.get("reasoning", "") if isinstance(decomposition, dict) else ""
    raw = decomposition.get("raw", "") if isinstance(decomposition, dict) else ""

    if verbose:
        print(f"-> SQ1: {q1}")
        for i, sq_item in enumerate(subqueries[1:], start=2):
            print(f"-> SQ{i}: {sq_item.get('query', '')}")
        if reasoning:
            print(f"-> Reasoning: {reasoning}")

    return {
        "index": item.get("index"),
        "type": item.get("type"),
        "mode": item.get("mode"),
        "question": question,
        "sub_q1": q1,
        "sub_q2": q2,
        "subqueries": subqueries,
        "should_decompose": len(subqueries) > 1,
        "num_subqueries": len(subqueries),
        "reasoning": reasoning
    }


def run(
    input_path: str,
    output_path: str,
    model: str,
    base_url: str,
    api_key: str,
    limit: Optional[int] = None,
    start: int = 0,
    verbose: bool = False,
) -> None:
    rows = load_jsonl(input_path)
    if start > 0:
        rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]

    total = len(rows)
    iterator = tqdm(rows, total=total, desc="Decomposing", unit="q") if tqdm is not None else rows

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(iterator, start=1):
            output = process_item(row, model=model, base_url=base_url, api_key=api_key, verbose=verbose)
            f.write(json.dumps(output, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            if verbose:
                print(f"-> Wrote output {idx}/{total}: {output.get('index')}")
                print(f"   SQ1: {output.get('sub_q1')}")
                print(f"   SQ2: {output.get('sub_q2')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Natural query to adaptive decomposition")
    parser.add_argument("--input", default="/root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl")
    parser.add_argument("--output", default="/root/autodl-tmp/new_model//root/autodl-tmp/new_model/step2_decompose/output/sq.jsonl")
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step execution logs")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key provided. Use --api-key or export LLM_API_KEY.")

    run(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        limit=args.limit,
        start=args.start,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
