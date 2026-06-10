import time
import requests
from typing import Dict, List, Tuple


"""MMQA Document (passage) query client.

Thin HTTP wrapper around the shared ColBERT passage-retrieval service
(default port 1215, started by table_service/setup_service_nba_datalake.py).

Mirrors `query_table.py`: `execute_doc_query` returns `(clean_title_list,
full_results)` where `full_results["organic_results"]` is a list of normalized
hits sharing the same shape Table hits use, so `DocRetrieverRAG` can mirror
`TableRetrieverRAG` without any special-casing downstream.
"""


def _split_title_body(raw_text: str) -> Tuple[str, str]:
    """ColBERT's collection stores each passage as 'Title | body'.

    The title always comes first, so split on the FIRST separator only.
    """
    text = str(raw_text or "").strip()
    if not text:
        return "", ""
    idx = text.find(" | ")
    if idx != -1:
        return text[:idx].strip(), text[idx + 3:].strip()
    idx = text.find("|")
    if idx != -1:
        return text[:idx].strip(), text[idx + 1:].strip()
    return "", text


def _normalize_doc_hit(hit: Dict, rank: int) -> Dict:
    raw_text = str(hit.get("text") or hit.get("passage") or hit.get("content") or "")
    title, body = _split_title_body(raw_text)
    doc_id = hit.get("pid", hit.get("doc_id", hit.get("id", "")))

    if not title:
        title = f"passage_{rank}"

    snippet_parts = []
    if doc_id != "":
        snippet_parts.append(f"Doc ID: {doc_id}")
    if title:
        snippet_parts.append(f"Title: {title}")
    snippet = " ; ".join(snippet_parts).strip()

    return {
        "rank": hit.get("rank", rank),
        "title": title,
        "snippet": snippet,
        "source": "doc",
        "link": "",
        # full passage text goes into article_text so the existing formatter
        # (which prints `article_text`) can consume it unchanged.
        "article_text": body or raw_text,
        "doc_id": str(doc_id),
        "score": hit.get("score"),
        "prob": hit.get("prob"),
    }


def execute_doc_query(query: str, doc_api_url: str, k: int = 5,
                      max_retries: int = 3) -> Tuple[List[str], Dict]:
    """Query the ColBERT passage service and normalize the top-k passages.

    Returns (clean_title_list, {"organic_results": [...]}).
    """
    params = {"query": query, "k": k}

    data = {}
    for attempt in range(max_retries):
        try:
            resp = requests.get(doc_api_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as e:
            print(f"[query_doc] API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                data = {}

    hits = data.get("topk", []) or data.get("results", []) or []

    clean_title_list: List[str] = []
    normalized_hits: List[Dict] = []
    for i, hit in enumerate(hits[:k], start=1):
        if not isinstance(hit, dict):
            continue
        normalized = _normalize_doc_hit(hit, i)
        normalized_hits.append(normalized)
        if normalized["title"]:
            clean_title_list.append(normalized["title"])

    full_results = {"organic_results": normalized_hits}
    return clean_title_list, full_results
