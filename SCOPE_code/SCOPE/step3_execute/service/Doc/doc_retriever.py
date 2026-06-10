from typing import List, Optional, Tuple
import time
import requests

try:
    from _common.cost_counter import bump_retrieval as _bump_retrieval
except Exception:  # pragma: no cover
    def _bump_retrieval(*_a, **_kw): pass


class DocRetriever:
    """Client wrapper for the ColBERT document (passage) retrieval service.

    Mirrors service/Table/table_retriever.py: it talks to the Flask service
    started by setup_service_nba_datalake.py (default port 1215) and normalizes
    the raw top-k passages into evidence dicts the DocSource layer consumes.
    """

    def __init__(self, api_url: str = "http://127.0.0.1:1215/api/search", timeout: int = 60):
        self.api_url = api_url
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  Service calls                                                      #
    # ------------------------------------------------------------------ #

    def retrieve_topk(self, query: str, k: int = 5, max_retries: int = 3) -> Optional[List[dict]]:
        """Retrieve the raw top-k passages with a simple retry mechanism."""
        _bump_retrieval("doc")
        params = {"query": query, "k": k}

        for attempt in range(max_retries):
            try:
                resp = requests.get(url=self.api_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                return payload.get("topk") or payload.get("results") or payload.get("data") or []
            except requests.exceptions.RequestException as e:
                print(f"[DocRetriever] API request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    return None

    def is_alive(self) -> bool:
        try:
            resp = requests.get(self.api_url, params={"query": "test", "k": 1}, timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ------------------------------------------------------------------ #
    #  Passage normalization                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _split_title_body(raw_text: str) -> Tuple[str, str]:
        """ColBERT's collection loader stores each passage as 'Title | body'."""
        text = str(raw_text or "").strip()
        if not text:
            return "", ""
        # Title is always first, so split on the FIRST separator only.
        idx = text.find(" | ")
        if idx != -1:
            return text[:idx].strip(), text[idx + 3:].strip()
        idx = text.find("|")
        if idx != -1:
            return text[:idx].strip(), text[idx + 1:].strip()
        return "", text

    def retrieve_topk_passages(self, query: str, k: int = 5) -> List[dict]:
        """Retrieve top-k passages normalized into doc evidence dicts.

        Each dict carries a `source: "doc"` tag and uses `doc_id` (never `pid`)
        so the reasoner can tell doc evidence apart from KG/Table evidence.
        """
        retrieved = self.retrieve_topk(query, k) or []
        passages: List[dict] = []
        for entry in retrieved:
            if not isinstance(entry, dict):
                continue
            raw_text = str(entry.get("text") or entry.get("passage") or entry.get("content") or "")
            title, body = self._split_title_body(raw_text)
            if not title:
                title = str(entry.get("title") or "").strip()
            doc_id = entry.get("pid", entry.get("doc_id", entry.get("id", "")))
            passages.append({
                "label": title,
                "title": title,
                "text": body or raw_text,
                "doc_id": str(doc_id),
                "rank": entry.get("rank", ""),
                "score": entry.get("score"),
                "prob": entry.get("prob"),
                "source": "doc",
            })
        return passages

    # ------------------------------------------------------------------ #
    #  Formatters for prompt consumption                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if max_chars and len(text) > max_chars:
            return text[:max_chars].rsplit(" ", 1)[0] + " ..."
        return text

    def format_passage(self, passage: dict, max_chars: int = 10000) -> str:
        title = str(passage.get("title") or passage.get("label") or "").strip()
        doc_id = str(passage.get("doc_id") or "").strip()
        body = self._truncate(passage.get("text") or passage.get("snippet") or "", max_chars)

        lines = []
        if doc_id:
            lines.append(f"Doc ID: {doc_id}")
        if title:
            lines.append(f"Title: {title}")
        if body:
            lines.append(f"Passage: {body}")
        return "\n".join(lines)

    def format_passages(self, passages: List[dict], max_chars: int = 10000) -> List[str]:
        return [self.format_passage(p, max_chars=max_chars) for p in passages]
