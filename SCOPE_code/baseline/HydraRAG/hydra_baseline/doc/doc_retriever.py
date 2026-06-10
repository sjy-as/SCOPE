"""HydraRAG 的 Doc 检索客户端。

瘦 HTTP 客户端，对接共享的 ColBERT 段落检索服务（默认端口 1215，由
`table_service/setup_service_nba_datalake.py` 启动）——与 atomr / deepservice /
new_model 共用同一个 doc 服务。

与 `table/table_retriever.py` 对齐：只负责取回 + 规范化 + 给 prompt 用的格式化，
不在本目录内放任何 ColBERT 索引/模型（保持 baseline 轻量）。
"""
from typing import List, Optional, Tuple
import sys
import time

import requests

try:
    if "/root/autodl-tmp" not in sys.path:
        sys.path.insert(0, "/root/autodl-tmp")
    from _common.cost_counter import bump_retrieval as _bump_retrieval
except Exception:  # pragma: no cover
    def _bump_retrieval(*_a, **_kw): pass


class DocRetriever:
    """ColBERT 段落检索服务的客户端封装。"""

    def __init__(self, api_url: str = "http://127.0.0.1:1215/api/search", timeout: int = 60):
        self.api_url = api_url
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  服务调用                                                            #
    # ------------------------------------------------------------------ #
    def retrieve_topk(self, query: str, k: int = 5, max_retries: int = 3) -> Optional[List[dict]]:
        """取回原始 top-k 段落，带简单重试。"""
        _bump_retrieval("doc")
        params = {"query": query, "k": k}
        for attempt in range(max_retries):
            try:
                resp = requests.get(url=self.api_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                return payload.get("topk") or payload.get("results") or []
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
    #  段落规范化                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_title_body(raw_text: str) -> Tuple[str, str]:
        """ColBERT collection 把每条段落存成 'Title | body'，标题在最前，只切第一个分隔符。"""
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

    def retrieve_topk_passages(self, query: str, k: int = 5) -> List[dict]:
        """取回 top-k 段落并规范化成 doc 证据 dict（带 `doc_id` 用于去重）。"""
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
                "doc_id": str(doc_id),
                "title": title,
                "text": body or raw_text,
                "rank": entry.get("rank", ""),
                "score": entry.get("score"),
                "prob": entry.get("prob"),
            })
        return passages

    # ------------------------------------------------------------------ #
    #  给 prompt 用的格式化                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if max_chars and len(text) > max_chars:
            return text[:max_chars].rsplit(" ", 1)[0] + " ..."
        return text

    def format_passage(self, passage: dict, max_chars: int = 4000) -> str:
        """把单条段落渲染成 LLM 可读文本块（含 doc_id，便于回指）。"""
        doc_id = str(passage.get("doc_id") or "").strip()
        title = str(passage.get("title") or "").strip()
        body = self._truncate(passage.get("text") or passage.get("snippet") or "", max_chars)

        lines = []
        if doc_id:
            lines.append(f"Doc ID: {doc_id}")
        if title:
            lines.append(f"Title: {title}")
        if body:
            lines.append(f"Passage: {body}")
        return "\n".join(lines)

    def format_passages(self, passages: List[dict], max_chars: int = 4000) -> List[str]:
        return [self.format_passage(p, max_chars=max_chars) for p in passages]
