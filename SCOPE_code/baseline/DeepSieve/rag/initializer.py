"""
rag/initializer.py

Initialize retrieval backends.
- Classic mode: vector RAG (naive/graph) for local/global.
- MMQA mode: source retrievers for table/kg.
"""

import os
import time
from typing import List, Dict, Tuple

from rag.naive_rag import NaiveRAG
from rag.graph_rag import GraphRAG_Improved
from rag.retrieve.query_table import execute_table_query
from rag.retrieve.query_doc import execute_doc_query
from rag.retrieve.query_kg import semantic_parsing_api, engine_exec_api


class TableRetrieverRAG:
    def __init__(self, table_api_url: str):
        self.table_api_url = table_api_url

    def rag_qa(self, question: str, k: int = 5) -> Dict:
        start = time.time()
        # Table evidence is fixed to top-5.
        table_top_k = 5
        _, full_results = execute_table_query(question, self.table_api_url, k=table_top_k)
        hits = (full_results or {}).get("organic_results", []) or []

        docs: List[str] = []
        scores: List[float] = []
        for h in hits[:table_top_k]:
            title = h.get("title", "")
            snippet = h.get("snippet", "")
            table_text = h.get("article_text", "")
            payload = "\n".join(x for x in [f"Title: {title}" if title else "", snippet, table_text] if x)
            docs.append(payload.strip())
            # Table service does not return a unified score; use rank-decay placeholder.
            scores.append(1.0 / (1 + len(scores)))

        if not docs:
            docs = [""]
            scores = [0.0]

        retrieval_time = time.time() - start
        return {
            "docs": docs,
            "doc_scores": scores,
            "metrics": {
                "retrieval_time": retrieval_time,
                "avg_similarity": float(sum(scores) / len(scores)) if scores else 0.0,
                "max_similarity": float(max(scores)) if scores else 0.0,
                "total_docs_searched": len(hits),
            },
        }


class DocRetrieverRAG:
    """Document (passage) retriever backed by the ColBERT service.

    Mirrors `TableRetrieverRAG`: it exposes `rag_qa` returning the same
    {docs, doc_scores, metrics} contract, so the routing pipeline can treat a
    doc source exactly like a table source. Used for the kg-doc benchmark.
    """

    def __init__(self, doc_api_url: str):
        self.doc_api_url = doc_api_url

    def rag_qa(self, question: str, k: int = 5) -> Dict:
        start = time.time()
        # Doc evidence is fixed to top-5 passages.
        doc_top_k = 5
        _, full_results = execute_doc_query(question, self.doc_api_url, k=doc_top_k)
        hits = (full_results or {}).get("organic_results", []) or []

        docs: List[str] = []
        scores: List[float] = []
        for h in hits[:doc_top_k]:
            title = h.get("title", "")
            passage = h.get("article_text", "")
            payload = "\n".join(x for x in [f"Title: {title}" if title else "", passage] if x)
            docs.append(payload.strip())
            # ColBERT returns a relevance score; fall back to rank-decay.
            sc = h.get("score")
            scores.append(float(sc) if sc is not None else 1.0 / (1 + len(scores)))

        if not docs:
            docs = [""]
            scores = [0.0]

        retrieval_time = time.time() - start
        return {
            "docs": docs,
            "doc_scores": scores,
            "metrics": {
                "retrieval_time": retrieval_time,
                "avg_similarity": float(sum(scores) / len(scores)) if scores else 0.0,
                "max_similarity": float(max(scores)) if scores else 0.0,
                "total_docs_searched": len(hits),
            },
        }


class KGRetrieverRAG:
    def __init__(self, kg_api_url: str):
        self.kg_api_url = kg_api_url

    def rag_qa(self, question: str, k: int = 5) -> Dict:
        start = time.time()

        program = semantic_parsing_api(question, self.kg_api_url)
        result = engine_exec_api(program, self.kg_api_url) or {}

        # KG evidence retrieval is fixed to top-100 by product requirement.
        kg_top_k = 100

        docs: List[str] = []
        scores: List[float] = []

        _ATTR_SKIP_KEYS = {"enwiki_title", "wikidata_id"}

        def _fmt_entity_attrs(snap):
            if not snap:
                return ""
            attrs = snap.get("attrs") or {}
            kept = {k: v for k, v in attrs.items() if k not in _ATTR_SKIP_KEYS and v}
            return str(kept) if kept else ""

        evidence = result.get("evidence") or []
        for ev in evidence[:kg_top_k]:
            head = ev.get("head_entity", "")
            rel = ev.get("relation", "")
            tail = ev.get("label", "")
            direction = ev.get("direction", "")
            meta = ev.get("meta", {}) or {}
            line = f"{head} --[{rel}/{direction}]--> {tail}"
            if meta:
                line += f" ; rel_meta={meta}"
            head_attrs_str = _fmt_entity_attrs(ev.get("head_entity_snapshot"))
            if head_attrs_str:
                line += f" ; head_attrs={head_attrs_str}"
            tail_attrs_str = _fmt_entity_attrs(ev.get("tail_entity_snapshot"))
            if tail_attrs_str:
                line += f" ; tail_attrs={tail_attrs_str}"
            docs.append(line)
            scores.append(1.0 / (1 + len(scores)))

        # 如果常规 evidence 全是 entity_only（self 边、meta 空），上面循环输出的 doc
        # 信息密度极低，这里追加一份纯实体属性 doc 作补充。
        for ev in evidence[:kg_top_k]:
            if (ev.get("direction") or "") != "self":
                continue
            head_attrs_str = _fmt_entity_attrs(ev.get("head_entity_snapshot"))
            if head_attrs_str:
                docs.append(f"KG entity: {ev.get('head_entity', '')} ; attrs={head_attrs_str}")
                scores.append(1.0 / (1 + len(scores)))

        inner_content = (((result.get("inner_content") or [{}])[0]).get("content") or [])
        if not docs and inner_content:
            for ans in inner_content[:kg_top_k]:
                docs.append(f"KG candidate answer: {ans}")
                scores.append(1.0 / (1 + len(scores)))

        if not docs:
            ans = result.get("answer", "")
            docs = [f"KG answer: {ans}" if ans else ""]
            scores = [0.0 if not ans else 1.0]

        retrieval_time = time.time() - start
        return {
            "docs": docs,
            "doc_scores": scores,
            "metrics": {
                "retrieval_time": retrieval_time,
                "avg_similarity": float(sum(scores) / len(scores)) if scores else 0.0,
                "max_similarity": float(max(scores)) if scores else 0.0,
                "total_docs_searched": len(evidence) if evidence else len(docs),
            },
        }


def initialize_mmqa_sources(enabled_sources: list, profiles: dict, use_routing: bool):
    """Build the live knowledge base for the MMQA-family benchmark.

    `enabled_sources` is the subset of {kg, table, doc} the run enabled (all
    three are individually switchable). Each gets its own source-specific
    retriever. Returns
    `(sources, merged_rag)` where `sources` is a list of
    {"name", "rag", "profile"} dicts (the routable knowledge base) and
    `merged_rag` is the single retriever used when routing is disabled.
    """
    table_api_url = os.environ.get("TABLE_API_URL", "http://127.0.0.1:1216/api/search")
    doc_api_url = os.environ.get("DOC_API_URL", "http://127.0.0.1:1215/api/search")
    kg_api_url = os.environ.get("KG_API_URL", "http://127.0.0.1:8002/query")

    builders = {
        "kg": lambda: KGRetrieverRAG(kg_api_url=kg_api_url),
        "table": lambda: TableRetrieverRAG(table_api_url=table_api_url),
        "doc": lambda: DocRetrieverRAG(doc_api_url=doc_api_url),
    }

    sources = []
    for name in enabled_sources:
        if name not in builders:
            raise ValueError(f"Unknown MMQA source: {name} (expected kg/table/doc)")
        sources.append({
            "name": name,
            "rag": builders[name](),
            "profile": (profiles or {}).get(name, ""),
        })

    merged_rag = None
    if not use_routing:
        # No-routing mode has no real "merged" retriever for source-specific
        # backends; prefer a non-KG retriever (table > doc), else KG.
        pref = (next((s for s in sources if s["name"] == "table"), None)
                or next((s for s in sources if s["name"] == "doc"), None)
                or sources[0])
        merged_rag = pref["rag"]

    print(f"🔍 MMQA knowledge base: {[s['name'] for s in sources]}  "
          f"(routing={'on' if use_routing else 'off'})")
    return sources, merged_rag


def initialize_rag_system(rag_type: str, use_routing: bool, source_a_docs: list, source_b_docs: list, dataset: str = ""):
    dataset_l = (dataset or "").lower().strip()

    if dataset_l == "mmqa":
        table_api_url = os.environ.get("TABLE_API_URL", "http://127.0.0.1:1216/api/search")
        kg_api_url = os.environ.get("KG_API_URL", "http://127.0.0.1:8002/query")

        table_rag = TableRetrieverRAG(table_api_url=table_api_url)
        kg_rag = KGRetrieverRAG(kg_api_url=kg_api_url)

        if use_routing:
            print("🔍 Using MMQA routing mode: initialized TABLE and KG retrievers")
            return table_rag, kg_rag, None

        # For no-routing mode in MMQA, fallback to table retriever by default.
        print("🔍 Using MMQA no-routing mode: defaulting to TABLE retriever")
        return None, None, table_rag

    if dataset_l == "mmqa_doc":
        # kg-doc benchmark: source A = Doc (ColBERT passages), source B = KG.
        doc_api_url = os.environ.get("DOC_API_URL", "http://127.0.0.1:1215/api/search")
        kg_api_url = os.environ.get("KG_API_URL", "http://127.0.0.1:8002/query")

        doc_rag = DocRetrieverRAG(doc_api_url=doc_api_url)
        kg_rag = KGRetrieverRAG(kg_api_url=kg_api_url)

        if use_routing:
            print("🔍 Using MMQA-Doc routing mode: initialized DOC and KG retrievers")
            return doc_rag, kg_rag, None

        # For no-routing mode, fall back to the doc retriever by default.
        print("🔍 Using MMQA-Doc no-routing mode: defaulting to DOC retriever")
        return None, None, doc_rag

    merged_docs = source_a_docs + source_b_docs

    if use_routing:
        if rag_type == "naive":
            source_a_rag = NaiveRAG(source_a_docs)
            source_b_rag = NaiveRAG(source_b_docs)
        elif rag_type == "graph":
            source_a_rag = GraphRAG_Improved(source_a_docs)
            source_b_rag = GraphRAG_Improved(source_b_docs)
        else:
            raise ValueError(f"Unsupported RAG type: {rag_type}")

        print(f"🔍 Using routing mode: initialized source A/B knowledge bases, RAG type: {rag_type}")
        return source_a_rag, source_b_rag, None

    if rag_type == "naive":
        merged_rag = NaiveRAG(merged_docs)
    elif rag_type == "graph":
        merged_rag = GraphRAG_Improved(merged_docs)
    else:
        raise ValueError(f"Unsupported RAG type: {rag_type}")

    print(f"🔍 Using no-routing mode: merged source A/B knowledge bases, RAG type: {rag_type}")
    return None, None, merged_rag
