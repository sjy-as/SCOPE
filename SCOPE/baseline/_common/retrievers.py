"""Thin retrieval wrappers shared by the RAG-style baselines.

KG uses the local in-process index from HydraRAG (no separate service).
Table goes through the BM25 HTTP service (:1216). Doc goes through the
ColBERT HTTP service (:1215). All three are the same services Atomr /
DeepSieve / HydraRAG / new_model already point at.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse HydraRAG's KG / Table / Doc primitives instead of duplicating
# hundreds of lines of indexer code. The hydra_baseline package is
# self-contained and its module-relative paths (data_sources/, etc.)
# resolve correctly regardless of who imports it.
_HYDRA_DIR = str(Path(os.getenv("SCOPE_HYDRARAG_DIR", str(Path(__file__).resolve().parents[1] / "HydraRAG" / "hydra_baseline"))).resolve())
if _HYDRA_DIR not in sys.path:
    sys.path.insert(0, _HYDRA_DIR)

from kg.kg_retriever import KGRetriever, EvidenceEntity  # noqa: E402
from kg.kg_explorer import KGExplorer, _fmt_props  # noqa: E402
from table.table_retriever import TableRetriever  # noqa: E402
from doc.doc_retriever import DocRetriever  # noqa: E402

DEFAULT_KG_DIR = str(Path(_HYDRA_DIR) / "data_sources" / "KG")
DEFAULT_TABLE_API = os.environ.get("TABLE_API_URL", "http://127.0.0.1:1216/api/search")
DEFAULT_DOC_API = os.environ.get("DOC_API_URL", "http://127.0.0.1:1215/api/search")


# ===================================================================== #
# KG: entity link + 1-hop edge dump as text                              #
# ===================================================================== #
def kg_evidence_text(
    kg_retriever: KGRetriever,
    query: str,
    entity_hints: Optional[List[str]] = None,
    max_entities: int = 3,
    max_edges_per_entity: int = 100,
) -> List[str]:
    """Return human-readable KG edge blocks for the query.

    Strategy: search candidate entities (use ``entity_hints`` if provided,
    otherwise fall back to the question string), then dump up to
    ``max_edges_per_entity`` outgoing + incoming edges per entity as
    ``head -[rel|props]-> tail`` lines.
    """
    seeds: List[EvidenceEntity] = []
    seen_qids: set = set()

    probes: List[str] = []
    if entity_hints:
        probes.extend(h for h in entity_hints if h and h.strip())
    if not probes:
        probes.append(query)

    for probe in probes:
        for ent in kg_retriever.search_entity(probe):
            if ent.qid and ent.qid not in seen_qids:
                seen_qids.add(ent.qid)
                seeds.append(ent)
                if len(seeds) >= max_entities:
                    break
        if len(seeds) >= max_entities:
            break

    blocks: List[str] = []
    for ent in seeds:
        lines: List[str] = [f"Entity: {ent.label} (qid={ent.qid})"]
        edges_dumped = 0
        for rel in kg_retriever.kg.relations:
            if edges_dumped >= max_edges_per_entity:
                break
            for t in kg_retriever.kg.relate(ent.qid, rel, inverse=False):
                tail_lab = kg_retriever.kg.qid_to_label(t.tail_qid) or t.tail_qid
                lines.append(f"  {ent.label} -[{rel}{_fmt_props(t.props)}]-> {tail_lab}")
                edges_dumped += 1
                if edges_dumped >= max_edges_per_entity:
                    break
            if edges_dumped >= max_edges_per_entity:
                break
            for t in kg_retriever.kg.relate(ent.qid, rel, inverse=True):
                head_lab = kg_retriever.kg.qid_to_label(t.head_qid) or t.head_qid
                lines.append(f"  {head_lab} -[{rel}{_fmt_props(t.props)}]-> {ent.label}")
                edges_dumped += 1
                if edges_dumped >= max_edges_per_entity:
                    break
        blocks.append("\n".join(lines))
    return blocks


# ===================================================================== #
# Table: BM25 service top-k formatted as text                            #
# ===================================================================== #
def table_evidence_text(
    table_retriever: TableRetriever, query: str, k: int = 5,
) -> List[str]:
    raw = table_retriever.retrieve_topk_tables(query, k=k) or []
    return table_retriever.format_tables(raw, full=True)


# ===================================================================== #
# Doc: ColBERT service top-k formatted as text                           #
# ===================================================================== #
def doc_evidence_text(
    doc_retriever: DocRetriever, query: str, k: int = 5,
) -> List[str]:
    passages = doc_retriever.retrieve_topk_passages(query, k=k)
    return doc_retriever.format_passages(passages)


def retrieve_all(
    *,
    query: str,
    enabled_sources: List[str],
    kg_retriever: Optional[KGRetriever],
    table_retriever: Optional[TableRetriever],
    doc_retriever: Optional[DocRetriever],
    entity_hints: Optional[List[str]] = None,
    k_table: int = 5,
    k_doc: int = 5,
) -> Dict[str, List[str]]:
    """One-shot retrieval across enabled sources. Returns text blocks."""
    out: Dict[str, List[str]] = {}
    if "kg" in enabled_sources and kg_retriever is not None:
        out["kg"] = kg_evidence_text(kg_retriever, query, entity_hints=entity_hints)
    if "table" in enabled_sources and table_retriever is not None:
        out["table"] = table_evidence_text(table_retriever, query, k=k_table)
    if "doc" in enabled_sources and doc_retriever is not None:
        out["doc"] = doc_evidence_text(doc_retriever, query, k=k_doc)
    return out


def format_evidence_block(by_source: Dict[str, List[str]]) -> str:
    """Concatenate per-source evidence blocks into one prompt section."""
    parts: List[str] = []
    for src in ("kg", "table", "doc"):
        items = by_source.get(src) or []
        if not items:
            continue
        parts.append(f"### {src.upper()} EVIDENCE")
        for i, blk in enumerate(items, 1):
            parts.append(f"[{src}#{i}]\n{blk}")
    return "\n\n".join(parts)
