"""ToG-2 baseline (Think-on-Graph 2.0, Ma et al., 2024).

Tight coupling between the KG and text retrievers. Each iteration:

  1. Knowledge-guided graph search
     - For every frontier entity, fetch its outgoing / incoming relations
       from the local KG.
     - LLM prunes the relation set to the most question-relevant W.
     - Walk those relations to obtain next-hop candidate entities.
  2. Knowledge-guided context retrieval
     - For each candidate entity, query the table BM25 / doc ColBERT
       services using ``<entity_label> : <question>`` and keep the
       service-reported scores as a Dense-Retrieval proxy.
     - Re-rank candidate entities by their best passage score and prune
       to the top-W frontier for the next iteration.
  3. Hybrid knowledge reasoning
     - Concatenate all collected triples + passages, ask the LLM if it
       can answer; if so emit ``The answer is X.`` and stop. Otherwise
       the LLM summarises remaining gaps and we continue.

Width W and depth D are kept modest (W=3, D=3) to bound LLM cost.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from _common.retrievers import EvidenceEntity, _fmt_props

WIDTH = 3
MAX_DEPTH = 3
PASSAGES_PER_ENTITY = 3


# --------------------------------------------------------------------- #
#  LLM helpers                                                          #
# --------------------------------------------------------------------- #
def _parse_json(raw: str) -> Optional[dict]:
    s = (raw or "").strip()
    if "```" in s:
        s = re.sub(r"```(?:json)?", "", s).strip("` \n")
    try:
        return json.loads(s[s.find("{"): s.rfind("}") + 1])
    except Exception:
        return None


def _extract_final(raw: str) -> str:
    m = re.search(r"(?i)the answer is\s*[:\-]?\s*(.+)", raw or "")
    if not m:
        return ""
    return m.group(1).split("\n")[0].strip().strip("\"'`").rstrip(".!?")


def _initial_entities(ctx) -> List[str]:
    prompt = (
        "Extract the named entities mentioned in the question that we should "
        "look up in a knowledge graph. Output a JSON array of strings (1-3 "
        "items, most specific entity names only).\n\n"
        f"Question: {ctx.question}\n\nEntities (JSON array):"
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=80, stage="tog2_init_entities")
    try:
        arr = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        return [str(s).strip() for s in arr if str(s).strip()][:3]
    except Exception:
        # naive fallback: keep capitalized phrases
        toks = re.findall(r"[A-Z][A-Za-z0-9'\-]+(?:\s+[A-Z][A-Za-z0-9'\-]+)*", ctx.question or "")
        return toks[:3] or [ctx.question]


def _link(ctx, labels: List[str]) -> List[EvidenceEntity]:
    seeds: List[EvidenceEntity] = []
    seen: set = set()
    for lab in labels:
        for ent in ctx.kg_retriever.search_entity(lab):
            if ent.qid and ent.qid not in seen:
                seen.add(ent.qid)
                seeds.append(ent)
                break
    return seeds


# --------------------------------------------------------------------- #
#  Per-iteration steps                                                  #
# --------------------------------------------------------------------- #
def _llm_prune_relations(ctx, entity: EvidenceEntity, relations: List[str]) -> List[str]:
    if len(relations) <= WIDTH:
        return relations
    prompt = (
        "Given a question and an entity, pick the relations from the KG that "
        "are most likely to lead to the answer. Output a JSON array of at "
        f"most {WIDTH} relation names taken verbatim from the list.\n\n"
        f"Question: {ctx.question}\n"
        f"Entity: {entity.label} (qid={entity.qid})\n"
        f"Available relations: {relations}\n\nPicked relations (JSON):"
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=120, stage="tog2_prune_rel")
    try:
        arr = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        picked = [r for r in arr if r in relations]
        if picked:
            return picked[:WIDTH]
    except Exception:
        pass
    return relations[:WIDTH]


def _walk(ctx, entity: EvidenceEntity, relation: str) -> List[Tuple[EvidenceEntity, dict, str]]:
    """Returns (neighbor, props, direction) tuples."""
    out: List[Tuple[EvidenceEntity, dict, str]] = []
    for t in ctx.kg_retriever.kg.relate(entity.qid, relation, inverse=False):
        lab = ctx.kg_retriever.kg.qid_to_label(t.tail_qid) or t.tail_qid
        out.append((EvidenceEntity(label=lab, qid=t.tail_qid, meta=t.props), t.props, "out"))
    for t in ctx.kg_retriever.kg.relate(entity.qid, relation, inverse=True):
        lab = ctx.kg_retriever.kg.qid_to_label(t.head_qid) or t.head_qid
        out.append((EvidenceEntity(label=lab, qid=t.head_qid, meta=t.props), t.props, "in"))
    return out


def _retrieve_context(ctx, entity: EvidenceEntity, relation: str,
                       retrieval_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    """Pull passages / table rows from the external services for an entity.

    Returns a list of {"source", "text", "score"} dicts. We use the
    service-reported scores as a Dense-Retrieval proxy (ColBERT for doc,
    BM25 for table).
    """
    query = f"{entity.label} {relation} {ctx.question}".strip()
    items: List[Dict[str, Any]] = []
    if ctx.doc_retriever is not None and "doc" in ctx.enabled_sources:
        retrieval_counts["doc"] = retrieval_counts.get("doc", 0) + 1
        for p in ctx.doc_retriever.retrieve_topk_passages(query, k=PASSAGES_PER_ENTITY):
            items.append({
                "source": "doc",
                "text": ctx.doc_retriever.format_passage(p),
                "score": float(p.get("score") or p.get("prob") or 0.0),
            })
    if ctx.table_retriever is not None and "table" in ctx.enabled_sources:
        retrieval_counts["table"] = retrieval_counts.get("table", 0) + 1
        tables = ctx.table_retriever.retrieve_topk_tables(query, k=PASSAGES_PER_ENTITY) or []
        for t in tables:
            items.append({
                "source": "table",
                "text": ctx.table_retriever.format_table_full(t),
                "score": float(t.get("score") or 0.0),
            })
    return items


def _reason(ctx, triples: List[str], passages: List[Dict[str, Any]]) -> Tuple[bool, str, str]:
    """Stage 3: can we answer yet? returns (can_answer, answer, clue_summary)."""
    triples_text = "\n".join(f"  - {t}" for t in triples) or "  (none)"
    passages_text = "\n\n".join(
        f"[{p['source']}#{i+1}] (score={p['score']:.3f})\n{p['text']}"
        for i, p in enumerate(passages[:12])
    ) or "(none)"
    prompt = (
        "You are answering a multi-hop question with a knowledge graph and "
        "text evidence. Decide whether you can already answer.\n\n"
        f"Question: {ctx.question}\n\n"
        f"### KG triples collected\n{triples_text}\n\n"
        f"### Passages collected\n{passages_text}\n\n"
        "Reply in JSON:\n"
        '{"can_answer": true/false, "answer": "X (only if can_answer=true)", '
        '"missing": "short summary of what is still missing (if can_answer=false)"}'
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=300, stage="tog2_reason")
    obj = _parse_json(raw) or {}
    can = bool(obj.get("can_answer"))
    ans = str(obj.get("answer") or "").strip().strip("\"'`").rstrip(".!?")
    miss = str(obj.get("missing") or "").strip()
    if can and not ans:
        ans = _extract_final(raw)
    return can, ans, miss


# --------------------------------------------------------------------- #
#  Driver                                                               #
# --------------------------------------------------------------------- #
def run(ctx) -> Dict[str, Any]:
    retrieval_counts: Dict[str, int] = {}

    if ctx.kg_retriever is None or "kg" not in ctx.enabled_sources:
        # No KG: retrieve directly from available sources (doc / table) and reason.
        query = ctx.question
        passages: List[Dict[str, Any]] = []
        if ctx.doc_retriever is not None and "doc" in ctx.enabled_sources:
            retrieval_counts["doc"] = retrieval_counts.get("doc", 0) + 1
            for p in ctx.doc_retriever.retrieve_topk_passages(query, k=5):
                passages.append({
                    "source": "doc",
                    "text": ctx.doc_retriever.format_passage(p),
                    "score": float(p.get("score") or p.get("prob") or 0.0),
                })
        if ctx.table_retriever is not None and "table" in ctx.enabled_sources:
            retrieval_counts["table"] = retrieval_counts.get("table", 0) + 1
            for t in ctx.table_retriever.retrieve_topk_tables(query, k=5) or []:
                passages.append({
                    "source": "table",
                    "text": ctx.table_retriever.format_table_full(t),
                    "score": float(t.get("score") or 0.0),
                })
        can, ans, _ = _reason(ctx, [], passages)
        if not ans:
            prompt = (
                "Read all collected evidence and emit a final answer in the form "
                "`The answer is X.`.\n\n"
                f"Question: {ctx.question}\n\nEvidence:\n"
                + "\n\n".join(p["text"] for p in passages[:10])
            )
            raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=120, stage="tog2_no_kg_final")
            ans = _extract_final(raw)
        active = [s for s in ("doc", "table") if s in ctx.enabled_sources]
        return {
            "sq1": [], "sq2": [], "final": [ans] if ans else [],
            "note": f"kg not enabled; retrieved from {','.join(active)}",
            "passages_retrieved": len(passages),
            "retrieval_counts": retrieval_counts,
        }

    init_labels = _initial_entities(ctx)
    # Each label in init_labels triggers one search_entity() call → 1 KG search per label.
    retrieval_counts["kg"] = len(init_labels)
    frontier = _link(ctx, init_labels)
    if not frontier:
        return {"sq1": [], "sq2": [], "final": [], "note": "no initial entities linked",
                "init_labels": init_labels, "retrieval_counts": retrieval_counts}

    all_triples: List[str] = []
    all_passages: List[Dict[str, Any]] = []
    iter_log: List[Dict[str, Any]] = []

    for depth in range(MAX_DEPTH):
        relations_global = list(ctx.kg_retriever.kg.relations)
        # 1) graph search: per-entity relation pruning + walk
        candidates: List[Tuple[EvidenceEntity, str, EvidenceEntity, dict, str]] = []
        for ent in frontier:
            picked_rels = _llm_prune_relations(ctx, ent, relations_global)
            for rel in picked_rels:
                for (neigh, props, direction) in _walk(ctx, ent, rel):
                    candidates.append((ent, rel, neigh, props, direction))
                    arrow = f"{ent.label} -[{rel}{_fmt_props(props)}]-> {neigh.label}" \
                        if direction == "out" \
                        else f"{neigh.label} -[{rel}{_fmt_props(props)}]-> {ent.label}"
                    all_triples.append(arrow)

        if not candidates:
            iter_log.append({"depth": depth, "candidates": 0, "note": "no kg neighbors"})
            break

        # 2) context retrieval per candidate + entity rerank by best score
        scored: List[Tuple[float, EvidenceEntity, str, List[Dict[str, Any]]]] = []
        seen_qids: set = set()
        for (src_ent, rel, neigh, _props, _dir) in candidates:
            if neigh.qid in seen_qids:
                continue
            seen_qids.add(neigh.qid)
            ctxs = _retrieve_context(ctx, neigh, rel, retrieval_counts)
            best = max((c["score"] for c in ctxs), default=0.0)
            scored.append((best, neigh, rel, ctxs))

        scored.sort(key=lambda x: x[0], reverse=True)
        kept = scored[:WIDTH]
        new_frontier: List[EvidenceEntity] = []
        for (_score, neigh, _rel, ctxs) in kept:
            new_frontier.append(neigh)
            all_passages.extend(ctxs)

        iter_log.append({
            "depth": depth,
            "frontier_in": [e.label for e in frontier],
            "candidates": len(candidates),
            "kept": [e.label for (_s, e, _r, _c) in kept],
            "new_passages": sum(len(c) for (_s, _e, _r, c) in kept),
        })

        # 3) reasoning gate
        can, ans, _missing = _reason(ctx, all_triples, all_passages)
        if can and ans:
            return {
                "sq1": [],
                "sq2": [],
                "final": [ans],
                "depth_reached": depth,
                "iters": iter_log,
                "triples": all_triples[-30:],
                "retrieval_counts": retrieval_counts,
            }
        frontier = new_frontier
        if not frontier:
            break

    # Ran out of depth: force a final answer with whatever was collected.
    can, ans, _missing = _reason(ctx, all_triples, all_passages)
    if not ans:
        prompt = (
            "Read all collected evidence and emit a final answer in the form "
            "`The answer is X.`.\n\n"
            f"Question: {ctx.question}\n\nKG triples:\n"
            + ("\n".join(all_triples[-50:]) or "(none)")
            + "\n\nPassages:\n"
            + ("\n\n".join(p["text"] for p in all_passages[:12]) or "(none)")
        )
        raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=120, stage="tog2_force_final")
        ans = _extract_final(raw)
    return {
        "sq1": [],
        "sq2": [],
        "final": [ans] if ans else [],
        "depth_reached": MAX_DEPTH,
        "iters": iter_log,
        "triples": all_triples[-30:],
        "note": "depth exhausted",
        "retrieval_counts": retrieval_counts,
    }
