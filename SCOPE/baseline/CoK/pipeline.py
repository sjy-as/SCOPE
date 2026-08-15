"""CoK baseline (Chain-of-Knowledge, Li et al., 2024).

Stages
------
1. Reasoning preparation: generate an initial CoT chain + candidate answer
   from the LLM's parametric knowledge, plus a self-rated confidence.
2. Adaptive triggering: if confidence < threshold, enter rectification.
3. Sentence-level rectification: for each reasoning sentence,
       a. pick the best source (kg / table / doc) given the sentence,
       b. retrieve evidence from that source using the sentence as query,
       c. rewrite the sentence to be consistent with the retrieved evidence.
4. Answer consolidation: read the rectified chain and emit the final answer.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from _common.retrievers import (
    doc_evidence_text, kg_evidence_text, table_evidence_text,
)

CONF_THRESHOLD = 0.9   # rectify when confidence falls below this
MAX_SENTENCES_TO_RECTIFY = 5


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #
def _parse_json(raw: str) -> Optional[dict]:
    s = (raw or "").strip()
    if "```" in s:
        s = re.sub(r"```(?:json)?", "", s).strip("` \n")
    try:
        return json.loads(s[s.find("{"): s.rfind("}") + 1])
    except Exception:
        return None


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_final(raw: str) -> str:
    s = (raw or "").strip()
    m = re.search(r"(?i)the answer is\s*[:\-]?\s*(.+)", s)
    if not m:
        m = re.search(r"(?i)^\s*answer\s*[:\-]\s*(.+)$", s, re.M)
    if m:
        s = m.group(1).split("\n")[0].strip()
    else:
        s = s.split("\n")[-1].strip()
    return s.strip("\"'`").rstrip(".!?")


def _retrieve_from(ctx, source: str, query: str) -> List[str]:
    if source == "kg" and ctx.kg_retriever is not None:
        return kg_evidence_text(ctx.kg_retriever, query)
    if source == "table" and ctx.table_retriever is not None:
        return table_evidence_text(ctx.table_retriever, query, k=5)
    if source == "doc" and ctx.doc_retriever is not None:
        return doc_evidence_text(ctx.doc_retriever, query, k=5)
    return []


# --------------------------------------------------------------------- #
#  Pipeline stages                                                      #
# --------------------------------------------------------------------- #
def _initial_chain(ctx) -> Tuple[List[str], str, float]:
    """Stage 1: ask LLM for chain + answer + confidence (parametric only)."""
    prompt = (
        "Answer the question by writing a short chain of 2-5 reasoning sentences, "
        "then give a final answer and a self-rated confidence in [0, 1].\n\n"
        f"Question: {ctx.question}\n\n"
        "Reply strictly in JSON with this schema:\n"
        '{"chain": ["sentence1", "sentence2", ...], "answer": "X", "confidence": 0.8}'
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=400, stage="cok_initial")
    obj = _parse_json(raw) or {}
    chain = [str(s).strip() for s in (obj.get("chain") or []) if str(s).strip()]
    if not chain:
        chain = _split_sentences(raw)
    answer = str(obj.get("answer") or _extract_final(raw)).strip().strip("\"'`").rstrip(".!?")
    try:
        conf = float(obj.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    return chain, answer, max(0.0, min(1.0, conf))


def _pick_source(ctx, sentence: str) -> str:
    """Stage 3a: let the LLM pick one of enabled sources for this sentence."""
    options = [s for s in ctx.enabled_sources if s in ("kg", "table", "doc")]
    if not options:
        return ""
    if len(options) == 1:
        return options[0]
    prompt = (
        "We need to fact-check the following reasoning sentence. Choose the most "
        "useful knowledge source from the list. Output ONLY the source name.\n\n"
        f"Sources: {options}\n"
        " - kg: NBA knowledge graph of players, teams, awards, coaches, venues\n"
        " - table: Wikipedia-style tabular data (season schedules, stats)\n"
        " - doc: Wikipedia text passages\n\n"
        f"Question: {ctx.question}\n"
        f"Sentence: {sentence}\n\n"
        "Source:"
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=10, stage="cok_pick_source")
    pick = (raw or "").strip().lower().split()[0] if raw else ""
    return pick if pick in options else options[0]


def _rectify_sentence(ctx, sentence: str, source: str, evidence: List[str]) -> str:
    """Stage 3c: rewrite the sentence so it agrees with retrieved evidence."""
    if not evidence:
        return sentence
    ev = "\n\n".join(f"[{source}#{i+1}]\n{e}" for i, e in enumerate(evidence[:5]))
    prompt = (
        "Rewrite the following reasoning sentence so it is consistent with the "
        "retrieved evidence. Keep it concise (one sentence). If the evidence "
        "confirms the original, return it unchanged. If the evidence shows the "
        "sentence is wrong, replace it with the correct version. Do not add "
        "extra sentences or commentary.\n\n"
        f"EVIDENCE\n{ev}\n\n"
        f"Original sentence: {sentence}\n"
        "Corrected sentence:"
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=120, stage="cok_rectify")
    line = (raw or "").strip().split("\n")[0].strip()
    return line or sentence


def _consolidate(ctx, rectified_chain: List[str]) -> str:
    """Stage 4: emit the final answer from the rectified chain."""
    prompt = (
        "Read the rectified reasoning chain and output the final answer. "
        "End with: `The answer is X.`\n\n"
        f"Question: {ctx.question}\n\n"
        f"Reasoning chain:\n" + "\n".join(f"- {s}" for s in rectified_chain) + "\n"
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=120, stage="cok_consolidate")
    return _extract_final(raw)


# --------------------------------------------------------------------- #
#  Entry                                                                #
# --------------------------------------------------------------------- #
def run(ctx) -> Dict[str, Any]:
    chain, initial_answer, conf = _initial_chain(ctx)

    if conf >= CONF_THRESHOLD or not ctx.enabled_sources:
        return {
            "sq1": [],
            "sq2": [],
            "final": [initial_answer] if initial_answer else [],
            "initial_chain": chain,
            "initial_answer": initial_answer,
            "confidence": conf,
            "rectified": False,
        }

    rectified: List[str] = []
    per_sentence_meta: List[Dict[str, Any]] = []
    for sent in chain[:MAX_SENTENCES_TO_RECTIFY]:
        source = _pick_source(ctx, sent)
        evidence = _retrieve_from(ctx, source, sent) if source else []
        new_sent = _rectify_sentence(ctx, sent, source, evidence) if evidence else sent
        rectified.append(new_sent)
        per_sentence_meta.append({
            "original": sent, "source": source,
            "evidence_count": len(evidence), "rectified": new_sent,
        })
    # Carry any tail untouched (saves LLM calls on very long chains).
    rectified.extend(chain[MAX_SENTENCES_TO_RECTIFY:])

    final = _consolidate(ctx, rectified)
    return {
        "sq1": [],
        "sq2": [],
        "final": [final] if final else [],
        "initial_chain": chain,
        "initial_answer": initial_answer,
        "confidence": conf,
        "rectified": True,
        "rectified_chain": rectified,
        "rectify_meta": per_sentence_meta,
    }
