"""Standard RAG baseline.

No query decomposition. The raw question is sent to all enabled sources
in parallel:
  * KG: search candidate entities from the question text, dump 1-hop edges
  * Table: BM25 top-5 tables via the shared :1216 service
  * Doc: ColBERT top-5 passages via the shared :1215 service

All evidence is concatenated and the LLM produces a final answer.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from _common.retrievers import format_evidence_block, retrieve_all


def _extract_answer(raw: str) -> str:
    s = (raw or "").strip()
    m = re.search(r"(?i)the answer is\s*[:\-]?\s*(.+)", s)
    if m:
        s = m.group(1).strip()
    else:
        m = re.search(r"(?i)^\s*answer\s*[:\-]\s*(.+)$", s, re.M)
        if m:
            s = m.group(1).strip()
        else:
            s = s.split("\n")[-1].strip()
    s = s.strip().strip("\"'`").rstrip(".!?")
    return s


def run(ctx) -> Dict[str, Any]:
    evidence = retrieve_all(
        query=ctx.question,
        enabled_sources=ctx.enabled_sources,
        kg_retriever=ctx.kg_retriever,
        table_retriever=ctx.table_retriever,
        doc_retriever=ctx.doc_retriever,
    )
    ev_block = format_evidence_block(evidence)

    prompt = (
        "You are answering a multi-hop question using the retrieved evidence below. "
        "Use only the evidence; do not invent facts. Provide a short, direct answer.\n\n"
        f"EVIDENCE\n{ev_block or '(none)'}\n\n"
        f"Question: {ctx.question}\n"
        "End your reply with a single line: `The answer is X.` where X is the final entity/value."
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=400, stage="standard_rag_answer")
    ans = _extract_answer(raw)
    return {
        "sq1": [],
        "sq2": [],
        "final": [ans] if ans else [],
        "evidence_counts": {k: len(v) for k, v in evidence.items()},
        "raw_response": raw,
    }
