"""IRCoT baseline (Trivedi et al., 2023): interleave retrieval and CoT.

  step 0: retrieve initial evidence with the original question
  loop until terminator or max_iters:
      reason: ask LLM to emit the NEXT single CoT sentence, conditioned on
              (question, evidence so far, CoT so far). If it includes
              "the answer is" terminate.
      retrieve: feed that new sentence back as a query to all enabled sources
                and accumulate the new top-k chunks.
  final answer: ask LLM to read everything and emit ``The answer is X.``
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from _common.retrievers import format_evidence_block, retrieve_all

MAX_ITERS = 4

FEWSHOT_CHAIN = """Example chain (illustrative, follow the per-sentence style):
Question: Who coached the team that won the 2007 NBA Championship?
Sentence 1: The 2007 NBA Championship was won by the San Antonio Spurs.
Sentence 2: The San Antonio Spurs in 2007 were coached by Gregg Popovich.
Sentence 3: So the answer is Gregg Popovich.
"""


def _has_terminator(sentence: str) -> bool:
    return bool(re.search(r"(?i)\b(so|therefore)\b.*\banswer is\b", sentence)
                or re.search(r"(?i)the answer is\b", sentence))


def _extract_final(sentence: str) -> str:
    m = re.search(r"(?i)the answer is\s*[:\-]?\s*(.+)", sentence)
    if not m:
        return ""
    return m.group(1).split("\n")[0].strip().strip("\"'`").rstrip(".!?")


def _merge_evidence(acc: Dict[str, List[str]], new: Dict[str, List[str]]) -> Dict[str, List[str]]:
    out = {k: list(v) for k, v in acc.items()}
    for k, blocks in new.items():
        existing = set(out.setdefault(k, []))
        for blk in blocks:
            if blk not in existing:
                out[k].append(blk)
                existing.add(blk)
    return out


def run(ctx) -> Dict[str, Any]:
    evidence: Dict[str, List[str]] = retrieve_all(
        query=ctx.question,
        enabled_sources=ctx.enabled_sources,
        kg_retriever=ctx.kg_retriever,
        table_retriever=ctx.table_retriever,
        doc_retriever=ctx.doc_retriever,
    )

    chain: List[str] = []
    terminated = False
    for it in range(MAX_ITERS):
        prompt = (
            "Generate the NEXT single sentence of a chain-of-thought reasoning "
            "for the question. Use the evidence below. Do not output multiple "
            "sentences. If you have enough information, end the sentence with "
            "`So the answer is X.`.\n\n"
            f"{FEWSHOT_CHAIN}\n"
            f"### EVIDENCE\n{format_evidence_block(evidence) or '(none)'}\n\n"
            f"### Question\n{ctx.question}\n\n"
            f"### CoT so far\n"
            + ("\n".join(f"Sentence {i+1}: {s}" for i, s in enumerate(chain)) or "(empty)")
            + f"\n\nSentence {len(chain)+1}:"
        )
        sent, _ = ctx.llm.query_gpt4o(prompt, max_tokens=120, stage=f"reason_step_{it}")
        sent = sent.strip().split("\n")[0].strip()
        if not sent:
            break
        chain.append(sent)
        if _has_terminator(sent):
            terminated = True
            break
        # Retrieve more evidence using the new sentence as a query.
        new_ev = retrieve_all(
            query=sent,
            enabled_sources=ctx.enabled_sources,
            kg_retriever=ctx.kg_retriever,
            table_retriever=ctx.table_retriever,
            doc_retriever=ctx.doc_retriever,
        )
        evidence = _merge_evidence(evidence, new_ev)

    final = _extract_final(chain[-1]) if chain else ""
    if not final:
        # Force a final-answer pass with everything collected.
        prompt = (
            "Read the evidence and the reasoning chain. Output a single line: "
            "`The answer is X.` where X is the final entity/value.\n\n"
            f"### EVIDENCE\n{format_evidence_block(evidence) or '(none)'}\n\n"
            f"### Question\n{ctx.question}\n\n"
            f"### Reasoning so far\n" + "\n".join(chain) + "\n"
        )
        raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=120, stage="final_answer")
        m = re.search(r"(?i)the answer is\s*[:\-]?\s*(.+)", raw or "")
        final = (m.group(1) if m else (raw or "").strip().split("\n")[-1]).strip().strip("\"'`").rstrip(".!?")

    return {
        "sq1": [],
        "sq2": [],
        "final": [final] if final else [],
        "reasoning_chain": chain,
        "terminated_naturally": terminated,
        "evidence_counts": {k: len(v) for k, v in evidence.items()},
    }
