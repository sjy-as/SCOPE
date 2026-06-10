"""Chain-of-Thought baseline (2-shot, no retrieval).

LLM is shown two worked multi-hop examples each ending in
"Therefore, the answer is X." and asked to think step-by-step before
emitting a final answer line. The pipeline parses the answer out.
"""
from __future__ import annotations

import re
from typing import Any, Dict

FEWSHOT = """Example 1.
Question: Who coached the team that won the 2007 NBA Championship?
Let's think step by step. The 2007 NBA Championship was won by the San Antonio Spurs. The San Antonio Spurs were coached by Gregg Popovich. Therefore, the answer is Gregg Popovich.

Example 2.
Question: In what year was the player who won the 1996 NBA Most Valuable Player Award drafted?
Let's think step by step. The 1996 NBA Most Valuable Player Award was won by Michael Jordan. Michael Jordan was drafted in 1984. Therefore, the answer is 1984.
"""


def _extract_answer(raw: str) -> str:
    s = (raw or "").strip()
    m = re.search(r"(?i)therefore[, ]*the answer is\s*[:\-]?\s*(.+)", s)
    if m:
        s = m.group(1).strip()
    else:
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
    prompt = (
        "Answer the question by reasoning step by step. End with a single sentence "
        "of the form `Therefore, the answer is X.` where X is the final entity/value.\n\n"
        f"{FEWSHOT}\n"
        f"Now answer this question.\n"
        f"Question: {ctx.question}\n"
        "Let's think step by step."
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=400, stage="cot_answer")
    ans = _extract_answer(raw)
    return {
        "sq1": [],
        "sq2": [],
        "final": [ans] if ans else [],
        "reasoning": raw,
    }
