"""Standard 2-shot prompting baseline (no retrieval).

The LLM sees two worked examples from the same multi-hop benchmark
schema, then is asked the new question. It must answer from its own
parametric knowledge.
"""
from __future__ import annotations

import re
from typing import Any, Dict

FEWSHOT = """Example 1.
Question: Who coached the team that won the 2007 NBA Championship?
Answer: Gregg Popovich

Example 2.
Question: In what year was the player who won the 1996 NBA Most Valuable Player Award drafted?
Answer: 1984
"""


def _extract_answer(raw: str) -> str:
    """Strip leading 'Answer:' / quotes / trailing punctuation."""
    s = (raw or "").strip()
    m = re.search(r"(?i)answer\s*:\s*(.+)", s)
    if m:
        s = m.group(1).strip()
    s = s.strip().strip("\"'`").rstrip(".!?")
    return s.split("\n")[0].strip()


def run(ctx) -> Dict[str, Any]:
    prompt = (
        "Answer the question with a short, direct answer. "
        "Output only the answer entity/value on a single line; no explanation.\n\n"
        f"{FEWSHOT}\n"
        f"Now answer this question.\n"
        f"Question: {ctx.question}\n"
        "Answer:"
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=80, stage="standard_answer")
    ans = _extract_answer(raw)
    return {
        "sq1": [],
        "sq2": [],
        "final": [ans] if ans else [],
        "raw_response": raw,
    }
