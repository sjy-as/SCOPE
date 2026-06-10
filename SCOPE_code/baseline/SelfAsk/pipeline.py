"""Self-Ask baseline (Press et al., 2022) — no retrieval.

The LLM is shown two demonstrations of the Self-Ask scaffold
(``Are follow up questions needed here: Yes`` → ``Follow up:`` → ``Intermediate answer:`` ... → ``So the final answer is:``).
It then continues the pattern on the new question, answering each
follow-up from its own knowledge until it emits the final answer.

We also harvest the first two intermediate answers as ``sq1`` / ``sq2``
so the predictions line up with the new_model's per-hop eval flags.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

FEWSHOT = """Question: Who coached the team that won the 2007 NBA Championship?
Are follow up questions needed here: Yes.
Follow up: Which team won the 2007 NBA Championship?
Intermediate answer: The San Antonio Spurs won the 2007 NBA Championship.
Follow up: Who was the head coach of the San Antonio Spurs in 2007?
Intermediate answer: Gregg Popovich was the head coach of the San Antonio Spurs in 2007.
So the final answer is: Gregg Popovich.

Question: In what year was the player who won the 1996 NBA Most Valuable Player Award drafted?
Are follow up questions needed here: Yes.
Follow up: Who won the 1996 NBA Most Valuable Player Award?
Intermediate answer: Michael Jordan won the 1996 NBA Most Valuable Player Award.
Follow up: In what year was Michael Jordan drafted?
Intermediate answer: Michael Jordan was drafted in 1984.
So the final answer is: 1984.
"""


_INT_RE = re.compile(r"Intermediate answer:\s*(.+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"So the final answer is:\s*(.+)", re.IGNORECASE)


def _strip(s: str) -> str:
    return s.strip().strip("\"'`").rstrip(".!?").strip()


def _parse(raw: str):
    intermediates: List[str] = []
    for m in _INT_RE.finditer(raw):
        intermediates.append(_strip(m.group(1).split("\n")[0]))
    m = _FINAL_RE.search(raw)
    final = _strip(m.group(1).split("\n")[0]) if m else ""
    if not final and intermediates:
        final = intermediates[-1]
    return intermediates, final


def run(ctx) -> Dict[str, Any]:
    prompt = (
        "Decompose the question into follow-up questions and answer each from "
        "your own knowledge. Follow the demonstration format exactly. End with "
        "a single line: `So the final answer is: X.`\n\n"
        f"{FEWSHOT}\n"
        f"Question: {ctx.question}\n"
        "Are follow up questions needed here:"
    )
    raw, _ = ctx.llm.query_gpt4o(prompt, max_tokens=600, stage="self_ask")
    intermediates, final = _parse(raw)
    sq1 = [intermediates[0]] if len(intermediates) >= 1 else []
    sq2 = [intermediates[1]] if len(intermediates) >= 2 else []
    return {
        "sq1": sq1,
        "sq2": sq2,
        "final": [final] if final else [],
        "intermediates": intermediates,
        "reasoning": raw,
    }
