"""LLM-as-judge eval, ported from new_model/run.py.

Used by all seven new baselines so their predictions.jsonl carry the
same ``eval`` / ``eval_judge`` fields the new_model run already emits,
and a comparable summary.json can be diffed across methods.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMClient

EMPTY_EVAL = {
    "sq1_exact": False, "sq1_partial": False,
    "sq2_exact": False, "sq2_partial": False,
    "final_exact": False, "final_partial": False,
}


def as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [str(v).strip() for v in x if str(v).strip()]


def _judge_call(
    llm: LLMClient, question: str, items: List[Dict[str, Any]], max_attempts: int = 3,
) -> Dict[str, Dict[str, str]]:
    if not items:
        return {}
    expected = {it.get("name") for it in items if it.get("name") in {"sq1", "sq2", "final"}}
    prompt = (
        "You are a format-tolerant string-matching evaluator. For each item,\n"
        "compare Predicted to Gold ONLY. Treat Gold as the authoritative,\n"
        "ground-truth answer — even if it looks historically wrong, anachronistic,\n"
        "or contradicts your own world knowledge. DO NOT fact-check Gold against\n"
        "reality. DO NOT use the Question to second-guess Gold.\n\n"
        "Choose one verdict per item:\n"
        "  exact   : Predicted and Gold refer to the same entity / value. IGNORE\n"
        "            differences in case, articles, trailing punctuation, plurals,\n"
        "            paraphrasing, and sentence wrapping. If Predicted is a full\n"
        "            sentence containing all of Gold's information, it is exact.\n"
        "  partial : Predicted covers some Gold elements but not all.\n"
        "  miss    : empty, refusal, or a different entity/value than Gold.\n\n"
        f"Question (for context only): {question}\n\n"
        f"Items:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Output strict JSON only, no prose, no code fences:\n"
        "{\"results\": [{\"name\": \"sq1\", \"verdict\": \"exact|partial|miss\", \"reason\": \"...\"}, ...]}"
    )
    for attempt in range(1, max_attempts + 1):
        try:
            raw, _ = llm.query_gpt4o(prompt, max_tokens=600, stage="judge")
        except Exception as e:
            print(f"[eval] judge attempt {attempt} post failed: {e}")
            time.sleep(min(2 ** attempt, 8))
            continue
        m = raw[raw.find("{"): raw.rfind("}") + 1] if "{" in raw else ""
        obj: Dict[str, Any] = {}
        try:
            obj = json.loads(m) if m else {}
        except Exception:
            obj = {}
        out: Dict[str, Dict[str, str]] = {}
        for r in obj.get("results", []) or []:
            name = r.get("name")
            if name in {"sq1", "sq2", "final"}:
                v = (r.get("verdict") or "miss").strip().lower()
                if v not in {"exact", "partial", "miss"}:
                    v = "miss"
                out[name] = {"verdict": v, "reason": r.get("reason", "")}
        if expected and expected.issubset(out.keys()):
            return out
        print(f"[eval] judge attempt {attempt} parse-incomplete, retrying")
        time.sleep(min(2 ** attempt, 8))
    return {}


def eval_one(
    *,
    llm: Optional[LLMClient],
    question: str,
    pred: Dict[str, Any],
    gold: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """Returns (flags, judge_details). Flags follow EMPTY_EVAL shape."""
    if not gold:
        return dict(EMPTY_EVAL), {}
    g1 = as_list((gold.get("q1") or {}).get("answer") or (gold.get("q1") or {}).get("answers"))
    g2 = as_list((gold.get("q2") or {}).get("answer") or (gold.get("q2") or {}).get("answers"))
    p1 = as_list(pred.get("sq1"))
    p2 = as_list(pred.get("sq2"))
    pf = as_list(pred.get("final"))

    items: List[Dict[str, Any]] = []
    if g1 and p1:
        items.append({"name": "sq1", "predicted": p1, "gold": g1})
    elif g1:
        items.append({"name": "sq1", "predicted": [], "gold": g1})
    if g2:
        items.append({"name": "sq2", "predicted": p2, "gold": g2})
        items.append({"name": "final", "predicted": pf, "gold": g2})

    judge = _judge_call(llm, question, items) if (items and llm is not None) else {}
    flags = dict(EMPTY_EVAL)
    for name in ("sq1", "sq2", "final"):
        v = (judge.get(name) or {}).get("verdict", "miss")
        flags[f"{name}_exact"] = (v == "exact")
        flags[f"{name}_partial"] = (v in {"exact", "partial"})
    return flags, judge
