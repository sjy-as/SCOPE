"""
Per-question cost counter.

Each model run process maintains:
  - a thread-local counter per in-flight question (so workers don't bleed)
  - a process-level aggregator keyed by qid

Typical usage in a per-question handler:

    from _common.cost_counter import question_scope, bump_llm, bump_retrieval

    def process_one(qid, question):
        with question_scope(qid):
            run_pipeline(...)   # bump_llm() / bump_retrieval() called inside

At end of the run:

    from _common.cost_counter import dump_summary
    dump_summary(out_dir / "cost_summary.json")

Counting rules
  - LLM: each LLM call bumps once, except calls whose stage is in VERIFY_STAGES.
  - Retrieval: 1 call per (KG entity search | Table service hit | Doc service hit).
    Relation queries / sub-edges don't count.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

VERIFY_STAGES = {"answer_verify"}

_local = threading.local()
_agg_lock = threading.Lock()
_aggregator: Dict[Any, Dict[str, Any]] = {}


@contextmanager
def question_scope(qid: Any):
    _local.counter = {"llm_calls": 0, "retrieval": defaultdict(int)}
    _local.qid = qid
    try:
        yield
    finally:
        c = _local.counter or {"llm_calls": 0, "retrieval": {}}
        with _agg_lock:
            _aggregator[qid] = {
                "llm_calls": int(c["llm_calls"]),
                "retrieval": dict(c["retrieval"]),
            }
        _local.counter = None
        _local.qid = None


def bump_llm(stage: Optional[str] = None, n: int = 1) -> None:
    if stage in VERIFY_STAGES:
        return
    c = getattr(_local, "counter", None)
    if c is not None:
        c["llm_calls"] += n


def bump_retrieval(source: Optional[str], n: int = 1) -> None:
    if not source:
        return
    c = getattr(_local, "counter", None)
    if c is not None:
        key = str(source).strip().lower()
        if key:
            c["retrieval"][key] += n


def reset_aggregator() -> None:
    with _agg_lock:
        _aggregator.clear()


def snapshot() -> Dict[Any, Dict[str, Any]]:
    with _agg_lock:
        return {k: {"llm_calls": v["llm_calls"],
                    "retrieval": dict(v["retrieval"])}
                for k, v in _aggregator.items()}


def aggregate_totals() -> Dict[str, Any]:
    per_q = snapshot()
    total_llm = sum(c["llm_calls"] for c in per_q.values())
    total_ret: Dict[str, int] = defaultdict(int)
    for c in per_q.values():
        for s, n in c["retrieval"].items():
            total_ret[s] += n
    total_ret = dict(total_ret)
    return {
        "n_questions": len(per_q),
        "llm_calls": total_llm,
        "retrieval": total_ret,
        "total_retrieval": sum(total_ret.values()),
        "per_question": {str(k): v for k, v in per_q.items()},
    }


def dump_summary(out_path: Path) -> Dict[str, Any]:
    out = aggregate_totals()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def seed_from_existing(in_path: Path) -> int:
    """Seed the in-memory aggregator from a prior cost_summary.json.

    Used on `--resume` so the new process inherits per-question cost data for
    questions that were finished in a previous session (which the resumed run
    will skip). New question_scope() exits for the same qid will overwrite,
    which is the desired behaviour.

    Returns the number of per-question entries loaded.
    """
    in_path = Path(in_path)
    if not in_path.exists() or in_path.stat().st_size == 0:
        return 0
    try:
        with in_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[cost] seed_from_existing: failed to read {in_path}: {e}", flush=True)
        return 0
    per_q = data.get("per_question") or {}
    if not isinstance(per_q, dict):
        return 0
    with _agg_lock:
        for qid, c in per_q.items():
            if not isinstance(c, dict):
                continue
            _aggregator[qid] = {
                "llm_calls": int(c.get("llm_calls", 0) or 0),
                "retrieval": dict(c.get("retrieval") or {}),
            }
    return len(per_q)


def format_summary_line(summary: Dict[str, Any]) -> str:
    """One-line printable summary for the runner's banner."""
    llm = summary.get("llm_calls", 0)
    ret = summary.get("retrieval", {}) or {}
    tot_ret = summary.get("total_retrieval", sum(ret.values()))
    ret_str = " ".join(f"{k}:{v}" for k, v in sorted(ret.items())) or "(none)"
    return f"LLM calls = {llm}  |  retrieval = {ret_str}  (total {tot_ret})"
