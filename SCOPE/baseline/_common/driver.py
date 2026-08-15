"""Common run-driver for the simple LLM / RAG baselines.

Each baseline's run.py builds a ``BaselineRunner``, wires in a
``pipeline_fn(question, ctx) -> result_dict`` (with at least
``sq1`` / ``sq2`` / ``final`` keys), then calls ``run()``.

The driver handles:
  * parsing --input / --output-dir / --kb / --workers / --resume / LLM args
  * loading the jsonl, skipping already-done indexes when --resume
  * spawning threads
  * per-row trace writes under <output_dir>/traces/idx_<index>.trace.json
  * appending predictions.jsonl with eval flags + LLM judge details
  * summary.json with the same six counters new_model emits
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .eval import EMPTY_EVAL, eval_one
from .llm import LLMClient
from .retrievers import (
    DEFAULT_DOC_API, DEFAULT_KG_DIR, DEFAULT_TABLE_API,
    DocRetriever, KGRetriever, TableRetriever,
)

try:
    code_root = str(Path(__file__).resolve().parents[2])
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    from _common.cost_counter import (
        question_scope as _question_scope,
        dump_summary as _dump_cost_summary,
        seed_from_existing as _seed_cost_summary,
        format_summary_line as _format_cost_line,
        reset_aggregator as _reset_cost_agg,
    )
except Exception:  # pragma: no cover
    from contextlib import nullcontext as _question_scope  # type: ignore
    def _dump_cost_summary(_p): return {}
    def _seed_cost_summary(_p): return 0
    def _format_cost_line(_s): return ""
    def _reset_cost_agg(): pass

_VALID_SOURCES = ["kg", "table", "doc"]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _parse_kb(raw: str) -> List[str]:
    want = {t.strip().lower() for t in (raw or "").split(",") if t.strip()}
    bad = sorted(want - set(_VALID_SOURCES))
    if bad:
        raise SystemExit(f"[run] unknown --kb source(s): {bad}  (allowed: kg, table, doc)")
    if not want:
        raise SystemExit("[run] --kb must enable at least one source from {kg, table, doc}")
    return [s for s in _VALID_SOURCES if s in want]


@dataclass
class RunContext:
    """Per-question execution context passed to ``pipeline_fn``."""
    question: str
    index: Any
    enabled_sources: List[str]
    llm: LLMClient
    kg_retriever: Optional[KGRetriever]
    table_retriever: Optional[TableRetriever]
    doc_retriever: Optional[DocRetriever]
    extra: Dict[str, Any] = field(default_factory=dict)


PipelineFn = Callable[[RunContext], Dict[str, Any]]


def make_parser(name: str, *, needs_retrieval: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"{name} baseline driver")
    ap.add_argument("--input", required=True, help="qa_bench jsonl path")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max", type=int, default=None, help="only run the first N rows")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="append to existing predictions.jsonl and skip done indexes")
    ap.add_argument("--llm-url", default=os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    ap.add_argument("--judge", action="store_true",
                    help="also run LLM-as-judge eval inline (default off; use eval script later)")
    if needs_retrieval:
        ap.add_argument("--kb", default="kg",
                        help="comma-separated subset of {kg,table,doc}")
        ap.add_argument("--kg-dir", default=DEFAULT_KG_DIR)
        ap.add_argument("--table-url", default=DEFAULT_TABLE_API)
        ap.add_argument("--doc-url", default=DEFAULT_DOC_API)
    return ap


def run_baseline(
    args: argparse.Namespace,
    *,
    name: str,
    pipeline_fn: PipelineFn,
    needs_retrieval: bool = True,
    build_extra: Optional[Callable[[argparse.Namespace, LLMClient], Dict[str, Any]]] = None,
) -> None:
    if not args.api_key:
        raise SystemExit(f"[{name}] --api-key (or env LLM_API_KEY) required")

    in_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    questions = _load_jsonl(in_path)
    if args.start:
        questions = questions[args.start:]
    if args.max is not None:
        questions = questions[: args.max]

    llm = LLMClient(api_key=args.api_key, base_url=args.llm_url, model=args.llm_model)

    enabled: List[str] = []
    kg_retriever = table_retriever = doc_retriever = None
    if needs_retrieval:
        enabled = _parse_kb(args.kb)
        if "kg" in enabled:
            print(f"[{name}] loading KG from {args.kg_dir} ...")
            kg_retriever = KGRetriever(args.kg_dir, llm=llm)
            print(f"[{name}]   KG loaded: {len(kg_retriever.kg.qid2label)} entities, "
                  f"{len(kg_retriever.kg.relations)} relations")
        if "table" in enabled:
            table_retriever = TableRetriever(api_url=args.table_url)
            if not table_retriever.is_alive():
                print(f"[{name}]   [WARN] Table BM25 {args.table_url} unreachable")
        if "doc" in enabled:
            doc_retriever = DocRetriever(api_url=args.doc_url)
            if not doc_retriever.is_alive():
                print(f"[{name}]   [WARN] Doc ColBERT {args.doc_url} unreachable")

    extra = build_extra(args, llm) if build_extra else {}

    _reset_cost_agg()

    print("=" * 70)
    print(f"{name} baseline")
    print(f"Input      : {in_path}  ({len(questions)} rows)")
    print(f"Output dir : {out_dir}")
    print(f"LLM        : {args.llm_model} @ {args.llm_url}")
    if needs_retrieval:
        print(f"KB         : {','.join(enabled)}")
    print(f"Workers    : {args.workers}   inline-judge: {bool(args.judge)}")
    print("=" * 70)

    done: set = set()
    eval_rows: List[Dict[str, Any]] = []
    ok = 0
    if args.resume and pred_path.exists():
        for line in pred_path.open("r", encoding="utf-8"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "index" in obj:
                done.add(obj["index"])
                eval_rows.append(obj.get("eval") or dict(EMPTY_EVAL))
                if not obj.get("error"):
                    ok += 1
        print(f"[{name}] resume: {len(done)} rows already done, will skip")
        _seeded = _seed_cost_summary(out_dir / "cost_summary.json")
        if _seeded:
            print(f"[{name}] resume: seeded cost aggregator with {_seeded} prior entries")

    gold_map = {r["index"]: r for r in questions if "index" in r}

    work: List[tuple] = []
    for rec in questions:
        idx = rec.get("index")
        q = (rec.get("question") or "").strip()
        if not q or idx in done:
            continue
        work.append((idx, q, gold_map.get(idx)))

    def process_one(idx: Any, question: str, gold: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ctx = RunContext(
            question=question, index=idx, enabled_sources=enabled, llm=llm,
            kg_retriever=kg_retriever, table_retriever=table_retriever,
            doc_retriever=doc_retriever, extra=extra,
        )
        llm.start_trace()
        with _question_scope(idx):
            try:
                res = pipeline_fn(ctx)
                err = None
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                res = {"sq1": [], "sq2": [], "final": []}
                err = str(e)
        llm_calls = llm.pop_trace()

        pred = {
            "index": idx,
            "question": question,
            "sq1": res.get("sq1") or [],
            "sq2": res.get("sq2") or [],
            "final": res.get("final") or [],
            "error": err,
            "extra": {k: v for k, v in res.items() if k not in {"sq1", "sq2", "final"}},
        }
        if args.judge:
            flags, judge = eval_one(llm=llm, question=question, pred=pred, gold=gold)
            pred["eval"] = flags
            pred["eval_judge"] = judge
        else:
            pred["eval"] = dict(EMPTY_EVAL)
            pred["eval_judge"] = {}

        _write_json(traces_dir / f"idx_{idx}.trace.json", {
            "index": idx, "question": question, "pipeline": res,
            "llm_calls": llm_calls, "error": err,
        })
        return pred

    write_lock = threading.Lock()
    total = len(work)
    completed = 0

    open_mode = "a" if args.resume else "w"
    with pred_path.open(open_mode, encoding="utf-8") as wf:
        def emit(pred: Dict[str, Any]) -> None:
            nonlocal completed, ok
            with write_lock:
                completed += 1
                if not pred.get("error"):
                    ok += 1
                eval_rows.append(pred["eval"])
                tag = "OK " if not pred.get("error") else "ERR"
                final_preview = (pred["final"][:1] or [""])[0][:60]
                print(f"[{completed}/{total}] {tag} idx={pred['index']}  final={final_preview!r}")
                wf.write(json.dumps(pred, ensure_ascii=False) + "\n")
                wf.flush()

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(process_one, *w) for w in work]
                for fut in as_completed(futs):
                    emit(fut.result())
        else:
            for w in work:
                emit(process_one(*w))

    n = len(eval_rows)
    summary = {
        "input": str(in_path),
        "output_dir": str(out_dir),
        "method": name,
        "knowledge_base": enabled if needs_retrieval else [],
        "llm_model": args.llm_model,
        "total": n,
        "succeeded": ok,
        "failed": n - ok,
        "llm_calls": llm.call_count,
        "tokens_in": llm.token_in,
        "tokens_out": llm.token_out,
    }
    for k in ("sq1_exact", "sq1_partial", "sq2_exact", "sq2_partial",
              "final_exact", "final_partial"):
        summary[k] = sum(1 for r in eval_rows if r.get(k))
    _write_json(summary_path, summary)

    cost_path = out_dir / "cost_summary.json"
    cost_summary = _dump_cost_summary(cost_path)

    print("\n" + "=" * 70)
    print(f"Done. {ok}/{n} succeeded, {n - ok} failed.")
    print(f"Predictions -> {pred_path}")
    print(f"Traces      -> {traces_dir}/idx_<index>.trace.json")
    print(f"Summary     -> {summary_path}")
    print(f"Cost summary-> {cost_path}")
    print(f"LLM calls   : {llm.call_count}  (in={llm.token_in} out={llm.token_out})")
    print(f"[cost] {_format_cost_line(cost_summary)}")
    if not args.judge:
        print("[hint] re-run with --judge to compute eval flags inline, "
              "or call _common/eval_cli.py later.")
    print("=" * 70)
