"""
python run_route_only.py --routing-mode graph --kb kg,doc   --input "/root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl" --output-dir "/root/autodl-tmp/new_model/result/route/kg_doc/graph"   --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"


--kb selects the live knowledge base (data sources). All of kg / table / doc
are opt-in; pass exactly the ones you want. Examples:
    --kb kg            KG only
    --kb table         Table only
    --kb doc           Doc only
    --kb kg,table      KG + Table  (matches kg-table-160 benchmark)
    --kb kg,doc        KG + Doc    (matches kg-doc-160 benchmark)
    --kb kg,table,doc  KG + Table + Doc
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pipeline as P
from step2_decompose import route as route_mod
from step2_decompose import route_baselines as route_baseline_mod
from step2_decompose import sq as sq_mod


_VALID_SOURCES = ["kg", "table", "doc"]


def _parse_kb(raw: str) -> List[str]:
    """Parse the --kb knowledge-base flag into an ordered source list.

    Accepts a comma-separated subset of {kg, table, doc}. Every source is
    opt-in. Returns the enabled sources in canonical (kg, table, doc) order.
    """
    want = {tok.strip().lower() for tok in (raw or "").split(",") if tok.strip()}
    bad = sorted(want - set(_VALID_SOURCES))
    if bad:
        raise SystemExit(
            f"[run_route_only] unknown --kb source(s): {bad}  (allowed: kg, table, doc)"
        )
    if not want:
        raise SystemExit(
            "[run_route_only] --kb must enable at least one source from {kg, table, doc}"
        )
    return [s for s in _VALID_SOURCES if s in want]


# AtomR uses Title-case labels (KG / Table / Text) in its few-shot prompt.
# When the live KB includes 'doc' we expose it to AtomR as "Text".
_ATOMR_LABEL = {"kg": "KG", "table": "Table", "doc": "Text"}


def _atomr_available_sources(enabled_sources: List[str]) -> List[str]:
    return [_ATOMR_LABEL[s] for s in enabled_sources if s in _ATOMR_LABEL]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _route_one(
    question: str,
    fusion_bundle: Dict[str, Any],
    routing_mode: str,
    profiles_path: Optional[str],
    enabled_sources: List[str],
    prompt_version: str,
) -> Dict[str, Any]:
    decomposition = sq_mod.decompose_question(
        question,
        model=P.LLM_MODEL,
        base_url=P.LLM_BASE_URL,
        api_key=P.LLM_API_KEY,
    )
    subqueries = decomposition.get("subqueries") or []
    q1 = ((subqueries[0] or {}).get("query") if len(subqueries) > 0 else question) or question
    q2 = ((subqueries[1] or {}).get("query") if len(subqueries) > 1 else question) or question

    def _route_subquery(subquery: str) -> Dict[str, Any]:
        if routing_mode == "graph":
            return route_mod.source_routing(
                matched_kind="none",
                matched_item=None,
                bundle=fusion_bundle,
                query=subquery,
                model=P.LLM_MODEL,
                base_url=P.LLM_BASE_URL,
                api_key=P.LLM_API_KEY,
                verbose=False,
                ref_type="entity",
                profiles_path=profiles_path,
                prompt_version=prompt_version,
                enabled_sources=enabled_sources,
            )
        if routing_mode == "atomr":
            return route_baseline_mod.route_atomr_few_shot(
                query=subquery,
                model=P.LLM_MODEL,
                base_url=P.LLM_BASE_URL,
                api_key=P.LLM_API_KEY,
                available_sources=_atomr_available_sources(enabled_sources),
                enabled_sources=enabled_sources,
                verbose=False,
            )
        if routing_mode == "deepsieve":
            if not profiles_path:
                raise ValueError("deepsieve routing requires profiles_path")
            return route_baseline_mod.route_deepsieve(
                query=subquery,
                profiles_path=profiles_path,
                enabled_sources=enabled_sources,
                model=P.LLM_MODEL,
                base_url=P.LLM_BASE_URL,
                api_key=P.LLM_API_KEY,
                verbose=False,
            )
        raise ValueError(f"Unsupported routing mode: {routing_mode}")

    r1 = _route_subquery(q1)
    r2 = _route_subquery(q2)
    return {
        "index": -1,
        "type": None,
        "question": question,
        "sub_q1": q1,
        "sub_q2": q2,
        "reasoning": decomposition.get("reasoning", ""),
        "rewrite_reasoning": decomposition.get("reasoning", ""),
        "match1": None,
        "match2": None,
        "route1": r1,
        "route2": r2,
        "decomposition": decomposition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONLY the routing stage of pipeline_v2")
    parser.add_argument("--input",  default="qa_bench/kg-table-160.jsonl")
    parser.add_argument("--output-dir", default="result_route_only")
    parser.add_argument("--max",    type=int, default=None)
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--llm-url",
        default=os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1"))
    parser.add_argument("--llm-model",
        default=os.getenv("LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--api-key",
        default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--routing-mode",
        default=os.getenv("ROUTING_MODE", "graph"),
        choices=["graph", "atomr", "deepsieve"])
    parser.add_argument("--profiles-path",
        default=str((Path(__file__).resolve().parent / "data_sources/source_profiles.json")))
    parser.add_argument("--prompt-version",
        default=os.getenv("PROMPT_VERSION", "v2"),
        choices=["v1", "v2"],
        help="Prompt version for routing_mode=graph. v2 is ref-type-aware + "
             "profile-context (default); v1 is the legacy prompt.")
    parser.add_argument("--kb",
        default=os.getenv("KB", "kg"),
        help="Knowledge base = the live data sources. Comma-separated subset of "
             "{kg,table,doc}; every source is opt-in. The router confines "
             "primary/fallback routing to these sources and the semantic "
             "catalog is filtered to them. Examples: 'kg' (KG only, default), "
             "'table', 'doc', 'kg,table', 'kg,doc', 'kg,table,doc'.")
    args = parser.parse_args()

    enabled_sources = _parse_kb(args.kb)

    if not args.api_key:
        raise SystemExit("[run_route_only] --api-key (or LLM_API_KEY env) required for step2 LLM calls.")
    if args.routing_mode == "deepsieve" and not Path(args.profiles_path).exists():
        raise SystemExit(
            f"[run_route_only] --routing-mode=deepsieve requires --profiles-path "
            f"to exist (got {args.profiles_path})."
        )

    # Apply LLM overrides on the pipeline_v2 module (sq.process_item uses these).
    P.LLM_BASE_URL = args.llm_url
    P.LLM_MODEL    = args.llm_model
    P.LLM_API_KEY  = args.api_key
    print(f"[run_route_only] step2 LLM    : model={P.LLM_MODEL}  url={P.LLM_BASE_URL}")

    root = Path(__file__).resolve().parent
    in_path = (root / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    out_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    routes_path  = out_dir / "routes.jsonl"
    summary_path = out_dir / "summary.json"

    questions = _load_jsonl(in_path)
    if args.start:
        questions = questions[args.start:]
    if args.max is not None:
        questions = questions[: args.max]

    print("=" * 70)
    print(f"Input         : {in_path}")
    print(f"Output dir    : {out_dir}")
    print(f"Questions     : {len(questions)}")
    print(f"Routing mode  : {args.routing_mode}"
          + (f"  (profiles={args.profiles_path})" if args.routing_mode in {"deepsieve", "graph"} else ""))
    print(f"Knowledge base: {', '.join(enabled_sources)}")
    print(f"Prompt ver.   : {args.prompt_version}"
          + ("  (only affects graph mode)" if args.routing_mode != "graph" else ""))
    print("=" * 70)

    # Stage 1 (loads the semantic catalog used by routing).
    # Filter the semantic catalog to the live KB so routing only ever sees
    # entries for the enabled sources.
    fusion_state = P.stage1_check(enabled_sources)
    bundle = fusion_state["bundle"]

    work: List[Tuple[int, int, str]] = []
    for i, rec in enumerate(questions, start=1):
        idx = rec.get("index", i - 1)
        question = (rec.get("question") or "").strip()
        if not question:
            continue
        work.append((i, idx, question))

    def process_one(i: int, idx: int, question: str) -> Tuple[int, int, Dict[str, Any], bool]:
        try:
            r = _route_one(
                question=question,
                fusion_bundle=bundle,
                routing_mode=args.routing_mode,
                profiles_path=args.profiles_path,
                enabled_sources=enabled_sources,
                prompt_version=args.prompt_version,
            )
            out = {
                "index": idx,
                "question": question,
                "routing_mode": args.routing_mode,
                "knowledge_base": enabled_sources,
                "sub_q1": r.get("sub_q1"),
                "sub_q2": r.get("sub_q2"),
                "rewrite_reasoning": r.get("rewrite_reasoning"),
                "match1": r.get("match1"),
                "match2": r.get("match2"),
                "route1": r.get("route1"),
                "route2": r.get("route2"),
                "error": None,
            }
            return (i, idx, out, True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return (i, idx, {
                "index": idx,
                "question": question,
                "routing_mode": args.routing_mode,
                "knowledge_base": enabled_sources,
                "error": str(e),
            }, False)

    write_lock = threading.Lock()
    workers = max(1, int(args.workers or 1))
    total = len(work)
    completed = 0
    ok = 0
    rows_out: List[Dict[str, Any]] = []

    with routes_path.open("w", encoding="utf-8") as wf:
        def _emit(i: int, idx: int, row: Dict[str, Any], success: bool) -> None:
            nonlocal completed, ok
            with write_lock:
                completed += 1
                if success:
                    ok += 1
                rows_out.append(row)
                wf.write(json.dumps(row, ensure_ascii=False) + "\n")
                wf.flush()
                os.fsync(wf.fileno())
                tag = "OK " if success else "ERR"
                r1 = (row.get("route1") or {})
                r2 = (row.get("route2") or {})
                # Mark routes that came from a parse-failure fallback so we
                # can spot silently-defaulted decisions in the log.
                m1 = " [FALLBACK]" if r1.get("parse_failed") else ""
                m2 = " [FALLBACK]" if r2.get("parse_failed") else ""
                print(f"[{completed}/{total}] {tag} index={idx}  "
                      f"r1={r1.get('primary_source')}->{r1.get('fallback_source')}{m1}  "
                      f"r2={r2.get('primary_source')}->{r2.get('fallback_source')}{m2}")

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process_one, *w) for w in work]
                for fut in as_completed(futs):
                    i, idx, row, success = fut.result()
                    _emit(i, idx, row, success)
                    if (not success) and (not args.continue_on_error):
                        # Cancel pending and bail.
                        for f in futs:
                            f.cancel()
                        raise RuntimeError(f"route_only failed at index={idx}: {row.get('error')}")
        else:
            for w in work:
                i, idx, row, success = process_one(*w)
                _emit(i, idx, row, success)
                if (not success) and (not args.continue_on_error):
                    raise RuntimeError(f"route_only failed at index={idx}: {row.get('error')}")

    # Aggregate per-mode counts.
    prim_counts = Counter()
    fb_counts   = Counter()
    layer_counts = Counter()
    parse_failed_counts = Counter()
    for row in rows_out:
        for k in ("route1", "route2"):
            r = row.get(k) or {}
            if r.get("primary_source"):
                prim_counts[(k, r["primary_source"])] += 1
            if r.get("fallback_source"):
                fb_counts[(k, r["fallback_source"])] += 1
            if r.get("routing_layer"):
                layer_counts[(k, r["routing_layer"])] += 1
            if r.get("parse_failed"):
                parse_failed_counts[k] += 1

    summary = {
        "input": str(in_path),
        "output_dir": str(out_dir),
        "routing_mode": args.routing_mode,
        "knowledge_base": enabled_sources,
        "prompt_version": args.prompt_version,
        "profiles_path": args.profiles_path if args.routing_mode in {"deepsieve", "graph"} else None,
        "total": total,
        "succeeded": ok,
        "failed": total - ok,
        # Number of route decisions that came from a parse-failure fallback
        # (model returned empty / unparseable content). High values mean the
        # "primary_source" counts below are mostly defaults, not real picks.
        "parse_failed_counts": {k: parse_failed_counts.get(k, 0) for k in ("route1", "route2")},
        "primary_source_counts": {f"{k}.{v}": c for (k, v), c in sorted(prim_counts.items())},
        "fallback_source_counts": {f"{k}.{v}": c for (k, v), c in sorted(fb_counts.items())},
        "routing_layer_counts": {f"{k}.{v}": c for (k, v), c in sorted(layer_counts.items())},
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Routes       -> {routes_path}")
    print(f"Summary      -> {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
