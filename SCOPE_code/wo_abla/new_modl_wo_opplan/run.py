"""
Ablation: new_model decompose + semantic routing + DeepSieve RAG retrieval.

Identical front-end to new_model (decomposition into 2 sub-questions +
semantic graph routing), but replaces new_model's operator-tree executor with
DeepSieve's plain retrieve-then-answer loop.

Pipeline summary
────────────────
Stage 1  : Load & filter semantic catalog (new_model).
Stage 2a : Decompose question → 2 sub-questions (new_model sq.decompose_question).
           Route each sub-question (new_model route.source_routing).
Stage 3  : For each sub-question:
             a. Retrieve top-k evidence from the primary source (DeepSieve RAG).
             b. LLM answers from the retrieved evidence (DeepSieve prompt style).
             c. If no answer, re-retrieve from the fallback source and re-answer.
             d. Resolve [1] reference in sq2 with sq1's answer before retrieval.
Stage 4  : Synthesise final answer from sub-question chain (DeepSieve fusion).

Example runs
────────────
python /root/autodl-tmp/baseline/new_model_deepsieve_retrieval/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --kb kg,table \
  --workers 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"

python /root/autodl-tmp/baseline/new_model_deepsieve_retrieval/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --kb kg,doc \
  --workers 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"

python /root/autodl-tmp/baseline/new_model_deepsieve_retrieval/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/table-doc-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/table-doc-160.jsonl \
  --kb table,doc \
  --workers 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Ensure our pipeline.py takes priority over new_model's pipeline.py
# ---------------------------------------------------------------------------
_THIS_DIR  = Path(__file__).resolve().parent
_NM_DIR    = Path("/root/autodl-tmp/new_model")
_DS_DIR    = Path("/root/autodl-tmp/baseline/DeepSieve")

# Append supporting paths without pushing them to the front.
for _p in [str(_NM_DIR), str(_NM_DIR / "step2_decompose"), str(_DS_DIR)]:
    if _p not in sys.path:
        sys.path.append(_p)

# Force THIS_DIR to index 0.
# Python may have already added it automatically when launching the script,
# but subsequent inserts above can shift it behind new_model/pipeline.py.
# Remove-then-reinsert guarantees our pipeline.py is always found first.
_s = str(_THIS_DIR)
if _s in sys.path:
    sys.path.remove(_s)
sys.path.insert(0, _s)

import pipeline as P   # our pipeline.py (THIS_DIR is first)


# =====================================================================
# Helpers
# =====================================================================

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _build_gold_map(path: Optional[Path]) -> Dict[Any, Dict[str, Any]]:
    if not path or not path.exists():
        return {}
    out: Dict[Any, Dict[str, Any]] = {}
    for r in _load_jsonl(path):
        idx = r.get("index")
        if idx is not None:
            out[idx] = r
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    return [str(v).strip() for v in x if str(v).strip()]


_VALID_SOURCES = ["kg", "table", "doc"]


def _parse_kb(raw: str) -> List[str]:
    want = {tok.strip().lower() for tok in (raw or "").split(",") if tok.strip()}
    bad  = sorted(want - set(_VALID_SOURCES))
    if bad:
        raise SystemExit(f"[run] unknown --kb source(s): {bad}  (allowed: kg, table, doc)")
    if not want:
        raise SystemExit("[run] --kb must enable at least one source from {kg, table, doc}")
    return [s for s in _VALID_SOURCES if s in want]


# =====================================================================
# LLM-judge evaluation (mirrors new_model run.py)
# =====================================================================

_EMPTY_EVAL = {
    "sq1_exact": False, "sq1_partial": False,
    "sq2_exact": False, "sq2_partial": False,
    "final_exact": False, "final_partial": False,
}


def _post_llm_simple(prompt: str, url: str, model: str, api_key: str) -> str:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    endpoint = url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    sess = requests.Session()
    retry = Retry(total=3, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["POST"], raise_on_status=False)
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    resp = sess.post(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.0, "max_tokens": 600},
        timeout=300,
        proxies={"http": None, "https": None},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json_loose(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1] if "\n" in s else s[3:]
        s = s.rsplit("```", 1)[0].strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            try:
                obj = json.loads(s[a:b+1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def _llm_judge_eval(
    question: str,
    items: List[Dict[str, Any]],
    llm_url: str,
    llm_model: str,
    api_key: str,
    max_attempts: int = 3,
) -> Dict[str, Dict[str, str]]:
    if not items:
        return {}
    prompt = (
        "You are a format-tolerant string-matching evaluator. For each item,\n"
        "compare Predicted to Gold ONLY. DO NOT fact-check Gold. Your sole job\n"
        "is to decide whether Predicted conveys the same information as Gold.\n\n"
        "Choose one verdict per item:\n"
        "  exact   : same entity/value (ignore case, articles, punctuation).\n"
        "  partial : covers some Gold elements but not all.\n"
        "  miss    : empty, refusal, or different entity/value.\n\n"
        f"Question (context only): {question}\n\n"
        f"Items:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Output strict JSON only:\n"
        '{"results": [{"name": "sq1", "verdict": "exact|partial|miss", "reason": "..."}, ...]}'
    )
    expected = {it["name"] for it in items if it.get("name") in {"sq1", "sq2", "final"}}
    for attempt in range(1, max_attempts + 1):
        try:
            raw = _post_llm_simple(prompt, llm_url, llm_model, api_key)
        except Exception as e:
            print(f"[eval] judge attempt {attempt}/{max_attempts} failed: {e}")
            time.sleep(min(2 ** attempt, 8))
            continue
        obj = _parse_json_loose(raw) or {}
        out: Dict[str, Dict[str, str]] = {}
        for r in obj.get("results") or []:
            name = r.get("name")
            if name in {"sq1", "sq2", "final"}:
                v = (r.get("verdict") or "miss").strip().lower()
                if v not in {"exact", "partial", "miss"}:
                    v = "miss"
                out[name] = {"verdict": v, "reason": r.get("reason", "")}
        if expected.issubset(out.keys()):
            return out
        print(f"[eval] judge attempt {attempt} incomplete: got {sorted(out.keys())}, expected {sorted(expected)}")
        time.sleep(min(2 ** attempt, 8))
    return {}


def _eval_one(
    question: str,
    pred: Dict[str, Any],
    gold: Optional[Dict[str, Any]],
    llm_url: str,
    llm_model: str,
    api_key: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    if not gold:
        return dict(_EMPTY_EVAL), {}

    g1 = _as_list((gold.get("q1") or {}).get("answer") or (gold.get("q1") or {}).get("answers"))
    g2 = _as_list((gold.get("q2") or {}).get("answer") or (gold.get("q2") or {}).get("answers"))
    p1 = _as_list(pred.get("sq1"))
    p2 = _as_list(pred.get("sq2"))
    pf = _as_list(pred.get("final"))

    items: List[Dict[str, Any]] = []
    if g1:
        items.append({"name": "sq1", "predicted": p1, "gold": g1})
    if g2:
        items.append({"name": "sq2", "predicted": p2, "gold": g2})
        items.append({"name": "final", "predicted": pf, "gold": g2})

    judge: Dict[str, Dict[str, str]] = {}
    if items and llm_url and api_key:
        judge = _llm_judge_eval(question, items, llm_url, llm_model, api_key)

    flags = dict(_EMPTY_EVAL)
    for name in ("sq1", "sq2", "final"):
        v = (judge.get(name) or {}).get("verdict", "miss")
        flags[f"{name}_exact"]   = (v == "exact")
        flags[f"{name}_partial"] = (v in {"exact", "partial"})
    return flags, judge


# =====================================================================
# Per-question processor
# =====================================================================

def process_one(
    i: int, idx: Any, question: str, gold: Optional[Dict[str, Any]],
    fusion_state: Dict[str, Any],
    rag_sources: Dict[str, Any],
    enabled_sources: List[str],
    profiles_path: str,
    prompt_version: str,
    traces_dir: Path,
    llm_url: str, llm_model: str, api_key: str,
) -> Tuple[int, Any, Dict[str, Any], bool]:
    row_trace_dir = traces_dir / f"idx_{idx}"
    row_trace_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = P.run_pipeline_one(
            question=question,
            fusion_state=fusion_state,
            rag_sources=rag_sources,
            enabled_sources=enabled_sources,
            profiles_path=profiles_path,
            prompt_version=prompt_version,
        )

        # Write per-sq trace files
        for sq_res in res.get("sq_results", []):
            sq_id = sq_res.get("sq_id") or sq_res.get("subquery_id") or "sq_unknown"
            _write_json(row_trace_dir / f"{sq_id}.trace.json", sq_res)
        _write_json(row_trace_dir / "decompose.trace.json", res.get("decompose", {}))
        _write_json(row_trace_dir / "final.trace.json", {
            "question":     question,
            "final_answer": res.get("final_answer"),
            "final_reason": res.get("final_reason"),
        })

        pred = {
            "index":    idx,
            "question": question,
            "sq1":      res["sq_answers"].get("sq1", []),
            "sq2":      res["sq_answers"].get("sq2", []),
            "final":    [res["final_answer"]] if res.get("final_answer") else [],
            "error":    None,
        }
        flags, judge = _eval_one(
            question=question, pred=pred, gold=gold,
            llm_url=llm_url, llm_model=llm_model, api_key=api_key,
        )
        pred["eval"]       = flags
        pred["eval_judge"] = judge
        return (i, idx, pred, True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        pred = {
            "index": idx, "question": question,
            "sq1": [], "sq2": [], "final": [],
            "error": str(e), "eval": dict(_EMPTY_EVAL), "eval_judge": {},
        }
        return (i, idx, pred, False)


# =====================================================================
# CLI
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="new_model_deepsieve_retrieval: decompose+route (new_model) + RAG execute (DeepSieve)")
    parser.add_argument("--input",      default="qa_bench/kg-table-26.jsonl")
    parser.add_argument("--gold",       default="qa_bench/kg-table-26.jsonl")
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--max",    type=int, default=None)
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip indexes already present in predictions.jsonl.")
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--llm-url",
                        default=os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1"))
    parser.add_argument("--llm-model",  default=os.getenv("LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--api-key",    default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--prompt-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--profiles-path",
                        default=str(_NM_DIR / "data_sources/source_profiles.json"))
    parser.add_argument("--kb", default=os.getenv("KB", "kg"),
                        help="Live knowledge base: comma-separated subset of {kg,table,doc}.")
    args = parser.parse_args()

    enabled_sources = _parse_kb(args.kb)

    if not args.api_key:
        raise SystemExit("[run] --api-key (or LLM_API_KEY env) must be provided.")

    # Push LLM settings into pipeline module so all helpers see them
    P.LLM_BASE_URL = args.llm_url
    P.LLM_MODEL    = args.llm_model
    P.LLM_API_KEY  = args.api_key

    # Also push into env vars so DeepSieve's utils.llm_call and rag submodules pick them up
    os.environ["LLM_BASE_URL"]   = args.llm_url
    os.environ["LLM_MODEL"]      = args.llm_model
    os.environ["LLM_API_KEY"]    = args.api_key
    os.environ["OPENAI_API_BASE"] = args.llm_url
    os.environ["OPENAI_BASE_URL"] = args.llm_url
    os.environ["OPENAI_API_KEY"]  = args.api_key
    os.environ["OPENAI_MODEL"]    = args.llm_model
    os.environ["ATOMR_KG_PARSER_MODEL"] = args.llm_model

    print(f"[run] LLM : model={args.llm_model}  url={args.llm_url}")
    print(f"[run] KB  : {', '.join(enabled_sources)}")

    root     = _THIS_DIR
    in_path  = (root / args.input).resolve()  if not Path(args.input).is_absolute()  else Path(args.input)
    gold_path = (root / args.gold).resolve()   if not Path(args.gold).is_absolute()   else Path(args.gold)
    out_dir  = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir   = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    pred_path    = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    questions = _load_jsonl(in_path)
    if args.start:
        questions = questions[args.start:]
    if args.max is not None:
        questions = questions[:args.max]
    gold_map = _build_gold_map(gold_path)

    print("=" * 70)
    print(f"Input       : {in_path}")
    print(f"Gold        : {gold_path}  ({len(gold_map)} entries)")
    print(f"Output dir  : {out_dir}")
    print(f"Questions   : {len(questions)}")
    print(f"KB          : {', '.join(enabled_sources)}")
    print(f"Prompt ver. : {args.prompt_version}")
    print("=" * 70)

    # Stage 1 — catalog
    fusion_state = P.stage1_check(enabled_sources)

    # Init DeepSieve RAG sources
    print("\n" + "=" * 70)
    print("Initializing DeepSieve RAG sources...")
    print("=" * 70)
    rag_sources = P.build_rag_sources(enabled_sources)

    # Build work list (skip already-done if --resume)
    done_idx: set = set()
    eval_rows: List[Dict[str, Any]] = []
    ok = 0
    if args.resume and pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if obj.get("index") is not None:
                    done_idx.add(obj["index"])
                    eval_rows.append(obj.get("eval") or dict(_EMPTY_EVAL))
                    if not obj.get("error"):
                        ok += 1
        print(f"[resume] {len(done_idx)} indexes already done, will skip them")

    work: List[Tuple[int, Any, str, Optional[Dict[str, Any]]]] = []
    for i, rec in enumerate(questions, start=1):
        idx      = rec.get("index", i - 1)
        question = (rec.get("question") or "").strip()
        if not question:
            continue
        if idx in done_idx:
            print(f"[{i}/{len(questions)}] index={idx} -- already done, skipping")
            continue
        gold = gold_map.get(idx)
        work.append((i, idx, question, gold))

    # Process
    common_kwargs = dict(
        fusion_state=fusion_state,
        rag_sources=rag_sources,
        enabled_sources=enabled_sources,
        profiles_path=args.profiles_path,
        prompt_version=args.prompt_version,
        traces_dir=traces_dir,
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        api_key=args.api_key,
    )
    write_lock = threading.Lock()
    total     = len(work)
    completed = 0
    workers   = max(1, int(args.workers or 1))

    open_mode = "a" if args.resume else "w"
    with pred_path.open(open_mode, encoding="utf-8") as wf:
        def _emit(i: int, idx: Any, pred: Dict[str, Any], success: bool) -> None:
            nonlocal completed, ok
            with write_lock:
                completed += 1
                eval_rows.append(pred["eval"])
                if success:
                    ok += 1
                tag = "OK " if success else "ERR"
                print(f"[{completed}/{total}] {tag} index={idx}")
                wf.write(json.dumps(pred, ensure_ascii=False) + "\n")
                wf.flush()
                os.fsync(wf.fileno())

        if workers > 1:
            print(f"\n[run] parallel workers={workers}")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process_one, *w, **common_kwargs) for w in work]
                for fut in as_completed(futs):
                    i, idx, pred, success = fut.result()
                    _emit(i, idx, pred, success)
        else:
            for w in work:
                print("\n" + "#" * 70)
                print(f"[{w[0]}/{len(questions)}] index={w[1]}")
                print("#" * 70)
                i, idx, pred, success = process_one(*w, **common_kwargs)
                _emit(i, idx, pred, success)
                if not success and not args.continue_on_error:
                    raise RuntimeError(f"pipeline failed at index={idx}: {pred.get('error')}")

    # Summary
    n = len(eval_rows)
    summary = {
        "input":          str(in_path),
        "gold":           str(gold_path),
        "output_dir":     str(out_dir),
        "knowledge_base": enabled_sources,
        "prompt_version": args.prompt_version,
        "total":     n,
        "succeeded": ok,
        "failed":    n - ok,
    }
    for k in ("sq1_exact", "sq1_partial", "sq2_exact", "sq2_partial",
              "final_exact", "final_partial"):
        summary[k] = sum(1 for r in eval_rows if r.get(k))
    _write_json(summary_path, summary)

    print("\n" + "=" * 70)
    print("Run finished.")
    print(f"Processed  : {summary['total']}")
    print(f"Succeeded  : {summary['succeeded']}")
    print(f"Failed     : {summary['failed']}")
    for k in ("sq1_exact", "sq1_partial", "sq2_exact", "sq2_partial",
              "final_exact", "final_partial"):
        print(f"  {k}: {summary[k]}")
    print(f"Predictions  -> {pred_path}")
    print(f"Per-sq traces-> {traces_dir}/idx_<index>/*.trace.json")
    print(f"Summary      -> {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
