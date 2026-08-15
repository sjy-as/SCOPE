"""Stand-alone post-hoc LLM-judge for any baseline's predictions.jsonl.

Use when you ran the pipeline without --judge and want to add eval
flags later, or to re-judge with a different LLM.

  python3 -m baseline._common.eval_cli \
    --pred  baseline/StandardPrompt/result/kg-table/predictions.jsonl \
    --gold  SCOPE_code/SCOPE/qa_bench/kg-table-1147.jsonl \
    --api-key "$LLM_API_KEY"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow `python3 eval_cli.py ...` from inside _common/ as well as
# `python3 -m baseline._common.eval_cli`.
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from _common.eval import EMPTY_EVAL, eval_one  # noqa: E402
from _common.llm import LLMClient  # noqa: E402


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc LLM-judge for baseline predictions")
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--llm-url", default=os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1"))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "deepseek-chat"))
    ap.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("--api-key required")

    pred_path = Path(args.pred).resolve()
    gold_path = Path(args.gold).resolve()
    out_path = Path(args.output).resolve() if args.output else pred_path.parent / "eval_report.json"

    preds = _load_jsonl(pred_path)
    gold_map = {r["index"]: r for r in _load_jsonl(gold_path) if "index" in r}
    llm = LLMClient(api_key=args.api_key, base_url=args.llm_url, model=args.llm_model)

    lock = threading.Lock()
    done = [0]

    def judge_row(p):
        gold = gold_map.get(p.get("index"))
        flags, judge = eval_one(
            llm=llm, question=p.get("question", ""), pred=p, gold=gold,
        )
        with lock:
            done[0] += 1
            v = "exact" if flags["final_exact"] else ("partial" if flags["final_partial"] else "miss")
            print(f"[{done[0]}/{len(preds)}] idx={p.get('index')}  final={v}")
        return {**p, "eval": flags, "eval_judge": judge}

    results = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(judge_row, p) for p in preds]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for p in preds:
            results.append(judge_row(p))

    results.sort(key=lambda r: (r.get("index") is None, r.get("index")))

    # Rewrite predictions.jsonl with eval populated.
    with pred_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(results)
    summary = {
        "pred": str(pred_path), "gold": str(gold_path),
        "total": n, "llm_model": args.llm_model,
    }
    for k in ("sq1_exact", "sq1_partial", "sq2_exact", "sq2_partial",
              "final_exact", "final_partial"):
        summary[k] = sum(1 for r in results if (r.get("eval") or {}).get(k))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    print("\nSummary:", json.dumps(summary, indent=2))
    print(f"Report -> {out_path}")
    print(f"Predictions rewritten in-place with eval flags -> {pred_path}")


if __name__ == "__main__":
    main()
