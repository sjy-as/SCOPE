"""HydraRAG baseline 评测脚本（只评最终答案）。

用 LLM-as-judge 把 predictions.jsonl 里的 `final` 和 gold 对比。
对 qa_bench 的 2-hop 题，主问题的标准答案就是 q2 的答案（最后一跳）。
判定标准与 new_model/run.py 的 _llm_judge_eval 一致：exact / partial / miss。

示例：
  cd /root/autodl-tmp/baseline/HydraRAG/hydra_baseline
  python3 eval/evaluate.py \
    --pred result/run1/predictions.jsonl \
    --gold /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
    --api-key "sk-xxxx"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from llm import LLMClient


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    return [str(v).strip() for v in x if str(v).strip()]


def _gold_final(rec: Dict[str, Any]) -> List[str]:
    """主问题（最后一跳）的标准答案 = q2 的 answer/answers。"""
    q2 = rec.get("q2") or {}
    return _as_list(q2.get("answer") or q2.get("answers"))


_JUDGE_PROMPT = """You are a format-tolerant string-matching evaluator. Compare Predicted to Gold ONLY.
Treat Gold as the authoritative ground-truth answer even if it looks wrong. DO NOT fact-check Gold.

Choose one verdict:
  exact   : Predicted and Gold refer to the same entity/value. IGNORE differences in case,
            articles, trailing punctuation, plurals, paraphrasing, sentence wrapping. If
            Predicted is a sentence containing all of Gold's information, it is exact.
  partial : Predicted covers some Gold elements but not all.
  miss    : empty, a refusal, or a different entity/value than Gold.

Question (context only, do not fact-check): {question}
Predicted: {predicted}
Gold: {gold}

Output strict JSON only, no prose, no code fences:
{{"verdict": "exact|partial|miss", "reason": "..."}}"""


def _judge_one(llm: LLMClient, question: str, predicted: List[str],
               gold: List[str]) -> Dict[str, str]:
    if not gold:
        return {"verdict": "miss", "reason": "no gold answer"}
    prompt = _JUDGE_PROMPT.format(
        question=question,
        predicted=json.dumps(predicted, ensure_ascii=False),
        gold=json.dumps(gold, ensure_ascii=False),
    )
    resp, _ = llm.query_gpt4o(prompt, max_tokens=200, stage="judge")
    verdict, reason = "miss", ""
    try:
        m = resp[resp.find("{"): resp.rfind("}") + 1]
        obj = json.loads(m)
        verdict = str(obj.get("verdict", "miss")).strip().lower()
        reason = str(obj.get("reason", ""))
    except Exception:  # noqa: BLE001
        low = resp.lower()
        verdict = "exact" if "exact" in low else ("partial" if "partial" in low else "miss")
    if verdict not in ("exact", "partial", "miss"):
        verdict = "miss"
    return {"verdict": verdict, "reason": reason}


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-judge evaluation for HydraRAG baseline")
    ap.add_argument("--pred", required=True, help="predictions.jsonl 路径")
    ap.add_argument("--gold", required=True, help="带 q2 标准答案的 jsonl")
    ap.add_argument("--output", default="", help="评测报告输出路径（默认 pred 同目录 eval_report.json）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--llm-url", default=os.getenv("LLM_BASE_URL", config.LLM_BASE_URL))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", config.LLM_MODEL))
    ap.add_argument("--api-key", default=os.getenv("LLM_API_KEY", config.LLM_API_KEY))
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("[eval] 需要 --api-key（或环境变量 LLM_API_KEY）")

    pred_path = Path(args.pred).resolve()
    gold_path = Path(args.gold).resolve()
    out_path = Path(args.output).resolve() if args.output else (pred_path.parent / "eval_report.json")

    preds = _load_jsonl(pred_path)
    gold_map = {r["index"]: r for r in _load_jsonl(gold_path) if "index" in r}
    llm = LLMClient(api_key=args.api_key, base_url=args.llm_url, model=args.llm_model)

    print("=" * 70)
    print(f"Pred : {pred_path}  ({len(preds)} rows)")
    print(f"Gold : {gold_path}  ({len(gold_map)} rows)")
    print("=" * 70)

    lock = threading.Lock()
    done = [0]

    def judge_row(pred: Dict[str, Any]) -> Dict[str, Any]:
        idx = pred.get("index")
        gold = gold_map.get(idx) or {}
        question = pred.get("question") or gold.get("question", "")
        g_final = _gold_final(gold)
        p_final = _as_list(pred.get("final"))
        verdict = _judge_one(llm, question, p_final, g_final)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(preds)}] index={idx}  {verdict['verdict']:8s} "
                  f"pred={p_final}  gold={g_final}")
        return {
            "index": idx, "question": question,
            "predicted_final": p_final, "gold_final": g_final,
            "verdict": verdict["verdict"], "reason": verdict["reason"],
            "error": pred.get("error"),
        }

    results: List[Dict[str, Any]] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(judge_row, p) for p in preds]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for p in preds:
            results.append(judge_row(p))

    results.sort(key=lambda r: (r["index"] is None, r["index"]))
    n = len(results)
    n_exact = sum(1 for r in results if r["verdict"] == "exact")
    n_partial = sum(1 for r in results if r["verdict"] in ("exact", "partial"))
    n_err = sum(1 for r in results if r.get("error"))

    report = {
        "pred": str(pred_path),
        "gold": str(gold_path),
        "total": n,
        "final_exact": n_exact,
        "final_partial": n_partial,
        "final_miss": n - n_partial,
        "errors": n_err,
        "exact_acc": round(n_exact / n, 4) if n else 0.0,
        "partial_acc": round(n_partial / n, 4) if n else 0.0,
        "details": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"Total            : {n}")
    print(f"final exact      : {n_exact}  ({report['exact_acc'] * 100:.1f}%)")
    print(f"final exact+part : {n_partial}  ({report['partial_acc'] * 100:.1f}%)")
    print(f"errors           : {n_err}")
    print(f"Report -> {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
