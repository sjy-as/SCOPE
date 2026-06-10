"""HydraRAG baseline 驱动脚本。

读入 qa_bench 的 jsonl，逐题跑 HydraRAG pipeline：
  - 把预测写到  <output_dir>/predictions.jsonl
  - 每题完整 trace 写到  <output_dir>/traces/idx_<index>.trace.json
  - 汇总写到  <output_dir>/summary.json

评测（LLM-judge）单独用 eval/evaluate.py 跑，详见 README。

知识库由 --kb 决定：KG 永远在，table / doc 可开关。外部补充头按开启的源
逐个挂上（KG 先检索、外部源补充并融合）：
  --kb kg            只有 KG
  --kb kg,table      KG + Table BM25（默认外部头）
  --kb kg,doc        KG + Doc ColBERT
  --kb kg,table,doc  KG + Table + Doc，两个外部头一起融合

示例（KG + Table）：
  cd /root/autodl-tmp/baseline/HydraRAG/hydra_baseline
  python3 run.py \
    --input /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
    --output-dir result/run1 \
    --kb kg,table \
    --workers 8 \
    --api-key "sk-xxxx" \
    --llm-url "https://api.chatanywhere.tech/v1" \
    --llm-model "deepseek-chat"

示例（KG + Doc，需先启动 ColBERT 段落检索服务 :1215）：
  python3 run.py \
    --kb kg,doc \
    --input /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
    --output-dir result/run_doc \
    --workers 8 \
    --api-key "sk-xxxx" \
    --llm-url "https://api.chatanywhere.tech/v1" \
    --llm-model "deepseek-chat"
"""
from __future__ import annotations

import sys as _sys
if "/root/autodl-tmp" not in _sys.path:
    _sys.path.insert(0, "/root/autodl-tmp")
try:
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

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# 让 hydra_baseline 根目录在 sys.path 上，包内模块可直接 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from llm import LLMClient
from kg.kg_retriever import KGRetriever
from kg.kg_explorer import KGExplorer
from table.table_retriever import TableRetriever
from table.table_to_kg import TableToKG
from doc.doc_retriever import DocRetriever
from doc.doc_to_kg import DocToKG
from pipeline import HydraPipeline


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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


_VALID_KB = ["kg", "table", "doc"]


def _parse_kb(raw: str) -> List[str]:
    """把 --kb（逗号分隔的 {kg,table,doc} 子集）解析成有序的活源列表。
    'kg' 永远包含。"""
    want = {t.strip().lower() for t in (raw or "").split(",") if t.strip()}
    want.add("kg")  # KG 永远是知识库的一部分
    bad = sorted(want - set(_VALID_KB))
    if bad:
        raise SystemExit(f"[run] unknown --kb source(s): {bad}  (allowed: kg, table, doc)")
    return [s for s in _VALID_KB if s in want]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run HydraRAG baseline over a question jsonl")
    ap.add_argument("--input", default="/root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl")
    ap.add_argument("--output-dir", default="result/run1")
    ap.add_argument("--max", type=int, default=None, help="只跑前 N 题")
    ap.add_argument("--start", type=int, default=0, help="从第几题开始")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数，1=串行")
    ap.add_argument("--resume", action="store_true", help="续跑，跳过已完成的 index")
    ap.add_argument("--llm-url", default=os.getenv("LLM_BASE_URL", config.LLM_BASE_URL))
    ap.add_argument("--llm-model", default=os.getenv("LLM_MODEL", config.LLM_MODEL))
    ap.add_argument("--api-key", default=os.getenv("LLM_API_KEY", config.LLM_API_KEY))
    ap.add_argument("--kg-dir", default=config.KG_DIR)
    ap.add_argument("--table-url", default=config.TABLE_API_URL)
    ap.add_argument("--doc-url", default=config.DOC_API_URL)
    ap.add_argument("--kb", default="kg",
                    help="知识库 = 活的数据源。逗号分隔的 {kg,table,doc} 子集，"
                         "'kg' 永远包含。table/doc 各对应一个外部补充头，"
                         "开启的一起融合。"
                         "示例：'kg'、'kg,table'、'kg,doc'、'kg,table,doc'。")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("[run] 需要 --api-key（或环境变量 LLM_API_KEY）")

    enabled = _parse_kb(args.kb)

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

    print("=" * 70)
    print(f"HydraRAG baseline")
    print(f"Input      : {in_path}  ({len(questions)} questions)")
    print(f"Output dir : {out_dir}")
    print(f"LLM        : {args.llm_model} @ {args.llm_url}")
    print(f"KG dir     : {args.kg_dir}")
    print(f"Knowledge base: {', '.join(enabled)}")
    if "table" in enabled:
        print(f"Table API  : {args.table_url}")
    if "doc" in enabled:
        print(f"Doc API    : {args.doc_url}")
    print(f"Workers    : {args.workers}   Max iterations: {config.MAX_ITERATIONS}")
    print("=" * 70)

    # ---- 共享重对象（只构建一次）----
    print("Loading local KG ...")
    t0 = time.time()
    llm = LLMClient(api_key=args.api_key, base_url=args.llm_url, model=args.llm_model)
    kg_retriever = KGRetriever(args.kg_dir, llm=llm)
    print(f"  KG loaded: {len(kg_retriever.kg.qid2label)} entities, "
          f"{len(kg_retriever.kg.relations)} relations  ({time.time() - t0:.1f}s)")

    # ---- 外部补充源检索器（按 --kb 开启的源构建；KG 不需要外部检索服务）----
    table_retriever = None
    doc_retriever = None
    if "table" in enabled:
        table_retriever = TableRetriever(api_url=args.table_url)
        if not table_retriever.is_alive():
            print(f"  [WARN] Table BM25 服务 {args.table_url} 不可达 —— "
                  f"Table 补充头将取空，请先启动检索服务。")
    if "doc" in enabled:
        doc_retriever = DocRetriever(api_url=args.doc_url)
        if not doc_retriever.is_alive():
            print(f"  [WARN] Doc ColBERT 服务 {args.doc_url} 不可达 —— "
                  f"Doc 补充头将取空，请先启动检索服务。")

    # ---- 续跑：收集已完成 index ----
    done: set = set()
    if args.resume and pred_path.exists():
        for line in pred_path.open("r", encoding="utf-8"):
            try:
                done.add(json.loads(line)["index"])
            except Exception:  # noqa: BLE001
                pass
        print(f"[resume] 已完成 {len(done)} 题，跳过")

    work = []
    for i, rec in enumerate(questions):
        idx = rec.get("index", i)
        q = (rec.get("question") or "").strip()
        if not q or idx in done:
            continue
        work.append((idx, q))

    write_lock = threading.Lock()
    total = len(work)
    completed = 0
    ok = 0

    _reset_cost_agg()

    if args.resume:
        _seed_path = out_dir / "cost_summary.json"
        _seeded = _seed_cost_summary(_seed_path)
        if _seeded:
            print(f"[hydrarag] resume: seeded cost aggregator with {_seeded} prior entries from {_seed_path}")

    def process_one(idx: int, question: str) -> Dict[str, Any]:
        # 每题用独立的 KGExplorer（含自己的子图缓存），避免并发共享可变状态
        explorer = KGExplorer(
            kg_retriever, max_hop=config.KG_MAX_HOP,
            max_degree=config.KG_MAX_DEGREE,
        )
        # 外部补充头：按 --kb 开启的源逐个挂上（TableToKG / DocToKG 接口一致）。
        external_heads = []
        if table_retriever is not None:
            external_heads.append(TableToKG(table_retriever, llm, k=config.K_TABLE,
                                            verbose=(args.workers == 1)))
        if doc_retriever is not None:
            external_heads.append(DocToKG(doc_retriever, llm, k=config.K_DOC,
                                          verbose=(args.workers == 1)))
        pipe = HydraPipeline(explorer, external_heads, llm,
                             verbose=(args.workers == 1), kb=",".join(enabled))
        with _question_scope(idx):
            try:
                res = pipe.run(question, index=idx)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                res = {"index": idx, "question": question, "final": "",
                       "answer_entities": [], "iterations": [], "n_iterations": 0,
                       "selected_evidence": [], "n_llm_calls": 0, "llm_calls": [],
                       "error": str(e)}
        return res

    open_mode = "a" if args.resume else "w"
    with pred_path.open(open_mode, encoding="utf-8") as wf:
        def emit(res: Dict[str, Any]) -> None:
            nonlocal completed, ok
            idx = res.get("index")
            _write_json(traces_dir / f"idx_{idx}.trace.json", res)
            pred = {
                "index": idx,
                "question": res.get("question", ""),
                "sq1": [],                       # HydraRAG 为整体式，只对齐最终答案
                "sq2": [],
                "final": [res["final"]] if res.get("final") else [],
                "answer_entities": res.get("answer_entities", []),
                "n_iterations": res.get("n_iterations", 0),
                "n_llm_calls": res.get("n_llm_calls", 0),
                "error": res.get("error"),
            }
            with write_lock:
                completed += 1
                if not res.get("error"):
                    ok += 1
                tag = "OK " if not res.get("error") else "ERR"
                print(f"[{completed}/{total}] {tag} index={idx}  final={pred['final']}")
                wf.write(json.dumps(pred, ensure_ascii=False) + "\n")
                wf.flush()

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(process_one, idx, q) for idx, q in work]
                for fut in as_completed(futs):
                    emit(fut.result())
        else:
            for idx, q in work:
                print("\n" + "#" * 70 + f"\n# index={idx}\n" + "#" * 70)
                emit(process_one(idx, q))

    summary = {
        "input": str(in_path),
        "output_dir": str(out_dir),
        "knowledge_base": enabled,
        "llm_model": args.llm_model,
        "total": total,
        "succeeded": ok,
        "failed": total - ok,
        "total_llm_calls": llm.call_count,
        "total_tokens_in": llm.token_in,
        "total_tokens_out": llm.token_out,
        "max_iterations": config.MAX_ITERATIONS,
    }
    _write_json(summary_path, summary)

    cost_path = out_dir / "cost_summary.json"
    cost_summary = _dump_cost_summary(cost_path)

    print("\n" + "=" * 70)
    print(f"Done. {ok}/{total} succeeded, {total - ok} failed.")
    print(f"Predictions -> {pred_path}")
    print(f"Traces      -> {traces_dir}/idx_<index>.trace.json")
    print(f"Summary     -> {summary_path}")
    print(f"Cost summary-> {cost_path}")
    print(f"LLM calls   : {llm.call_count}  (in={llm.token_in} out={llm.token_out} tokens)")
    print(f"[cost] {_format_cost_line(cost_summary)}")
    print("=" * 70)
    print("评测：python3 eval/evaluate.py "
          f"--pred {pred_path} --gold {in_path} --api-key <KEY>")


if __name__ == "__main__":
    main()
