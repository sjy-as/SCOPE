"""
Main execution entry point for multi-source reasoning with operator trees.

The knowledge base is selected with --kb:
    kg-table : reason over KG + Table sources   (default)
    kg-doc   : reason over KG + Doc  sources

Usage:
    cd /root/autodl-tmp/new_model
    python3 -m step3_execute.main --kb kg-table --index 1
    python3 -m step3_execute.main --kb kg-doc   --index 1 --verbose
    python3 -m step3_execute.main --kb kg-doc   --all --max 5
"""

import argparse
import json
from pathlib import Path
from typing import Dict

from step3_execute.reasoner import MultiSourceReasoner, OperatorResult
from step3_execute.knowledge_sources.query_table import TableSource
from step3_execute.knowledge_sources.query_kg import KGSource
from step3_execute.knowledge_sources.query_doc import DocSource
from step3_execute.knowledge_sources.query_llm import LLMClient
from step3_execute.service.KG.kg_retriever import KGRetriever

# ── 默认配置 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent              # .../step3_execute
_ROOT = _HERE.parent                                 # .../new_model

DEFAULT_TABLE_API = "http://127.0.0.1:1216/api/search"
DEFAULT_DOC_API   = "http://127.0.0.1:1215/api/search"
DEFAULT_KG_DIR    = str(_ROOT / "data_sources" / "KG")
DEFAULT_K_TABLE   = 5
DEFAULT_K_DOC     = 5

# 每种知识库对应的输入计划文件 / 答案文件
KB_INPUTS = {
    "kg-table": _HERE / "input" / "sq-plan_kg-table.jsonl",
    "kg-doc":   _HERE / "input" / "sq-plan_kg-doc.jsonl",
}
KB_ANSWERS = {
    "kg-table": _ROOT / "qa_bench" / "kg-table-160.jsonl",
    "kg-doc":   _ROOT / "qa_bench" / "kg-doc-160.jsonl",
}


# ── 初始化所有组件 ────────────────────────────────────────────────────────
def build_reasoner(kb: str = "kg-table", verbose: bool = True) -> MultiSourceReasoner:
    """根据知识库类型初始化 LLM / KGSource / (TableSource | DocSource)。"""
    # LLM
    try:
        llm = LLMClient()
        print("✓ LLMClient initialized")
    except Exception as e:
        print(f"⚠ LLMClient failed: {e}")
        llm = None

    # KG source —— kg-table 和 kg-doc 都需要
    try:
        kg_retriever = KGRetriever(kg_dir=DEFAULT_KG_DIR, llm=llm)
        kg_source = KGSource(kg_retriever=kg_retriever, llm=llm, verbose=verbose)
        print("✓ KGSource initialized")
    except Exception as e:
        print(f"⚠ KGSource failed: {e}")
        kg_source = None

    table_source = None
    doc_source = None

    if kb == "kg-table":
        table_source = TableSource(
            retriever_api_url=DEFAULT_TABLE_API,
            k=DEFAULT_K_TABLE,
            llm=llm,
            verbose=verbose,
        )
        print("✓ TableSource initialized")
    elif kb == "kg-doc":
        try:
            doc_source = DocSource(
                retriever_api_url=DEFAULT_DOC_API,
                k=DEFAULT_K_DOC,
                llm=llm,
                verbose=verbose,
            )
            print("✓ DocSource initialized")
        except Exception as e:
            print(f"⚠ DocSource failed: {e}")
            doc_source = None
    else:
        raise ValueError(f"Unknown knowledge base: {kb} (expected 'kg-table' or 'kg-doc')")

    return MultiSourceReasoner(
        table_source=table_source,
        kg_source=kg_source,
        doc_source=doc_source,
        llm=llm,
    )


# ── 加载 ground truth（可选，用于验证）────────────────────────────────────
def load_ground_truth(answer_jsonl: str) -> Dict[int, dict]:
    gt = {}
    try:
        with open(answer_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                gt[obj["index"]] = obj
    except FileNotFoundError:
        print(f"⚠ Ground truth file not found: {answer_jsonl} (skipping verification)")
    except Exception as e:
        print(f"⚠ Could not load ground truth: {e}")
    return gt


# ── 验证结果 ──────────────────────────────────────────────────────────────
def verify_result(
    sq_results: Dict[str, OperatorResult],
    record: dict,
    gt: dict,
) -> None:
    """打印 sq1/sq2 最终答案并与 ground truth 对比。"""
    print("\n" + "=" * 70)
    print("FINAL ANSWERS")
    print("=" * 70)

    sub_queries = record.get("sub_queries", [])
    sq_ids = [sq["id"] for sq in sub_queries]

    for sq_id in sq_ids:
        result = sq_results.get(sq_id)
        if result:
            print(f"  [{sq_id}] answers : {result.answers[:10]}")
            print(f"  [{sq_id}] evidence: {len(result.evidence)} items")
        else:
            print(f"  [{sq_id}] no result")

    # 最终综合答案
    final = sq_results.get("__final__")
    if final and final.answers:
        print(f"\n  [FINAL] {final.answers[0]}")

    # ground truth 验证
    idx = record.get("index", -1)
    if gt and idx in gt:
        print("\n" + "-" * 70)
        print("GROUND TRUTH")
        print("-" * 70)
        gt_rec = gt[idx]

        for gt_key, sq_key in (("q1", "sq1"), ("q2", "sq2")):
            if gt_key not in gt_rec:
                continue
            gt_ans = gt_rec[gt_key].get("answer") or gt_rec[gt_key].get("answers", [])
            res = sq_results.get(sq_key)
            got = res.answers if res else []
            gt_set, got_set = set(gt_ans), set(got)
            hits = gt_set & got_set
            print(f"  [{sq_key}] expected: {gt_ans}")
            print(f"  [{sq_key}] got     : {got}")
            if gt_set == got_set:
                print(f"  [{sq_key}] ✓ EXACT MATCH")
            elif hits:
                print(f"  [{sq_key}] ~ PARTIAL MATCH  hits={hits}")
            else:
                print(f"  [{sq_key}] ✗ NO MATCH")
    else:
        print("  (no ground truth available)")


# ── 执行单条记录 ──────────────────────────────────────────────────────────
def run_single(record: dict, reasoner: MultiSourceReasoner, gt: dict) -> Dict[str, OperatorResult]:
    idx = record.get("index", "?")
    question = record.get("question", "")
    print("\n" + "=" * 70)
    print(f"[index={idx}] {question}")
    print("=" * 70)

    sq_results = reasoner.execute_subqueries(record)
    verify_result(sq_results, record, gt)
    return sq_results


# ── 主入口 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-source reasoner executor")
    parser.add_argument("--kb", type=str, default="kg-table", choices=["kg-table", "kg-doc"],
                        help="Knowledge base: kg-table or kg-doc")
    parser.add_argument("--index", type=int, default=None, help="0-based line index in input JSONL")
    parser.add_argument("--all", action="store_true", help="Run all records")
    parser.add_argument("--max", type=int, default=None, help="Max records to run (used with --all)")
    parser.add_argument("--input", type=str, default=None, help="Input JSONL path (overrides --kb default)")
    parser.add_argument("--answer", type=str, default=None, help="Answer JSONL path (overrides --kb default)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    args = parser.parse_args()

    input_path = args.input or str(KB_INPUTS[args.kb])
    answer_path = args.answer or str(KB_ANSWERS[args.kb])

    # 初始化
    print(f"Knowledge base: {args.kb}")
    reasoner = build_reasoner(kb=args.kb, verbose=args.verbose)
    gt = load_ground_truth(answer_path)

    # 读取输入
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\nLoaded {len(records)} records from {input_path}")

    if args.index is not None:
        # 先找 record["index"] == args.index，找不到则用行号
        target = next((r for r in records if r.get("index") == args.index), None)
        if target is None and args.index < len(records):
            target = records[args.index]
        if target is None:
            print(f"[Error] index={args.index} not found")
            return
        run_single(target, reasoner, gt)

    elif args.all:
        to_run = records[:args.max] if args.max else records
        all_results = []
        for rec in to_run:
            try:
                sq_res = run_single(rec, reasoner, gt)
                all_results.append({"index": rec.get("index"), "results": {
                    sq_id: {"answers": r.answers} for sq_id, r in sq_res.items()
                }})
            except Exception as e:
                print(f"[Error] index={rec.get('index')}: {e}")
        print(f"\nDone. Ran {len(all_results)} records.")

    else:
        print("No action specified. Use --index N or --all. Example:")
        print("  python3 -m step3_execute.main --kb kg-doc --index 1")
        print("  python3 -m step3_execute.main --kb kg-doc --all --max 3")


if __name__ == "__main__":
    main()
