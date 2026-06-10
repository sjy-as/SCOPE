"""
DeepSieve RAG-only pipeline runner.

Supports:
- Classic datasets (local/global source profiles)
- MMQA dataset (table/kg source profiles)

cd /root/autodl-tmp/baseline/deepservice && python3 runner/main_rag_only.py \
  --dataset mmqa \
  --dataset-path /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl \
  --kb kg,doc \
  --output-dir /root/autodl-tmp/new_model/eval/result/answer/kg_doc/deepserive \
  --use_routing --decompose --use_reflection --max_reflexion_times 2 \
  --concurrency 8 \
  --api-key "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA" \
  --llm-url "https://api.chatanywhere.tech/v1"


"""

import os
import sys
import json
import argparse
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

if "/root/autodl-tmp" not in sys.path:
    sys.path.insert(0, "/root/autodl-tmp")
try:
    from _common.cost_counter import (
        question_scope as _question_scope,
        dump_summary as _dump_cost_summary,
        seed_from_existing as _seed_cost_summary,
        format_summary_line as _format_cost_line,
        reset_aggregator as _reset_cost_agg,
    )
except Exception:
    from contextlib import nullcontext as _question_scope  # type: ignore
    def _dump_cost_summary(_p): return {}
    def _seed_cost_summary(_p): return 0
    def _format_cost_line(_s): return ""
    def _reset_cost_agg(): pass

# Ensure project root (deepservice) is on PYTHONPATH when executed as a script.
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rag.initializer import initialize_rag_system, initialize_mmqa_sources
from pipeline.reasoning_pipeline import plan_subqueries_with_llm, get_fused_final_answer
from pipeline.subquery_executor import execute_subquery
from utils.data_load import load_queries, load_corpus_and_profiles, load_source_profiles
from utils.metrics import evaluate_answer, calculate_overall_metrics, score_answer_prediction
from utils.trace_recorder import get_recorder


_VALID_KB = ["kg", "table", "doc"]


def _parse_kb(raw: str):
    """Parse --kb (comma-separated subset of {kg,table,doc}) into an ordered
    source list. All three are individually switchable."""
    want = {t.strip().lower() for t in (raw or "").split(",") if t.strip()}
    if not want:
        raise SystemExit(
            "[main_rag_only] --kb must list at least one of kg/table/doc"
        )
    bad = sorted(want - set(_VALID_KB))
    if bad:
        raise SystemExit(
            f"[main_rag_only] unknown --kb source(s): {bad}  (allowed: kg, table, doc)"
        )
    return [s for s in _VALID_KB if s in want]


def get_save_dir(decompose: bool, use_routing: bool, use_reflection: bool, dataset: str, rag_type: str, run_tag: str = "", kb: str = ""):
    save_dir = "outputs/"
    save_dir += rag_type
    save_dir += "_"
    save_dir += dataset
    kb_tag = (kb or "").strip()
    if kb_tag:
        save_dir += "_kb-" + kb_tag.replace(",", "-")
    save_dir += "_"
    tag = (run_tag or "").strip()
    if tag:
        save_dir += "_" + tag
    if not use_routing:
        save_dir += "_no_routing"
    if not decompose:
        save_dir += "_no_decompose"
    if not use_reflection:
        save_dir += "_no_reflection"
    return save_dir


def save_overall_results(save_dir, overall_metrics, queries_and_truth, all_metrics):
    overall_txt_path = os.path.join(save_dir, "overall_results.txt")
    with open(overall_txt_path, "w", encoding="utf-8") as f:
        f.write("📊 Overall Performance Summary:\n")
        f.write(f"- Average Exact Match: {overall_metrics['avg_exact_match']:.4f}\n")
        f.write(f"- Average F1 Score: {overall_metrics['avg_f1']:.4f}\n")
        f.write(f"- Average Retrieval Time: {overall_metrics['avg_retrieval_time']:.4f}s\n")
        f.write(f"- Average Documents Searched: {overall_metrics['avg_docs_searched']:.1f}\n")
        f.write(f"- Average Similarity Score: {overall_metrics['avg_similarity']:.4f}\n")
        f.write(f"- Average Prompt Tokens per Subquery: {overall_metrics['avg_prompt_tokens_per_subquery']:.2f}\n")
        f.write(f"- Average Total Tokens per Query: {overall_metrics['avg_total_tokens_per_query']:.2f}\n")
        f.write(f"- Average Execution Time per Query: {overall_metrics.get('avg_execution_time', 0):.4f}s\n")
        f.write(f"- Average SQ1 Score: {overall_metrics.get('avg_sq1_score', 0):.4f}\n")
        f.write(f"- Average SQ2 Score: {overall_metrics.get('avg_sq2_score', 0):.4f}\n")
        f.write(f"- Average Total Score: {overall_metrics.get('avg_total_score', 0):.4f}\n")

    overall_json_path = os.path.join(save_dir, "overall_results.json")
    with open(overall_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "queries": queries_and_truth,
            "overall_metrics": overall_metrics,
            "all_query_metrics": all_metrics,
        }, f, indent=2)

    print("\n📊 Overall Performance Summary:")
    print(f"- Average Exact Match: {overall_metrics['avg_exact_match']:.4f}")
    print(f"- Average F1 Score: {overall_metrics['avg_f1']:.4f}")
    print(f"- Average Retrieval Time: {overall_metrics['avg_retrieval_time']:.4f}s")
    print(f"- Average Documents Searched: {overall_metrics['avg_docs_searched']:.1f}")
    print(f"- Average Similarity Score: {overall_metrics['avg_similarity']:.4f}")
    print(f"- Average Prompt Tokens per Subquery: {overall_metrics['avg_prompt_tokens_per_subquery']:.2f}")
    print(f"- Average Total Tokens per Query: {overall_metrics['avg_total_tokens_per_query']:.2f}")
    print(f"- Average Execution Time per Query: {overall_metrics.get('avg_execution_time', 0):.4f}s")
    print(f"- Average SQ1 Score: {overall_metrics.get('avg_sq1_score', 0):.4f}")
    print(f"- Average SQ2 Score: {overall_metrics.get('avg_sq2_score', 0):.4f}")
    print(f"- Average Total Score: {overall_metrics.get('avg_total_score', 0):.4f}")

    print("\n✅ Results saved to:")
    print(f"   - {overall_txt_path}")
    print(f"   - {overall_json_path}")


def save_single_query_results(
    save_dir,
    idx,
    multi_hop_query,
    ground_truth,
    final_answer,
    final_reason,
    fusion_token_count,
    fallback_answer,
    fusion_prompt,
    eval_results,
    eval_results_fallback,
    performance_metrics,
    results,
    fused_answer_texts,
):
    query_results_path = os.path.join(save_dir, f"query_{idx}_results.jsonl")
    with open(query_results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "query_info", "query": multi_hop_query, "ground_truth": ground_truth}) + "\n")
        f.write(json.dumps({
            "type": "final_answer",
            "final_answer": final_answer,
            "final_reason": final_reason,
            "fusion_prompt_tokens": fusion_token_count,
            "fallback_answer": fallback_answer,
            "fusion_equals_fallback": fallback_answer.strip().lower() == final_answer.strip().lower(),
        }) + "\n")
        f.write(json.dumps({
            "type": "evaluation_metrics",
            "fusion": {"exact_match": eval_results["exact_match"], "f1": eval_results["f1"]},
            "fallback": {"exact_match": eval_results_fallback["exact_match"], "f1": eval_results_fallback["f1"]},
            "sq1_score": performance_metrics.get("sq1_score", 0),
            "sq2_score": performance_metrics.get("sq2_score", 0),
            "total_score": performance_metrics.get("total_score", 0),
        }) + "\n")
        f.write(json.dumps({
            "type": "performance_metrics",
            "total_retrieval_time": performance_metrics["total_retrieval_time"],
            "avg_retrieval_time": performance_metrics["avg_retrieval_time"],
            "total_docs_searched": performance_metrics["total_docs_searched"],
            "avg_similarity": performance_metrics["avg_similarity"],
            "max_similarity": performance_metrics["max_similarity"],
            "token_cost": {
                "total_prompt_tokens": performance_metrics["total_prompt_tokens"],
                "avg_prompt_tokens": performance_metrics["avg_prompt_tokens"],
                "max_prompt_tokens": performance_metrics["max_prompt_tokens"],
                "min_prompt_tokens": performance_metrics["min_prompt_tokens"],
            },
        }) + "\n")

        for r in results:
            f.write(json.dumps({
                "type": "execution_result",
                "subquery_id": r["subquery_id"],
                "original_query": r["original_query"],
                "actual_query": r["actual_query"],
                "variables_used": r.get("variables_used", None),
                "routing": r["routing"],
                "answer": r["answer"],
                "reason": r["reason"],
                "docs": [
                    {"text": doc, "score": r["doc_scores"][i]}
                    for i, doc in enumerate(r["docs"])
                ],
            }) + "\n")

        for step in fused_answer_texts:
            f.write(json.dumps({"type": "fused_answer_step", "text": step}) + "\n")

    fusion_prompt_path = os.path.join(save_dir, f"query_{idx}_fusion_prompt.txt")
    with open(fusion_prompt_path, "w", encoding="utf-8") as f_prompt:
        f_prompt.write(fusion_prompt)

    return performance_metrics


def process_subqueries(
    performance_metrics,
    query_plan,
    variable_values,
    sources,
    merged_rag,
    use_routing,
    use_reflection,
    max_reflexion_times,
    openai_api_key,
    openai_model,
    openai_base_url,
    save_dir,
    idx,
    multi_hop_query,
    ground_truth,
    q1_answers,
    q2_answers,
    results,
    fused_answer_texts,
):
    recorder = get_recorder()

    for subquery_info in query_plan["subqueries"]:
        sq_id = subquery_info.get("id", f"q{len(results) + 1}")
        recorder.set_stage(f"subq_{sq_id}")
        recorder.merge_meta({
            "subquery_id": sq_id,
            "subquery_info": subquery_info,
            "variable_values_in": dict(variable_values),
            "use_routing": use_routing,
            "use_reflection": use_reflection,
        })

        subquery_result = execute_subquery(
            subquery_info,
            variable_values,
            sources,
            merged_rag,
            use_routing,
            use_reflection,
            max_reflexion_times,
            openai_api_key,
            openai_model,
            openai_base_url,
        )
        results.append(subquery_result)
        fused_answer_texts.append(
            f"{subquery_result['subquery_id']}: {subquery_result['actual_query']} → {subquery_result['answer']} (reason: {subquery_result['reason']})"
        )
        performance_metrics["total_retrieval_time"] += subquery_result["retrieval_time"]
        performance_metrics["total_docs_searched"] += subquery_result["docs_searched"]
        performance_metrics["avg_similarity_scores"].append(subquery_result["avg_similarity"])
        performance_metrics["max_similarity_scores"].append(subquery_result["max_similarity"])

        recorder.merge_meta({
            "actual_query": subquery_result.get("actual_query"),
            "routing": subquery_result.get("routing"),
            "initial_routing": subquery_result.get("initial_routing"),
            "final_routing": subquery_result.get("final_routing"),
            "num_attempts": subquery_result.get("num_attempts"),
            "attempts": subquery_result.get("attempts"),
            "answer": subquery_result.get("answer"),
            "reason": subquery_result.get("reason"),
            "success": subquery_result.get("success"),
            "variables_used": subquery_result.get("variables_used"),
            "docs_preview": [
                {"score": (subquery_result.get("doc_scores") or [None] * len(subquery_result.get("docs", [])))[i],
                 "text": (d if len(str(d)) <= 1200 else str(d)[:1200] + "...<truncated>")}
                for i, d in enumerate(subquery_result.get("docs", []))
            ],
            "retrieval_time": subquery_result.get("retrieval_time"),
            "docs_searched": subquery_result.get("docs_searched"),
            "avg_similarity": subquery_result.get("avg_similarity"),
            "max_similarity": subquery_result.get("max_similarity"),
            "prompt_token_count": subquery_result.get("prompt_token_count"),
        })
        recorder.dump_stage(f"subq_{sq_id}.trace.json")

    performance_metrics["avg_retrieval_time"] = performance_metrics["total_retrieval_time"] / len(query_plan["subqueries"])
    performance_metrics["avg_similarity"] = float(np.mean(performance_metrics["avg_similarity_scores"]))
    performance_metrics["max_similarity"] = float(np.max(performance_metrics["max_similarity_scores"]))

    performance_metrics["avg_prompt_tokens"] = performance_metrics["total_prompt_tokens"] / len(query_plan["subqueries"])
    performance_metrics["max_prompt_tokens"] = max(performance_metrics["prompt_token_counts"], default=0)
    performance_metrics["min_prompt_tokens"] = min(performance_metrics["prompt_token_counts"], default=0)

    fallback_answer = results[-1]["answer"] if results else ""
    performance_metrics["evaluation_metrics"]["fallback_answer"] = fallback_answer

    recorder.set_stage("final")
    recorder.merge_meta({
        "original_question": multi_hop_query,
        "ground_truth": ground_truth,
        "fallback_answer": fallback_answer,
        "subquery_chain": [
            {
                "subquery_id": r.get("subquery_id"),
                "actual_query": r.get("actual_query"),
                "routing": r.get("routing"),
                "answer": r.get("answer"),
                "reason": r.get("reason"),
            }
            for r in results
        ],
    })

    final_answer, final_reason, fusion_token_count, fusion_prompt = get_fused_final_answer(
        multi_hop_query,
        results,
        api_key=openai_api_key,
        model=openai_model,
        base_url=openai_base_url,
    )

    performance_metrics["evaluation_metrics"]["final_answer"] = final_answer
    performance_metrics["evaluation_metrics"]["final_reason"] = final_reason
    performance_metrics["fusion_prompt_tokens"] = fusion_token_count

    performance_metrics["total_prompt_tokens"] += fusion_token_count
    performance_metrics["prompt_token_counts"].append(fusion_token_count)

    sq1_score = score_answer_prediction(
        results[0]["answer"] if results else "",
        q1_answers,
        api_key=openai_api_key,
        model=openai_model,
        base_url=openai_base_url,
    )
    sq2_score = score_answer_prediction(
        final_answer,
        q2_answers or ground_truth,
        api_key=openai_api_key,
        model=openai_model,
        base_url=openai_base_url,
    )

    eval_results = evaluate_answer(final_answer, ground_truth)
    eval_results_fallback = evaluate_answer(fallback_answer, ground_truth)
    performance_metrics["evaluation_metrics"].update(eval_results)
    performance_metrics["evaluation_metrics_fallback"].update(eval_results_fallback)
    performance_metrics["sq1_score"] = 0.5 * float(sq1_score["score"])
    performance_metrics["sq2_score"] = 0.5 * float(sq2_score["score"])
    performance_metrics["total_score"] = performance_metrics["sq1_score"] + performance_metrics["sq2_score"]

    performance_metrics = save_single_query_results(
        save_dir,
        idx,
        multi_hop_query,
        ground_truth,
        final_answer,
        final_reason,
        fusion_token_count,
        fallback_answer,
        fusion_prompt,
        eval_results,
        eval_results_fallback,
        performance_metrics,
        results,
        fused_answer_texts,
    )

    recorder.merge_meta({
        "final_answer": final_answer,
        "final_reason": final_reason,
        "fusion_token_count": fusion_token_count,
        "fusion_prompt": fusion_prompt,
        "eval_fusion": eval_results,
        "eval_fallback": eval_results_fallback,
        "sq1_score": performance_metrics.get("sq1_score"),
        "sq2_score": performance_metrics.get("sq2_score"),
        "total_score": performance_metrics.get("total_score"),
    })
    recorder.dump_stage("final.trace.json")

    return performance_metrics


def _process_one_query(
    idx,
    query_info,
    total,
    decompose,
    sources,
    merged_rag,
    use_routing,
    use_reflection,
    max_reflexion_times,
    openai_api_key,
    openai_model,
    openai_base_url,
    save_dir,
):
    """Run the full pipeline for a single query. Thread-safe: every per-query state
    (variable_values / results / performance_metrics / output file / trace cursor)
    lives in this function's stack or in thread-local storage."""

    multi_hop_query = query_info["query"]
    ground_truth = query_info["ground_truth"]
    q1_answers = query_info.get("q1_answers", [])
    q2_answers = query_info.get("q2_answers", [])

    print(f"\n📝 Processing query {idx}/{total}:")
    print(f"Query: {multi_hop_query}")
    print(f"Ground Truth: {ground_truth}")

    recorder = get_recorder()
    recorder.set_idx(idx)
    recorder.set_stage("decompose")
    _q_scope = _question_scope(idx)
    _q_scope.__enter__()
    recorder.merge_meta({
        "query_index": idx,
        "question": multi_hop_query,
        "ground_truth": ground_truth,
        "q1_answers": q1_answers,
        "q2_answers": q2_answers,
        "decompose_enabled": decompose,
    })

    query_plan = plan_subqueries_with_llm(decompose, multi_hop_query, openai_api_key, openai_model, openai_base_url)
    if not query_plan or not query_plan["subqueries"]:
        print(f"❌ [idx={idx}] Subquery planning failed, skipping current query.")
        recorder.merge_meta({"status": "subquery_planning_failed", "query_plan": query_plan})
        recorder.dump_stage("decompose.trace.json")
        recorder.reset_idx()
        _q_scope.__exit__(None, None, None)
        return None

    recorder.merge_meta({"query_plan": query_plan, "status": "ok"})
    recorder.dump_stage("decompose.trace.json")

    variable_values = {}
    results, fused_answer_texts = [], []
    performance_metrics = {
        "total_retrieval_time": 0,
        "total_docs_searched": 0,
        "avg_similarity_scores": [],
        "max_similarity_scores": [],
        "subquery_metrics": [],
        "total_prompt_tokens": 0,
        "prompt_token_counts": [],
        "evaluation_metrics": {
            "exact_match": 0,
            "f1": 0,
            "final_answer": "",
            "ground_truth": ground_truth,
        },
        "evaluation_metrics_fallback": {
            "exact_match": 0,
            "f1": 0,
            "ground_truth": ground_truth,
        },
    }

    performance_metrics = process_subqueries(
        performance_metrics,
        query_plan,
        variable_values,
        sources,
        merged_rag,
        use_routing,
        use_reflection,
        max_reflexion_times,
        openai_api_key,
        openai_model,
        openai_base_url,
        save_dir,
        idx,
        multi_hop_query,
        ground_truth,
        q1_answers,
        q2_answers,
        results,
        fused_answer_texts,
    )
    recorder.reset_idx()
    _q_scope.__exit__(None, None, None)
    return performance_metrics


def _load_completed_metrics(save_dir, idx):
    """Try to reconstruct an all_metrics entry from a previously written
    query_<idx>_results.jsonl. Returns metrics dict on success, None if the file
    is missing/corrupt/incomplete (caller should re-run that idx)."""
    path = os.path.join(save_dir, f"query_{idx}_results.jsonl")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    except Exception:
        return None

    by_type = {}
    for r in rows:
        t = r.get("type")
        if t and t not in by_type:
            by_type[t] = r

    # File must contain at least the four header rows we always write.
    needed = {"query_info", "final_answer", "evaluation_metrics", "performance_metrics"}
    if not needed.issubset(by_type.keys()):
        return None

    fa = by_type["final_answer"]
    em = by_type["evaluation_metrics"]
    pm = by_type["performance_metrics"]
    qi = by_type["query_info"]

    fusion = em.get("fusion") or {}
    fallback = em.get("fallback") or {}
    token_cost = pm.get("token_cost") or {}

    return {
        "total_retrieval_time": pm.get("total_retrieval_time", 0),
        "total_docs_searched": pm.get("total_docs_searched", 0),
        "avg_retrieval_time": pm.get("avg_retrieval_time", 0),
        "avg_similarity": pm.get("avg_similarity", 0),
        "max_similarity": pm.get("max_similarity", 0),
        "total_prompt_tokens": token_cost.get("total_prompt_tokens", 0),
        "avg_prompt_tokens": token_cost.get("avg_prompt_tokens", 0),
        "max_prompt_tokens": token_cost.get("max_prompt_tokens", 0),
        "min_prompt_tokens": token_cost.get("min_prompt_tokens", 0),
        "fusion_prompt_tokens": fa.get("fusion_prompt_tokens", 0),
        "evaluation_metrics": {
            "exact_match": fusion.get("exact_match", 0),
            "f1": fusion.get("f1", 0),
            "final_answer": fa.get("final_answer", ""),
            "final_reason": fa.get("final_reason", ""),
            "fallback_answer": fa.get("fallback_answer", ""),
            "ground_truth": qi.get("ground_truth", ""),
        },
        "evaluation_metrics_fallback": {
            "exact_match": fallback.get("exact_match", 0),
            "f1": fallback.get("f1", 0),
            "ground_truth": qi.get("ground_truth", ""),
        },
        "sq1_score": em.get("sq1_score", 0),
        "sq2_score": em.get("sq2_score", 0),
        "total_score": em.get("total_score", 0),
        "execution_time": 0,  # not preserved; will be backfilled at the end if needed
        "_resumed_from_disk": True,
    }


def single_query_execution(
    decompose,
    all_metrics,
    queries_and_truth,
    sources,
    merged_rag,
    use_routing,
    use_reflection,
    max_reflexion_times,
    openai_api_key,
    openai_model,
    openai_base_url,
    save_dir,
    concurrency=1,
    resume=False,
):
    total = len(queries_and_truth)
    concurrency = max(1, int(concurrency))
    print(f"[Batch] total queries = {total}, concurrency = {concurrency}, resume = {resume}")

    common_kwargs = dict(
        total=total,
        decompose=decompose,
        sources=sources,
        merged_rag=merged_rag,
        use_routing=use_routing,
        use_reflection=use_reflection,
        max_reflexion_times=max_reflexion_times,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        save_dir=save_dir,
    )

    indexed_metrics = []
    metrics_lock = threading.Lock()
    failures = []

    # Resume: scan save_dir, reuse query_<idx>_results.jsonl for already-completed
    # entries, and only schedule the unfinished ones.
    pending = []
    skipped = 0
    for idx, query_info in enumerate(queries_and_truth, 1):
        if resume:
            cached = _load_completed_metrics(save_dir, idx)
            if cached is not None:
                indexed_metrics.append((idx, cached))
                skipped += 1
                continue
        pending.append((idx, query_info))

    if resume:
        print(f"[Resume] reused {skipped} completed queries; {len(pending)} remaining to run")

    if not pending:
        print("[Resume] nothing left to run, skipping execution loop")
    elif concurrency == 1:
        for idx, query_info in pending:
            try:
                pm = _process_one_query(idx=idx, query_info=query_info, **common_kwargs)
                if pm is not None:
                    indexed_metrics.append((idx, pm))
            except Exception as e:
                failures.append((idx, str(e)))
                print(f"[Worker Error] idx={idx}, error={e}")
                traceback.print_exc()
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            fut2idx = {}
            for idx, query_info in pending:
                fut = pool.submit(_process_one_query, idx=idx, query_info=query_info, **common_kwargs)
                fut2idx[fut] = idx

            done = 0
            for fut in as_completed(fut2idx):
                idx = fut2idx[fut]
                done += 1
                try:
                    pm = fut.result()
                    if pm is not None:
                        with metrics_lock:
                            indexed_metrics.append((idx, pm))
                    print(f"[Done {done}/{len(pending)}] idx={idx}")
                except Exception as e:
                    failures.append((idx, str(e)))
                    print(f"[Worker Error] idx={idx}, error={e}")
                    traceback.print_exc()

    if failures:
        print(f"\n!! failures ({len(failures)}):")
        for idx, err in failures:
            print(f"  idx={idx}, err={err}")

    indexed_metrics.sort(key=lambda x: x[0])
    all_metrics.extend(pm for _, pm in indexed_metrics)
    return all_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSieve RAG-only pipeline")
    parser.add_argument("--decompose", action="store_true", help="Enable query decomposition")
    parser.add_argument("--use_routing", action="store_true", help="Enable source routing")
    parser.add_argument("--use_reflection", action="store_true", help="Enable reflection on failed queries")
    parser.add_argument("--max_reflexion_times", type=int, default=2, help="Max retry times for reflection")

    parser.add_argument("--dataset", type=str, default="hotpot_qa", help="Dataset name (e.g. hotpot_qa or mmqa)")
    parser.add_argument("--kb", type=str, default=os.environ.get("KB", "kg"),
                        help="Knowledge base for MMQA-family datasets (--dataset mmqa / mmqa_doc): "
                             "the live data sources. Comma-separated subset of {kg,table,doc}; "
                             "all three are individually switchable. N-source routing is confined "
                             "to these sources, with profiles loaded from new_model/data_sources/"
                             "source_profiles.json. Examples: 'kg', 'kg,table', 'table,doc', "
                             "'kg,table,doc'. Ignored for classic datasets (local/global).")
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, default="",
                        help="Optional path to a QA JSONL/JSON file. For --dataset mmqa, overrides "
                             "the default data/MMQA/kg-table-t2k-65.jsonl. For other datasets, overrides "
                             "data/rag/{dataset}_qa.json. If empty, falls back to the default path.")
    parser.add_argument("--run-tag", "--run_tag", dest="run_tag", type=str, default="",
                        help="Optional tag appended to the output dir name "
                             "(outputs/{rag_type}_{dataset}__{tag}...). Use to keep parallel "
                             "runs on different QA files from overwriting each other.")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of samples to evaluate")

    parser.add_argument("--openai_model", type=str, default=os.environ.get("OPENAI_MODEL", "deepseek-chat"))
    parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--openai_base_url", type=str, default=os.environ.get("OPENAI_API_BASE"))

    # atomr-style aliases (hyphenated). Take precedence over the underscored flags above when set.
    parser.add_argument("--llm-url", dest="llm_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""),
                        help="OpenAI-compatible base_url, e.g. https://api.chatanywhere.tech/v1. Overrides --openai_base_url.")
    parser.add_argument("--llm-model", dest="llm_model", type=str, default=os.environ.get("DEEPSIEVE_LLM_MODEL", ""),
                        help="Model name, e.g. deepseek-chat. Overrides --openai_model.")
    parser.add_argument("--api-key", dest="api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""),
                        help="API key. Overrides --openai_api_key.")
    parser.add_argument("--concurrency", type=int,
                        default=int(os.environ.get("DEEPSIEVE_CONCURRENCY", "1")),
                        help="Number of queries processed in parallel via ThreadPoolExecutor; 1 = serial.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip queries whose query_<idx>_results.jsonl already exists in save_dir "
                             "and is well-formed. Reuse their metrics in the final overall_results aggregation.")

    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="",
                        help="Explicit output directory. When set, overrides the auto-generated "
                             "outputs/{rag_type}_{dataset}_... path. Per-query results, "
                             "overall_results.json/txt and traces/ all go here.")
    parser.add_argument("--rag_type", type=str, choices=["naive", "graph"], default=os.environ.get("RAG_TYPE", "naive"))
    parser.add_argument("--trace_dir", type=str, default=os.environ.get("DEEPSIEVE_TRACE_DIR", ""),
                        help="Directory to write per-query trace files (idx_<N>/decompose|subq_*|final.trace.json). "
                             "Falls back to DEEPSIEVE_TRACE_DIR env var.")
    return parser.parse_args()


def main(args):
    _reset_cost_agg()
    dataset_l = args.dataset.lower().strip()
    is_mmqa = dataset_l in ("mmqa", "mmqa_doc")
    queries_and_truth = load_queries(args.dataset, args.sample_size, dataset_path=getattr(args, "dataset_path", "") or "")
    # --output-dir, when given, is used verbatim; otherwise fall back to the
    # auto-generated outputs/{rag_type}_{dataset}_kb-.../ path.
    out_dir = (getattr(args, "output_dir", "") or "").strip()
    if out_dir:
        save_dir = out_dir if os.path.isabs(out_dir) else os.path.abspath(out_dir)
    else:
        save_dir = get_save_dir(args.decompose, args.use_routing, args.use_reflection, args.dataset, args.rag_type,
                                run_tag=getattr(args, "run_tag", "") or "",
                                kb=(args.kb if is_mmqa else ""))
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Output dir = {save_dir}")

    if getattr(args, "resume", False):
        _seed_path = os.path.join(save_dir, "cost_summary.json")
        _seeded = _seed_cost_summary(_seed_path)
        if _seeded:
            print(f"[deepsieve] resume: seeded cost aggregator with {_seeded} prior entries from {_seed_path}")

    trace_dir = (getattr(args, "trace_dir", "") or "").strip()
    if not trace_dir:
        # Default: traces sit alongside the output dir for the same run.
        trace_dir = os.path.join(save_dir, "traces")
    if not os.path.isabs(trace_dir):
        trace_dir = os.path.abspath(trace_dir)
    recorder = get_recorder()
    recorder.set_output_dir(trace_dir)
    print(f"🧾 Trace enabled, dir = {trace_dir}")

    # atomr-style hyphen flags take precedence; fall back to underscore versions / env.
    openai_model = (getattr(args, "llm_model", "") or "").strip() or args.openai_model
    openai_api_key = (getattr(args, "api_key", "") or "").strip() or args.openai_api_key
    openai_base_url = (getattr(args, "llm_url", "") or "").strip() or args.openai_base_url
    if not openai_api_key:
        raise ValueError("❌ Please set --api-key (or OPENAI_API_KEY env var).")
    if not openai_base_url:
        raise ValueError("❌ Please set --llm-url (or OPENAI_API_BASE / OPENAI_BASE_URL env var).")
    print(f"[LLM] base_url={openai_base_url}, model={openai_model}")

    # Sync CLI/decided config into env vars so submodules that build their own
    # LLM client (notably rag/retrieve/query_llm.py via query_kg.py) share the
    # same gateway/key/model. Mirrors atomr's main.py:177-183 pattern.
    os.environ["OPENAI_API_BASE"] = openai_base_url
    os.environ["OPENAI_BASE_URL"] = openai_base_url
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_MODEL"] = openai_model
    os.environ["ATOMR_KG_PARSER_MODEL"] = openai_model

    if is_mmqa:
        # MMQA-family: the knowledge base is set by --kb (kg always on,
        # table/doc opt-in). Per-source profiles come from new_model's
        # source_profiles.json; N-source routing is confined to the live set.
        enabled_sources = _parse_kb(args.kb)
        profiles = load_source_profiles(enabled_sources)
        sources, merged_rag = initialize_mmqa_sources(
            enabled_sources, profiles, args.use_routing
        )
        print(f"✅ Knowledge base: {', '.join(enabled_sources)}")
    else:
        # Classic datasets keep the original local/global dual-source setup.
        source_a_docs, source_b_docs, source_a_profile, source_b_profile = \
            load_corpus_and_profiles(args.dataset)
        print(f"✅ Loaded source A docs: {len(source_a_docs)} (local)")
        print(f"✅ Loaded source B docs: {len(source_b_docs)} (global)")
        source_a_rag, source_b_rag, merged_rag = initialize_rag_system(
            args.rag_type,
            args.use_routing,
            source_a_docs,
            source_b_docs,
            dataset=args.dataset,
        )
        sources = [
            {"name": "local", "rag": source_a_rag, "profile": source_a_profile},
            {"name": "global", "rag": source_b_rag, "profile": source_b_profile},
        ]

    all_metrics = []
    start_time = time.perf_counter()
    all_metrics = single_query_execution(
        args.decompose,
        all_metrics,
        queries_and_truth,
        sources,
        merged_rag,
        args.use_routing,
        args.use_reflection,
        args.max_reflexion_times,
        openai_api_key,
        openai_model,
        openai_base_url,
        save_dir,
        concurrency=getattr(args, "concurrency", 1),
        resume=getattr(args, "resume", False),
    )

    total_elapsed = time.perf_counter() - start_time
    for item in all_metrics:
        item["execution_time"] = item.get("execution_time", 0)
    if all_metrics:
        per_query = total_elapsed / len(all_metrics)
        for item in all_metrics:
            item["execution_time"] = per_query
    overall_metrics = calculate_overall_metrics(all_metrics)
    overall_metrics["total_elapsed_time"] = total_elapsed
    save_overall_results(save_dir, overall_metrics, queries_and_truth, all_metrics)

    _cost_path = os.path.join(save_dir, "cost_summary.json")
    _cost_summary = _dump_cost_summary(_cost_path)
    print(f"Cost summary-> {_cost_path}")
    print(f"[cost] {_format_cost_line(_cost_summary)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
