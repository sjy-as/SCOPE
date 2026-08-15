import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, ".."))
if CUR_DIR not in sys.path:
    sys.path.append(CUR_DIR)

from query_knowledge_source.query_llm import OpenAICaller
from global_reasoner import GlobalReasoner
from main_mline import run_query
from trace_recorder import get_recorder


def _load_build_reasoning_tree():
    import importlib.util

    tree_gre_path = os.path.join(CUR_DIR, "test", "tree_gre.py")
    spec = importlib.util.spec_from_file_location("tree_gre_module", tree_gre_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import build_reasoning_tree from {tree_gre_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_reasoning_tree


build_reasoning_tree = _load_build_reasoning_tree()


# --kb 知识库 -> AtomR 的知识源集合。三者皆可开关；kg->KG, table->Table, doc->Text。
_KB_TO_ATOMR_SOURCE = {"kg": "KG", "table": "Table", "doc": "Text"}


def _parse_kb(raw: str) -> set:
    """把 --kb（逗号分隔的 {kg,table,doc} 子集）解析成 AtomR 的知识源集合。
    三者均为可选；doc 映射到 AtomR 的 Text 段落检索源。"""
    want = {tok.strip().lower() for tok in (raw or "").split(",") if tok.strip()}
    if not want:
        raise SystemExit("[main] --kb 不能为空，至少需要 kg/table/doc 之一")
    bad = sorted(want - set(_KB_TO_ATOMR_SOURCE))
    if bad:
        raise SystemExit(
            f"[main] unknown --kb source(s): {bad}  (allowed: kg, table, doc)"
        )
    return {_KB_TO_ATOMR_SOURCE[s] for s in want}


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(CUR_DIR, path))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_gold_from_answer_entry(entry: Dict[str, Any]) -> List[str]:
    if "answers" in entry and isinstance(entry["answers"], list):
        return [str(x) for x in entry["answers"]]

    q2 = entry.get("q2", {}) if isinstance(entry.get("q2", {}), dict) else {}
    if "answers" in q2 and isinstance(q2["answers"], list):
        return [str(x) for x in q2["answers"]]
    if "answer" in q2 and isinstance(q2["answer"], list):
        return [str(x) for x in q2["answer"]]

    q1 = entry.get("q1", {}) if isinstance(entry.get("q1", {}), dict) else {}
    if "answers" in q1 and isinstance(q1["answers"], list):
        return [str(x) for x in q1["answers"]]
    if "answer" in q1 and isinstance(q1["answer"], list):
        return [str(x) for x in q1["answer"]]

    return []


def build_answer_map(answer_path: str) -> Dict[int, List[str]]:
    answer_rows = read_jsonl(answer_path)
    answer_map: Dict[int, List[str]] = {}
    for row in answer_rows:
        idx = int(row.get("index", -1))
        if idx < 0:
            continue
        answer_map[idx] = extract_gold_from_answer_entry(row)
    return answer_map


def load_done_indices(path: str) -> set:
    """读取已完成的 index 集合。文件不存在或为空返回空集合，损坏行跳过。"""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "index" in row:
                try:
                    done.add(int(row["index"]))
                except Exception:
                    continue
    return done


def _process_one_entry(args, entry, gold, position, total, openai_caller, reasoner, recorder, write_lock,
                        f_tree, f_pred):
    """处理一条 entry 的完整流程：build tree → execute → 写文件 → dump trace。

    每个 worker 线程独立调用。trace_recorder cursor 是 thread-local，互不干扰。"""
    q_index = int(entry.get("index", position - 1))
    question = str(entry.get("question", "")).strip()

    print(f"\n[{position}/{total}] index={q_index}")
    print(f"Q: {question}")

    # trace: 切到当前问题（thread-local）
    recorder.set_idx(q_index)

    # 1) build reasoning tree
    recorder.set_stage("decompose")
    recorder.merge_meta({"question": question, "gold": gold})
    try:
        tree = build_reasoning_tree(question=question, dataset_name=args.dataset_name, openai_caller=openai_caller)
    except Exception as e:
        print(f"[Tree Error] index={q_index}, error={e}")
        tree = "TREE_PARSING_ERROR"
    recorder.merge_meta({
        "reasoning_tree": tree,
        "status": "tree_parse_error" if tree == "TREE_PARSING_ERROR" else "ok",
    })
    recorder.dump_stage("decompose.trace.json")

    tree_row = {
        "index": q_index,
        "question": question,
        "gold": gold,
        "reasoning_tree": tree,
    }
    with write_lock:
        f_tree.write(json.dumps(tree_row, ensure_ascii=False) + "\n")
        f_tree.flush()
        os.fsync(f_tree.fileno())

    # 2) execute tree
    predicted = ""
    recorder.set_stage("execute")
    if tree != "TREE_PARSING_ERROR":
        try:
            print(f"[Exec Start] index={q_index}")
            _, _, predicted = run_query(
                reasoner=reasoner,
                dataset_name=args.dataset_name.lower().strip(),
                question=question,
                q_index=q_index,
                gold=gold,
                build_tree_only=False,
                execute_tree_only=True,
                input_tree=tree,
            )
        except Exception as e:
            print(f"[Exec Error] index={q_index}, error={e}")
            traceback.print_exc()
    retrieval_counts = recorder.get_retrieval_counts()
    recorder.dump_stage("execute.trace.json", extra={"retrieval_counts": retrieval_counts, "predicted": predicted})
    recorder.reset_idx()

    pred_row = {
        "index": q_index,
        "question": question,
        "predicted": predicted,
        "gold": gold,
        "retrieval_counts": retrieval_counts,
    }
    with write_lock:
        f_pred.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
        f_pred.flush()
        os.fsync(f_pred.fileno())

    return q_index


def run_batch(args):
    dataset_path = resolve_path(args.dataset_path)
    answer_path = resolve_path(args.answer_path)
    output_trees_path = resolve_path(args.output_trees_path)
    output_predictions_path = resolve_path(args.output_predictions_path)

    os.makedirs(os.path.dirname(output_trees_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_predictions_path), exist_ok=True)

    # 初始化 trace 目录（参数优先于环境变量）
    trace_dir = (args.trace_dir or os.environ.get("ATOMR_TRACE_DIR", "")).strip()
    recorder = get_recorder()
    if trace_dir:
        recorder.set_output_dir(resolve_path(trace_dir))
        print(f"[Trace] enabled, dir = {recorder._output_dir}")

    entries = read_jsonl(dataset_path)
    answer_map = build_answer_map(answer_path)

    # 把 CLI 同步到环境变量，让 query_kg.py 内部独立创建的 OpenAICaller 也能用同一套配置
    if args.llm_url:
        os.environ["OPENAI_BASE_URL"] = args.llm_url
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.llm_model:
        os.environ["ATOMR_LLM_MODEL"] = args.llm_model
        os.environ["ATOMR_KG_PARSER_MODEL"] = args.llm_model

    # 单一 LLM 客户端（线程安全），所有 worker 复用
    openai_caller = OpenAICaller(
        base_url=args.llm_url,
        api_key=args.api_key,
        model=args.llm_model,
        cache_path=args.llm_cache_path,
    )
    print(f"[LLM] base_url={openai_caller.base_url}, model={openai_caller.default_model}, "
          f"cache={'on' if openai_caller.use_cache else 'off'}")

    knowledge_sources = _parse_kb(args.kb)
    print(f"[kb={args.kb}] available knowledge sources = {sorted(knowledge_sources)}")
    reasoner = GlobalReasoner(
        openai_caller=openai_caller,
        available_knowledge_sources=knowledge_sources,
        text_retriever_url=args.text_retriever_url,
        table_retriever_url=args.table_retriever_url,
        kg_query_language=args.kg_query_language,
        k=args.k,
    )

    total = len(entries)
    concurrency = max(1, int(args.concurrency))

    # ---- 断点续跑：以 predictions 文件为权威进度（一条 prediction 表示该 index 完整跑完）----
    done_indices: set = set()
    file_mode = "w"
    if args.resume:
        done_indices = load_done_indices(output_predictions_path)
        # trees 文件如果有但 prediction 没有 → 仍重跑该 index（可能 tree 跑出来了但 exec 没完成或没写入）
        if done_indices:
            file_mode = "a"
            print(f"[Resume] found {len(done_indices)} completed predictions, will skip them")
        else:
            print(f"[Resume] no existing predictions found, starting fresh")

    pending = [(i, e) for i, e in enumerate(entries, 1)
               if int(e.get("index", i - 1)) not in done_indices]
    print(f"[Batch] total entries = {total}, pending = {len(pending)}, concurrency = {concurrency}")

    write_lock = threading.Lock()
    failures = []

    with open(output_trees_path, file_mode, encoding="utf-8") as f_tree, \
         open(output_predictions_path, file_mode, encoding="utf-8") as f_pred:

        if concurrency == 1:
            for i, entry in pending:
                gold = answer_map.get(int(entry.get("index", i - 1)), [])
                try:
                    _process_one_entry(args, entry, gold, i, total, openai_caller, reasoner,
                                       recorder, write_lock, f_tree, f_pred)
                except Exception as e:
                    failures.append((entry.get("index"), str(e)))
                    print(f"[Worker Error] index={entry.get('index')}, error={e}")
                    traceback.print_exc()
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                fut2idx = {}
                for i, entry in pending:
                    gold = answer_map.get(int(entry.get("index", i - 1)), [])
                    fut = pool.submit(_process_one_entry, args, entry, gold, i, total,
                                      openai_caller, reasoner, recorder, write_lock, f_tree, f_pred)
                    fut2idx[fut] = entry.get("index")

                done = 0
                for fut in as_completed(fut2idx):
                    idx = fut2idx[fut]
                    done += 1
                    try:
                        q_index = fut.result()
                        print(f"[Done {done}/{len(pending)}] index={q_index}")
                    except Exception as e:
                        failures.append((idx, str(e)))
                        print(f"[Worker Error] index={idx}, error={e}")
                        traceback.print_exc()

    print("\n[Batch Finished]")
    print(f"trees: {output_trees_path}")
    print(f"predictions: {output_predictions_path}")
    print(f"total LLM calls: {openai_caller.total_llm_calls}, prompt tokens: {openai_caller.total_prompt_tokens}")
    if failures:
        print(f"!! failures ({len(failures)}):")
        for idx, err in failures:
            print(f"  index={idx}, err={err}")


def main():
    parser = argparse.ArgumentParser(description="MMQA batch: generate tree then execute tree for each entry")
    parser.add_argument("--dataset-name", default="mmqa")
    parser.add_argument("--dataset-path", default="../datasets/MMQA/kg-table.jsonl")
    parser.add_argument("--answer-path", default="../datasets/MMQA/kg-table-answer.jsonl")
    parser.add_argument("--output-trees-path", default="../results/mmqa_tree.jsonl")
    parser.add_argument("--output-predictions-path", default="../results/mmqa_predictions.jsonl")

    parser.add_argument("--text-retriever-url", default="http://127.0.0.1:1214/api/search")
    parser.add_argument("--table-retriever-url", default="http://127.0.0.1:1216/api/search")
    parser.add_argument("--kg-query-language", default="local")
    parser.add_argument("--kb", default="kg",
                        help="知识库 = 活的知识源。逗号分隔的 {kg,table,doc} 子集，"
                             "三者均为可开关；doc 对应 AtomR 的 Text 段落检索源。"
                             "示例：'kg'（默认）、'kg,table'、'table,doc'、'kg,table,doc'。")

    # ---- LLM 直连配置 ----
    parser.add_argument("--llm-url", default=os.environ.get("OPENAI_BASE_URL", ""),
                        help="OpenAI 兼容网关 base_url，例如 https://api.chatanywhere.tech/v1。也可读 OPENAI_BASE_URL。")
    parser.add_argument("--llm-model", default=os.environ.get("ATOMR_LLM_MODEL", "deepseek-chat"),
                        help="模型名，例如 deepseek-chat / gpt-4o-mini / DeepSeek-V3.2-Fast。也可读 ATOMR_LLM_MODEL。")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                        help="OpenAI 兼容 API key。也可读 OPENAI_API_KEY。")

    parser.add_argument("--llm-cache-path", default="../../openai_service/llm_cache/cache.jsonl")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--trace-dir", default="",
                        help="按 idx 分目录写入轨迹 (LLM prompts/responses + 节点元数据)。也可读 ATOMR_TRACE_DIR。")
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("ATOMR_CONCURRENCY", "8")),
                        help="并发处理 entry 的 worker 数量；1=纯串行（兼容老行为）。")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：以 predictions 文件已写入的 index 为完成标记，跳过已完成项；"
                             "trees/predictions 用 append 模式打开。trees 文件中可能产生重复 index "
                             "（同一 index 重跑过 tree），下游用最后一条即可。")

    args = parser.parse_args()
    start = time.time()
    run_batch(args)
    print(f"elapsed: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
