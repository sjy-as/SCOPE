'''
cd /root/autodl-tmp/baseline/atomr/src && OPENAI_API_KEY="sk-51dapR57sBDeMgCY4dcoHWGp9qI7bBI35uG49LOYqvQubp6S" OPENAI_BASE_URL="https://api.chatanywhere.tech/v1" ATOMR_LLM_MODEL="deepseek-chat" python3 main_mline.py   --dataset-name mmqa   --dataset-path /root/autodl-tmp/new_model/qa_bench/kg-doc-160.jsonl   --output-trees-path /root/autodl-tmp/new_model/eval/result/answer/kg_doc/atomr/kgdoc_trees.jsonl   --output-predictions-path /root/autodl-tmp/new_model/eval/result/answer/kg_doc/atomr/kgdoc_pred.jsonl   --kb kg,doc   --workers 8   --resume
'''


import json
import time
import re
import os
import sys
from typing import Optional
from tqdm import tqdm
import traceback
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import torch

from global_reasoner import GlobalReasoner
from prompts.tree_generation_prompts import format_tree_generation_prompt_blendqa, format_tree_generation_prompt_mmqa
from prompts.answer_formulation_prompts import format_knowledge_source_selection_prompt
from query_knowledge_source.query_llm import OpenAICaller
from utils import extract_json_tree, extract_function_parameters, extract_ref_indices, find_qa_pairs_given_ref_indices, extract_knowledge_sources
from calculate_metrics import calculate_em, calculate_f1
from collections import defaultdict
from trace_recorder import get_recorder

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


def _safe_node_slug(text: str, maxlen: int = 40) -> str:
    """把 sub-question 字符串变成可作文件名的小段。"""
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(text or ""))[:maxlen].strip("_")
    return s or "node"

NUM_PROCESSES = 4  # Concurrency for [build_trees_only] mode only

num_llm_calls = 0
num_text_retriever_calls = 0
num_table_retriever_calls = 0
num_kg_retriever_calls = 0
num_direct_Answer = 0
num_direct_RAG_func_failure = 0
num_direct_RAG_unknown = 0
num_Search_calls = 0
num_Relate_calls = 0
num_Filter_calls = 0
num_END_calls = 0
num_answer_from_child_qa_pairs_calls = 0

# =========================
# Logging Config
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # DEBUG / INFO / WARN
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", 220))
MAX_LIST_PREVIEW = int(os.getenv("MAX_LIST_PREVIEW", 4))
SHOW_FULL_TABLE = os.getenv("SHOW_FULL_TABLE", "0") == "1"


def log_debug(*args):
    if LOG_LEVEL == "DEBUG":
        print(*args)


def log_info(*args):
    if LOG_LEVEL in {"DEBUG", "INFO"}:
        print(*args)


def _truncate(x, n=MAX_EVIDENCE_CHARS):
    s = str(x)
    return s if len(s) <= n else s[:n] + "...(truncated)"


def preview_list(lst, k=MAX_LIST_PREVIEW):
    if not isinstance(lst, list):
        return _truncate(lst)
    if len(lst) <= k:
        return [_truncate(i) for i in lst]
    return [_truncate(i) for i in lst[:k]] + [f"...(+{len(lst)-k} more)"]


def summarize_supporting_knowledge(supporting_knowledge):
    """
    Returns:
      source_hits: dict(Text/Table/KG -> hit count)
      preview: compact preview list for terminal
    """
    source_hits = {"Text": 0, "Table": 0, "KG": 0}
    preview = []

    if isinstance(supporting_knowledge, dict):
        for src in ["Text", "Table", "KG"]:
            val = supporting_knowledge.get(src, None)
            if val:
                if isinstance(val, list):
                    source_hits[src] += len(val)
                    pv = val if (SHOW_FULL_TABLE and src == "Table") else preview_list(val)
                else:
                    source_hits[src] += 1
                    pv = _truncate(val)
                preview.append({"source": src, "evidence_preview": pv})
    elif supporting_knowledge:
        preview.append({"source": "UnknownFormat", "evidence_preview": _truncate(supporting_knowledge)})

    return source_hits, preview


def print_reasoner_trace(trace, indent: str = "  "):
    if not isinstance(trace, dict) or not trace:
        return

    op = trace.get("operation", "unknown")
    print(f"{indent}[Trace] operation={op}")

    selected = trace.get("selected_sources", []) or []
    print(f"{indent}[Trace] selected_sources={selected}")

    source_details = trace.get("source_details", {}) or {}
    for src in ["Text", "Table", "KG"]:
        det = source_details.get(src) or {}
        hit_count = det.get("hit_count", 0)
        if hit_count <= 0:
            continue
        print(f"{indent}[Trace][{src}] hit_count={hit_count}")
        raw_preview = det.get("raw_answers_preview", []) or []
        for i, item in enumerate(raw_preview[:3], start=1):
            print(f"{indent}  raw[{i}]: {_truncate(item, 180)}")
        support_preview = det.get("supporting_knowledge_preview", "")
        if support_preview:
            print(f"{indent}  support_preview: {_truncate(support_preview, 180)}")

    clean = trace.get("clean_answer_list", []) or []
    print(f"{indent}[Trace] clean_answer_list={preview_list(clean, k=8)}")
    paraphrase = trace.get("paraphrase_answer", "")
    if paraphrase:
        print(f"{indent}[Trace] paraphrase={_truncate(paraphrase, 220)}")



tree_generation_prompts = {
    'mmqa': format_tree_generation_prompt_mmqa
}


def print_results(em, cover_em, f1, num_total_entries, num_tree_parsing_failures, num_execution_failures, s):
    global num_llm_calls
    global num_text_retriever_calls
    global num_table_retriever_calls
    global num_kg_retriever_calls
    global num_direct_Answer
    global num_direct_RAG_func_failure
    global num_direct_RAG_unknown
    global num_Search_calls
    global num_Relate_calls
    global num_Filter_calls
    global num_END_calls
    global num_answer_from_child_qa_pairs_calls
    print("\n" + "*****" * 3 + " Evaluation Results " + "*****" * 3)
    print("Total num exact match:", em)
    print("Total num cover exact match:", cover_em)
    print("Total num entries:", num_total_entries)
    if num_total_entries == 0:
        print("EM Score:", "N/A")
        print("Cover EM Score:", "N/A")
        print("F1 Score:", "N/A")
    else:
        print("EM Score:", em / num_total_entries)
        print("Cover EM Score:", cover_em / num_total_entries)
        print("F1 Score:", f1 / num_total_entries)
    total_num_retriever_calls = num_text_retriever_calls + num_table_retriever_calls + num_kg_retriever_calls
    if total_num_retriever_calls == 0:
        print("Total num_text_retriever_calls:", f"{num_text_retriever_calls}, N/A")
        print("Total num_table_retriever_calls:", f"{num_table_retriever_calls}, N/A")
        print("Total num_kg_retriever_calls:", f"{num_kg_retriever_calls}, N/A")
    else:
        print("Total num_text_retriever_calls:", f"{num_text_retriever_calls}, {(num_text_retriever_calls/total_num_retriever_calls)*100:.2f}%")
        print("Total num_table_retriever_calls:", f"{num_table_retriever_calls}, {(num_table_retriever_calls/total_num_retriever_calls)*100:.2f}%")
        print("Total num_kg_retriever_calls:", f"{num_kg_retriever_calls}, {(num_kg_retriever_calls/total_num_retriever_calls)*100:.2f}%")
    print("Total num_llm_calls:", num_llm_calls)
    if num_llm_calls == 0:
        print("Total num_direct_Answer:", f"{num_direct_Answer}, N/A")
        print("Total num_direct_RAG_func_failure:", f"{num_direct_RAG_func_failure}, N/A")
        print("Total num_direct_RAG_unknown:", f"{num_direct_RAG_unknown}, N/A")
        print("Total num_Search_calls:", f"{num_Search_calls}, N/A")
        print("Total num_Relate_calls:", f"{num_Relate_calls}, N/A")
        print("Total num_Filter_calls:", f"{num_Filter_calls}, N/A")
        print("Total num_END_calls:", f"{num_END_calls}, N/A")
        print("Total num_answer_from_child_qa_pairs_calls:", f"{num_answer_from_child_qa_pairs_calls}, N/A")
    else:
        print("Total num_direct_Answer:", f"{num_direct_Answer}, {(num_direct_Answer/num_llm_calls)*100:.2f}%")
        print("Total num_direct_RAG_func_failure:", f"{num_direct_RAG_func_failure}, {(num_direct_RAG_func_failure/num_llm_calls)*100:.2f}%")
        print("Total num_direct_RAG_unknown:", f"{num_direct_RAG_unknown}, {(num_direct_RAG_unknown/num_llm_calls)*100:.2f}%")
        print("Total num_Search_calls:", f"{num_Search_calls}, {(num_Search_calls/num_llm_calls)*100:.2f}%")
        print("Total num_Relate_calls:", f"{num_Relate_calls}, {(num_Relate_calls/num_llm_calls)*100:.2f}%")
        print("Total num_Filter_calls:", f"{num_Filter_calls}, {(num_Filter_calls/num_llm_calls)*100:.2f}%")
        print("Total num_END_calls:", f"{num_END_calls}, {(num_END_calls/num_llm_calls)*100:.2f}%")
        print("Total num_answer_from_child_qa_pairs_calls:", f"{num_answer_from_child_qa_pairs_calls}, {(num_answer_from_child_qa_pairs_calls/num_llm_calls)*100:.2f}%")
    print("Total num_tree_parsing_failures:", num_tree_parsing_failures)
    print("Total num_execution_failures:", num_execution_failures)
    print("\nTotal time (seconds):", time.time() - s)


def build_reasoning_tree(dataset_name, q, q_index, reasoner: GlobalReasoner):
    prompt = tree_generation_prompts[dataset_name](q)
    llm_response, finish_reason = reasoner.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
    if finish_reason == "length":  # retry once if the provider still truncates the tree
        llm_response, finish_reason = reasoner.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        print("build_reasoning_tree() finish_reason = \"length\", reran with max_tokens=10000.")
    try:
        reasoning_tree = extract_json_tree(llm_response)
    except Exception as e:
        print(f"\n!!! Reasoning tree parsing error at question {q_index}: '{q}'")
        print(f"Error: {e}")
        print(f"LLM Response: {llm_response}")
        print("Saving reasoning_tree = \"TREE_PARSING_ERROR\".")
        return "TREE_PARSING_ERROR"
    
    return reasoning_tree


def extract_gold_answers(entry):
    """Pull gold answers from a dataset row, tolerant of layout differences.

    kg-table rows may carry a top-level `answers`; kg-doc rows carry both a
    top-level `answers` and a `q2.answers`. Fall back through q2 -> q1 so the
    same loader works for kg-table and kg-doc benchmarks.
    """
    if isinstance(entry.get("answers"), list) and entry["answers"]:
        return [str(x) for x in entry["answers"]]
    for key in ("q2", "q1"):
        sub = entry.get(key)
        if isinstance(sub, dict):
            vals = sub.get("answers") or sub.get("answer")
            if isinstance(vals, list) and vals:
                return [str(x) for x in vals]
    return []


def build_tree_concurrently(dataset_name, entry, reasoner: GlobalReasoner):  # for multiprocessing
        question = entry["question"].strip()
        question_index = entry["index"]
        gold_answers = extract_gold_answers(entry)
        reasoning_tree = build_reasoning_tree(dataset_name, question, question_index, reasoner)

        return {"index": question_index, "question": question, "gold": gold_answers, "reasoning_tree": reasoning_tree}
    

def postorder_traversal(question_tree, node):  # performs post-order traversal recursively
    results = []
    children = question_tree.get(node, [])
    current_node_dict = {}
    
    if isinstance(children, list):
        child_questions = []
        for child in children:
            child_result = postorder_traversal(question_tree, child)
            results.extend(child_result)
            child_questions.append(child)
        current_node_dict[node] = child_questions  # append child questions for non-leaf nodes
    else:
        current_node_dict[node] = children  # append leaf function/[END] mark for leaf nodes
        
    results.append(current_node_dict) 
    
    return results


def convert_question_tree_to_list(question_tree, q, q_index):  
    try:
        root_question = next(iter(question_tree))
        ordered_questions = postorder_traversal(question_tree, root_question)
        print(f"[Convert Q Tree to List] index={q_index}, ordered_questions={ordered_questions}")
    except Exception as e:
        raise Exception(f"!!! Postorder question list convertion error at question {q_index}: '{q}'\nError: {e}")
    
    return ordered_questions


def execute_postorder_q_list(reasoner: GlobalReasoner, dataset_name: str, reasoning_tree: dict, postorder_q_list: list, q_index: Optional[int] = 0, gold: Optional[list] = None):
    global num_llm_calls
    global num_text_retriever_calls
    global num_table_retriever_calls
    global num_kg_retriever_calls
    global num_direct_Answer
    global num_direct_RAG_func_failure
    global num_direct_RAG_unknown
    global num_Search_calls
    global num_Filter_calls
    global num_Relate_calls
    global num_END_calls
    global num_answer_from_child_qa_pairs_calls
    
    num_llm_calls_before_cur_entry = num_llm_calls

    node_logs = []  # structured logs for each node
    
    """ 
    Local helper functions 
    """
    
    def extract_index_from_dict(d):
        key = next(iter(d.keys()))
        match = re.search(r'^(\d+)\.', key)
        return int(match.group(1)) if match else float('inf')
            
            
    def execute_select_knowledge_sources(reasoner, question, q_ref_indices, ref_qa_list):
        global num_llm_calls
        # if len(reasoner.available_knowledge_sources) <= 2:  # directly use available knowledge sources if less than 2 available
        #     selected_knowledge_sources = reasoner.available_knowledge_sources
        if True:  # let LLM planner select knowledge sources
            temp_q_for_knowledge_selection = question  # format temporary question for knowledge source selection  
            try:
                for ref in q_ref_indices:
                    temp_q_for_knowledge_selection = temp_q_for_knowledge_selection.replace(ref, ref_qa_list[ref]["answers"][0])  # simply replace each ref index with first answer
                prompt = format_knowledge_source_selection_prompt(temp_q_for_knowledge_selection, reasoner.available_knowledge_sources)
                llm_response, finish_reason = reasoner.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
                num_llm_calls += 1
                selected_knowledge_sources = extract_knowledge_sources(llm_response, reasoner.available_knowledge_sources)
            except Exception as e:
                print("!!! [Knowledge source selection failure]:", e)
                print("Setting selected_knowledge_sources=reasoner.available_knowledge_sources.")
                selected_knowledge_sources = reasoner.available_knowledge_sources
                    
        print("Selected knowledge sources:", selected_knowledge_sources)
        return selected_knowledge_sources
    
    
    def execute_direct_RAG(question, q_ref_indices, ref_qa_list, selected_knowledge_sources, direct_rag_for_unknown=False):
        global num_direct_RAG_func_failure, num_direct_RAG_unknown, num_llm_calls, num_text_retriever_calls, num_table_retriever_calls, num_kg_retriever_calls
        if (direct_rag_for_unknown):
            print("(entered direct_RAG() due to empty child aggregation answer)")
        else:
            print("(entered direct_RAG() due to function execution failure)")
        print("question:", question)
                    
        if selected_knowledge_sources is None:
            print("selected_knowledge_sources is None, calling execute_select_knowledge_sources()")
            selected_knowledge_sources = execute_select_knowledge_sources(reasoner, question, q_ref_indices, ref_qa_list)
                    
        # Rewrite questions with ref indices
        directRAG_questions_rewritten = []
        if "among" in question.lower():  # likely filter question, directly replace ref index with answer list
            question_rewritten_ans_lists = question
            for ref in q_ref_indices:
                question_rewritten_ans_lists = question_rewritten_ans_lists.replace(ref, str(ref_qa_list[ref]["answers"]))
            directRAG_questions_rewritten.append(question_rewritten_ans_lists)
        else:  # non-filter questions
            if len(q_ref_indices) == 0:
                directRAG_questions_rewritten.append(question)
            elif len(q_ref_indices) == 1:
                ref = q_ref_indices[0]
                ref_answers = ref_qa_list[ref]["answers"]
                for ans in ref_answers:
                    directRAG_questions_rewritten.append(question.replace(ref, ans))  # rewrite question with each possible ref answer
            elif len(q_ref_indices) == 2:  # supports maximum 2 ref indices
                ref1 = q_ref_indices[0]
                ref2 = q_ref_indices[1]
                ref_answers_1 = ref_qa_list[ref1]["answers"]
                ref_answers_2 = ref_qa_list[ref2]["answers"]
                for ans1 in ref_answers_1:  # rewrite question with each possible double ref answer combinations
                    for ans2 in ref_answers_2:
                        cur_question_rewritten = question.replace(ref1, ans1)
                        cur_question_rewritten = cur_question_rewritten.replace(ref2, ans2)
                        directRAG_questions_rewritten.append(cur_question_rewritten)  
            else:  # >= 3 indices, directly formulate single question_rewritten by replacing each question index with answer list
                question_rewritten_ans_lists = question
                for ref in q_ref_indices:
                    question_rewritten_ans_lists = question_rewritten_ans_lists.replace(ref, str(ref_qa_list[ref]["answers"]))
                directRAG_questions_rewritten.append(question_rewritten_ans_lists)
                    
        for question_rewritten in directRAG_questions_rewritten:
            print("direct_RAG() cur_question_rewritten:", question_rewritten)
                        
            clean_answer_list, paraphrase_answer, supporting_knowledge = reasoner.direct_RAG(dataset_name, selected_knowledge_sources, question_rewritten)
            print("clean_answer_list:", clean_answer_list)
            print("paraphrase_answer:", paraphrase_answer)
                        
            # Store answers to global answer dict
            global_qa_dict_combined_answers[question]["clean_answer_list"] += clean_answer_list
            global_qa_dict_combined_answers[question]["paraphrase_answer"] += f"{paraphrase_answer}; "
            global_qa_dict[question]["clean_answer_list"].append(clean_answer_list)
            global_qa_dict[question]["paraphrase_answer"].append(paraphrase_answer)
            global_qa_dict[question]["supporting_knowledge"].append(supporting_knowledge)
                        
            # Update statistics
            if direct_rag_for_unknown:
                num_direct_RAG_unknown += 1
            else:
                num_direct_RAG_func_failure += 1
            num_llm_calls += 1
            if "Text" in selected_knowledge_sources:
                num_text_retriever_calls += 1
            if "Table" in selected_knowledge_sources:
                num_table_retriever_calls += 1
            if "KG" in selected_knowledge_sources:
                num_kg_retriever_calls += 1
    
    """
    Main logic
    """
    
    root_question = next(iter(postorder_q_list[-1].keys()))  
    
    print("\n" + "======" * 6)
    print(f"Question {q_index}:", root_question)
    if gold is not None:
        print("Gold:", gold)
        
    print("\n*** Step 1 'build_reasoning_tree' (completed) ***")
    print("Reasoning Tree:", reasoning_tree)
    
    print("\n*** Step 2 'convert_question_tree_to_list' (completed) ***")
    print("Postorder Question List:", postorder_q_list)
    
    print("\n*** Step 3 'execute_postorder_q_list' ***")
    
    global_qa_dict = {}  # stores sub-questions and answers for the whole reasoning tree
    global_qa_dict_combined_answers = {}  # combines all sub-answers for each sub-question

    _trace_rec = get_recorder()
    _root_question_for_trace = next(iter(postorder_q_list[-1].keys()))

    for _node_pos, node in enumerate(postorder_q_list, 1):
        (question, children), = node.items()
        print("\nQuestion:", question)
        print("Children:", children)

        # ---- TRACE: 切到当前子节点 ----
        _is_root_node = (question == _root_question_for_trace)
        _stage_role = "final" if _is_root_node else f"subq_{_node_pos}"
        _trace_rec.set_stage(_stage_role)
        _trace_rec.merge_meta({
            "node_position": _node_pos,
            "is_root": _is_root_node,
            "question": question,
            "children": children if isinstance(children, str) else list(children),
        })

        cur_node_log = {
        "question": question,
        "children_type": type(children).__name__,
        "operator": None,
        "selected_sources": [],
        "source_hits": {"Text": 0, "Table": 0, "KG": 0},
        "supporting_knowledge_preview": [],
        "clean_answer_list": [],
        "paraphrase_answer": "",
        "is_empty": False,
        "fallback": None,  # direct_RAG_func_failure / direct_RAG_unknown / direct_Answer
        }


        # Initialize answer dicts and selected knowledge sources
        global_qa_dict[question] = {}
        global_qa_dict[question]["clean_answer_list"] = []
        global_qa_dict[question]["paraphrase_answer"] = []
        global_qa_dict[question]["supporting_knowledge"] = []
        global_qa_dict[question]["function"] = "N/A"
        global_qa_dict_combined_answers[question] = {} 
        global_qa_dict_combined_answers[question]["paraphrase_answer"] = ""
        global_qa_dict_combined_answers[question]["clean_answer_list"] = []
        selected_knowledge_sources = None  
        
        # Extract [Question ref indices] and record ref indices with actual answers
        q_ref_indices = extract_ref_indices(question) 
        ref_qa_list, ref_qa_paraphrase = find_qa_pairs_given_ref_indices(q_ref_indices, global_qa_dict_combined_answers)
        
        ##### Execute Node 
        
        ### Case A - current node is a leaf node with an atomic function
        if isinstance(children, str) and '[END]' not in children:  
            function_str = children
            
            ### Select external knowledge sources
            selected_knowledge_sources = execute_select_knowledge_sources(reasoner, question, q_ref_indices, ref_qa_list)
            
            ### Parse and execute function, redirects to direct_RAG() if function execution fails
            try:  
                function_parameters = extract_function_parameters(function_str)  
                global_qa_dict[question]["function"] = function_str
                
                ### Extract [Fun tion ref indices] and update ref_qa_list
                func_ref_indices = extract_ref_indices(function_str)  # question and function may refer to different indices, need to extract indices from both
                all_ref_indices_set = set(q_ref_indices + func_ref_indices)
                ref_qa_list, ref_qa_paraphrase = find_qa_pairs_given_ref_indices(all_ref_indices_set, global_qa_dict_combined_answers)
            
                ### Call Search() function
                if function_str.lower().startswith("search"):
                    
                    def execute_Search(question, function_str, function_parameters):
                        global num_llm_calls, num_Search_calls, num_text_retriever_calls, num_table_retriever_calls, num_kg_retriever_calls
                        print("function_str:", function_str)

                        # Check function parameters
                        if not (len(function_parameters) == 1 or len(function_parameters) == 2):
                            raise Exception(f"Invalid number of function parameters: expected 1 or 2, got {len(function_parameters)}.\nfunction_str: {function_str}")
                        
                        # Initialize question and parameters that may need ref index replacement
                        question_template = question
                        head_entity_param_template = function_parameters[0]
                        descriptor_param_template = ""
                        
                        # Get head entities（寻找实体引用）
                        head_entity_refs = []  # referenced answers in head entity param
                        head_entity_ref_idx_list = extract_ref_indices(function_parameters[0])
                        if len(head_entity_ref_idx_list) == 0:  # first func parameter is entity name string
                            head_entity_refs = [function_parameters[0]]
                        elif len(head_entity_ref_idx_list) == 1:   # first func parameter contains ref index, needs entity replacement
                            ref = head_entity_ref_idx_list[0]
                            head_entity_refs = ref_qa_list[ref]["answers"]
                            question_template = question_template.replace(ref, "[HEAD]")
                            head_entity_param_template = head_entity_param_template.replace(ref, "[HEAD]")
                        else:
                            raise Exception(f"Invalid number of ref indices for Search() parameter 1 \"entity_name\": expected 0 or 1, got {len(head_entity_ref_idx_list)}.\nfunction_str: {function_str}")
                        
                        # Get descriptors      
                        str_descriptor_refs = []  # referenced answers in descriptor param
                        if len(function_parameters) == 1:  # no descriptors
                            str_descriptor_refs = [""]
                        elif len(function_parameters) == 2:  # contains descriptors
                            descriptor_ref_idx_list = extract_ref_indices(function_parameters[1])
                            print("descriptor_ref_idx_list:", descriptor_ref_idx_list)
                            if len(descriptor_ref_idx_list) == 0:  # descriptor doesn't contain ref index
                                str_descriptor_refs = [function_parameters[1]]
                                descriptor_param_template = function_parameters[1]
                            elif len(descriptor_ref_idx_list) == 1:  # descriptor contains ref index
                                ref = descriptor_ref_idx_list[0]
                                str_descriptor_refs = ref_qa_list[ref]["answers"]
                                question_template = question_template.replace(ref, "[DES]")
                                descriptor_param_template = function_parameters[1].replace(ref, "[DES]")
                            else:
                                raise Exception(f"Invalid number of ref indices for Search() parameter 2 \"descriptor\": expected 0 or 1, got {len(descriptor_ref_idx_list)}.\nfunction_str: {function_str}")
                        
                        print("question_template:", question_template)
                        print("\nhead_entity_refs:", head_entity_refs)
                        print("str_descriptor_refs:", str_descriptor_refs)
                        
                        # Execute function
                        for entity in head_entity_refs:
                            for descriptor in str_descriptor_refs:
                                # Prepare function parameters
                                cur_question = question_template.replace("[HEAD]", entity)
                                cur_question = cur_question.replace("[DES]", descriptor)
                                cur_head_entity_param = head_entity_param_template.replace("[HEAD]", entity)
                                cur_descriptor_param = descriptor_param_template.replace("[DES]", descriptor)
                                # Append descriptor to question if not already included
                                if cur_descriptor_param != "" and cur_descriptor_param.lower() not in cur_question.lower():  
                                    cur_question = f"{cur_question} ({cur_descriptor_param})"
                                
                                # Call function
                                clean_answer_list, paraphrase_answer, supporting_knowledge = reasoner.Search(dataset_name, selected_knowledge_sources, cur_question, cur_head_entity_param, cur_descriptor_param)
                                print("clean_answer_list:", clean_answer_list)
                                
                                # Store answers to global answer dict
                                global_qa_dict_combined_answers[question]["clean_answer_list"] += clean_answer_list
                                global_qa_dict_combined_answers[question]["paraphrase_answer"] += f"{paraphrase_answer}; "
                                global_qa_dict[question]["clean_answer_list"].append(clean_answer_list)
                                global_qa_dict[question]["paraphrase_answer"].append(paraphrase_answer)
                                global_qa_dict[question]["supporting_knowledge"].append(supporting_knowledge)
                                
                                # Update statistics
                                num_Search_calls += 1
                                num_llm_calls += 1
                                if "Text" in selected_knowledge_sources:
                                    num_text_retriever_calls += 1
                                if "Table" in selected_knowledge_sources:
                                    num_table_retriever_calls += 1
                                if "KG" in selected_knowledge_sources:
                                    num_kg_retriever_calls += 1
                                    
                                    
                    execute_Search(question, function_str, function_parameters)
                    
                ### Call Relate() function  
                elif function_str.lower().startswith("relate"):
                    
                    def execute_Relate(question, function_str, function_parameters):
                        global num_llm_calls, num_Relate_calls, num_text_retriever_calls, num_table_retriever_calls, num_kg_retriever_calls
                        print("function_str:", function_str)
                        
                        # Check function parameters
                        if not len(function_parameters) == 2:
                            raise Exception(f"Invalid number of function parameters: expected 2, got {len(function_parameters)}.\nfunction_str: {function_str}")
                        
                        # Initialize question and parameters that may need ref index replacement
                        question_template = question
                        head_entity_param_template = function_parameters[0]
                        relation_param_template = function_parameters[1]
                        
                        # Get head entities
                        head_entity_refs = []  # referenced answers in head entity param
                        head_entity_ref_idx_list = extract_ref_indices(function_parameters[0])
                        if len(head_entity_ref_idx_list) == 0:  # first func parameter is entity name string
                            head_entity_refs = [function_parameters[0]]
                        elif len(head_entity_ref_idx_list) == 1:   # first func parameter contains ref index, needs entity replacement
                            ref = head_entity_ref_idx_list[0]
                            head_entity_refs = ref_qa_list[ref]["answers"]
                            question_template = question_template.replace(ref, "[HEAD]")
                            head_entity_param_template = head_entity_param_template.replace(ref, "[HEAD]")
                        else:
                            raise Exception(f"Invalid number of ref indices for Relate() parameter 1 \"entity_name\": expected 0 or 1, got {len(head_entity_ref_idx_list)}.\nfunction_str: {function_str}")
                        
                        # Get relations      
                        relation_refs = []  # referenced answers in relation param
                        relations_ref_idx_list = extract_ref_indices(function_parameters[1])
                        if len(relations_ref_idx_list) == 0:  # relation doesn't contain ref index
                            relation_refs = [function_parameters[1]]
                            relation_param_template = function_parameters[1]
                        elif len(relations_ref_idx_list) == 1:  # relation contains one ref index
                            ref = relations_ref_idx_list[0]
                            relation_refs = ref_qa_list[ref]["answers"]
                            question_template = question_template.replace(ref, "[REL]")
                            relation_param_template = relation_param_template.replace(ref, "[REL]")
                        else:
                            raise Exception(f"Invalid number of ref indices for Relate() parameter 2 \"relation\": expected 0 or 1, got {len(relations_ref_idx_list)}.\nfunction_str: {function_str}")
                        
                        # Reformat question (might contain different refs compared to function, reformatting just in case)
                        q_template_ref_indices = extract_ref_indices(question_template)  # extract any additional ref indices
                        for ref in q_template_ref_indices: 
                            question_template = question_template.replace(ref, ref_qa_list[ref]["answers"][0])  # TODO: only extracting first answer for now, should be enough
                        
                        print("question_template:", question_template)
                        print("head_entity_refs:", head_entity_refs)
                        print("relation_refs:", relation_refs)
                        
                        # Execute function
                        for entity in head_entity_refs:
                            for relation in relation_refs:
                                # Prepare function parameters
                                cur_question = question_template.replace("[HEAD]", entity)
                                cur_question = cur_question.replace("[REL]", relation)
                                cur_head_entity_param = head_entity_param_template.replace("[HEAD]", entity)
                                cur_relation_param = relation_param_template.replace("[REL]", relation)
                                print("\nRelate() cur_question:", cur_question)
                                
                                # Call function
                                clean_answer_list, paraphrase_answer, supporting_knowledge = reasoner.Relate(dataset_name, selected_knowledge_sources, cur_question, cur_head_entity_param, cur_relation_param)                           
                                print("clean_answer_list:", clean_answer_list)
                                
                                # Store answers to global answer dict
                                global_qa_dict_combined_answers[question]["clean_answer_list"] += clean_answer_list
                                global_qa_dict_combined_answers[question]["paraphrase_answer"] += f"{paraphrase_answer}; "
                                global_qa_dict[question]["clean_answer_list"].append(clean_answer_list)
                                global_qa_dict[question]["paraphrase_answer"].append(paraphrase_answer)
                                global_qa_dict[question]["supporting_knowledge"].append(supporting_knowledge)
                                
                                # Update statistics
                                num_Relate_calls += 1 
                                num_llm_calls += 1
                                if "Text" in selected_knowledge_sources:
                                    num_text_retriever_calls += 1
                                if "Table" in selected_knowledge_sources:
                                    num_table_retriever_calls += 1
                                if "KG" in selected_knowledge_sources:
                                    num_kg_retriever_calls += 1
                    
                    execute_Relate(question, function_str, function_parameters)
                        
                ### Call Filter() function    
                elif function_str.lower().startswith("filter"):
                    def execute_Filter(question, function_str, function_parameters):
                        global num_Filter_calls, num_llm_calls, num_text_retriever_calls, num_table_retriever_calls, num_kg_retriever_calls
                        print("function_str:", function_str)
                        
                        # Check parameters
                        if not len(function_parameters) == 2:
                            raise Exception(f"Invalid number of function parameters: expected 2, got {len(function_parameters)}.\nfunction_str: {function_str}")
                        
                        # Initialize question and parameters that may need ref index replacement
                        question_template = question
                        head_entities_param_template = function_parameters[0]
                        condition_param_template = function_parameters[1]
                            
                        # Get head entities
                        head_entities_refs = []  # referenced answers in head entities param
                        head_entities_ref_idx_list = extract_ref_indices(function_parameters[0])
                        print("head_entities_ref_idx_list:", head_entities_ref_idx_list)
                        if len(head_entities_ref_idx_list) == 0:  # first func parameter doesn't contain ref index
                            head_entities_refs = [function_parameters[0]]
                        elif len(head_entities_ref_idx_list) == 1:   # first func parameter contains ref index, needs entity replacement
                            ref = head_entities_ref_idx_list[0]
                            head_entities_refs = ref_qa_list[ref]["answers"]
                            # for Filter() function, directly replace entity ref with serialized list of entities
                            question_template = question_template.replace(ref, ", ".join(head_entities_refs))
                            head_entities_param_template = head_entities_param_template.replace(ref, ", ".join(head_entities_refs))
                        else:
                            raise Exception(f"Invalid number of ref indices for Filter() parameter 1 \"entities\": expected 0 or 1, got {len(head_entities_ref_idx_list)}.\nfunction_str: {function_str}")
                        
                        # Get conditions   
                        condition_refs = []  # referenced answers in relation param
                        conditions_ref_idx_list = extract_ref_indices(function_parameters[1])
                        print("conditions_ref_idx_list:", conditions_ref_idx_list)
                        if len(conditions_ref_idx_list) == 0:  # relation doesn't contain ref index
                            condition_refs = [function_parameters[1]]
                        elif len(conditions_ref_idx_list) == 1:  # relation contains one ref index
                            ref = conditions_ref_idx_list[0]
                            condition_refs = ref_qa_list[ref]["answers"]
                            question_template = question_template.replace(ref, "[COND]")
                            condition_param_template = condition_param_template.replace(ref, "[COND]")
                        else:
                            raise Exception(f"Invalid number of ref indices for Filter() parameter 2 \"condition\": expected 0 or 1, got {len(conditions_ref_idx_list)}.\nfunction_str: {function_str}")
                        
                        print("question_template:", question_template)
                        print("head_entities_refs:", head_entities_refs)
                        print("condition_refs:", condition_refs)
                            
                        # Execute function
                        cur_head_entities_param = head_entities_param_template  # static head entities for Filter(); only used for printing debug messages
                        for condition in condition_refs:
                            # Prepare function parameters
                            cur_question = question_template.replace("[COND]", condition)
                            cur_condition_param = condition_param_template.replace("[COND]", condition)
                            # Append condition to question if not already included
                            if cur_condition_param != "" and cur_condition_param.lower() not in cur_question.lower():  
                                cur_question = f"{cur_question} (Filter condition: {cur_condition_param})"
                            print("\nFilter() cur_question:", cur_question)
                            
                            clean_answer_list, paraphrase_answer, supporting_knowledge = reasoner.Filter(dataset_name, selected_knowledge_sources, cur_question, head_entities_refs, cur_condition_param)
                            print("clean_answer_list:", clean_answer_list)
                                    
                            # Store answers to global answer dict
                            global_qa_dict_combined_answers[question]["clean_answer_list"] += clean_answer_list
                            global_qa_dict_combined_answers[question]["paraphrase_answer"] += f"{paraphrase_answer}; "
                            global_qa_dict[question]["clean_answer_list"].append(clean_answer_list)
                            global_qa_dict[question]["paraphrase_answer"].append(paraphrase_answer)
                            global_qa_dict[question]["supporting_knowledge"].append(supporting_knowledge)
                            
                            # Update statistics
                            num_Filter_calls += 1 
                            num_llm_calls += 1
                            if "Text" in selected_knowledge_sources:
                                num_text_retriever_calls += 1
                            if "Table" in selected_knowledge_sources:
                                num_table_retriever_calls += 1
                            if "KG" in selected_knowledge_sources:
                                num_kg_retriever_calls += 1
                        
                    execute_Filter(question, function_str, function_parameters)
                    
            except Exception as e:  # if function execution failed, prefer LLM synthesis from prior evidence, then fallback to direct_RAG()
                print("\n!!! [Function Execution failure]:", function_str)
                print("Exception:", e)
                print("Traceback:", traceback.format_exc())

                used_llm_ref_fallback = False
                if len(ref_qa_list) > 0:
                    try:
                        ref_qa_pairs_formatted = []
                        for ref in sorted(ref_qa_list.keys()):
                            ref_q = ref_qa_list[ref]["question"]
                            ref_ans = ref_qa_list[ref]["answers"]
                            ref_para = ref_qa_paraphrase.get(ref, {}).get("answers", "")
                            if isinstance(ref_ans, list) and len(ref_ans) > 0:
                                ref_qa_pairs_formatted.append({ref_q: f"{ref_para}. {str(ref_ans)}" if ref_para else str(ref_ans)})

                        if len(ref_qa_pairs_formatted) > 0:
                            print("[Fallback] AnswerFromQAPairs with ref evidence:", ref_qa_pairs_formatted)
                            clean_answer_list, paraphrase_answer, supporting_knowledge = reasoner.AnswerFromQAPairs(dataset_name, question, ref_qa_pairs_formatted)
                            print("fallback clean_answer_list:", clean_answer_list)
                            print("fallback paraphrase_answer:", paraphrase_answer)

                            global_qa_dict_combined_answers[question]["clean_answer_list"] += clean_answer_list
                            global_qa_dict_combined_answers[question]["paraphrase_answer"] += f"{paraphrase_answer}; "
                            global_qa_dict[question]["clean_answer_list"].append(clean_answer_list)
                            global_qa_dict[question]["paraphrase_answer"].append(paraphrase_answer)
                            global_qa_dict[question]["supporting_knowledge"].append({"child_qa_pairs": ref_qa_pairs_formatted})

                            num_llm_calls += 1
                            used_llm_ref_fallback = not (
                                len(clean_answer_list) == 0 or str(paraphrase_answer).lower().strip() == "unknown"
                            )
                    except Exception as fallback_e:
                        print("[Fallback] AnswerFromQAPairs failed, will use direct_RAG.")
                        print("Fallback exception:", fallback_e)

                if not used_llm_ref_fallback:
                    execute_direct_RAG(question, q_ref_indices, ref_qa_list, selected_knowledge_sources, direct_rag_for_unknown=False)
                    
        ### Non-leaf node or [END] leaf node, answer using child/sibling QA pairs
        elif (isinstance(children, list) and len(children) > 0) or (isinstance(children, str) and '[END]' in children):  
            child_qa_pairs_list = []
            is_childQA = False  # flag to denote child aggregation QA (apart from [END] ref aggregation answer)
            
            # Case (1): Non-leaf node, answer from child QA pairs
            if isinstance(children, list):  
                for child in children:
                    child_q = child
                    try:
                        child_clean_answer_list = global_qa_dict_combined_answers[child_q]["clean_answer_list"]
                        child_paraphrase_answer = global_qa_dict_combined_answers[child_q]["paraphrase_answer"]
                    except Exception as e:
                        raise Exception(f"!!! Child question '{child_q}' not found in global_qa_dict.\nError: {e}")
                    child_qa_pairs_list.append({"question": child_q, "clean_answer_list": child_clean_answer_list, "paraphrase_answer": child_paraphrase_answer})
                is_childQA = True
                num_answer_from_child_qa_pairs_calls += 1
            
            # Case (2): [END] leaf node, answer from ref QA pairs
            else:  
                for ref in q_ref_indices:
                    ref_q = ref_qa_list[ref]["question"]
                    ref_clean_answer_list = ref_qa_list[ref]["answers"]
                    ref_paraphrase_answer = ref_qa_paraphrase[ref]["answers"] 
                    child_qa_pairs_list.append({"question": ref_q, "clean_answer_list": ref_clean_answer_list, "paraphrase_answer": ref_paraphrase_answer})
                num_END_calls += 1
                        
            # Format and augment child qa pairs
            child_qa_pairs_formatted = []
            added_child_q = set()  # record added child questions to avoid duplication during cross layer recordings
            for qa_pair in child_qa_pairs_list:
                child_q = qa_pair["question"]
                child_clean_answer_list = qa_pair["clean_answer_list"]
                child_paraphrase_answer = qa_pair["paraphrase_answer"]
                child_q_ref_indices = extract_ref_indices(child_q)
                if len(child_q_ref_indices) == 0:
                    child_qa_pairs_formatted.append({child_q: str(child_clean_answer_list)})  # no ref idx = no ambiguity, only append clean answer list
                    added_child_q.add(child_q)
                else:
                    child_qa_pairs_formatted.append({child_q: child_paraphrase_answer + ". " + str(child_clean_answer_list)})  # append paraphrase answer + clean answer list
                    added_child_q.add(child_q)
                    # Also append qa pairs referenced in child questions
                    child_cross_layer_ref_qa_list, child_cross_layer_ref_qa_paraphrase = find_qa_pairs_given_ref_indices(child_q_ref_indices, global_qa_dict_combined_answers)
                    for cross_layer_ref in child_q_ref_indices:
                        cross_layer_ref_q = child_cross_layer_ref_qa_list[cross_layer_ref]["question"]
                        if cross_layer_ref_q not in added_child_q:  # avoid duplication
                            child_qa_pairs_formatted.append({cross_layer_ref_q: str(child_cross_layer_ref_qa_list[cross_layer_ref]["answers"])})  # clean answer list should be enough
            
            child_qa_pairs_formatted = sorted(child_qa_pairs_formatted, key=extract_index_from_dict)  # sort in ascending index order
            print("child_qa_pairs_formatted:", child_qa_pairs_formatted)
            
            # Call function
            clean_answer_list, paraphrase_answer, supporting_knowledge = reasoner.AnswerFromQAPairs(dataset_name, question, child_qa_pairs_formatted)
            print("clean_answer_list:", clean_answer_list)
            
            num_llm_calls += 1
            
            # call direct_RAG() if childQA yields empty answer
            if is_childQA and (paraphrase_answer.lower().strip() == "unknown" or len(clean_answer_list) == 0):
                print("\n!!! [Empty answer for ChildQA]")
                    
                execute_direct_RAG(question, q_ref_indices, ref_qa_list, selected_knowledge_sources, direct_rag_for_unknown=True)
                continue  # cur node's answers already stored in execute_direct_RAG() function, can continue to next node
                                
            # Store answers to global answer dict
            global_qa_dict_combined_answers[question]["clean_answer_list"] += clean_answer_list
            global_qa_dict_combined_answers[question]["paraphrase_answer"] += f"{paraphrase_answer}; "

            global_qa_dict[question]["clean_answer_list"].append(clean_answer_list)
            global_qa_dict[question]["paraphrase_answer"].append(paraphrase_answer)
            global_qa_dict[question]["supporting_knowledge"].append(supporting_knowledge)
                        
        else:  # use direct_RAG() for empty child or unknown child type
            execute_direct_RAG(question, q_ref_indices, ref_qa_list, selected_knowledge_sources, direct_rag_for_unknown=True)
            
        
        cur_ans_paraphrase = global_qa_dict_combined_answers[question]["paraphrase_answer"]
        if len(cur_ans_paraphrase) >= 2 and cur_ans_paraphrase[-2:] == "; ":
            cur_ans_paraphrase = cur_ans_paraphrase[:-2]  # remove last "; " for paraphrase_answer
            global_qa_dict_combined_answers[question]["paraphrase_answer"] = cur_ans_paraphrase

        # ---- TRACE: 节点结束 → 落盘 ----
        if not _is_root_node:
            _node_combined = global_qa_dict_combined_answers.get(question, {}) or {}
            _node_full = global_qa_dict.get(question, {}) or {}
            _src_hits, _sk_preview = summarize_supporting_knowledge(
                (_node_full.get("supporting_knowledge") or [{}])[-1] if _node_full.get("supporting_knowledge") else {}
            )
            _trace_rec.merge_meta({
                "function_call": _node_full.get("function", "N/A"),
                "clean_answer_list": _node_combined.get("clean_answer_list", []),
                "paraphrase_answer": _node_combined.get("paraphrase_answer", ""),
                "selected_sources": cur_node_log.get("selected_sources", []),
                "source_hits": _src_hits,
                "supporting_knowledge_preview": _sk_preview,
                "fallback": cur_node_log.get("fallback"),
                "status": "ok",
            })
            _slug = _safe_node_slug(question)
            _trace_rec.dump_stage(f"subq_{_node_pos}_{_slug}.trace.json")


    ### Formulate final answer
    final_answer_list = global_qa_dict_combined_answers[root_question]["clean_answer_list"]
    final_answer_paraphrase = global_qa_dict_combined_answers[root_question]["paraphrase_answer"]
    _used_direct_answer_fallback = False

    if len(final_answer_list) == 0:  # if final answer is empty, call LLM to directly answer
        print("!!! final_answer_list = [], calling direct_Answer().")
        print("question:", root_question)
        final_answer_list, final_answer_paraphrase = reasoner.direct_Answer(dataset_name, root_question)
        print("final_answer_list:", final_answer_list)
        print("final_answer_paraphrase:", final_answer_paraphrase)

        global_qa_dict_combined_answers[root_question]["supporting_knowledge"] = "direct_Answer()"
        global_qa_dict[root_question]["supporting_knowledge"].append("direct_Answer()")
        num_llm_calls += 1
        num_direct_Answer += 1
        _used_direct_answer_fallback = True
    
    print("\n===================== Final Answer =====================") 
    print("Question:", root_question) 
    print("Full Postorder Reasoning Path: ")
    for sub_q in global_qa_dict:
        clean_answer_list = global_qa_dict_combined_answers[sub_q]["clean_answer_list"]
        function = global_qa_dict[sub_q]["function"]
        supporting_knowledge = global_qa_dict[sub_q]["supporting_knowledge"]
        print("Question:", sub_q)
        print("Clean Answer List:", clean_answer_list)
        print("Function Call:", function)
        # print("Supporting Knowledge:", supporting_knowledge)
        print()
    
    print("Paraphrase Answer:", final_answer_paraphrase)
    
    # format Predicted Answer list into single string
    predicted_answer = ", ".join(str(entry) for entry in final_answer_list)
    print("Predicted Answer:", predicted_answer)
    if gold is not None:
        print("Gold:", gold)
    
    print("num_llm_calls for cur question execution:", num_llm_calls - num_llm_calls_before_cur_entry)

    # ---- TRACE: 最终答案落盘 (root node 的 final.trace.json) ----
    try:
        _all_subq_summary = []
        for _sq in global_qa_dict:
            _all_subq_summary.append({
                "question": _sq,
                "function": global_qa_dict[_sq].get("function", "N/A"),
                "clean_answer_list": global_qa_dict_combined_answers.get(_sq, {}).get("clean_answer_list", []),
                "paraphrase_answer": global_qa_dict_combined_answers.get(_sq, {}).get("paraphrase_answer", ""),
            })
        _trace_rec.set_stage("final")
        _trace_rec.merge_meta({
            "root_question": root_question,
            "predicted_answer": predicted_answer,
            "final_answer_list": final_answer_list,
            "final_answer_paraphrase": final_answer_paraphrase,
            "used_direct_answer_fallback": _used_direct_answer_fallback,
            "gold": gold,
            "all_subq_summary": _all_subq_summary,
            "num_llm_calls_this_question": num_llm_calls - num_llm_calls_before_cur_entry,
            "status": "ok",
        })
        _trace_rec.dump_stage("final.trace.json")
    except Exception as _e:
        print(f"[Trace] final.trace.json dump failed: {_e}")

    return predicted_answer, global_qa_dict_combined_answers, global_qa_dict
    

### Function to run single query
def run_query(reasoner: GlobalReasoner, dataset_name: str, question: str, q_index: Optional[int] = 0, gold: Optional[list] = None, 
              build_tree_only: Optional[bool] = False, execute_tree_only: Optional[bool] = False, 
              input_tree: Optional[str] = None): 
    if (int(build_tree_only) + int(execute_tree_only)) != 1:
        raise Exception("Need to set exactly one mode to True: ['build_tree_only', 'execute_tree_only']")
    if (execute_tree_only and input_tree is None):
        raise Exception("\"execute_tree_only\" mode selected, but \"input_tree\" is None.")
    question = question.strip()  # remove possible extra spaces

    # Set default return values
    reasoning_tree = None
    postorder_q_list = None
    predicted_answer = ""
        
    if build_tree_only:
        reasoning_tree = build_reasoning_tree(dataset_name, question, q_index, reasoner)
        return reasoning_tree, postorder_q_list, ""
    
    # 代码优化后，直接基于树执行
    if execute_tree_only:
        reasoning_tree = input_tree
        if isinstance(reasoning_tree, str) and reasoning_tree == "TREE_PARSING_ERROR":
            print(f"\n### Tree parsing error for Question {q_index}: \"{question}\"") 
            print("Returning empty answers.")
            return reasoning_tree, None, ""
        postorder_q_list = convert_question_tree_to_list(reasoning_tree, question, q_index)
        predicted_answer, tree_clean, tree_full = execute_postorder_q_list(reasoner, dataset_name, reasoning_tree, postorder_q_list, q_index, gold) 
        
        return reasoning_tree, postorder_q_list, predicted_answer
 

### Function to evaluate jsonl dataset
def evaluate_dataset(reasoner: GlobalReasoner, dataset_name: str, dataset_path: Optional[str] = "", input_trees_path: Optional[str] = "", output_trees_path: Optional[str] = "", output_predictions_path: Optional[str] = "", build_trees_only: Optional[bool] = False, execute_trees_only: Optional[bool] = False, workers: Optional[int] = None, resume: bool = False):
    

    if (int(build_trees_only) + int(execute_trees_only)) != 1:
        raise Exception("Need to set exactly one mode to True: ['build_trees_only', 'execute_trees_only']")
    
    global num_llm_calls
    em = 0
    cover_em = 0
    f1 = 0.0
    num_entries = 0
    num_tree_parsing_failures = 0
    num_execution_failures = 0
    num_continuous_failed_executions = 0
    s = time.time()
    workers = workers or NUM_PROCESSES  # concurrency for both build and execute stages

    # Mode 1 - only build reasoning trees given dataset questions
    if build_trees_only:
        with open(dataset_path, 'r', encoding='utf-8') as dataset:  # Load test entries
            entries = [json.loads(line) for line in dataset]

        # --resume: reuse trees already written to output_trees_path. An index
        # counts as "done" only if its stored tree is NOT a TREE_PARSING_ERROR,
        # so both missing and previously-failed trees get (re)built.
        prebuilt = {}  # index -> stored tree result dict
        if resume and output_trees_path and os.path.exists(output_trees_path):
            with open(output_trees_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "index" in r:
                        prebuilt[r["index"]] = r
            done = {i for i, r in prebuilt.items()
                    if r.get("reasoning_tree") != "TREE_PARSING_ERROR"}
            print(f"[build][resume] {len(prebuilt)} trees found in {output_trees_path}; "
                  f"{len(done)} usable — will skip those, (re)build the rest")
        else:
            done = set()

        final_results = [prebuilt[e["index"]] for e in entries if e.get("index") in done]
        todo = [e for e in entries if e.get("index") not in done]

        # Build trees on a THREAD pool — NOT multiprocessing.Pool. The reasoner
        # holds an httpx client / threading.RLock and is therefore unpicklable,
        # so Pool.apply_async failed to dispatch every task and (having no
        # error_callback, and never being .get()'d) swallowed the errors
        # silently — yielding an empty trees file. OpenAICaller is documented
        # thread-safe, so threads are the correct concurrency model: the shared
        # reasoner is passed by reference, no pickling involved.
        def _build_one(entry):
            try:
                return build_tree_concurrently(dataset_name, entry, reasoner)
            except Exception as e:  # surface the failure instead of dropping it
                print(f"\n!!! [build] index={entry.get('index')} failed: {e}")
                return {"index": entry.get("index"),
                        "question": entry.get("question", ""),
                        "gold": extract_gold_answers(entry),
                        "reasoning_tree": "TREE_PARSING_ERROR"}

        if not todo:
            print("[build] all trees already built — skipping tree generation")
        else:
            print(f"[build] tree generation concurrency = {workers} threads "
                  f"({len(todo)}/{len(entries)} to build)")
            # Write each tree as it completes. This loop body runs in the main
            # thread (as_completed yields here), so append+write is already
            # serialized — no lock needed. A killed run keeps its progress on
            # disk and --resume picks it up; the file is re-sorted at the end.
            with open(output_trees_path, 'w', encoding='utf-8') as tf:
                for r in final_results:        # reused (resumed) trees first
                    tf.write(json.dumps(r) + '\n')
                tf.flush()
                pbar = tqdm(total=len(todo), desc="Building Reasoning Trees")
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_build_one, entry) for entry in todo]
                    for fut in as_completed(futures):
                        res = fut.result()
                        final_results.append(res)
                        tf.write(json.dumps(res) + '\n')
                        tf.flush()
                        os.fsync(tf.fileno())
                        pbar.update(1)
                pbar.close()

        # Rewrite the trees file in index order (tidy, deterministic output).
        final_results.sort(key=lambda x: x["index"])
        with open(output_trees_path, 'w', encoding='utf-8') as output_trees:
            for result in final_results:
                reasoning_tree = result["reasoning_tree"]
                if isinstance(reasoning_tree, str) and reasoning_tree == "TREE_PARSING_ERROR":
                    num_tree_parsing_failures += 1
                output_trees.write(json.dumps(result) + '\n')

        print("Total tree parsing failures:", num_tree_parsing_failures)
        return
    
    # Mode 2 - execute pre-built reasoning trees (concurrent, thread pool)
    if execute_trees_only:
        if input_trees_path is None:
            raise Exception("Error: 'execute_trees_only=True', but 'input_trees_path' is None")

        with open(input_trees_path, 'r', encoding='utf-8') as input_trees:
            tree_entries = [json.loads(line) for line in input_trees]

        # --resume: reuse predictions already written to output_predictions_path.
        # An index counts as "done" only if its stored prediction is non-empty,
        # so both missing and previously-failed (empty) predictions get rerun.
        predictions = []  # final list of prediction dicts (reused + freshly run)
        done = set()
        if resume and output_predictions_path and os.path.exists(output_predictions_path):
            done_pred = {}
            with open(output_predictions_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "index" in r:
                        done_pred[r["index"]] = r
            for idx, p in done_pred.items():
                if str(p.get("predicted") or "").strip():
                    done.add(idx)
                    predictions.append(p)
                    # fold the reused prediction into the metric totals
                    _em, _cover = calculate_em(p.get("predicted", ""), p.get("gold", []))
                    em += _em
                    cover_em += _cover
                    f1 += calculate_f1(p.get("predicted", ""), p.get("gold", []))
                    num_entries += 1
            print(f"[execute][resume] {len(done_pred)} predictions found in "
                  f"{output_predictions_path}; {len(done)} usable — will skip those")

        todo = [e for e in tree_entries if e.get("index") not in done]

        # Execute on a thread pool. run_query / the reasoner are thread-safe
        # (same model as the build stage). The metric accumulators below are
        # plain locals, so every mutation is guarded by `metrics_lock`.
        # NOTE: the module-level stat counters inside execute_postorder_q_list
        # (num_llm_calls, num_*_calls) are diagnostics only; under concurrency
        # they may be off by a few — EM / Cover-EM / F1 stay exact.
        metrics_lock = threading.Lock()

        # Incremental prediction sink: every finished entry is appended (under
        # metrics_lock, so concurrent writes are serialized) and flushed, so a
        # killed run keeps its progress on disk for --resume. The file is
        # re-sorted into index order at the very end.
        pred_fh = open(output_predictions_path, 'w', encoding='utf-8')
        for p in predictions:              # reused (resumed) predictions first
            pred_fh.write(json.dumps(p) + '\n')
        pred_fh.flush()

        def _execute_one(entry):
            nonlocal em, cover_em, f1, num_entries, num_tree_parsing_failures, num_execution_failures
            question = entry["question"].strip()
            question_index = entry["index"]
            gold_answers = entry["gold"]
            reasoning_tree = entry["reasoning_tree"]
            predicted_answer = ""  # default empty value

            _rec = get_recorder()
            _rec.set_idx(question_index)

            with _question_scope(question_index):
                try:
                    _, postorder_q_list, predicted_answer = run_query(
                        reasoner=reasoner, dataset_name=dataset_name, question=question,
                        q_index=question_index, gold=gold_answers,
                        build_tree_only=build_trees_only, execute_tree_only=execute_trees_only,
                        input_tree=reasoning_tree)
                except Exception as e:  # rare — usually a retrieval API error
                    print(f"\n!!! [Question execution failure] Question {question_index}: '{question}'")
                    print("Exception:", e)
                    print("Traceback:", traceback.format_exc())
                    print("Storing empty predicted_answer.")
                    with metrics_lock:
                        num_execution_failures += 1

            retrieval_counts = _rec.get_retrieval_counts()
            _rec.reset_idx()

            cur_em, cur_cover_em = calculate_em(predicted_answer, gold_answers)
            cur_f1 = calculate_f1(predicted_answer, gold_answers)
            print(f"index={question_index}  EM={bool(cur_em)}  CoverEM={bool(cur_cover_em)}  F1={cur_f1}")

            pred = {"index": question_index, "question": question,
                    "predicted": predicted_answer, "gold": gold_answers,
                    "retrieval_counts": retrieval_counts}
            with metrics_lock:
                if isinstance(reasoning_tree, str) and reasoning_tree == "TREE_PARSING_ERROR":
                    num_tree_parsing_failures += 1
                em += cur_em
                cover_em += cur_cover_em
                f1 += cur_f1
                num_entries += 1
                predictions.append(pred)
                pred_fh.write(json.dumps(pred) + '\n')   # incremental checkpoint
                pred_fh.flush()
                os.fsync(pred_fh.fileno())
                if num_entries % 10 == 0:
                    print_results(em, cover_em, f1, num_entries, num_tree_parsing_failures, num_execution_failures, s)

        try:
            if not todo:
                print("[execute] all predictions already present — nothing to execute")
            else:
                print(f"[execute] tree execution concurrency = {workers} threads "
                      f"({len(todo)}/{len(tree_entries)} to execute)")
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_execute_one, entry) for entry in todo]
                    for fut in as_completed(futures):
                        try:
                            fut.result()  # surface any unexpected (non-run_query) error
                        except Exception as e:
                            print(f"!!! [execute worker crashed] {e}")
        finally:
            pred_fh.close()

        # Rewrite predictions in index order (tidy, deterministic output).
        predictions.sort(key=lambda x: x["index"])
        with open(output_predictions_path, 'w', encoding='utf-8') as output_predictions:
            for p in predictions:
                output_predictions.write(json.dumps(p) + '\n')

        print_results(em, cover_em, f1, num_entries, num_tree_parsing_failures, num_execution_failures, s)
        return


def evaluate_dataset_single_source(dataset_name, dataset_path, output_trees_path, output_predictions_path, text_retriever_url, llm_cache_path="../openai_service/llm_cache/cache.jsonl", k=3):
    
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in {"hotpotqa", "2wikimultihop", "musique"}: 
        raise Exception(f"Unsupported single-source dataset: {dataset_name}. List of supported datasets: HotpotQA, 2WikiMultiHop, Musique.")
    
    knowledge_sources = {"Text"} 
    openai_caller = OpenAICaller(cache_path=llm_cache_path) 
    global_reasoner = GlobalReasoner(openai_caller=openai_caller, available_knowledge_sources=knowledge_sources, text_retriever_url=text_retriever_url, k=k)
    
    # Stage 1 - Atomic Reasoning Planning (Tree Generation)
    evaluate_dataset(reasoner=global_reasoner,
                     dataset_name=dataset_name,
                     dataset_path=dataset_path,
                     output_trees_path=output_trees_path, 
                     build_trees_only=True)
    
    # Stage 2 - Atomic Reasoning Execution (Tree Execution)
    evaluate_dataset(reasoner=global_reasoner,
                     dataset_name=dataset_name,
                     input_trees_path=output_trees_path, 
                     output_predictions_path=output_predictions_path, 
                     execute_trees_only=True)
    



# --kb 知识库 -> AtomR 的知识源集合。三者皆可开关；kg -> KG, table -> Table，
# doc -> Text（AtomR 的段落检索源）。
_KB_TO_ATOMR_SOURCE = {"kg": "KG", "table": "Table", "doc": "Text"}


def _parse_kb(raw):
    """把 --kb（逗号分隔的 {kg,table,doc} 子集）解析成 AtomR 的知识源集合。
    三者均为可选；doc 映射到 AtomR 的 Text 段落检索源。"""
    want = {tok.strip().lower() for tok in (raw or "").split(",") if tok.strip()}
    if not want:
        raise SystemExit("[main_mline] --kb 不能为空，至少需要 kg/table/doc 之一")
    bad = sorted(want - set(_KB_TO_ATOMR_SOURCE))
    if bad:
        raise SystemExit(
            f"[main_mline] unknown --kb source(s): {bad}  (allowed: kg, table, doc)"
        )
    return {_KB_TO_ATOMR_SOURCE[s] for s in want}


def evaluate_dataset_multi_source(
    dataset_name,
    dataset_path,
    output_trees_path,
    output_predictions_path,
    text_retriever_url,
    table_retriever_url,
    kg_query_language: str = "local",
    kopl_parser_url="https://viskop.xlore.cn/programApi",
    kopl_engine_url="https://viskop.xlore.cn/large",
    llm_cache_path="../../openai_service/llm_cache/cache.jsonl",
    k=3,
    run_mode="both",  # 新增：控制运行模式
    kb="kg",          # 知识库：逗号分隔的 {kg,table,doc} 子集，kg 永远在
    workers=None,     # build / execute 两阶段的并发线程数；None -> NUM_PROCESSES
    resume=False      # 断点续跑：跳过已建好的树 / 已出预测的题
):
    dataset_name = dataset_name.lower().strip()
    if dataset_name not in {"blendqa", "mmqa", "crag", "cmdbench"}:
        raise Exception(f"Unsupported multi-source dataset: {dataset_name}. List of supported datasets: BlendQA, CRAG, Cmdbench, MMQA.")

    # 知识库由 --kb 决定：kg/table/doc 三者均可开关。知识源选择只在这些
    # 活源里挑，没开的源对 GlobalReasoner 不可见。
    knowledge_sources = _parse_kb(kb)
    print(f"[kb={kb}] available knowledge sources = {sorted(knowledge_sources)}")
    openai_caller = OpenAICaller(cache_path=llm_cache_path)
    global_reasoner = GlobalReasoner(
        openai_caller=openai_caller,
        available_knowledge_sources=knowledge_sources,
        text_retriever_url=text_retriever_url,
        table_retriever_url=table_retriever_url,
        kg_query_language=kg_query_language,
        kopl_parser_url=kopl_parser_url,
        kopl_engine_url=kopl_engine_url,
        k=k,
    )
    
    # 阶段 1 - 仅生成树
    if run_mode in ["build", "both"]:
        print(f"\n=== Starting Stage 1: Tree Generation ({dataset_name}) ===")
        evaluate_dataset(reasoner=global_reasoner,
                         dataset_name=dataset_name,
                         dataset_path=dataset_path,
                         output_trees_path=output_trees_path,
                         build_trees_only=True,
                         workers=workers,
                         resume=resume)
    
    # 阶段 2 - 仅执行树
    if run_mode in ["execute", "both"]:
        print(f"\n=== Starting Stage 2: Tree Execution ({dataset_name}) ===")
        evaluate_dataset(reasoner=global_reasoner,
                         dataset_name=dataset_name,
                         input_trees_path=output_trees_path,
                         output_predictions_path=output_predictions_path,
                         execute_trees_only=True,
                         workers=workers,
                         resume=resume)


if __name__ == "__main__":
    import argparse

    # 提高多进程兼容性
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Run full dataset evaluation for multi-source reasoning.")
    
    # 数据集与路径配置
    parser.add_argument("--dataset-name", default="mmqa", help="Dataset name (e.g., mmqa, blendqa)")
    parser.add_argument("--dataset-path", required=True, help="Path to the input JSONL dataset")
    parser.add_argument("--output-trees-path", required=True, help="Path to save or load generated reasoning trees")
    parser.add_argument("--output-predictions-path", default="", help="Path to save final predictions (required if mode is execute or both)")
    
    # 运行模式
    parser.add_argument("--mode", choices=["build", "execute", "both"], default="both",
                        help="build: only generate trees; execute: only run execution; both: run full pipeline")
    parser.add_argument("--kb", default="kg",
                        help="知识库 = 活的知识源。逗号分隔的 {kg,table,doc} 子集，"
                             "三者均为可开关；doc 对应 AtomR 的 Text 段落检索源。"
                             "知识源选择只在这些活源里挑。"
                             "示例：'kg'（只有 KG，默认）、'kg,table'、'table,doc'、'kg,table,doc'。")

    # API 与环境配置
    # text-retriever-url 默认指向共享的 ColBERT 段落检索服务（doc 服务，端口 1215）。
    parser.add_argument("--text-retriever-url", default="http://127.0.0.1:1215/api/search")
    parser.add_argument("--table-retriever-url", default="http://127.0.0.1:1216/api/search")
    parser.add_argument("--kg-query-language", default="local")
    parser.add_argument("--llm-cache-path", default="../../openai_service/llm_cache/cache.jsonl")
    parser.add_argument("--k", type=int, default=3, help="Top-K retrieval limit")
    parser.add_argument("--trace-dir", default="",
                        help="按 idx 分目录写入轨迹 (LLM prompts/responses + 节点元数据)。也可读 ATOMR_TRACE_DIR。")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发线程数，作用于 Stage-1 树生成与 Stage-2 树执行两个阶段"
                             "（OpenAICaller 线程安全）。默认 8。")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：Stage-1 跳过 output-trees-path 里已建好的树"
                             "（TREE_PARSING_ERROR 的会重建）；Stage-2 跳过 "
                             "output-predictions-path 里已出的非空预测（空预测会重跑）。"
                             "树文件已完整时 build 阶段整段跳过、直接执行。")
    parser.add_argument("--max-failures", type=int, default=10, help="Max continuous execution failures before exiting")

    args = parser.parse_args()

    if args.mode in ["execute", "both"] and not args.output_predictions_path:
        parser.error("--output-predictions-path is required when --mode is 'execute' or 'both'")

    trace_dir = (args.trace_dir or os.environ.get("ATOMR_TRACE_DIR", "")).strip()
    if trace_dir:
        get_recorder().set_output_dir(os.path.abspath(trace_dir))
        print(f"[Trace] enabled, dir = {os.path.abspath(trace_dir)}")

    s = time.time()
    _reset_cost_agg()

    # 动态注入配置，确保上游调用的容错率可以被控制
    # 提示：你需要在 evaluate_dataset 中将 `if num_continuous_failed_executions >= 3:`
    # 修改为 `if num_continuous_failed_executions >= MAX_FAILURES:`
    global MAX_FAILURES
    MAX_FAILURES = args.max_failures

    os.makedirs(os.path.dirname(os.path.abspath(args.output_trees_path)), exist_ok=True)
    if args.output_predictions_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_predictions_path)), exist_ok=True)

    if args.resume:
        _seed_out_dir = os.path.dirname(os.path.abspath(args.output_predictions_path)) \
            if args.output_predictions_path else \
            os.path.dirname(os.path.abspath(args.output_trees_path))
        _seed_path = os.path.join(_seed_out_dir, "cost_summary.json")
        _seeded = _seed_cost_summary(_seed_path)
        if _seeded:
            print(f"[atomr] resume: seeded cost aggregator with {_seeded} prior entries from {_seed_path}")

    evaluate_dataset_multi_source(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        output_trees_path=args.output_trees_path,
        output_predictions_path=args.output_predictions_path,
        text_retriever_url=args.text_retriever_url,
        table_retriever_url=args.table_retriever_url,
        kg_query_language=args.kg_query_language,
        llm_cache_path=args.llm_cache_path,
        k=args.k,
        run_mode=args.mode,
        kb=args.kb,
        workers=args.workers,
        resume=args.resume,
    )
    
    print(f"\n[All Done] Total pipeline execution time: {time.time() - s:.2f} seconds.")

    # 成本汇总：把累计的 LLM/检索调用次数写到 out_dir/cost_summary.json
    _cost_out_dir = os.path.dirname(os.path.abspath(args.output_predictions_path)) \
        if args.output_predictions_path else \
        os.path.dirname(os.path.abspath(args.output_trees_path))
    _cost_path = os.path.join(_cost_out_dir, "cost_summary.json")
    _cost_summary = _dump_cost_summary(_cost_path)
    print(f"Cost summary-> {_cost_path}")
    print(f"[cost] {_format_cost_line(_cost_summary)}")