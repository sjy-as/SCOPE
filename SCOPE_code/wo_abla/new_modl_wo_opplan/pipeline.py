"""
Ablation: new_model decompose + semantic routing + DeepSieve RAG retrieval.

Combines:
  - new_model's query decomposition (sq.decompose_question → 2 sub-questions)
  - new_model's semantic graph routing (route.source_routing → primary + fallback)
  - DeepSieve's RAG retrieval + LLM direct-answering per sub-question
  - Fallback retrieval from secondary source when primary yields no answer
  - DeepSieve-style final answer synthesis from the sub-question answer chain

What this ablation removes (vs new_model):
  - Operator tree planning (no Search/Relate/Filter/Math steps)
  - Semantic parsing (entities / conditions / return-type extraction)
  - new_model's MultiSourceReasoner execution engine

What this ablation keeps (vs new_modl_wo_opplan):
  - The same decomposition + routing front-end
  - The same fallback-source retry logic
  But replaces the executor with a plain retrieve-then-answer loop.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------------------------
_THIS_DIR     = Path(__file__).resolve().parent
_NEW_MODEL    = Path("/root/autodl-tmp/new_model")
_DEEPSIEVE    = Path("/root/autodl-tmp/baseline/DeepSieve")

for _p in [str(_NEW_MODEL), str(_NEW_MODEL / "step2_decompose"), str(_DEEPSIEVE)]:
    if _p not in sys.path:
        sys.path.append(_p)

_s = str(_THIS_DIR)
if _s in sys.path:
    sys.path.remove(_s)
sys.path.insert(0, _s)

# ---------------------------------------------------------------------------
# Module-level constants (overridden by run.py before any pipeline call)
# ---------------------------------------------------------------------------
MERGED_FUSION_PATH = _NEW_MODEL / "step1_oag/fusion/output/semantic_list_3sources.json"
TABLE_API_URL = os.getenv("TABLE_API_URL", "http://127.0.0.1:1216/api/search")
DOC_API_URL   = os.getenv("DOC_API_URL",   "http://127.0.0.1:1215/api/search")
KG_API_URL    = os.getenv("KG_API_URL",    "http://127.0.0.1:8002/query")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "deepseek-chat")

ALL_SOURCES = ["kg", "table", "doc"]
K_RETRIEVE  = 5


# =====================================================================
# Stage 1 — semantic catalog
# =====================================================================

def _filter_catalog_by_sources(
    catalog: Dict[str, Any], enabled_sources: List[str],
) -> Dict[str, Any]:
    want = {str(s).strip().lower() for s in enabled_sources}
    filtered: Dict[str, Any] = {}
    for concept, info in catalog.items():
        if not isinstance(info, dict):
            continue
        sources = info.get("sources") or {}
        kept = {name: val for name, val in sources.items()
                if str(name).strip().lower() in want}
        if not kept:
            continue
        new_info = dict(info)
        new_info["sources"] = kept
        filtered[concept] = new_info
    return filtered


def stage1_check(enabled_sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load and filter the semantic catalog (identical to new_model)."""
    print("\n" + "=" * 70)
    print("STAGE 1: semantic catalog check")
    print("=" * 70)

    if not MERGED_FUSION_PATH.exists():
        raise FileNotFoundError(f"[Stage1] Missing: {MERGED_FUSION_PATH}")
    print(f"  Found: {MERGED_FUSION_PATH}")

    with MERGED_FUSION_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    if not isinstance(catalog, dict):
        raise ValueError(f"[Stage1] Expected dict, got {type(catalog).__name__}")

    if enabled_sources:
        before = len(catalog)
        catalog = _filter_catalog_by_sources(catalog, enabled_sources)
        print(f"  knowledge base   : {', '.join(enabled_sources)}")
        print(f"  catalog filtered : {before} -> {len(catalog)} concepts")

    from step2_decompose.semantic import build_catalog_lookup
    catalog_lookup = build_catalog_lookup(catalog)
    print(f"  semantic items   : {len(catalog)}")
    print(f"  concept lookup   : {len(catalog_lookup)}")

    return {
        "bundle": catalog,
        "catalog": catalog_lookup,
        "path": str(MERGED_FUSION_PATH),
    }


# =====================================================================
# Stage 2a — decompose + route
# (new_model: sq.decompose_question + route.source_routing)
# =====================================================================

def stage2a_decompose_route(
    question: str,
    fusion_bundle: Dict[str, Any],
    enabled_sources: Optional[List[str]] = None,
    profiles_path: Optional[str] = None,
    prompt_version: str = "v2",
) -> Dict[str, Any]:
    """Decompose question into 2 sub-questions and route each one.

    Uses new_model's decomposition and graph routing unchanged.
    Returns a dict with sub_q1/sub_q2 and route1/route2.
    """
    from step2_decompose import sq as sq_mod
    from step2_decompose import route as route_mod

    print("\n" + "=" * 70)
    print(f"STAGE 2a: decompose + route  (prompt_version={prompt_version})")
    print("=" * 70)

    # Decompose
    decomposition = sq_mod.decompose_question(
        question,
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )
    subqueries = decomposition.get("subqueries") or []
    sub_q1 = ((subqueries[0] or {}).get("query") if len(subqueries) > 0 else question) or question
    sub_q2 = ((subqueries[1] or {}).get("query") if len(subqueries) > 1 else "") or ""

    # Route each sub-question via new_model's semantic graph router
    def _route(sq_text: str) -> Dict[str, Any]:
        return route_mod.source_routing(
            matched_kind="none",
            matched_item=None,
            bundle=fusion_bundle,
            query=sq_text,
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            verbose=False,
            ref_type="entity",
            profiles_path=profiles_path,
            prompt_version=prompt_version,
            enabled_sources=enabled_sources,
        )

    route1 = _route(sub_q1)
    route2 = _route(sub_q2) if sub_q2 else {}

    print(f"  sub_q1 : {sub_q1}")
    print(f"  route1 : primary={route1.get('primary_source')}  fallback={route1.get('fallback_source')}")
    if sub_q2:
        print(f"  sub_q2 : {sub_q2}")
        print(f"  route2 : primary={route2.get('primary_source')}  fallback={route2.get('fallback_source')}")

    return {
        "sub_q1": sub_q1,
        "sub_q2": sub_q2,
        "route1": route1,
        "route2": route2,
        "decomposition": decomposition,
        "reasoning": decomposition.get("reasoning", ""),
    }


# =====================================================================
# Stage 3 — DeepSieve RAG retrieval + LLM answering
# =====================================================================

def _retrieve_and_answer(
    question: str,
    rag,
    k: int = K_RETRIEVE,
) -> Dict[str, Any]:
    """Retrieve top-k docs from a RAG backend, then ask the LLM for an answer.

    Returns a dict with keys: answer, reason, success, docs, doc_scores.
    `success` is 1 when the LLM is confident and 0 otherwise.
    """
    from utils.llm_call import call_openai_chat

    retrieved = rag.rag_qa(question, k=k)
    docs      = retrieved.get("docs", [])
    doc_scores = retrieved.get("doc_scores", [])

    prompt = (
        "Answer the following question based on the provided evidence documents.\n\n"
        "Please respond strictly in JSON format with the following fields:\n"
        '  "answer": the direct, concise answer (entity/value/fact). '
        'Leave empty ("") if not found.\n'
        '  "reason": brief explanation of how you arrived at this answer.\n'
        '  "success": 1 if confidently found, 0 otherwise.\n\n'
        "Format:\n"
        '{"answer": "...", "reason": "...", "success": 1}\n\n'
        "If the answer cannot be inferred, return:\n"
        '{"answer": "", "reason": "no relevant information found", "success": 0}\n\n'
        f"Question: {question}\n\n"
        "Evidence Documents:\n"
    )
    for doc in docs:
        prompt += f"- {doc}\n"
    prompt += "\nOnly output valid JSON. Do not add explanation or markdown fences."

    response = call_openai_chat(prompt, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL)

    answer, reason, success = "", "", 0
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        parsed  = json.loads(cleaned)
        answer  = str(parsed.get("answer", "")).strip()
        reason  = str(parsed.get("reason", "")).strip()
        success = int(parsed.get("success", 0))
    except Exception as e:
        print(f"  [rag-answer] JSON parse error: {e}")
        reason  = f"parse_error: {e}"
        success = 0

    return {
        "answer":     answer,
        "reason":     reason,
        "success":    success,
        "docs":       docs,
        "doc_scores": doc_scores,
        "metrics":    retrieved.get("metrics", {}),
    }


def _substitute_ref(text: str, prior_answers: Dict[str, str]) -> str:
    """Replace [1] tokens in text with the answer from sub-question 1.

    new_model's decomposition produces sub_q2 with [1] as a reference
    placeholder for the answer to sub_q1.
    """
    result = text
    sq1_ans = prior_answers.get("sq1", "")
    if sq1_ans:
        result = result.replace("[1]", sq1_ans)
    return result


def stage3_execute_deepsieve(
    sq_id: str,
    sq_text: str,
    route: Dict[str, Any],
    rag_sources: Dict[str, Any],
    prior_answers: Dict[str, str],
) -> Dict[str, Any]:
    """Execute one sub-query using DeepSieve RAG retrieval.

    1. Resolve any [1] reference in sq_text with prior sub-query answers.
    2. Retrieve + answer from the primary source.
    3. If no answer (success=0 or empty answer), retry with the fallback source.

    Returns a result dict compatible with DeepSieve's subquery_result format,
    with extra keys for fallback tracking.
    """
    actual_text = _substitute_ref(sq_text, prior_answers)
    primary  = (route.get("primary_source") or "").strip().lower()
    fallback = (route.get("fallback_source") or "").strip().lower()

    primary_attempt: Optional[Dict[str, Any]] = None
    fallback_attempt: Optional[Dict[str, Any]] = None
    used_source = primary

    # --- Primary attempt ---
    if primary and primary in rag_sources:
        print(f"\n  [{sq_id}] primary='{primary}'  query='{actual_text[:100]}'")
        primary_attempt = _retrieve_and_answer(actual_text, rag_sources[primary]["rag"])
        print(f"  [{sq_id}] primary answer='{primary_attempt['answer']}'  "
              f"success={primary_attempt['success']}")
    else:
        print(f"\n  [{sq_id}] primary source '{primary}' not in rag_sources, skipping")

    # --- Fallback attempt (when primary empty or low-confidence) ---
    primary_found = (
        primary_attempt is not None
        and primary_attempt.get("success", 0) == 1
        and primary_attempt.get("answer", "")
    )
    if not primary_found and fallback and fallback != primary and fallback in rag_sources:
        print(f"  [{sq_id}] primary empty/low-confidence → fallback='{fallback}'")
        fallback_attempt = _retrieve_and_answer(actual_text, rag_sources[fallback]["rag"])
        print(f"  [{sq_id}] fallback answer='{fallback_attempt['answer']}'  "
              f"success={fallback_attempt['success']}")
        if fallback_attempt.get("answer") or fallback_attempt.get("success", 0) == 1:
            used_source = fallback

    # Pick the result from the used source
    if used_source == fallback and fallback_attempt is not None:
        best = fallback_attempt
    elif primary_attempt is not None:
        best = primary_attempt
    else:
        best = {
            "answer": "", "reason": "no source available",
            "success": 0, "docs": [], "doc_scores": [], "metrics": {},
        }

    return {
        "sq_id":           sq_id,
        "original_query":  sq_text,
        "actual_query":    actual_text,
        "subquery_id":     sq_id,
        "answer":          best["answer"],
        "reason":          best["reason"],
        "success":         best["success"],
        "docs":            best["docs"],
        "doc_scores":      best["doc_scores"],
        "metrics":         best.get("metrics", {}),
        "routing":         used_source,
        "primary_source":  primary,
        "fallback_source": fallback,
        "used_fallback":   used_source == fallback,
        "primary_attempt": {
            "answer":  primary_attempt.get("answer", "") if primary_attempt else "",
            "success": primary_attempt.get("success", 0) if primary_attempt else 0,
        },
        "fallback_attempt": {
            "answer":  fallback_attempt.get("answer", "") if fallback_attempt else "",
            "success": fallback_attempt.get("success", 0) if fallback_attempt else 0,
        },
    }


# =====================================================================
# Stage 4 — final answer synthesis
# =====================================================================

def _get_fused_final_answer(
    original_question: str,
    subquery_results: List[Dict[str, Any]],
) -> Tuple[str, str]:
    """LLM synthesises the final answer from the ordered sub-query chain.

    Mirrors DeepSieve's get_fused_final_answer but uses this module's LLM
    settings so a single --api-key / --llm-url covers the whole pipeline.
    """
    from utils.llm_call import call_openai_chat

    prompt = (
        "You are a multi-hop reasoning assistant. Generate the final answer "
        "to a multi-hop question based on the following reasoning steps.\n\n"
        f"Original Question: {original_question}\n\n"
        "Subquestion Reasoning Steps:\n"
    )
    for r in subquery_results:
        prompt += f"{r['subquery_id']}: {r['actual_query']} → {r['answer']}\n"
        prompt += f"Reason: {r['reason']}\n\n"

    prompt += (
        "\nBased on the above reasoning steps, what is the final answer to the "
        "original question?\n\n"
        "Please respond in JSON format:\n"
        '{"answer": "final_answer", "reason": "final_reasoning"}\n'
        "Only output valid JSON. Do not add explanation or markdown fences."
    )

    response = call_openai_chat(prompt, LLM_API_KEY, LLM_MODEL, LLM_BASE_URL)
    answer, reason = "", ""
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed  = json.loads(cleaned.strip())
        answer  = parsed.get("answer", "").strip()
        reason  = parsed.get("reason", "").strip()
        print(f"  final_answer: {answer}")
    except Exception as e:
        print(f"  [fusion] parse error: {e}")
    return answer, reason


# =====================================================================
# Top-level: run one question
# =====================================================================

def build_rag_sources(enabled_sources: Optional[List[str]] = None) -> Dict[str, Any]:
    """Initialise DeepSieve RAG retrievers for every enabled source.

    Returns a dict: source_name → {"rag": <retriever>, "profile": <str>}.
    """
    from rag.initializer import (
        TableRetrieverRAG, DocRetrieverRAG, KGRetrieverRAG,
    )

    sources_to_build = [s for s in ALL_SOURCES
                        if not enabled_sources or s in enabled_sources]
    result: Dict[str, Any] = {}

    builders = {
        "table": lambda: TableRetrieverRAG(table_api_url=TABLE_API_URL),
        "doc":   lambda: DocRetrieverRAG(doc_api_url=DOC_API_URL),
        "kg":    lambda: KGRetrieverRAG(kg_api_url=KG_API_URL),
    }
    for name in sources_to_build:
        if name in builders:
            result[name] = {"rag": builders[name](), "profile": name}
            print(f"  RAG source initialized: {name}")
    return result


def run_pipeline_one(
    question: str,
    fusion_state: Dict[str, Any],
    rag_sources: Dict[str, Any],
    enabled_sources: Optional[List[str]] = None,
    profiles_path: Optional[str] = None,
    prompt_version: str = "v2",
) -> Dict[str, Any]:
    """Run the full pipeline for a single question.

    Returns a result dict with:
      question, sub_q1, sub_q2, route1, route2,
      sq_results ([sq1_result, sq2_result]),
      sq_answers {sq1: [...], sq2: [...]},
      final_answer, final_reason, decompose.
    """
    print("\n" + "#" * 70)
    print(f"PIPELINE: {question}")
    print("#" * 70)

    bundle = fusion_state["bundle"]

    # ---- Stage 2a: decompose + route ----
    s2a = stage2a_decompose_route(
        question=question,
        fusion_bundle=bundle,
        enabled_sources=enabled_sources,
        profiles_path=profiles_path,
        prompt_version=prompt_version,
    )

    sub_q1 = s2a["sub_q1"]
    sub_q2 = s2a["sub_q2"]
    route1 = s2a["route1"]
    route2 = s2a["route2"]

    # ---- Stage 3: DeepSieve RAG retrieval per sub-question ----
    print("\n" + "=" * 70)
    print("STAGE 3: DeepSieve RAG retrieval + LLM answering")
    print("=" * 70)

    prior_answers: Dict[str, str] = {}
    sq_results: List[Dict[str, Any]] = []

    # sq1
    print(f"\n--- sq1 ---")
    sq1_result = stage3_execute_deepsieve(
        sq_id="sq1",
        sq_text=sub_q1,
        route=route1,
        rag_sources=rag_sources,
        prior_answers=prior_answers,
    )
    sq_results.append(sq1_result)
    prior_answers["sq1"] = sq1_result["answer"]

    # sq2 (only when the decomposition produced a second sub-question)
    sq2_result: Optional[Dict[str, Any]] = None
    if sub_q2:
        print(f"\n--- sq2 ---")
        sq2_result = stage3_execute_deepsieve(
            sq_id="sq2",
            sq_text=sub_q2,
            route=route2,
            rag_sources=rag_sources,
            prior_answers=prior_answers,
        )
        sq_results.append(sq2_result)
        prior_answers["sq2"] = sq2_result["answer"]

    # ---- Stage 4: final answer synthesis ----
    print("\n" + "=" * 70)
    print("STAGE 4: final answer synthesis")
    print("=" * 70)

    final_answer, final_reason = _get_fused_final_answer(question, sq_results)

    return {
        "question":     question,
        "sub_q1":       sub_q1,
        "sub_q2":       sub_q2,
        "route1":       route1,
        "route2":       route2,
        "decompose":    s2a,
        "sq_results":   sq_results,
        "sq_answers": {
            "sq1": [sq1_result["answer"]] if sq1_result["answer"] else [],
            "sq2": [sq2_result["answer"]] if sq2_result and sq2_result["answer"] else [],
        },
        "final_answer": final_answer,
        "final_reason": final_reason,
    }
