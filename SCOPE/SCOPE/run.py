"""
Driver for pipeline.

Reads a jsonl of questions and, per row:
  1. Runs pipeline.run_pipeline_one
  2. Appends one summary line to <output_dir>/predictions.jsonl
  3. Writes one trace.json per sub-query under
       <output_dir>/traces/idx_<index>/<sq_id>.trace.json
  4. Writes the decompose-stage trace to
       <output_dir>/traces/idx_<index>/decompose.trace.json
  5. Writes the final-answer-synthesis trace to
       <output_dir>/traces/idx_<index>/final.trace.json

run:
cd SCOPE_code/SCOPE && python3 run.py \
  --input ../CMQA/qa_bench/table-doc-1120.jsonl \
  --gold ../CMQA/qa_bench/table-doc-1120.jsonl \
  --output-dir result/table_doc_demo \
  --routing-mode graph \
  --prompt-version v1 \
  --kb doc,table \
  --workers 24 \
  --llm-url "https://api.deepseek.com" \
  --api-key ""

--kb selects the live knowledge base (data sources). All of kg / table / doc
are opt-in; pass exactly the ones you want. Examples:
  --kb kg            KG only (default)
  --kb table         Table only
  --kb doc           Doc only
  --kb kg,table      KG + Table
  --kb kg,table,doc  KG + Table + Doc  (router picks per-hop among all three)
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline as P
import consolidation as C
from _common.cost_counter import question_scope, dump_summary, format_summary_line, reset_aggregator, seed_from_existing


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _build_gold_map(path: Optional[Path]) -> Dict[Tuple[Any, str], Dict[str, Any]]:
    """Key gold rows by (index, question) so datasets that reuse the same
    entity index across many question phrasings don't overwrite each other.
    Callers should look up with _gold_lookup(gold_map, idx, question)."""
    if not path or not path.exists():
        return {}
    out: Dict[Tuple[Any, str], Dict[str, Any]] = {}
    for r in _load_jsonl(path):
        idx = r.get("index")
        # Indices may be ints (kg-table) or strings like "Q137003" (kg-doc).
        if idx is None:
            continue
        q = (r.get("question") or "").strip()
        out[(idx, q)] = r
    return out


def _gold_lookup(
    gold_map: Dict[Tuple[Any, str], Dict[str, Any]],
    idx: Any,
    question: str,
) -> Optional[Dict[str, Any]]:
    if not gold_map:
        return None
    g = gold_map.get((idx, (question or "").strip()))
    if g is not None:
        return g
    # Fallback: legacy callers may pass an idx that only has one gold row.
    for (i, _q), row in gold_map.items():
        if i == idx:
            return row
    return None


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


def _uniq_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _normalize_source_name(name: str) -> str:
    s = (name or "").strip().lower()
    if s in {"kg", "knowledge graph", "knowledge_graph"}:
        return "KG"
    if s in {"table", "tabular"}:
        return "Table"
    return name.strip() or "Unknown"


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
            f"[run] unknown --kb source(s): {bad}  (allowed: kg, table, doc)"
        )
    if not want:
        raise SystemExit(
            "[run] --kb must enable at least one source from {kg, table, doc}"
        )
    return [s for s in _VALID_SOURCES if s in want]


_EMPTY_EVAL = {"sq1_exact": False, "sq1_partial": False,
               "sq2_exact": False, "sq2_partial": False,
               "final_exact": False, "final_partial": False}


def _compact_answers(value: Any, limit: int = 5) -> List[str]:
    vals = _as_list(value)
    return vals[:limit]


def _compact_sq_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    route = trace.get("route") or {}
    semantic = trace.get("semantic_parse") or {}
    execution = trace.get("execution") or {}
    primary = execution.get("primary_attempt") or {}
    fallback = execution.get("fallback_attempt") or {}
    return {
        "sq_id": trace.get("sq_id"),
        "text": trace.get("rewritten_text"),
        "route": {
            "matched_concept": route.get("matched_concept"),
            "primary_source": route.get("primary_source"),
            "fallback_source": route.get("fallback_source"),
            "reasoning": route.get("reasoning"),
        },
        "semantic_parse": {
            "Entities": semantic.get("Entities") or [],
            "Relation": semantic.get("Relation") or [],
            "Return Type": semantic.get("Return Type") or [],
            "Conditions": semantic.get("Conditions") or [],
            "need_math": semantic.get("need_math"),
        },
        "plan": trace.get("plan"),
        "primary_answers": _compact_answers(primary.get("answers")),
        "fallback_source": fallback.get("source") if isinstance(fallback, dict) else None,
        "fallback_answers": _compact_answers(fallback.get("answers") if isinstance(fallback, dict) else []),
        "status": trace.get("status"),
    }


def _build_consolidation_observation(
    question: str,
    pred: Dict[str, Any],
    gold: Optional[Dict[str, Any]],
    res: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    gold_q1 = _compact_answers((gold or {}).get("q1", {}).get("answer") or (gold or {}).get("q1", {}).get("answers"))
    gold_q2 = _compact_answers((gold or {}).get("q2", {}).get("answer") or (gold or {}).get("q2", {}).get("answers"))
    return {
        "index": pred.get("index"),
        "question": question,
        "pred": {
            "sq1": _compact_answers(pred.get("sq1")),
            "sq2": _compact_answers(pred.get("sq2")),
            "final": _compact_answers(pred.get("final")),
        },
        "gold": {"q1": gold_q1, "q2": gold_q2},
        "eval": pred.get("eval") or {},
        "error": pred.get("error"),
        "decompose": (res or {}).get("decompose") or {},
        "sq_traces": [_compact_sq_trace(t) for t in ((res or {}).get("sq_traces") or [])],
    }


def _consolidate_experience(
    records: List[Dict[str, Any]],
    llm_url: str,
    llm_model: str,
    api_key: str,
    max_items: int,
) -> Optional[Dict[str, Any]]:
    if not records or not C.enabled():
        return None
    current = C.snapshot().get("items") or []
    prompt = (
        "You are the Reflection & Consolidation module for a multi-source QA system.\n"
        "Analyze the recent execution records and update a compact set of reusable experience items.\n"
        "Focus on actionable guidance that can improve later source routing, semantic parsing, and operator planning.\n"
        "Prefer lessons like: when to choose another source immediately, when KG schema is too closed, "
        "how to parse [k] references, how to form table/doc search anchors, and when a fallback pattern indicates a routing mistake.\n\n"
        "Rules:\n"
        "- Keep only lessons supported by the records.\n"
        "- Do not memorize specific gold answers.\n"
        "- Do not add broad slogans; each item should say when it applies and what to do.\n"
        "- Merge with existing items, removing duplicates or weak items.\n"
        f"- Return at most {max_items} items.\n\n"
        "Allowed stage values: general, routing, semantic, planning, execution.\n"
        "Return strict JSON only with this schema:\n"
        "{\"items\": [{\"stage\": \"routing|semantic|planning|execution|general\", "
        "\"when\": \"condition\", \"action\": \"recommended behavior\", "
        "\"lesson\": \"short rationale\", \"evidence\": \"brief record ids/outcome\"}]}\n\n"
        f"Existing experience:\n{json.dumps(current, ensure_ascii=False, indent=2)}\n\n"
        f"Recent records:\n{json.dumps(records, ensure_ascii=False, indent=2)}"
    )
    try:
        raw = P._post_llm(prompt, llm_model, llm_url, api_key, max_tokens=10000)
        obj = P._parse_json_loose(raw) or {}
        items = obj.get("items") or []
        state = C.replace_items(items, source="reflection_consolidation_llm")
        print(f"[consolidation] updated revision={state.get('revision')} items={len(state.get('items') or [])}")
        return state
    except Exception as e:
        print(f"[consolidation] update failed: {e}")
        return None


# =====================================================================
# Concept description generation from fusion bundle
# =====================================================================

def _extract_concept_payload(concept_name: str, concept_obj: Dict[str, Any]) -> Dict[str, Any]:
    sources = concept_obj.get("sources") or {}
    payload_sources: Dict[str, Any] = {}
    for source_name, source_obj in sources.items():
        norm_source = _normalize_source_name(source_name)
        source_desc = (source_obj or {}).get("description") or ""
        samples = (source_obj or {}).get("sample") or []
        payload_sources[norm_source] = {
            "description": source_desc,
            "samples": samples,
        }
    return {
        "concept": concept_name,
        "concept_description": concept_obj.get("description") or "",
        "sources": payload_sources,
    }


def _build_concept_description_prompt(concept_payload: Dict[str, Any]) -> str:
    concept_name = concept_payload.get("concept", "")
    concept_description = concept_payload.get("concept_description", "")
    sources = concept_payload.get("sources") or {}
    source_names = ", ".join(sources.keys()) if sources else ""

    return (
        "You are generating a structured concept profile for downstream QA and retrieval.\n"
        "The overall concept description should be broad and source-agnostic.\n"
        "Then, for each source (KG or Table), derive a concise source-specific description\n"
        "by combining the source metadata, sample elements, and the concept meaning.\n\n"
        "Rules:\n"
        "1) The overall description should describe the concept in a broad way.\n"
        "2) For each source, write a source-specific description that reflects only that source's\n"
        "   schema/style and explicitly mentions the important elements/attributes.\n"
        "3) Also extract the key elements/attributes for that source. Use IDs for KG entities\n"
        "   when appropriate, and header fields for tables.\n"
        "4) Provide a few compact data examples from the given samples, but do not invent new facts.\n"
        "5) Keep the output concise, factual, and JSON-only.\n\n"
        f"Concept name: {concept_name}\n"
        f"Existing broad description: {concept_description}\n"
        f"Available sources: {source_names}\n\n"
        f"Source payload:\n{json.dumps(sources, ensure_ascii=False, indent=2)}\n\n"
        "Return strict JSON only with this schema:\n"
        "{\n"
        '  "concept": "...",\n'
        '  "overall_description": "...",\n'
        '  "sources": {\n'
        '    "KG": {"description": "...", "elements": ["..."], "data_examples": ["..."]},\n'
        '    "Table": {"description": "...", "elements": ["..."], "data_examples": ["..."]}\n'
        "  }\n"
        "}"
    )


def generate_concept_descriptions(
    fusion_path: Path,
    output_path: Path,
    llm_url: str,
    llm_model: str,
    api_key: str,
) -> Dict[str, Any]:
    with fusion_path.open("r", encoding="utf-8") as f:
        fusion = json.load(f)

    results: Dict[str, Any] = {}
    for concept_name, concept_obj in fusion.items():
        if not isinstance(concept_obj, dict):
            continue
        payload = _extract_concept_payload(concept_name, concept_obj)
        prompt = _build_concept_description_prompt(payload)
        raw = P._post_llm(prompt, llm_model, llm_url, api_key, max_tokens=10000)
        parsed = P._parse_json_loose(raw) or {}

        # Backfill safe defaults if the model omits something.
        sources_out: Dict[str, Any] = {}
        for source_name in ["KG", "Table"]:
            src_in = (parsed.get("sources") or {}).get(source_name) or {}
            src_payload = payload["sources"].get(source_name) or {}
            samples = src_payload.get("samples") or []
            elements = src_in.get("elements") or []
            if not elements:
                if source_name == "KG" and samples:
                    first = samples[0]
                    elements = [k for k in first.keys() if k not in {":START_ID(Player)", ":END_ID(Award)", "time"}]
                    if not elements:
                        elements = [":START_ID(Player)", ":END_ID(Award)", "time"]
                elif source_name == "Table" and samples:
                    first = samples[0]
                    elements = list((first.get("header") or [])[:])
            data_examples = src_in.get("data_examples") or []
            if not data_examples and samples:
                data_examples = samples[:3]
            sources_out[source_name] = {
                "description": src_in.get("description") or src_payload.get("description") or "",
                "elements": _uniq_keep_order([str(x) for x in elements]),
                "data_examples": data_examples,
            }

        results[concept_name] = {
            "concept": concept_name,
            "overall_description": parsed.get("overall_description") or concept_obj.get("description") or "",
            "sources": sources_out,
            "raw_llm_output": raw,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def _llm_judge_eval(
    question: str,
    items: List[Dict[str, Any]],
    llm_url: str,
    llm_model: str,
    api_key: str,
    max_attempts: int = 3,
) -> Dict[str, Dict[str, str]]:
    """One LLM call evaluates several (predicted, gold) pairs.

    items = [{"name": "sq1", "predicted": [...], "gold": [...]}, ...]
    Returns {"sq1": {"verdict": "exact|partial|miss", "reason": "..."}, ...}

    Retries up to `max_attempts` times if the call raises, returns unparseable
    content, or yields zero valid verdicts. Returns {} only if every attempt
    fails — callers should treat that as "unjudged", not "miss".
    """
    if not items:
        return {}
    expected_names = {it.get("name") for it in items if it.get("name") in {"sq1", "sq2", "final"}}
    prompt = (
        "You are a format-tolerant string-matching evaluator. For each item,\n"
        "compare Predicted to Gold ONLY. Treat Gold as the authoritative,\n"
        "ground-truth answer — even if it looks historically wrong, anachronistic,\n"
        "or contradicts your own world knowledge. DO NOT fact-check Gold against\n"
        "reality. DO NOT use the Question to second-guess Gold. Your sole job is\n"
        "to decide whether Predicted conveys the same information as Gold.\n\n"
        "Choose one verdict per item:\n"
        "  exact   : Predicted and Gold refer to the same entity / value. IGNORE\n"
        "            differences in case, articles (the/a), trailing punctuation,\n"
        "            plurals, surface paraphrasing, and sentence wrapping. If\n"
        "            Predicted is a full sentence that contains all of Gold's\n"
        "            information, count it as exact. If Predicted's string\n"
        "            equals (case-insensitively) any string in Gold, it is exact.\n"
        "  partial : Predicted covers some Gold elements but not all, OR Gold is\n"
        "            partially wrapped/restated.\n"
        "  miss    : empty, refusal ('does not contain', 'not listed', 'cannot\n"
        "            find', etc.), or a different entity/value than Gold.\n\n"
        "REMINDER: 'wrong according to real-world facts' is NOT a reason to mark\n"
        "miss. If Predicted matches Gold textually/semantically, it is exact even\n"
        "if you personally believe Gold is historically inaccurate.\n\n"
        f"Question (for context only, not for fact-checking): {question}\n\n"
        f"Items:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Output strict JSON only, no prose, no code fences:\n"
        "{\"results\": [{\"name\": \"sq1\", \"verdict\": \"exact|partial|miss\", \"reason\": \"...\"}, ...]}"
    )

    last_err: str = ""
    for attempt in range(1, max_attempts + 1):
        try:
            raw = P._post_llm(prompt, llm_model, llm_url, api_key, max_tokens=10000)
        except Exception as e:
            last_err = f"post failed: {e}"
            print(f"[eval] LLM judge attempt {attempt}/{max_attempts} {last_err}")
            time.sleep(min(2 ** attempt, 8))
            continue

        obj = P._parse_json_loose(raw) or {}
        results = obj.get("results") or []
        out: Dict[str, Dict[str, str]] = {}
        for r in results:
            name = r.get("name")
            if name in {"sq1", "sq2", "final"}:
                verdict = (r.get("verdict") or "miss").strip().lower()
                if verdict not in {"exact", "partial", "miss"}:
                    verdict = "miss"
                out[name] = {"verdict": verdict, "reason": r.get("reason", "")}

        if expected_names and expected_names.issubset(out.keys()):
            return out
        last_err = f"got verdicts for {sorted(out.keys())}, expected {sorted(expected_names)} (raw[:200]={raw[:200]!r})"
        print(f"[eval] LLM judge attempt {attempt}/{max_attempts} parse-incomplete: {last_err}")
        time.sleep(min(2 ** attempt, 8))

    print(f"[eval] LLM judge gave up after {max_attempts} attempts; last_err={last_err}")
    return {}


def _eval_one(
    question: str,
    pred: Dict[str, Any],
    gold: Optional[Dict[str, Any]],
    llm_url: str = "",
    llm_model: str = "",
    api_key: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """Returns (eval_flags, judge_details)."""
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
        flags[f"{name}_exact"] = (v == "exact")
        flags[f"{name}_partial"] = (v in {"exact", "partial"})
    return flags, judge


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline over a question jsonl")
    parser.add_argument("--input",  default="qa_bench/kg-table-26.jsonl")
    # Gold answers are inlined in the question file (q1.answer / q2.answers),
    # so default the gold path to the same file.
    parser.add_argument("--gold",   default="qa_bench/kg-table-26.jsonl")
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--max",    type=int, default=None)
    parser.add_argument("--start",  type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true",
        help="Append to existing predictions.jsonl and skip indexes already present.")
    parser.add_argument("--workers", type=int, default=8,
        help="Number of questions to process concurrently. 1 = serial (preserves old log layout).")
    parser.add_argument("--llm-url",
        default=os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1"),
        help="LLM base URL (used for step2 decomposition / semantic / planning)")
    parser.add_argument("--llm-model",
        default=os.getenv("LLM_MODEL", "deepseek-chat"),
        help="LLM model name for step2")
    parser.add_argument("--api-key",
        default=os.getenv("LLM_API_KEY", ""),
        help="LLM API key for step2")
    parser.add_argument("--exec-llm-url",
        default=os.getenv("EXEC_LLM_BASE_URL", ""),
        help="LLM base URL used inside step3 reasoner (LLMClient). "
             "If empty, falls back to the step3 module's hard-coded URL.")
    parser.add_argument("--exec-llm-model",
        default=os.getenv("EXEC_LLM_MODEL", ""),
        help="LLM model name used inside step3 reasoner. "
             "If empty, falls back to LLMClient default.")
    parser.add_argument("--exec-api-key",
        default=os.getenv("EXEC_LLM_API_KEY", ""),
        help="LLM API key used inside step3 reasoner. "
             "If empty, falls back to LLMClient default.")
    parser.add_argument("--routing-mode",
        default=os.getenv("ROUTING_MODE", "graph"),
        choices=["graph", "atomr", "deepsieve"],
        help="Routing ablation: 'graph' (event/semantic dual-layer, default), "
             "'atomr' (few-shot KG/Table/Text), "
             "'deepsieve' (source-profile table/kg).")
    parser.add_argument("--profiles-path",
        default=str((Path(__file__).resolve().parent / "data_sources/source_profiles.json")),
        help="JSON file with 'table' and 'kg' source profiles. "
             "Required when --routing-mode=deepsieve; also used by graph mode v2 prompts.")
    parser.add_argument("--prompt-version",
        default=os.getenv("PROMPT_VERSION", "v2"),
        choices=["v1", "v2"],
        help="Prompt version for routing_mode=graph. v1=legacy, v2=ref-type-aware + entity-attribute escape valve + profile context (default).")
    parser.add_argument("--kb",
        default=os.getenv("KB", "kg"),
        help="Knowledge base = the live data sources. Comma-separated subset of "
             "{kg,table,doc}; every source is opt-in. The router confines "
             "primary/fallback routing to these sources and the semantic "
             "catalog is filtered to them. Examples: 'kg' (KG only, default), "
             "'table', 'doc', 'kg,table', 'table,doc', 'kg,table,doc'.")
    parser.add_argument("--concept-fusion-path",
        default=str(Path(__file__).resolve().parent / "step1_oag/fusion/output/merged_fusion.json"),
        help="Input fusion JSON file containing concepts, overall descriptions, and source samples.")
    parser.add_argument("--concept-output-path",
        default=str(Path(__file__).resolve().parent / "step1_oag/fusion/output/concept_descriptions_llm.json"),
        help="Where to write the generated concept descriptions.")
    parser.add_argument("--generate-concept-descriptions", action="store_true",
        help="Generate concept descriptions from the fusion file and exit.")
    parser.add_argument("--no-decompose", action="store_true",
        help="Ablation: skip question decomposition; treat the full question as a single sub-query.")
    parser.add_argument("--no-fallback", action="store_true",
        help="Ablation: disable fallback-source retry when primary source returns empty.")
    parser.add_argument("--use-internal-knowledge-fallback", action="store_true",
        help="If the synthesized final answer is empty (either because a sub-query "
             "halted early, or because the Stage-4 synthesis returned nothing), "
             "ask the LLM to answer from its own internal knowledge instead of "
             "leaving the final answer blank. Default: OFF.")
    parser.add_argument(
        "--inject-plan-semlist-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to inject semantic-list matched-concept metadata into the operator-tree planning prompt.",
    )
    parser.add_argument(
        "--consolidation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Reflection & Consolidation: periodically summarize recent traces into reusable prompt guidance.",
    )
    parser.add_argument("--consolidation-interval", type=int, default=50,
        help="Number of completed questions between consolidation updates. Use workers=1 for strict sequential adaptation.")
    parser.add_argument("--consolidation-path", default="",
        help="Where to store learned experience JSON. Default: <output-dir>/consolidation/experience.json")
    parser.add_argument("--consolidation-max-items", type=int, default=24,
        help="Maximum learned guidance items kept in the prompt artifact.")
    parser.add_argument("--reset-consolidation", action="store_true",
        help="Start with an empty consolidation artifact even if the path already exists.")
    args = parser.parse_args()

    enabled_sources = _parse_kb(args.kb)

    if not args.api_key:
        raise SystemExit(
            "[run_v2] --api-key (or LLM_API_KEY env) must be provided "
            "for step2 LLM calls."
        )

    if args.generate_concept_descriptions:
        out = generate_concept_descriptions(
            fusion_path=Path(args.concept_fusion_path),
            output_path=Path(args.concept_output_path),
            llm_url=args.llm_url,
            llm_model=args.llm_model,
            api_key=args.api_key,
        )
        print(f"[run_v2] generated concept descriptions -> {args.concept_output_path} ({len(out)} concepts)")
        return

    # Apply overrides to the pipeline_v2 module-level constants so the new
    # defaults take effect for every helper inside the pipeline.
    P.LLM_BASE_URL = args.llm_url
    P.LLM_MODEL    = args.llm_model
    P.LLM_API_KEY  = args.api_key
    # If --exec-* not supplied, route step3 LLM calls to the same endpoint
    # so one --llm-url / --api-key works for the whole pipeline.
    P.EXEC_LLM_BASE_URL = args.exec_llm_url or args.llm_url
    P.EXEC_LLM_MODEL    = args.exec_llm_model or args.llm_model
    P.EXEC_LLM_API_KEY  = args.exec_api_key or args.api_key
    print(f"[run_v2] step2 LLM    : model={P.LLM_MODEL}  url={P.LLM_BASE_URL}")
    print(f"[run_v2] step3 LLM    : model={P.EXEC_LLM_MODEL}  url={P.EXEC_LLM_BASE_URL}")

    root = Path(__file__).resolve().parent
    in_path  = (root / args.input).resolve()  if not Path(args.input).is_absolute()  else Path(args.input)
    gold_path = (root / args.gold).resolve()  if not Path(args.gold).is_absolute()   else Path(args.gold)
    out_dir   = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    consolidation_path = Path(args.consolidation_path) if args.consolidation_path else (out_dir / "consolidation" / "experience.json")
    C.configure(
        enabled=args.consolidation,
        path=str(consolidation_path),
        max_items=args.consolidation_max_items,
        reset=args.reset_consolidation,
    )
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    pred_path    = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"

    questions = _load_jsonl(in_path)
    if args.start:
        questions = questions[args.start:]
    if args.max is not None:
        questions = questions[: args.max]

    gold_map = _build_gold_map(gold_path)

    if args.routing_mode == "deepsieve" and not Path(args.profiles_path).exists():
        raise SystemExit(
            f"[run_v2] --routing-mode=deepsieve requires --profiles-path "
            f"to exist (got {args.profiles_path})."
        )

    print("=" * 70)
    print(f"Input        : {in_path}")
    print(f"Gold         : {gold_path}  ({len(gold_map)} entries, "
          f"{len({k[0] for k in gold_map})} unique indices)")
    print(f"Output dir   : {out_dir}")
    print(f"Predictions  : {pred_path}")
    print(f"Traces dir   : {traces_dir}")
    print(f"Questions    : {len(questions)}")
    print(f"Routing mode : {args.routing_mode}"
          + (f"  (profiles={args.profiles_path})"
             if args.routing_mode in {"deepsieve", "graph"} else ""))
    print(f"Knowledge base: {', '.join(enabled_sources)}")
    print(f"Prompt ver.  : {args.prompt_version}"
          + ("  (only affects graph mode)" if args.routing_mode != "graph" else ""))
    print(f"Plan SemList : {args.inject_plan_semlist_metadata}")
    print(f"Consolidation: {args.consolidation}  interval={args.consolidation_interval}  path={consolidation_path}")
    print(f"Internal-KB fallback: {args.use_internal_knowledge_fallback} (when final answer is empty)")
    print("=" * 70)

    # The catalog is filtered to the live knowledge base, so routing and the
    # semantic catalog only ever see content for the enabled sources.
    fusion_state = P.stage1_check(enabled_sources)

    print("\n" + "=" * 70)
    print("Initializing execution engine...")
    print("=" * 70)
    reasoner = P.build_executor(enabled_sources=enabled_sources)

    P.TRACER.install()

    eval_rows: List[Dict[str, Any]] = []
    ok = 0

    done_idx: set = set()
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
        print(f"[run_v2] resume: {len(done_idx)} indexes already in {pred_path.name}, will skip them")

    # Build the work list (skip already-done indexes and empty questions).
    work: List[Tuple[int, int, str, Optional[Dict[str, Any]]]] = []
    for i, rec in enumerate(questions, start=1):
        idx = rec.get("index", i - 1)
        question = (rec.get("question") or "").strip()
        if not question:
            continue
        if idx in done_idx:
            print(f"[{i}/{len(questions)}] index={idx} -- already done, skipping")
            continue
        gold = _gold_lookup(gold_map, idx, question)
        work.append((i, idx, question, gold))

    reset_aggregator()

    if args.resume:
        _seed_path = out_dir / "cost_summary.json"
        _seeded = seed_from_existing(_seed_path)
        if _seeded:
            print(f"[run_v2] resume: seeded cost aggregator with {_seeded} prior entries from {_seed_path}")

    def process_one(
        i: int, idx: int, question: str, gold: Optional[Dict[str, Any]],
    ) -> Tuple[int, int, Dict[str, Any], bool, Dict[str, Any]]:
        row_trace_dir = traces_dir / f"idx_{idx}"
        row_trace_dir.mkdir(parents=True, exist_ok=True)
        with question_scope(idx):
            try:
                res = P.run_pipeline_one(
                    question=question,
                    fusion_state=fusion_state,
                    reasoner=reasoner,
                    ground_truth=gold,
                    routing_mode=args.routing_mode,
                    profiles_path=args.profiles_path,
                    prompt_version=args.prompt_version,
                    enabled_sources=enabled_sources,
                    no_decompose=args.no_decompose,
                    no_fallback=args.no_fallback,
                    include_plan_semlist_metadata=args.inject_plan_semlist_metadata,
                    use_internal_knowledge_fallback=args.use_internal_knowledge_fallback,
                )
                for sq_trace in res.get("sq_traces", []):
                    sq_id = sq_trace.get("sq_id") or "sq_unknown"
                    _write_json(row_trace_dir / f"{sq_id}.trace.json", sq_trace)
                _write_json(row_trace_dir / "decompose.trace.json", res.get("decompose", {}))
                _write_json(row_trace_dir / "final.trace.json", {
                    "question": question,
                    "final_answer": res.get("final_answer"),
                    "llm_calls": res.get("final_synthesis_llm_calls", []),
                })
                pred = {
                    "index": idx,
                    "question": question,
                    "sq1": res["sq_answers"].get("sq1", []),
                    "sq2": res["sq_answers"].get("sq2", []),
                    "final": [res["final_answer"]] if res.get("final_answer") else [],
                    "error": None,
                }
                flags, judge = _eval_one(
                    question=question, pred=pred, gold=gold,
                    llm_url=args.llm_url, llm_model=args.llm_model, api_key=args.api_key,
                )
                pred["eval"] = flags
                pred["eval_judge"] = judge
                obs = _build_consolidation_observation(question, pred, gold, res)
                return (i, idx, pred, True, obs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                pred = {
                    "index": idx,
                    "question": question,
                    "sq1": [],
                    "sq2": [],
                    "final": [],
                    "error": str(e),
                    "eval": dict(_EMPTY_EVAL),
                    "eval_judge": {},
                }
                obs = _build_consolidation_observation(question, pred, gold, None)
                return (i, idx, pred, False, obs)

    write_lock = threading.Lock()
    total = len(work)
    completed = 0
    workers = max(1, int(args.workers or 1))

    recent_consolidation_records: List[Dict[str, Any]] = []
    open_mode = "a" if args.resume else "w"
    with pred_path.open(open_mode, encoding="utf-8") as wf:
        def _emit(i: int, idx: int, pred: Dict[str, Any], success: bool, obs: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
            nonlocal completed, ok
            batch = None
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
                if args.consolidation:
                    recent_consolidation_records.append(obs)
                    interval = max(1, int(args.consolidation_interval or 50))
                    if len(recent_consolidation_records) >= interval:
                        batch = list(recent_consolidation_records)
                        recent_consolidation_records.clear()
            return batch

        if workers > 1:
            print(f"\n[run_v2] running with workers={workers} "
                  f"(results written in completion order; rows carry an `index` field)")
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(process_one, *w) for w in work]
                for fut in as_completed(futs):
                    i, idx, pred, success, obs = fut.result()
                    batch = _emit(i, idx, pred, success, obs)
                    if batch:
                        _consolidate_experience(batch, args.llm_url, args.llm_model, args.api_key, args.consolidation_max_items)
        else:
            for w in work:
                print("\n" + "#" * 70)
                print(f"[{w[0]}/{len(questions)}] index={w[1]}")
                print("#" * 70)
                i, idx, pred, success, obs = process_one(*w)
                batch = _emit(i, idx, pred, success, obs)
                if batch:
                    _consolidate_experience(batch, args.llm_url, args.llm_model, args.api_key, args.consolidation_max_items)
                if (not success) and (not args.continue_on_error):
                    raise RuntimeError(f"pipeline failed at index={idx}: {pred.get('error')}")

    if args.consolidation and recent_consolidation_records:
        print(f"[consolidation] final consolidation update for {len(recent_consolidation_records)} leftover record(s)")
        _consolidate_experience(recent_consolidation_records, args.llm_url, args.llm_model, args.api_key, args.consolidation_max_items)
        recent_consolidation_records.clear()

    n = len(eval_rows)
    summary = {
        "input": str(in_path),
        "gold": str(gold_path),
        "output_dir": str(out_dir),
        "knowledge_base": enabled_sources,
        "routing_mode": args.routing_mode,
        "prompt_version": args.prompt_version,
        "inject_plan_semlist_metadata": args.inject_plan_semlist_metadata,
        "consolidation_enabled": args.consolidation,
        "consolidation_interval": args.consolidation_interval,
        "consolidation_path": str(consolidation_path),
        "consolidation_revision": C.snapshot().get("revision"),
        "profiles_path": args.profiles_path if args.routing_mode in {"deepsieve", "graph"} else None,
        "total": n,
        "succeeded": ok,
        "failed": n - ok,
    }
    for k in ("sq1_exact", "sq1_partial", "sq2_exact", "sq2_partial",
              "final_exact", "final_partial"):
        summary[k] = sum(1 for r in eval_rows if r.get(k))
    _write_json(summary_path, summary)

    cost_path = out_dir / "cost_summary.json"
    cost_summary = dump_summary(cost_path)
    print("\n" + "=" * 70)
    print("Run finished.")
    print(f"Processed : {summary['total']}")
    print(f"Succeeded : {summary['succeeded']}")
    print(f"Failed    : {summary['failed']}")
    print(f"Predictions  -> {pred_path}")
    print(f"Per-sq traces-> {traces_dir}/idx_<index>/*.trace.json")
    print(f"Summary      -> {summary_path}")
    print(f"Cost summary -> {cost_path}")
    print(f"[cost] {format_summary_line(cost_summary)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
