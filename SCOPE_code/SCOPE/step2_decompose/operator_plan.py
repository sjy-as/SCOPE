"""
Operator-tree planner — LLM-driven and route-aware.

Input  : jsonl produced by semantic.py.
Output : same rows + op_plans (one plan per sub-query).

Build an operator tree from:
  - routed source information from semantic.py
  - semantic_parse output
  - matched concept context from semantic_list.json
  - a strict prompt that guides the LLM to emit the plan JSON
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from semantic import load_catalog_lookup, _sample_section_titles


_NUM_REF_RE = re.compile(r"\[(\d+)\]")


def _extract_refs(text: str) -> List[int]:
    if not text:
        return []
    return sorted({int(m.group(1)) for m in _NUM_REF_RE.finditer(text)})


def _to_ref_form(text: str) -> str:
    """Convert every [k] placeholder in *text* into ${ref_k} form."""
    return _NUM_REF_RE.sub(lambda m: "${ref_" + m.group(1) + "}", str(text or ""))


def _is_pure_ref(entity: str) -> bool:
    """True when *entity* is exactly a [k] / ${ref_k} placeholder."""
    e = str(entity or "").strip()
    return bool(re.fullmatch(r"\[\d+\]", e) or re.fullmatch(r"\$\{ref_\d+\}", e))


def _entities_to_anchor(entities: List[str]) -> str:
    """Join semantic_parse.Entities into a Search anchor, [k] -> ${ref_k}."""
    parts = []
    for e in entities or []:
        e = _to_ref_form(str(e).strip())
        if e:
            parts.append(e)
    return " ".join(parts).strip()


def _repair_search_entities(
    steps: List[Dict[str, Any]],
    semantic: Dict[str, Any],
    plan_type: str = "single_path",
    sq_id: str = "",
) -> None:
    """Re-ground Search.entity_name to semantic_parse.Entities.

    The planner LLM sometimes drops the parsed entity and puts a relation /
    abstract concept (e.g. 'Winner selection process') or an outright
    hallucination (e.g. 'Nobel Prize') into Search.entity_name. The entity to
    search MUST come from semantic_parse.Entities; here we deterministically
    repair any Search step whose entity_name is not grounded in them.

    Map plans iterate ${item} and are left untouched.
    """
    if plan_type == "map":
        return
    entities = semantic.get("Entities") if isinstance(semantic, dict) else None
    if not isinstance(entities, list):
        return
    entities = [str(e).strip() for e in entities if str(e).strip()]
    if not entities:
        return  # nothing to ground against

    anchor = _entities_to_anchor(entities)
    if not anchor:
        return

    concrete = [e.lower() for e in entities if not _is_pure_ref(e)]
    has_ref = any(_is_pure_ref(e) for e in entities)

    for st in steps:
        if not isinstance(st, dict) or st.get("op") != "Search":
            continue
        name = str(st.get("entity_name") or "").strip()
        norm = _to_ref_form(name).lower()
        if not norm:
            grounded = False
        elif "${item}" in norm:
            grounded = True  # map iteration variable
        elif has_ref and re.search(r"\$\{ref_\d+\}", norm):
            grounded = True  # uses an upstream reference
        elif concrete and any(c in norm or norm in c for c in concrete):
            grounded = True  # overlaps a parsed concrete entity
        else:
            grounded = False

        if not grounded:
            print(
                f"  [repair] {sq_id} Search.entity_name {name!r} not grounded in "
                f"semantic_parse.Entities {entities} -> {anchor!r}"
            )
            st["entity_name"] = anchor


def _session_with_retry() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504, 567],
        allowed_methods=["POST"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _call_llm(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 10000,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    resp = _session_with_retry().post(
        url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
        proxies={"http": None, "https": None},
    )
    if resp.status_code >= 400:
        print(f"[!!! API HTTP {resp.status_code}] body={resp.text[:400]}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _strip_llm_wrappers(raw: str) -> Optional[str]:
    s = (raw or "").strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    return s


def _extract_balanced_json_object(raw: str) -> Optional[str]:
    s = _strip_llm_wrappers(raw)
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    s = _strip_llm_wrappers(raw)
    candidates = [s]
    balanced = _extract_balanced_json_object(s)
    if balanced and balanced not in candidates:
        candidates.append(balanced)
    a = s.find("{")
    b = s.rfind("}")
    if a >= 0 and b > a:
        span = s[a : b + 1]
        if span not in candidates:
            candidates.append(span)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _repair_plan_json(
    raw: str,
    model: str,
    base_url: str,
    api_key: str,
) -> Optional[Dict[str, Any]]:
    repair_prompt = (
        "Convert the following model output into one strict JSON object only.\n"
        "Rules:\n"
        "- Output raw JSON only.\n"
        "- No markdown fences.\n"
        "- No explanation.\n"
        "- Preserve the original content if possible.\n"
        "- If the content is truncated or incomplete, return {}.\n\n"
        f"Model output:\n{raw}"
    )
    repaired = _call_llm(
        repair_prompt,
        model,
        base_url,
        api_key,
        temperature=0.0,
        max_tokens=10000,
    )
    return _parse_json(repaired)


# =====================================================================
# Prompt
# =====================================================================

_KG_SEARCH_RULE = (
    "Primary data source = kg.\n"
    "  - Search.entity_name MUST be the *relation-emitting entity* — i.e., the known complete entity,\n"
    "    the entity that performs the relation. Pick it from semantic_parse.Entities.\n"
    "    It is NOT the entity corresponding to the answer's Return Type.\n"
    "  - If the only concrete entity is a [k] reference, use ${ref_k}.\n"
    "  - Time tokens (years, season names) MUST NOT appear as Search.entity_name."
)

_TABLE_SEARCH_RULE = (
    "Primary data source = table.\n"
    "  - Search.entity_name MUST be an 'event-entity' composite name.\n"
    "  - Content: Build it ONLY from items in semantic_parse.Entities (with [k] converted to ${ref_k}).\n"
    "  - Pattern: Refer to the way items in 'page_title + section_titles' (see the semantic concept block above) are written,\n"
    "    and arrange/join these entities following a similar structure.\n"
    "    'page_title + section_titles' is only a *naming template*; do not copy its literal words directly.\n"
    "    The key point is: how to select entities, which fields to take, and how types are organized.\n"
    "  - When the sub-query contains [k], you need to determine whether ${ref_k} should appear in the composite name.\n"
    "    (These entity names should belong to Relate.entity or filter conditions.)"
)

_DOC_SEARCH_RULE = (
    "Primary data source = doc.\n"
    "  - Search.entity_name MUST be a concrete named entity taken from semantic_parse.Entities.\n"
    "    It is the subject the passage is about — NOT a relation, NOT the Return Type, and\n"
    "    NOT an abstract concept (e.g. 'winner selection process', 'selection criteria').\n"
    "  - If semantic_parse.Entities contains ONLY [k] reference(s), Search.entity_name MUST be\n"
    "    exactly the corresponding ${ref_k}. Do NOT invent a passage title in its place.\n"
    "  - NEVER output a named entity that is absent from semantic_parse.Entities and the\n"
    "    sub-query text. Do not guess or fill in a plausible-sounding award/person/event name.\n"
    "  - The matched concept's title / section_title examples are ONLY a loose phrasing hint;\n"
    "    they MUST NOT introduce entity words that are not in semantic_parse.Entities.\n"
    "  - Convert every [k] to ${ref_k}.\n"
    "  - These entity names should usually feed Relate.entity or passage-level filters downstream."
)



_PLAN_SHAPES = (
    "Plan shapes (pick the one that fits the question):\n"
    "  (1) single_path : Search -> Relate -> [Filter pre] -> [Math]\n"
    "                    -> [Filter post]\n"
    "  (2) map         : the question contains 'for each [k]'.\n"
    "                    plan_type='map', map_over='ref_k'.\n"
    "                    Inside the body, Search.entity_name and\n"
    "                    Relate.entity MUST be ${item}.\n"
    "  (3) intersect   : the question asks for X who satisfies BOTH A AND\n"
    "                    B (literal 'both ... and ...'). Emit two parallel\n"
    "                    chains:\n"
    "                       Search(A) -> Relate -> r1\n"
    "                       Search(B) -> Relate -> r2\n"
    "                       Filter(stage='pre',\n"
    "                              condition_struct={\"type\":\"intersect\",\n"
    "                                                \"refs\":[\"r1\",\"r2\"]})\n"
    "                              -> f_pre\n"
    "                    plan_type stays 'single_path'; final_ref='f_pre'."
)

_FIELD_RULES = (
    "Field rules:\n"
    "  - Search.entity_name  : a real named thing taken from semantic_parse.Entities\n"
    "    (convert every [k] to ${ref_k}). NEVER a step ref, the Return Type, the\n"
    "    matched_concept, a relation/concept, or an entity invented out of thin air.\n"
    "    If the only parsed entity is [k], entity_name MUST be exactly ${ref_k}.\n"
    "  - Relate.entity       : a step output ref (s1, r1, ...) or ${item}.\n"
    "  - Relate.relation     : a short verb phrase predicate. Take it from\n"
    "    semantic_parse.Relation if available. MUST NOT equal matched_concept.\n"
    "  - Relate.target_entity: the answer's Return Type. NOT a step ref,\n"
    "    NOT a proper-noun entity, NOT a Search.entity_name.\n"
    "  - Filter               : Conditions from semantic_parse.\n"
    "    stage='pre' filters raw rows; stage='post' filters Math output.\n"
    "  - Math                 : ONLY emit when semantic_parse.need_math is\n"
    "    true. operation in {count, max, min, aggregate, group_count}.\n"
    "  - [k] refs             : convert every [k] to ${ref_k}."
)


def _source_scoped_context(
    matched_info: Optional[Dict[str, Any]],
    source: str,
) -> Dict[str, Any]:
    """Project the matched-concept context onto the routed source only.

    The planner only needs the concept's schema on the source it is about to
    plan against; dumping every source's block dilutes the prompt.
    """
    if not matched_info:
        return {}
    src = (source or "kg").strip().lower()
    ctx: Dict[str, Any] = {
        "matched_concept": matched_info.get("canonical_name"),
        "overall_description": matched_info.get("description"),
    }
    if src == "table":
        ctx["Table"] = {
            "description": matched_info.get("table_description"),
            "elements": matched_info.get("table_elements") or [],
            "data_examples": matched_info.get("table_sample") or [],
        }
    elif src == "doc":
        ctx["Doc"] = {
            "description": matched_info.get("doc_description"),
            "elements": matched_info.get("doc_elements") or [],
            "data_examples": matched_info.get("doc_sample") or [],
        }
    else:
        ctx["KG"] = {
            "description": matched_info.get("kg_description"),
            "elements": matched_info.get("kg_elements") or [],
            "data_examples": matched_info.get("kg_sample") or [],
        }
    return ctx


def _build_prompt(
    sq: Dict[str, Any],
    ref_bindings: Dict[str, Any],
    matched_info: Optional[Dict[str, Any]],
    include_semlist_metadata: bool = True,
) -> str:
    text = (sq.get("text") or "").strip()
    execution = sq.get("execution") if isinstance(sq.get("execution"), dict) else {}
    semantic = sq.get("semantic_parse") if isinstance(sq.get("semantic_parse"), dict) else {}
    source = (execution.get("primary_source") or "kg").strip().lower()
    if source == "table":
        rule = _TABLE_SEARCH_RULE
    elif source == "doc":
        rule = _DOC_SEARCH_RULE
    else:
        rule = _KG_SEARCH_RULE
    titles = []
    if matched_info and source == "table":
        titles = _sample_section_titles(matched_info.get("table_sample") or [])
    elif matched_info and source == "doc":
        titles = _sample_section_titles(matched_info.get("doc_sample") or [])
    scoped_context = (
        _source_scoped_context(matched_info, source)
        if include_semlist_metadata
        else {}
    )
    matched_context_label = (
        "Matched concept context"
        if include_semlist_metadata
        else "Matched concept context (disabled by config)"
    )
    payload = {
        "subquery": text,
        "execution": execution,
        "semantic_parse": semantic,
        "matched_concept": matched_info.get("canonical_name") if matched_info else None,
        "matched_context": scoped_context,
        "ref_bindings": ref_bindings,
        "numeric_refs_in_text": _extract_refs(text),
        "table_section_titles_or_columns": titles,
    }
    return (
        "You are an operator-tree planner. Produce ONE execution plan using only "
        "Search, Relate, Filter, Math.\n\n"
        f"{matched_context_label}:\n{json.dumps(scoped_context, ensure_ascii=False)}\n\n"
        f"{rule}\n\n"
        f"{_FIELD_RULES}\n\n"
        "Output STRICT JSON with this schema and no prose:\n"
        "{\n"
        '  "plan_type": "single_path" | "map",\n'
        '  "map_over": "ref_k" | null,\n'
        '  "steps": [\n'
        '    {"op":"Search","entity_name":"...","descriptors":"","route":{},"out":"s1"},\n'
        '    {"op":"Relate","entity":"s1","relation":"...","target_entity":"...","in":"s1","out":"r1"},\n'
        '    {"op":"Filter","stage":"pre","entities":{"ref":"r1"},"condition":"...","condition_struct":{},"out":"f_pre"},\n'
        '    {"op":"Math","data":{"ref":"f_pre"},"operation":"count|max|min|aggregate|group_count","group_by":"...","out":"m1"},\n'
        '    {"op":"Filter","stage":"post","entities":{"ref":"m1"},"condition":"...","condition_struct":{},"out":"f_post"}\n'
        "  ],\n"
        '  "final_ref": "r1|f_pre|m1|f_post"\n'
        "}\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


_VALID_OPS = {"Search", "Relate", "Filter", "Math"}


def _validate(plan: Dict[str, Any]) -> bool:
    if not isinstance(plan, dict):
        return False
    if plan.get("plan_type") not in {"single_path", "map"}:
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    if not isinstance(plan.get("final_ref"), str) or not plan["final_ref"]:
        return False
    for st in steps:
        if not isinstance(st, dict) or st.get("op") not in _VALID_OPS:
            return False
    if plan["plan_type"] == "map" and not isinstance(plan.get("map_over"), str):
        return False
    return True


def _attach_route(steps: List[Dict[str, Any]], source: str, fallback: str, event_type: str) -> List[Dict[str, Any]]:
    route = {"primary_source": source, "fallback_source": fallback or None, "event_type": event_type or None}
    out: List[Dict[str, Any]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        if st.get("op") == "Search":
            st = dict(st)
            st["route"] = route
        out.append(st)
    return out


def _drop_math_if_not_needed(steps: List[Dict[str, Any]], need_math: bool) -> List[Dict[str, Any]]:
    return steps if need_math else [s for s in steps if s.get("op") != "Math"]


def _realign_final_ref(plan: Dict[str, Any]) -> None:
    steps = plan.get("steps") or []
    outs = {s.get("out") for s in steps if isinstance(s, dict)}
    if plan.get("final_ref") not in outs and steps:
        plan["final_ref"] = steps[-1].get("out", "")


def build_plan_for_subquery(
    sq: Dict[str, Any],
    ref_bindings: Optional[Dict[str, Any]],
    matched_info: Optional[Dict[str, Any]],
    llm_url: str,
    llm_model: str,
    api_key: str,
    include_semlist_metadata: bool = True,
) -> Dict[str, Any]:
    rb = ref_bindings or {}
    text = (sq.get("text") or "").strip()
    execution = sq.get("execution") if isinstance(sq.get("execution"), dict) else {}
    semantic = sq.get("semantic_parse") if isinstance(sq.get("semantic_parse"), dict) else {}
    source = (execution.get("primary_source") or "kg").strip().lower()
    fallback = (execution.get("fallback_source") or "").strip().lower()
    event_type = (execution.get("event_type") or "").strip()
    need_math = bool(semantic.get("need_math"))

    prompt = _build_prompt(
        sq=sq,
        ref_bindings=rb,
        matched_info=matched_info,
        include_semlist_metadata=include_semlist_metadata,
    )
    raw = _call_llm(prompt, llm_model, llm_url, api_key)
    obj = _parse_json(raw)
    if not _validate(obj):
        raise ValueError(f"LLM returned invalid plan for {sq.get('id')}: {str(raw)[:300]}")

    obj["steps"] = _attach_route(obj["steps"], source, fallback, event_type)
    obj["steps"] = _drop_math_if_not_needed(obj["steps"], need_math)
    _repair_search_entities(obj["steps"], semantic, obj.get("plan_type", "single_path"), sq.get("id") or "")
    _realign_final_ref(obj)

    bindings: Dict[str, Any] = {f"ref_{n}": {"ref": n} for n in _extract_refs(text)}
    bindings.update({str(k): v for k, v in rb.items()})

    plan_out = {
        "type": "map" if obj["plan_type"] == "map" else "single_path",
        "final_ref": obj["final_ref"],
        "planner_mode": "llm_v2",
    }
    if obj["plan_type"] == "map":
        plan_out.update({"over": obj.get("map_over"), "var": "item", "body": obj["steps"], "merge": "collect"})
    else:
        plan_out["steps"] = obj["steps"]

    return {
        "subquery_id": sq.get("id"),
        "subquery": text,
        "semantic": semantic,
        "execution": execution,
        "plan": plan_out,
        "bindings": bindings,
    }


def _slim_steps(steps: List[Dict[str, Any]], return_type: str = "") -> List[Dict[str, Any]]:
    slim: List[Dict[str, Any]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        op = st.get("op")
        s: Dict[str, Any] = {"op": op}
        if op == "Search":
            s["entity_name"] = st.get("entity_name", "")
            if st.get("descriptors"):
                s["descriptors"] = st.get("descriptors")
            if st.get("route"):
                s["route"] = st["route"]
        elif op == "Relate":
            s["entity"] = st.get("entity", "")
            s["relation"] = st.get("relation", "")
            tgt = st.get("target_entity") or return_type
            if tgt:
                s["target_entity"] = tgt
        elif op == "Filter":
            s["stage"] = st.get("stage", "pre")
            s["entities"] = st.get("entities", {})
            s["condition"] = st.get("condition", "")
            if st.get("condition_struct"):
                s["condition_struct"] = st.get("condition_struct")
        elif op == "Math":
            s["data"] = st.get("data", {})
            s["operation"] = st.get("operation", "")
            if st.get("group_by") is not None:
                s["group_by"] = st.get("group_by")
        s["out"] = st.get("out", "")
        slim.append(s)
    return slim


def _slim_plan(plan: Dict[str, Any], return_type: str = "") -> Dict[str, Any]:
    p = dict(plan)
    if p.get("type") == "map":
        p["body"] = _slim_steps(p.get("body") or [], return_type=return_type)
    else:
        p["steps"] = _slim_steps(p.get("steps") or [], return_type=return_type)
    return p


def _slim_output(item_out: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k in ("index", "type", "question"):
        if k in item_out:
            result[k] = item_out[k]

    op_plans: List[Dict[str, Any]] = item_out.get("op_plans") or []
    slim_sqs: List[Dict[str, Any]] = []
    for plan_obj in op_plans:
        execution = plan_obj.get("execution") or {}
        route_info: Dict[str, Any] = {}
        for k in ("primary_source", "fallback_source", "matched_concept"):
            if execution.get(k):
                route_info[k] = execution.get(k)
        return_type = str((plan_obj.get("semantic") or {}).get("Return Type") or "").strip()
        sq_slim: Dict[str, Any] = {
            "id": plan_obj.get("subquery_id", ""),
            "text": plan_obj.get("subquery", ""),
            "plan": _slim_plan(plan_obj.get("plan") or {}, return_type=return_type),
        }
        if route_info:
            sq_slim["route"] = route_info
        if plan_obj.get("bindings"):
            sq_slim["bindings"] = plan_obj["bindings"]
        slim_sqs.append(sq_slim)

    result["sub_queries"] = slim_sqs
    return result


def build_item_plan(
    item: Dict[str, Any],
    catalog_lookup: Optional[Dict[str, Dict[str, Any]]],
    llm_url: str,
    llm_model: str,
    api_key: str,
    include_semlist_metadata: bool = True,
    item_idx: Optional[int] = None,
    item_total: Optional[int] = None,
) -> Dict[str, Any]:
    sub_queries = item.get("sub_queries")
    out = dict(item)
    if not isinstance(sub_queries, list) or not sub_queries:
        out["op_plans"] = []
        return out

    plans: List[Dict[str, Any]] = []
    ref_bindings: Dict[str, Any] = {}
    total = len(sub_queries)
    for idx, sq in enumerate(sub_queries, start=1):
        sq_id = sq.get("id") or f"sq{idx}"
        sq_text = (sq.get("text") or "").strip()
        execution = sq.get("execution") if isinstance(sq.get("execution"), dict) else {}
        primary = execution.get("primary_source", "kg")
        fallback = execution.get("fallback_source", "")
        matched_concept = (execution.get("matched_concept") or "").strip()
        matched_info = (catalog_lookup or {}).get(matched_concept) if matched_concept else None

        prefix = f"[item {item_idx}/{item_total}] " if (item_idx and item_total) else ""
        print(f"{prefix}[{idx}/{total}] {sq_id}: {sq_text}")
        print(f"  primary={primary} | fallback={fallback or 'None'} | matched={matched_concept or 'None'}")

        plan = build_plan_for_subquery(
            sq=sq,
            ref_bindings=ref_bindings,
            matched_info=matched_info,
            llm_url=llm_url,
            llm_model=llm_model,
            api_key=api_key,
            include_semlist_metadata=include_semlist_metadata,
        )
        print(f"  plan_type={plan['plan']['type']} final={plan['plan']['final_ref']}")
        print("-" * 80)

        plans.append(plan)
        ref_bindings[f"ref_{idx}"] = {"from_subquery": sq_id, "final_ref": plan["plan"]["final_ref"]}

    out["op_plans"] = plans
    return out


def process_jsonl(
    input_path: str,
    output_path: str,
    llm_url: str,
    llm_model: str,
    api_key: str,
    fusion_path: Optional[str],
    include_semlist_metadata: bool,
    limit: Optional[int],
    start: int,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    catalog_lookup = load_catalog_lookup(fusion_path)
    if catalog_lookup:
        print(f"Loaded {len(catalog_lookup)} catalog entries from {fusion_path}")
    else:
        print(f"[warn] no catalog loaded (fusion_path={fusion_path}); matched event context will be empty.")

    with open(input_path, "r", encoding="utf-8") as fin:
        lines = [ln for ln in fin if ln.strip()]
    sliced = lines[start:]
    if limit is not None:
        sliced = sliced[:limit]
    total = len(sliced)
    print(f"Loaded {total} rows from {input_path}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for j, line in enumerate(sliced, start=1):
            item = json.loads(line)
            out = build_item_plan(
                item=item,
                catalog_lookup=catalog_lookup,
                llm_url=llm_url,
                llm_model=llm_model,
                api_key=api_key,
                include_semlist_metadata=include_semlist_metadata,
                item_idx=j,
                item_total=total,
            )
            fout.write(json.dumps(_slim_output(out), ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
    print(f"Saved {total} planned rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Operator-tree planner v2 (LLM-driven, route-aware).")
    parser.add_argument("--input", default="/root/autodl-tmp/new_model/step2_decompose_new/output/sq-semantic.jsonl")
    parser.add_argument("--output", default="/root/autodl-tmp/new_model/step2_decompose_new/output/sq-plan-v2.jsonl")
    parser.add_argument("--fusion-path", default="/root/autodl-tmp/new_model/step1_oag/fusion/output/semantic_list.json")
    parser.add_argument("--llm-url", default=os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1"))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-chat"))
    parser.add_argument(
        "--inject-semlist-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to inject semantic-list matched-concept metadata into the operator-tree planning prompt.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key. Use --api-key or export LLM_API_KEY.")

    process_jsonl(
        input_path=args.input,
        output_path=args.output,
        llm_url=args.llm_url,
        llm_model=args.model,
        api_key=args.api_key,
        fusion_path=args.fusion_path,
        include_semlist_metadata=args.inject_semlist_metadata,
        limit=args.limit,
        start=args.start,
    )


if __name__ == "__main__":
    main()
