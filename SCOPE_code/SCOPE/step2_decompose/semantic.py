"""
Semantic parser for sub-queries (route-aware, leakage-free).

Input  : jsonl produced by sq.py (each row has sub_q1/sub_q2 + route1/route2).
Output : same rows, augmented with `semantic_q1` / `semantic_q2` fields.

"""

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


# =====================================================================
# Regex helpers
# =====================================================================

_NUM_REF_RE = re.compile(r"\[(\d+)\]")


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _dedup_keep_order(items: List[str]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for x in items:
        if not isinstance(x, str):
            continue
        s = x.strip()
        if not s:
            continue
        k = _norm(s)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def _ensure_numeric_refs(question: str, entities: List[str]) -> List[str]:
    refs = sorted({m.group(0) for m in _NUM_REF_RE.finditer(question or "")})
    if not refs:
        return entities
    existing = {_norm(e) for e in entities}
    out = list(entities)
    for r in refs:
        if _norm(r) not in existing:
            out.insert(0, r)
    return _dedup_keep_order(out)


def _detect_need_math(question: str) -> bool:
    q = (question or "").strip()
    return bool(q)


def _normalize_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


# =====================================================================
# LLM utilities (kept consistent with sq.py)
# =====================================================================

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
    resp = requests.post(
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


def _parse_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            return {}
    return {}


def _call_and_parse(
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    required_keys: Optional[List[str]] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    last = ""
    for attempt in range(max_retries):
        try:
            raw = _call_llm(prompt, model, base_url, api_key)
            parsed = _parse_json_block(raw)
            if not parsed:
                last = "empty/invalid JSON"
            elif required_keys and any(k not in parsed for k in required_keys):
                last = f"missing keys {required_keys}"
            else:
                return parsed
        except Exception as e:
            last = str(e)
        if attempt < max_retries - 1:
            time.sleep(2)
        print(f"[Warning] semantic LLM attempt {attempt + 1}/{max_retries} failed: {last}")
    print(f"[Error] semantic LLM all retries failed; last={last}")
    return {}


# =====================================================================
# Catalog lookup (matched event / concept / relation -> rich context)
# =====================================================================

def _flatten_text(*parts: Any, limit: int = 400) -> str:
    """Flatten heterogeneous fields (str / list / dict) into one short string."""
    out: List[str] = []
    for part in parts:
        if isinstance(part, str) and part.strip():
            out.append(part.strip())
        elif isinstance(part, list):
            vals = [str(x).strip() for x in part if str(x).strip()]
            if vals:
                out.append("; ".join(vals))
        elif isinstance(part, dict):
            vals = []
            for k, v in part.items():
                if isinstance(v, (str, int, float)) and str(v).strip():
                    vals.append(f"{k}: {v}")
            if vals:
                out.append("; ".join(vals))
    s = " | ".join(out)
    return s[:limit]


def _sample_section_titles(samples: List[Any], max_titles: int = 6) -> List[str]:
    """Pull section / table titles or top-level keys from a few sample rows."""
    titles: List[str] = []
    for s in (samples or [])[:3]:
        if not isinstance(s, dict):
            continue
        for key in ("section_title", "title", "page_title", "name", "table_title"):
            v = s.get(key)
            if isinstance(v, str) and v.strip():
                titles.append(v.strip())
        # Also expose column-header-like keys
        cols = s.get("columns") or s.get("headers")
        if isinstance(cols, list):
            for c in cols:
                if isinstance(c, str) and c.strip():
                    titles.append(c.strip())
    seen: set = set()
    uniq: List[str] = []
    for t in titles:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(t)
        if len(uniq) >= max_titles:
            break
    return uniq


def build_catalog_lookup(data: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Transform a raw semantic_list.json dict into a per-concept context lookup.

    This is the in-memory counterpart of ``load_catalog_lookup``; callers that
    already hold the parsed semantic_list (e.g. pipeline.stage1_check) should
    use this so the matched-concept context is the structure that
    ``parse_semantic`` / ``operator_plan`` expect — NOT the raw catalog.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for key, info in (data or {}).items():
        if not isinstance(info, dict):
            continue
        sources = info.get("sources") or {}
        kg = sources.get("KG") or sources.get("kg") or {}
        tbl = sources.get("Table") or sources.get("table") or {}
        doc = sources.get("Doc") or sources.get("doc") or {}
        out[key] = {
            "kind": "concept",
            "canonical_name": info.get("concept") or key,
            "description": (info.get("overall_description") or "").strip(),
            "kg_description": (kg.get("description") or "").strip(),
            "kg_elements": kg.get("elements") or [],
            "kg_sample": kg.get("data_examples") or [],
            "table_description": (tbl.get("description") or "").strip(),
            "table_elements": tbl.get("elements") or [],
            "table_sample": tbl.get("data_examples") or [],
            "doc_description": (doc.get("description") or "").strip(),
            "doc_elements": doc.get("elements") or [],
            "doc_sample": doc.get("data_examples") or [],
        }
    return out


def load_catalog_lookup(semantic_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load semantic concept metadata from semantic_list.json.

    Expected format:
    {
      "concept_name": {
        "concept": "concept_name",
        "overall_description": "...",
        "sources": {
          "KG": {"description": "...", "elements": [...], "data_examples": [...]},
          "Table": {...}
        }
      }
    }
    """
    if not semantic_path or not os.path.exists(semantic_path):
        return {}
    with open(semantic_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return build_catalog_lookup(data)


def _render_matched_context(
    info: Optional[Dict[str, Any]],
    primary_source: str,
) -> str:
    """Render source-specific concept metadata based on routed source."""
    if not info:
        return "(no matched concept metadata; rely on general rules below)"

    src = (primary_source or "").strip().lower()
    parts: List[str] = []
    parts.append(f"- matched_concept: {info.get('canonical_name')}")
    desc = (info.get("description") or "").strip()
    if desc:
        parts.append(f"- overall_description: {desc[:300]}")

    if src == "kg":
        kg_desc = (info.get("kg_description") or "").strip()
        if kg_desc:
            parts.append(f"- kg_description: {kg_desc}")
        kg_elements = info.get("kg_elements") or []
        if kg_elements:
            parts.append(f"- kg_elements: {json.dumps(kg_elements, ensure_ascii=False)}")
        kg_sample = info.get("kg_sample") or []
        if kg_sample:
            preview = json.dumps(kg_sample[:2], ensure_ascii=False)
            parts.append(f"- kg_sample: {preview[:300]}")
    elif src == "table":
        tbl_desc = (info.get("table_description") or "").strip()
        if tbl_desc:
            parts.append(f"- table_description: {tbl_desc}")
        tbl_elements = info.get("table_elements") or []
        if tbl_elements:
            parts.append(f"- table_elements: {json.dumps(tbl_elements, ensure_ascii=False)}")
        titles = _sample_section_titles(info.get("table_sample") or [])
        if titles:
            parts.append(f"- table_section_titles_or_columns: {titles}")
        tbl_sample = info.get("table_sample") or []
        if tbl_sample and not titles:
            preview = json.dumps(tbl_sample[:1], ensure_ascii=False)
            parts.append(f"- table_sample: {preview[:300]}")
    elif src == "doc":
        doc_desc = (info.get("doc_description") or info.get("description") or "").strip()
        if doc_desc:
            parts.append(f"- doc_description: {doc_desc}")
        doc_samples = info.get("doc_sample") or info.get("table_sample") or info.get("kg_sample") or []
        titles = _sample_section_titles(doc_samples)
        if titles:
            parts.append(f"- doc_title_or_section_title_examples: {titles}")
        doc_elements = info.get("doc_elements") or info.get("kg_elements") or []
        if doc_elements:
            parts.append(f"- doc_elements: {json.dumps(doc_elements, ensure_ascii=False)}")
        doc_sample = info.get("doc_sample") or []
        if doc_sample and not titles:
            preview = json.dumps(doc_sample[:1], ensure_ascii=False)
            parts.append(f"- doc_sample: {preview[:300]}")

    return "\n".join(parts)


# =====================================================================
# Prompt (route-aware, no example queries)
# =====================================================================

_RULES_BY_SOURCE = {
    "kg": (
        "PRIMARY SOURCE = kg.\n"
        "Extraction priority:\n"
        "  - Entities: the head & tail of the relation, plus any [k] ref tokens.\n"
        "  - Relation: the relation/predicate name (verb phrase) connecting the head entity to the Return Type.\n"
        "KG parsing rules:\n"
        "  - If the matched KG example has a clear head entity and tail entity, then:\n"
        "    - Entities should prefer those head/tail entities from the KG example.\n"
        "    - Relation should be the relation connecting the head and tail entities.\n"
        "    - Return Type should be the target entity type.\n"
        "    - Conditions should capture the remaining elements in the KG concept.\n"
        "  - If the matched KG example is not a head-tail entity relation, but instead an attribute relation between entities, then:\n"
        "    - Entities should be the known entity/entities.\n"
        "    - Relation should be the attribute name.\n"
        "    - Return Type should be the attribute target name.\n"
        "    - Conditions should capture the other elements/modifiers."
    ),
    "table": (
        "PRIMARY SOURCE = table.\n"
        "The section / column titles shown in the matched-event context are the authoritative pattern for how an event name is written.\n"
        "Extraction priority:\n"
        "  - Entities:\n"
        "      (a) the people / teams / time involved;\n"
        "      (b) any [k] ref tokens.\n"
        "      (c) event name: use the naming pattern (structure) from the matched-event context titles, but fill it using entities from the user's question — never copy example entities unless the question explicitly contains them;\n"
        "  - Relation: the numeric / statistical columns the question asks for, mapped to a short verb phrase.\n"
        "  - Return Type should be that attribute / column name.\n"
        "  - Conditions should capture the remaining qualifiers."
    ),
    "doc": (
        "PRIMARY SOURCE = doc.\n"
        "The title / section_title style examples in the matched-event context are the authoritative pattern for how a document passage should be searched.\n"
        "Extraction priority:\n"
        "  - Entities:\n"
        "      (a) the main subject entity or event title implied by the question;\n"
        "      (b) the passage / section title pattern suggested by matched examples (for example: award name, season, team name, competition name, or biography subject);\n"
        "      (c) any [k] ref tokens.\n"
        "  - Relation: the passage fact asked for, often expressed as a short verb phrase or descriptor.\n"
        "  - Return Type: the answer type requested by the question.\n"
        "  - Conditions should capture time, competition, award, and descriptive modifiers."
    ),
}

def _build_semantic_prompt(
    question: str,
    primary_source: Optional[str],
    fallback_source: Optional[str],
    matched_info: Optional[Dict[str, Any]] = None,
) -> str:
    src = (primary_source or "kg").strip().lower()
    src_rules = _RULES_BY_SOURCE.get(src, _RULES_BY_SOURCE["kg"])
    matched_block = _render_matched_context(matched_info, src)

    hint_lines: List[str] = []
    if primary_source:
        hint_lines.append(f"- primary_source: {primary_source}")
    if fallback_source:
        hint_lines.append(f"- fallback_source: {fallback_source}")
    hint_block = "\n".join(hint_lines) if hint_lines else "(none)"

    return f"""You are a sports QA semantic parser.

    General rules:
    - Entities       : named things — people, teams, awards, events, venues,
                    seasons, years, competitions, and any [k] reference token
                    that appears in the question.
    - Relation       : a short canonical phrase describing the relation that
                    connects the head entity to the Return Type
                    (e.g. "play for", "coach", "win", "received").
    - Return Type    : what the question asks to return (player / team / coach /
                    award / streak / number / ...).
    - Conditions     : (a) temporal / contextual prepositional phrases such as
                    in/at/on/during/for/of/as/from + object;
                    (b) frequency constraints (e.g. "three consecutive times");
                    (c) extreme/aggregate phrases (highest, most, best, ...).
                    Award names that contain "Most Valuable Player Award"
                    are NOT extreme conditions; they are entities.
    - need_math      : true when the question requires aggregation / sorting /
                    counting, or any mathematical operation such as max/min,
                    highest/lowest, how many, total/sum, average, ratio,
                    consecutive counts, streaks, percentages, ...; otherwise false.

    Routing context (already decided by the upstream router):
    {hint_block}

    Source-aware parsing instructions:
    - Use the routed primary source to decide how to ground the answer.
    - Do not treat KG and Table the same way.

    Matched concept context:
    {matched_block}

    Source-specific rules:
    {src_rules}

            Output strictly valid JSON with this schema and no extra commentary:
            {{
            "Entities":    ["..."],
            "Relation":    ["..."],
            "Return Type": ["..."],
            "Conditions":  ["..."],
            "need_math":   true/false
            }}

    Question: "{question}"
    """


# =====================================================================
# Core parse
# =====================================================================

def parse_semantic(
    question: str,
    route: Optional[Dict[str, Any]],
    model: str,
    base_url: str,
    api_key: str,
    catalog_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    q = (question or "").strip()
    empty = {
            "Entities": [], "Relation": [], "Return Type": [], "Conditions": [], "need_math": False,
    }
    if not q:
        return empty

    route = route or {}
    matched_key = route.get("matched_concept")
    matched_info = (catalog_lookup or {}).get(matched_key) if matched_key else None

    prompt = _build_semantic_prompt(
        question=q,
        primary_source=route.get("primary_source"),
        fallback_source=route.get("fallback_source"),
        matched_info=matched_info,
    )
    parsed = _call_and_parse(
        prompt, model, base_url, api_key,
        required_keys=["Entities", "Return Type"],
    )

    def _list(x: Any) -> List[str]:
        # LLM 有时把单值字段返回成裸字符串而非数组，需包成列表而不是丢弃
        if isinstance(x, str):
            x = [x]
        if not isinstance(x, list):
            return []
        return [v.strip() for v in x if isinstance(v, str) and v.strip()]

    entities    = _dedup_keep_order(_list(parsed.get("Entities")))
    relation    = _dedup_keep_order(_list(parsed.get("Relation")))
    return_type = _dedup_keep_order(_list(parsed.get("Return Type")))
    conditions  = _dedup_keep_order(_list(parsed.get("Conditions")))
    need_math   = _normalize_boolish(parsed.get("need_math"))

    # ensure [k] tokens always live in entities
    entities = _ensure_numeric_refs(q, entities)


    return {
        "Entities":    entities,
        "Relation":    relation,
        "Return Type": return_type,
        "Conditions":  conditions,
        "need_math":   need_math,
    }


# =====================================================================
# File processing
# =====================================================================

def _build_sq_block(
    sq_id: str,
    sq_text: str,
    route: Dict[str, Any],
    semantic: Optional[Dict[str, Any]],
    match: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    route = route or {}
    return {
        "id": sq_id,
        "text": sq_text,
        "match": match or {},
        "execution": {
            "matched_concept": route.get("matched_concept"),
            "primary_source":  route.get("primary_source"),
            "fallback_source": route.get("fallback_source"),
        },
        "semantic_parse": semantic or {
            "Entities": [], "Relation": [], "Return Type": [], "Conditions": [], "need_math": False,
        },
    }


def _print_sq_debug(
    idx: Any,
    tag: str,
    question: str,
    route: Dict[str, Any],
    semantic: Optional[Dict[str, Any]],
) -> None:
    print(f"\n=== idx={idx} {tag} ===")
    print(f"  question        : {question}")
    print(f"  primary_source  : {route.get('primary_source')}")
    print(f"  fallback_source : {route.get('fallback_source')}")
    print(f"  matched_concept : {route.get('matched_concept')}")
    if semantic:
        print(f"  -- semantic_parse --")
        print(f"    Entities    : {semantic.get('Entities')}")
        print(f"    Relation    : {semantic.get('Relation')}")
        print(f"    Return Type : {semantic.get('Return Type')}")
        print(f"    Conditions  : {semantic.get('Conditions')}")
        print(f"    need_math   : {semantic.get('need_math')}")
    else:
        print(f"  -- semantic_parse -- (skipped: empty question)")


def process_item(
    item: Dict[str, Any],
    model: str,
    base_url: str,
    api_key: str,
    catalog_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    q1 = (item.get("sub_q1") or "").strip()
    q2 = (item.get("sub_q2") or "").strip()
    r1 = item.get("route1") or {}
    r2 = item.get("route2") or {}
    m1 = item.get("match1") or {}
    m2 = item.get("match2") or {}

    sem1 = parse_semantic(q1, r1, model, base_url, api_key,
                          catalog_lookup=catalog_lookup) if q1 else None
    if verbose:
        _print_sq_debug(item.get("index"), "sq1", q1, r1, sem1)

    sem2 = parse_semantic(q2, r2, model, base_url, api_key,
                          catalog_lookup=catalog_lookup) if q2 else None
    if verbose:
        _print_sq_debug(item.get("index"), "sq2", q2, r2, sem2)

    return {
        "index": item.get("index"),
        "type": item.get("type"),
        "question": item.get("question", ""),
        "sub_queries": [
            _build_sq_block("sq1", q1, r1, sem1, m1),
            _build_sq_block("sq2", q2, r2, sem2, m2),
        ],
    }


def run(
    input_path: str,
    output_path: str,
    model: str,
    base_url: str,
    api_key: str,
    fusion_path: Optional[str] = None,
    limit: Optional[int] = None,
    start: int = 0,
    verbose: bool = False,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    catalog_lookup = load_catalog_lookup(fusion_path)
    if catalog_lookup:
        print(f"Loaded {len(catalog_lookup)} concept entries from {fusion_path}")
    else:
        print(f"[warn] no semantic catalog loaded (fusion_path={fusion_path}); "
              "matched concept context will be empty.")

    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    if start:
        rows = rows[start:]
    if limit is not None:
        rows = rows[:limit]
    print(f"Loaded {len(rows)} rows from {input_path}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, item in enumerate(rows, 1):
            out = process_item(item, model, base_url, api_key,
                               catalog_lookup=catalog_lookup, verbose=verbose)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            if verbose:
                print(f"  -> wrote {i}/{len(rows)} idx={out.get('index')}")
    print(f"Saved {len(rows)} enriched rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Route-aware semantic parsing of sub-queries (leakage-free)."
    )
    parser.add_argument("--input",
        default="/root/autodl-tmp/new_model/step2_decompose_new/output/kg-table-26_test.jsonl")
    parser.add_argument("--output",
        default="/root/autodl-tmp/new_model/step2_decompose_new/output/sq-semantic.jsonl")
    parser.add_argument("--fusion-path",
        default="/root/autodl-tmp/new_model/step1_oag/fusion/output/semantic_list.json",
        help="Path to semantic_list.json (provides matched concept context).")
    parser.add_argument("--model",    default=os.getenv("LLM_MODEL", "deepseek-chat"))
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1"))
    parser.add_argument("--api-key",  default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--start",    type=int, default=0)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("No API key. Use --api-key or export LLM_API_KEY.")

    run(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        fusion_path=args.fusion_path,
        limit=args.limit,
        start=args.start,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
