import json
import os
import re
import sys
import difflib
from typing import Any, Dict, List, Optional, Tuple


"""MMQA KG query adapter (LLM-first, fallback-light).

Public APIs:
- semantic_parsing_api(question, api_url)
- engine_exec_api(program, api_url)

Design:
- Semantic parsing: parse question -> entities/relation/targets (LLM-first)
- Execution: resolve entities -> select relations -> collect/rank evidence
"""


# ============================================================
# 0) Bootstrap / shared resources
# ============================================================


def _ensure_import_path() -> None:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    service_kg_dir = os.path.join(cur_dir, "service", "KG")
    if service_kg_dir not in sys.path:
        sys.path.append(service_kg_dir)


_ensure_import_path()
from localkg_index import LocalKGIndex  # noqa: E402


try:
    # Preferred: current project layout when running from `src/`
    from query_knowledge_source.query_llm import OpenAICaller  # type: ignore # noqa: E402
except Exception:
    try:
        # Package-relative import fallback (when imported as a package module)
        from .query_llm import OpenAICaller  # type: ignore # noqa: E402
    except Exception:
        try:
            # Legacy layout fallback
            from rag.retrieve.query_llm import OpenAICaller  # type: ignore # noqa: E402
        except Exception:
            try:
                # Legacy layout fallback
                from retrieve.query_llm import OpenAICaller  # type: ignore # noqa: E402
            except Exception:
                # Last-resort local module fallback
                from query_llm import OpenAICaller  # type: ignore # noqa: E402


_KG_INDEX: Optional[LocalKGIndex] = None
_LLM_CALLER: Optional[OpenAICaller] = None


def _kg_debug_enabled() -> bool:
    return str(os.environ.get("ATOMR_KG_DEBUG", "0")).strip().lower() in {"1", "true", "yes", "on"}


def _kg_debug_print(msg: str) -> None:
    if _kg_debug_enabled():
        print(f"[KG-DEBUG] {msg}")


def _get_kg_index() -> LocalKGIndex:
    global _KG_INDEX
    if _KG_INDEX is None:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(cur_dir, "..", ".."))

        # Preferred override from environment
        env_kg_dir = os.environ.get("ATOMR_KG_DIR", "").strip()

        # Current repo layout candidate (baseline/atomr/data_sources/KG)
        default_kg_dir = os.path.join(project_root, "data_sources", "KG")

        # Legacy layout candidate (kept for backward compatibility)
        legacy_kg_dir = os.path.join(project_root, "data", "data_sources", "KG")

        if env_kg_dir:
            kg_dir = env_kg_dir
        elif os.path.isdir(default_kg_dir):
            kg_dir = default_kg_dir
        else:
            kg_dir = legacy_kg_dir

        _KG_INDEX = LocalKGIndex(kg_dir=kg_dir)
    return _KG_INDEX


def _llm_parser_enabled() -> bool:
    return str(os.environ.get("ATOMR_KG_ENABLE_LLM_PARSER", "1")).strip().lower() in {"1", "true", "yes", "on"}


def _get_llm_caller() -> Optional[OpenAICaller]:
    global _LLM_CALLER
    if not _llm_parser_enabled():
        return None
    if _LLM_CALLER is None:
        cache_path = os.environ.get("ATOMR_LLM_CACHE_PATH", "../../openai_service/llm_cache/cache.jsonl")
        # 直连模式：base_url/api_key/model 全部从环境变量获取
        # main.py 已把 CLI 参数同步到环境变量
        _LLM_CALLER = OpenAICaller(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            model=os.environ.get("ATOMR_KG_PARSER_MODEL") or os.environ.get("ATOMR_LLM_MODEL"),
            cache_path=cache_path,
        )
    return _LLM_CALLER


# ============================================================
# 1) Common text helpers
# ============================================================


def _normalize_text(s: str) -> str:
    return " ".join(str(s or "").split()).strip()


def _token_set_ratio(left: str, right: str) -> float:
    lset = sorted(set(_normalize_text(left.lower()).split()))
    rset = sorted(set(_normalize_text(right.lower()).split()))
    l = " ".join(lset)
    r = " ".join(rset)
    return difflib.SequenceMatcher(a=l, b=r).ratio() * 100.0


def _target_match_score(text: str, target: str) -> float:
    if not text or not target:
        return 0.0
    return _token_set_ratio(_normalize_text(text), _normalize_text(target))


def _label_similarity(kg: LocalKGIndex, query: str, cand: str) -> float:
    qn = kg.normalize_label_for_index(query) if hasattr(kg, "normalize_label_for_index") else _normalize_text(query.lower())
    cn = kg.normalize_label_for_index(cand) if hasattr(kg, "normalize_label_for_index") else _normalize_text(cand.lower())
    if not qn or not cn:
        return 0.0
    return _token_set_ratio(qn, cn)


def _parse_llm_json(raw: str) -> Optional[Dict]:
    txt = str(raw or "").strip()
    if not txt:
        return None

    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)

    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", txt)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _parse_llm_relation_choice(raw: str) -> str:
    txt = _normalize_text(str(raw or ""))
    if not txt:
        return ""

    obj = _parse_llm_json(txt)
    if isinstance(obj, dict):
        return _normalize_text(obj.get("relation", ""))

    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    return _normalize_text(txt.strip('"\''))


# ============================================================
# 2) Semantic parsing
# ============================================================


def _build_relation_target(question: str, entities: List[str], relation_text: str, parsed_target: Any = None) -> Dict[str, str]:
    target = parsed_target if isinstance(parsed_target, dict) else {}
    head = _normalize_text(target.get("head", ""))
    predicate = _normalize_text(target.get("predicate", ""))
    tail = _normalize_text(target.get("tail", ""))

    if not head and entities:
        head = _normalize_text(entities[0])
    if not predicate:
        predicate = _normalize_text(relation_text)

    return {"head": head, "predicate": predicate, "tail": tail}


def _build_relation_targets(
    question: str,
    entities: List[str],
    relation_text: str,
    parsed_targets: Any = None,
    parsed_target: Any = None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    if isinstance(parsed_targets, list):
        for item in parsed_targets[:8]:
            t = _build_relation_target(question, entities, relation_text, parsed_target=item)
            if t.get("head") or t.get("predicate") or t.get("tail"):
                out.append(t)

    if not out:
        out.append(_build_relation_target(question, entities, relation_text, parsed_target=parsed_target))

    dedup: List[Dict[str, str]] = []
    seen = set()
    for t in out:
        key = (t.get("head", ""), t.get("predicate", ""), t.get("tail", ""))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)

    return dedup


def _llm_parse(question: str) -> Optional[Dict]:
    caller = _get_llm_caller()
    if caller is None:
        return None

    prompt = (
        "You are a KG query parser. Extract a lightweight plan from the user question.\n"
        "Return ONLY JSON with schema:\n"
        "{\n"
        "  \"entities\": [string],\n"
        "  \"relation\": string,\n"
        "  \"relation_target\": {\"head\": string, \"predicate\": string, \"tail\": string},\n"
        "  \"relation_targets\": [{\"head\": string, \"predicate\": string, \"tail\": string}],\n"
        "  \"hop\": number,\n"
        "  \"constraints\": [string]\n"
        "}\n"
        "Rules:\n"
        "- entities should include concrete known entity mentions if possible.\n"
        "- relation should be short natural phrase, not a full sentence.\n"
        "- relation_targets should include one target per relevant head entity when possible.\n"
        "- each target.head should be a concrete entity mention.\n"
        "- hop defaults to 1.\n\n"

        """for example:
        question: For each player in Glen Selbo, what teams did they play for?
        "{\n"
        "  \"entities\": [\"Glen Selbo\"],\n"
        "  \"relation\": \"play for\",\n"
        "  \"relation_target\": {\"head\": \"Glen Selbo\", \"predicate\": \"play for\", \"tail\": \"teams\"},\n"
        "  \"relation_targets\": [{\"head\": \"Glen Selbo\", \"predicate\": \"play for\", \"tail\": \"teams\"}],\n"
        "  \"hop\": 1,\n"
        "  \"constraints\": [string]\n"
        "}
        
        """

        f"Question: {question}"
    )

    try:
        model = os.environ.get("ATOMR_KG_PARSER_MODEL") or os.environ.get("ATOMR_LLM_MODEL")
        response, _ = caller.query_deepseek(prompt=prompt, model=model, max_tokens=10000, temperature=0)
        parsed = _parse_llm_json(response)
        if not parsed:
            return None

        entities = parsed.get("entities") or []
        if not isinstance(entities, list):
            entities = []
        entities = [_normalize_text(x) for x in entities if _normalize_text(x)]

        relation = _normalize_text(parsed.get("relation", ""))

        hop = parsed.get("hop", 1)
        try:
            hop = int(hop)
        except Exception:
            hop = 1

        constraints = parsed.get("constraints") or []
        if not isinstance(constraints, list):
            constraints = []

        relation_targets = _build_relation_targets(
            question=question,
            entities=entities,
            relation_text=relation,
            parsed_targets=parsed.get("relation_targets"),
            parsed_target=parsed.get("relation_target"),
        )

        return {
            "entities": entities[:5],
            "relation": relation,
            "relation_target": relation_targets[0] if relation_targets else _build_relation_target(question, entities, relation),
            "relation_targets": relation_targets,
            "hop": 1 if hop <= 0 else min(hop, 2),
            "constraints": [str(c) for c in constraints[:5]],
            "parser": "llm",
        }
    except Exception as e:
        _kg_debug_print(f"llm_parse_failed={e}")
        return None


def semantic_parsing_api(question, api_url):
    q = _normalize_text(str(question or ""))
    parsed = _llm_parse(q)

    if not parsed:
        msg = "LLM解析失败"
        print(msg)
        return {
            "question": q,
            "backend": "mmqa_local_kg",
            "api_url": api_url,
            "error": msg,
            "entities": [],
            "relation_text": "",
            "relation_targets": [],
            "relation_target": {},
            "parser": "llm_failed",
        }

    program = {
        "question": q,
        "backend": "mmqa_local_kg",
        "api_url": api_url,
        "entities": parsed.get("entities") or [],
        "relation_text": parsed.get("relation") or q,
        "relation_targets": parsed.get("relation_targets") or [],
        "relation_target": parsed.get("relation_target") or {},
        "parser": parsed.get("parser", "llm"),
    }
    _kg_debug_print(f"semantic_program={program}")
    return program


# ============================================================
# 3) Execution helpers
# ============================================================


def _entity_snapshot(kg: LocalKGIndex, qid: str, fallback_label: str = "") -> Dict:
    return {
        "qid": qid,
        "label": kg.qid_to_label(qid) or fallback_label or qid,
        "types": kg.get_entity_types(qid),
        "attrs": kg.get_entity_props(qid),
    }


def _resolve_entity_qids(kg: LocalKGIndex, label: str) -> List[str]:
    raw = _normalize_text(label)
    if not raw:
        return []

    # 1. 尝试原样精确匹配
    qids = kg.resolve_label(raw)
    
    # 2. 尝试小写/归一化精确匹配
    if not qids:
        qids = kg.resolve_label_normalized(raw)

    # 3. 尝试去除括号后的精确匹配 (例如 "Apple (Company)" -> "Apple")
    if not qids:
        stripped = re.sub(r"\s*\([^)]*\)\s*$", "", raw).strip()
        if stripped and stripped != raw:
            qids = kg.resolve_label(stripped) or kg.resolve_label_normalized(stripped)

    # 如果找到了精确匹配，直接返回
    if qids:
        return qids[:5]

    # 删除了原来盲目用 Token 召回并直接 return 的逻辑。
    # 这里直接返回空，让外层主流程去调用更强大的 _fuzzy_entity_candidates 进行打分匹配。
    return []


def _fuzzy_entity_candidates(kg: LocalKGIndex, mention: str, limit: int = 5) -> List[Tuple[float, str]]:
    mention_norm = _normalize_text(mention)
    if not mention_norm:
        return []

    # 1) 倒排 token 召回候选
    tokens = [t for t in kg.tokenize_label(mention_norm) if len(t) >= 2]
    tokens_sorted = sorted(tokens, key=len, reverse=True)

    cand_qids = set()
    stop_tokens = {"la", "ny", "dc", "the", "and", "for", "of"}
    for t in tokens_sorted:
        if t.lower() in stop_tokens:
            continue
        cand_qids.update(kg.resolve_tokens([t]))
        if len(cand_qids) >= 1000:
            break

    if not cand_qids:
        return []

    # 2) 对候选重排（与后来的实现对齐：使用规范化 label 相似度）
    scored: List[Tuple[float, str]] = []
    for qid in cand_qids:
        cand_label = kg.qid_to_label(qid) or ""
        s = _label_similarity(kg, mention_norm, cand_label)
        scored.append((s, qid))

    scored.sort(key=lambda x: x[0], reverse=True)

    # 3) 阈值过滤（45/100 对齐后续工程的 0.45）
    keep = [(s, qid) for (s, qid) in scored if s >= 45.0]
    return keep[:limit]



def _make_entity_only_evidence(kg: LocalKGIndex, qid: str, parser_name: str, source: str) -> Dict:
    label = kg.qid_to_label(qid) or qid
    snap = _entity_snapshot(kg, qid, label)
    return {
        "label": label,
        "qid": qid,
        "meta": {},
        "head_entity": label,
        "head_qid": qid,
        "relation": "",
        "direction": "self",
        "parser": parser_name,
        "evidence_source": source,
        "head_entity_snapshot": snap,
        "tail_entity_snapshot": {},
    }


def _target_head_matches_entity(target_head: str, entity_label: str) -> bool:
    th = _normalize_text(target_head)
    if not th:
        return True
    el = _normalize_text(entity_label)
    if not el:
        return False
    return _target_match_score(th, el) >= 86.0


def _parse_mmqa_search_answer_labels(raw: str) -> List[str]:
    txt = str(raw or "")
    if not txt.strip():
        return []

    m = re.search(r"Answer\s*:\s*(.+)", txt, flags=re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
    else:
        ans = txt.strip()

    ans = ans.strip().strip("`")
    if ans.lower() in {"none", "[]", "null", ""}:
        return []

    try:
        obj = json.loads(ans)
        if isinstance(obj, list):
            return [_normalize_text(str(x)) for x in obj if _normalize_text(str(x))]
        if isinstance(obj, str):
            val = _normalize_text(obj)
            return [] if not val or val.lower() == "none" else [val]
    except Exception:
        pass

    parts = [p.strip().strip('"\'') for p in re.split(r",|\n", ans) if p.strip()]
    out = []
    for p in parts:
        pp = _normalize_text(p)
        if pp and pp.lower() != "none":
            out.append(pp)
    return out


def _llm_confirm_entity_qids(
    kg: LocalKGIndex,
    question: str,
    target_entity: str,
    candidate_qids: List[str],
) -> List[str]:
    caller = _get_llm_caller()
    if caller is None or not candidate_qids:
        return candidate_qids[:2]

    candidates = []
    for qid in candidate_qids[:8]:
        label = kg.qid_to_label(qid) or qid
        props = kg.get_entity_props(qid)
        candidates.append({"qid": qid, "label": label, "meta": props})

    prompt = (
        "You are an AI assistant helping to verify if retrieved candidate entities match the CORE target entity requested in the question.\n\n"
        f"Question: {question}\n"
        f"Target Entity to find: {target_entity}\n\n"
        "You will be provided with retrieved knowledge from KG. Evaluate each candidate's label and metadata to determine if it represents the core entity.\n"
        "Important: ignore extra temporal/event constraints in target text and focus on core entity identity.\n"
        "Output format:\n"
        "Answer: [comma-separated list of exact candidate labels that are correct matches, or 'None']\n"
        "Reasoning: [brief explanation]\n\n"
        "Retrieved Knowledge:\n"
        f"KG: {json.dumps(candidates, ensure_ascii=False)}\n"
        f"Question: {question}\n"
        f"Target Entity to find: {target_entity}\n"
        "Answer:"
    )

    try:
        model = os.environ.get("ATOMR_KG_PARSER_MODEL") or os.environ.get("ATOMR_LLM_MODEL")
        response, _ = caller.query_deepseek(prompt=prompt, model=model, max_tokens=10000, temperature=0)
        
        keep_labels = set(x.lower() for x in _parse_mmqa_search_answer_labels(response))
        if not keep_labels:
            return []

        keep_qids = []
        for qid in candidate_qids:
            label = _normalize_text(kg.qid_to_label(qid) or qid).lower()
            if label in keep_labels:
                keep_qids.append(qid)
        return keep_qids[:2]
    except Exception as e:
        _kg_debug_print(f"llm_confirm_entity_qids_failed={e}")
        return candidate_qids[:2]


def _resolve_target_types(kg: LocalKGIndex, target_entity: str) -> List[str]:
    raw = _normalize_text(target_entity).lower()
    if not raw:
        return []

    candidates = [raw, raw.replace(" ", "_"), raw.replace("_", " "), raw.split(" ")[-1]]
    type_keys = set(getattr(kg, "type2columns", {}).keys())

    out = []
    seen = set()
    for c in candidates:
        cc = _normalize_text(c).lower()
        if not cc or cc in seen:
            continue
        seen.add(cc)
        if cc in type_keys:
            out.append(cc)
    return out


def _llm_select_schema_column(
    question: str,
    relation_text: str,
    target_type: str,
    columns: List[str],
    missing_entity: str,
) -> str:
    if not columns:
        return ""

    caller = _get_llm_caller()
    if caller is not None:
        prompt = (
            "You are a database schema expert. Your task is to find the most appropriate column in a database table that can help answer the query.\n\n"
            f"Question: {question}\n"
            f"Missing Entity: {missing_entity}\n"
            f"Target Table: {target_type}\n"
            f"Relation to map: {relation_text}\n"
            f"Available Columns: {columns}\n\n"
            "Output ONLY the exact column name from the available list that best represents this relationship."
            " If no column is a good match, output 'None'.\n"
            "Answer:"
        )
        try:
            model = os.environ.get("ATOMR_KG_PARSER_MODEL") or os.environ.get("ATOMR_LLM_MODEL")
            response, _ = caller.query_deepseek(prompt=prompt, model=model, max_tokens=10000, temperature=0)
            ans = _normalize_text(str(response or "")).strip().strip("`").strip('"\'')
            m = re.search(r"Answer\s*:\s*(.+)", ans, flags=re.IGNORECASE)
            if m:
                ans = _normalize_text(m.group(1))
            for c in columns:
                if _normalize_text(c).lower() == ans.lower():
                    return c
        except Exception as e:
            _kg_debug_print(f"llm_select_schema_column_failed={e}")

    scored = []
    for c in columns:
        s = _token_set_ratio(relation_text, c.replace("_", " "))
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] < 35.0:
        return ""
    return scored[0][1]


def _fallback_entities_by_property_match(
    kg: LocalKGIndex,
    question: str,
    known_entity_label: str,
    relation_text: str,
    relation_targets: List[Dict[str, str]],
    parser_name: str,
) -> List[Dict]:
    out: List[Dict] = []

    for rt in relation_targets[:6]:
        target_entity = _normalize_text(rt.get("tail", ""))
        rel_hint = _normalize_text(rt.get("predicate", "")) or relation_text
        if not target_entity:
            continue

        for ttype in _resolve_target_types(kg, target_entity):
            columns = kg.get_columns_for_type(ttype)
            if not columns:
                continue

            col = _llm_select_schema_column(
                question=question,
                relation_text=rel_hint,
                target_type=ttype,
                columns=columns,
                missing_entity=known_entity_label,
            )
            if not col:
                continue

            matched_qids = kg.search_by_property_value(ttype, col, known_entity_label)
            for qid in matched_qids[:20]:
                label = kg.qid_to_label(qid) or qid
                props = kg.get_entity_props(qid)
                meta = dict(props)
                meta["fallback_target_type"] = ttype
                meta["fallback_column"] = col
                meta["fallback_match_value"] = known_entity_label
                out.append(
                    {
                        "label": label,
                        "qid": qid,
                        "meta": meta,
                        "head_entity": known_entity_label,
                        "head_qid": "",
                        "relation": rel_hint,
                        "direction": "fallback_property",
                        "parser": parser_name,
                        "evidence_source": "fallback_property_match",
                        "relation_target": rt,
                        "head_entity_snapshot": {},
                        "tail_entity_snapshot": _entity_snapshot(kg, qid, label),
                    }
                )

    dedup = []
    seen = set()
    for ev in out:
        key = (ev.get("qid"), ev.get("meta", {}).get("fallback_column"), ev.get("meta", {}).get("fallback_target_type"))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ev)
    return dedup[:30]


def _resolve_relation_with_target(
    question: str,
    relation_text: str,
    relation_names: List[str],
    relation_target: Dict[str, str],
    head_label: str,
    rel2_neighbor_info: Dict[str, List[Dict[str, str]]],
) -> str:
    if not relation_names:
        return ""

    rel_text = _normalize_text(relation_text)
    target_head = _normalize_text(relation_target.get("head", ""))
    target_pred = _normalize_text(relation_target.get("predicate", "")) or rel_text
    target_tail = _normalize_text(relation_target.get("tail", ""))

    caller = _get_llm_caller()
    if caller is not None:
        try:
            option_lines = []
            for rel in relation_names[:200]:
                neigh = rel2_neighbor_info.get(rel, [])[:8]
                neigh_txt = "; ".join(
                    f"{x.get('label', '')} (types: {x.get('types', 'unknown')})"
                    for x in neigh
                )
                option_lines.append(f"- {rel} || neighbors: {neigh_txt}")

            prompt = (
                "You are selecting the best KG relation with head/predicate/tail-target constraints.\n"
                "Your decision MUST jointly consider:\n"
                "1) relation semantic match to predicate/relation description, and\n"
                "2) whether neighbor entities match the requested target entity type.\n\n"
                "Return ONLY JSON: {\"relation\": \"<one option exactly>\"}.\n"
                "If no option is semantically compatible, return {\"relation\": \"\"}.\n\n"
                f"Question: {question}\n"
                f"Target pattern: [{target_head or head_label}, {target_pred or rel_text}, {target_tail}]\n"
                f"Current KG head entity: {head_label}\n"
                "Relation options with sampled neighbors and neighbor types:\n"
                + "\n".join(option_lines)
            )
            model = os.environ.get("ATOMR_KG_PARSER_MODEL") or os.environ.get("ATOMR_LLM_MODEL")
            response, _ = caller.query_deepseek(prompt=prompt, model=model, max_tokens=10000, temperature=0)
            _kg_debug_print(f"llm_relation_choice_response={response}")
            chosen = _parse_llm_relation_choice(response)

            # 策略A：LLM 返回空关系时，直接退出，不做后续 fallback 打分
            if not _normalize_text(chosen):
                return ""

            if chosen in relation_names:
                return chosen

            chosen_norm = chosen.lower().replace("_", " ")
            for rel in relation_names:
                if rel.lower().replace("_", " ") == chosen_norm:
                    return rel
        except Exception as e:
            _kg_debug_print(f"llm_relation_target_select_failed={e}")

    scored: List[Tuple[float, str]] = []
    for rel in relation_names:
        rel_low = rel.lower().replace("_", " ")
        relation_score = _token_set_ratio(target_pred or rel_text or question, rel_low)

        neighbor_info = rel2_neighbor_info.get(rel, [])[:12]
        tail_label_score = 0.0
        if target_tail and neighbor_info:
            tail_label_score = max(_token_set_ratio(target_tail, x.get("label", "")) for x in neighbor_info)

        type_score = 0.0
        if target_tail and neighbor_info:
            type_score = max(_token_set_ratio(target_tail, x.get("types", "")) for x in neighbor_info)

        head_score = _token_set_ratio(target_head, head_label) if target_head and head_label else 0.0
        total = 0.55 * relation_score + 0.15 * tail_label_score + 0.25 * type_score + 0.05 * head_score
        scored.append((total, rel))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return ""

    best_score, best_rel = scored[0]
    if best_score < 45.0:
        return ""
    return best_rel


def _evidence_text(ev: Dict) -> str:
    head = _normalize_text((ev.get("head_entity_snapshot") or {}).get("label", ev.get("head_entity", "")))
    rel = _normalize_text(ev.get("relation", ""))
    tail = _normalize_text((ev.get("tail_entity_snapshot") or {}).get("label", ev.get("label", "")))
    direction = _normalize_text(ev.get("direction", ""))
    attrs = json.dumps((ev.get("meta") or {}), ensure_ascii=False)
    return f"{head} {rel} {tail} {direction} {attrs}".strip()


def _score_evidence_with_question(question: str, evidence: List[Dict]) -> List[Dict]:
    out = []
    for ev in evidence:
        ev_text = _evidence_text(ev)
        score = _token_set_ratio(question, ev_text)
        ev2 = dict(ev)
        ev2["similarity_score"] = round(float(score), 4)
        out.append(ev2)

    out.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
    return out


# ============================================================
# 4) Query execution
# ============================================================


def _execute_program(program: Dict) -> Dict:
    kg = _get_kg_index()

    question = _normalize_text(program.get("question", ""))
    entities = program.get("entities") or []
    relation_text = program.get("relation_text") or question

    normalized_targets = _build_relation_targets(
        question=question,
        entities=entities,
        relation_text=relation_text,
        parsed_targets=program.get("relation_targets"),
        parsed_target=program.get("relation_target"),
    )

    parser_name = program.get("parser", "llm")

    evidence: List[Dict] = []
    unresolved_mentions: List[str] = []

    for ent in entities[:5]:
        qids = _resolve_entity_qids(kg, ent)
        if not qids:
            fuzzy = _fuzzy_entity_candidates(kg, ent, limit=5)
            if fuzzy:
                qids = [qid for _, qid in fuzzy]
            else:
                unresolved_mentions.append(ent)
                _kg_debug_print(f"entity_unresolved_or_low_conf={ent}, fuzzy_top={(fuzzy[0][0] if fuzzy else 0.0):.2f}")
                continue

        qids = _llm_confirm_entity_qids(kg=kg, question=question, target_entity=ent, candidate_qids=qids)
        if not qids:
            fallback_rows = _fallback_entities_by_property_match(
                kg=kg,
                question=question,
                known_entity_label=ent,
                relation_text=relation_text,
                relation_targets=normalized_targets,
                parser_name=parser_name,
            )
            if fallback_rows:
                evidence.extend(fallback_rows)
                continue

            unresolved_mentions.append(ent)
            continue

        _kg_debug_print(f"entity={ent} resolved_qids={qids}")

        for qid in qids[:2]:
            head_label = kg.qid_to_label(qid) or ent
            out_rel_map = kg.head_rel2triples.get(qid, {}) or {}
            in_rel_map = kg.tail_rel2triples.get(qid, {}) or {}

            out_rel_names = list(out_rel_map.keys())
            in_rel_names = list(in_rel_map.keys())

            selected_targets = [
                t for t in normalized_targets
                if _target_head_matches_entity(t.get("head", ""), head_label)
            ] or normalized_targets

            matched = False

            def _neighbor_info(rel_map, rel_names, inverse: bool = False) -> Dict[str, List[Dict[str, str]]]:
                rel2info: Dict[str, List[Dict[str, str]]] = {}
                for rel_name in rel_names:
                    info = []
                    for t in (rel_map.get(rel_name, []) or [])[:25]:
                        other_qid = t.head_qid if inverse else t.tail_qid
                        other_label = kg.qid_to_label(other_qid) or ""
                        if not other_label:
                            continue
                        other_types = []
                        try:
                            other_types = kg.get_entity_types(other_qid)
                        except Exception:
                            other_types = []
                        info.append(
                            {
                                "qid": other_qid,
                                "label": other_label,
                                "types": ", ".join(other_types) if other_types else "unknown",
                            }
                        )
                    rel2info[rel_name] = info
                return rel2info

            for relation_target in selected_targets[:3]:
                if out_rel_names:
                    resolved_out = _resolve_relation_with_target(
                        question=question,
                        relation_text=relation_text,
                        relation_names=out_rel_names,
                        relation_target=relation_target,
                        head_label=head_label,
                        rel2_neighbor_info=_neighbor_info(out_rel_map, out_rel_names, inverse=False),
                    )
                    if resolved_out:
                        for t in out_rel_map.get(resolved_out, []):
                            ans_qid = t.tail_qid
                            ans_label = kg.qid_to_label(ans_qid) or ans_qid
                            if not ans_label:
                                continue
                            matched = True
                            evidence.append(
                                {
                                    "label": str(ans_label),
                                    "qid": ans_qid,
                                    "meta": dict(t.props or {}),
                                    "head_entity": head_label,
                                    "head_qid": qid,
                                    "relation": resolved_out,
                                    "direction": "outgoing",
                                    "parser": parser_name,
                                    "evidence_source": "matched_triple",
                                    "relation_target": relation_target,
                                    "head_entity_snapshot": _entity_snapshot(kg, qid, head_label),
                                    "tail_entity_snapshot": _entity_snapshot(kg, ans_qid, ans_label),
                                }
                            )

                if in_rel_names:
                    resolved_in = _resolve_relation_with_target(
                        question=question,
                        relation_text=relation_text,
                        relation_names=in_rel_names,
                        relation_target=relation_target,
                        head_label=head_label,
                        rel2_neighbor_info=_neighbor_info(in_rel_map, in_rel_names, inverse=True),
                    )
                    if resolved_in:
                        for t in in_rel_map.get(resolved_in, []):
                            ans_qid = t.head_qid
                            ans_label = kg.qid_to_label(ans_qid) or ans_qid
                            if not ans_label:
                                continue
                            matched = True
                            evidence.append(
                                {
                                    "label": str(ans_label),
                                    "qid": ans_qid,
                                    "meta": dict(t.props or {}),
                                    "head_entity": head_label,
                                    "head_qid": qid,
                                    "relation": resolved_in,
                                    "direction": "incoming",
                                    "parser": parser_name,
                                    "evidence_source": "matched_triple",
                                    "relation_target": relation_target,
                                    "head_entity_snapshot": _entity_snapshot(kg, qid, head_label),
                                    "tail_entity_snapshot": _entity_snapshot(kg, ans_qid, ans_label),
                                }
                            )

            if not matched:
                evidence.append(_make_entity_only_evidence(kg, qid, parser_name, source="entity_only_no_relation_match"))

    # de-dup evidence by (head_qid, relation, direction, qid, meta)
    dedup_evidence: List[Dict] = []
    ev_seen = set()
    for ev in evidence:
        meta_key = tuple(sorted((ev.get("meta") or {}).items()))
        key = (ev.get("head_qid"), ev.get("relation"), ev.get("direction"), ev.get("qid"), meta_key)
        if key in ev_seen:
            continue
        ev_seen.add(key)
        dedup_evidence.append(ev)

    ranked_evidence = _score_evidence_with_question(question, dedup_evidence)[:100]

    return {
        "answer": "",
        "inner_content": [{"content": []}],
        "backend": "mmqa_local_kg",
        "program": program,
        "evidence": ranked_evidence,
        "unresolved_mentions": unresolved_mentions,
    }


def engine_exec_api(program, api_url):
    if isinstance(program, dict):
        normalized_program = dict(program)
        if "question" not in normalized_program:
            normalized_program["question"] = ""
    else:
        normalized_program = semantic_parsing_api(str(program), api_url)

    if normalized_program.get("error") == "LLM解析失败":
        return {
            "answer": "",
            "inner_content": [{"content": []}],
            "backend": "mmqa_local_kg",
            "program": normalized_program,
            "evidence": [],
            "error": "LLM解析失败",
        }

    result = _execute_program(normalized_program)
    _kg_debug_print(f"engine_result={result}")
    return result
