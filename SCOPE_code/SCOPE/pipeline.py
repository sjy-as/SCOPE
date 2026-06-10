"""
End-to-end pipeline.

Stage 1 : verify merged_fusion.json exists.
Stage 2a: call step2_decompose.sq.process_item to get
            sub_q1 / sub_q2 + match1/match2 + route1/route2 in one go.
Stage 2b: for each sub-query, call
            step2_decompose.semantic.parse_semantic
            step2_decompose.operator_plan.build_plan_for_subquery
Stage 3 : for each sub-query, call
            step3_execute.reasoner.MultiSourceReasoner
                .execute_single_subquery (with fallback retry on empty)
Stage 4 : LLM final-answer synthesis (uses reasoner._generate_final_answer).

Per-sub-query trace dict captures: rewritten text, match, route,
semantic_parse, plan, bindings, primary/fallback execution, verified
sub-answers, and ALL LLM calls made on its behalf.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_STEP2 = ROOT / "step2_decompose"
if str(_STEP2) not in sys.path:
    sys.path.insert(0, str(_STEP2))

# ---------------------------------------------------------------------------
# Defaults (override via env or pipeline kwargs)
# ---------------------------------------------------------------------------

# Single 3-source semantic catalog (KG + Table + Doc). The router filters it
# down to whichever sources the run's knowledge base actually enables.
MERGED_FUSION_PATH = ROOT / "step1_oag/fusion/output/semantic_list_3sources.json"
TABLE_API_URL = "http://127.0.0.1:1216/api/search"
DOC_API_URL = "http://127.0.0.1:1215/api/search"
KG_DIR = str(ROOT / "data_sources/KG")
# Step 2 (decomposition / semantic / planning) LLM defaults.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "deepseek-chat")

# Step 3 (executor LLMClient) LLM defaults. Empty string -> let LLMClient
# use its own hard-coded defaults inside step3_execute.
EXEC_LLM_BASE_URL = os.getenv("EXEC_LLM_BASE_URL", "")
EXEC_LLM_API_KEY  = os.getenv("EXEC_LLM_API_KEY", "")
EXEC_LLM_MODEL    = os.getenv("EXEC_LLM_MODEL", "")

K_TABLE = 5

# Canonical ordering of every data source. Each source (kg / table / doc) is
# opt-in via the run's --kb knowledge-base flag.
ALL_SOURCES = ["kg", "table", "doc"]


class PipelineExecutionError(RuntimeError):
    pass


# =====================================================================
# Generic proxy-bypassing LLM POST + JSON parser
# Shared by refusal-detection (pipeline) and LLM-as-judge eval (run).
# =====================================================================

def _post_llm(
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
    sess = requests.Session()
    retry = Retry(
        total=3, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504, 567],
        allowed_methods=["POST"], raise_on_status=False,
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    resp = sess.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=120,
        proxies={"http": None, "https": None},
    )
    if resp.status_code >= 400:
        print(f"[!!! LLM HTTP {resp.status_code}] body={resp.text[:300]}")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json_loose(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        a = s.find("{"); b = s.rfind("}")
        if a >= 0 and b > a:
            try:
                obj = json.loads(s[a:b+1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


# =====================================================================
# Refusal detection: turn "the table does not contain..." answers
# into empty answers, so the executor's fallback retry kicks in.
# =====================================================================

_REFUSAL_HINT_RE = re.compile(
    r"(?:does not contain|not (?:listed|present|found|available|in the table)|"
    r"no (?:information|match(?:ing)?|relevant|data|record)|"
    r"cannot (?:find|determine|answer)|"
    r"unable to (?:find|determine|locate)|"
    r"insufficient (?:information|data)|"
    r"i (?:don't|do not) (?:have|know))",
    re.IGNORECASE,
)


def _looks_like_refusal(answer: str) -> bool:
    if not answer or not isinstance(answer, str):
        return False
    s = answer.strip()
    if len(s) < 25:
        return False
    return bool(_REFUSAL_HINT_RE.search(s))


def _llm_classify_refusals(
    question: str,
    answers: List[str],
    llm_url: str,
    llm_model: str,
    api_key: str,
    trace: Optional[Dict[str, Any]] = None,
) -> List[bool]:
    """Returns a parallel list of bool flags. True = the answer is a
    refusal / no-info string and should be dropped. Cheap regex pre-filter
    avoids LLM calls for short factual answers (names, numbers, etc.)."""
    if not answers:
        return []
    suspicious_idx = [i for i, a in enumerate(answers) if _looks_like_refusal(a)]
    verdicts = [False] * len(answers)
    if not suspicious_idx:
        return verdicts

    items = [{"i": i, "answer": answers[i]} for i in suspicious_idx]
    prompt = (
        "You are evaluating QA outputs. For each Answer, decide whether it is a\n"
        "REAL answer to the Question or a REFUSAL (e.g. 'the data does not contain',\n"
        "'not listed in the table', 'cannot find', 'no information'). A real answer\n"
        "is something like a name / date / number / short list / short factual\n"
        "phrase, even if wrapped in a sentence.\n\n"
        f"Question: {question}\n\n"
        f"Answers (only suspicious ones shown):\n"
        f"{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Output strict JSON only, no prose, no code fences:\n"
        "{\"verdicts\": [{\"i\": <int>, \"refusal\": true|false}, ...]}"
    )
    try:
        raw = _post_llm(prompt, llm_model, llm_url, api_key, max_tokens=10000)
    except Exception as e:
        print(f"  [refusal-check] LLM call failed: {e}")
        return verdicts
    if trace is not None:
        trace.setdefault("llm_calls", []).append({
            "stage": "refusal_check", "ts": time.time(),
            "module": "pipeline.refusal", "model": llm_model,
            "prompt": prompt, "response": raw,
        })
    obj = _parse_json_loose(raw) or {}
    for v in obj.get("verdicts") or []:
        i = v.get("i")
        if isinstance(i, int) and 0 <= i < len(answers):
            verdicts[i] = bool(v.get("refusal"))
    return verdicts


def _drop_refusals(
    question: str,
    op_result,
    llm_url: str,
    llm_model: str,
    api_key: str,
    trace: Dict[str, Any],
):
    """Returns a new OperatorResult with refusal answers removed. Records
    the action under trace['refusal_filter']."""
    from step3_execute.reasoner import OperatorResult
    answers = list((op_result.answers if op_result else []) or [])
    if not answers:
        return op_result
    mask = _llm_classify_refusals(question, answers, llm_url, llm_model, api_key, trace=trace)
    if not any(mask):
        return op_result
    kept_answers = [a for a, r in zip(answers, mask) if not r]
    dropped = [a for a, r in zip(answers, mask) if r]
    trace.setdefault("refusal_filter", []).append({
        "input_answers": answers,
        "refusal_mask": mask,
        "dropped": dropped,
        "kept": kept_answers,
    })
    print(f"  [refusal-check] dropped {len(dropped)}/{len(answers)} answer(s) as refusals")
    return OperatorResult(kept_answers, op_result.evidence if op_result else [])


# =====================================================================
# LLM-call tracing
#
# We monkey-patch the `call_llm` / `_call_llm` symbols of the new step2
# modules and the LLMClient.query_gpt4o so every LLM round-trip during a
# pipeline run gets appended to the active trace's `llm_calls` list.
# =====================================================================

class LLMTracer:
    """Tee LLM calls into the *currently active* trace dict.

    Active trace + stage are stored per-thread so multiple pipelines can run
    concurrently without their LLM calls bleeding into each other's traces.
    """

    def __init__(self) -> None:
        self._tls = threading.local()
        self._installed = False
        self._origs: List[Tuple[Any, str, Callable]] = []

    def bind(self, trace: Optional[Dict[str, Any]], stage: str) -> None:
        self._tls.active = trace
        self._tls.stage = stage

    def _record(self, prompt: str, response: str, extra: Optional[Dict[str, Any]] = None) -> None:
        stage = getattr(self._tls, "stage", "global")
        try:
            from _common.cost_counter import bump_llm as _bump_llm
            _bump_llm(stage=stage)
        except Exception:
            pass
        active = getattr(self._tls, "active", None)
        if active is None:
            return
        entry = {
            "stage": stage,
            "ts": time.time(),
            "prompt": prompt,
            "response": response,
        }
        if extra:
            entry.update(extra)
        active.setdefault("llm_calls", []).append(entry)

    def install(self) -> None:
        if self._installed:
            return

        # step2_decompose.sq.call_llm
        sq_mod = importlib.import_module("step2_decompose.sq")
        orig_sq_call = sq_mod.call_llm

        def _wrap_sq_call(prompt, model, base_url, api_key, temperature=0.0, max_tokens=10000):
            resp = orig_sq_call(prompt, model, base_url, api_key, temperature, max_tokens)
            self._record(prompt, resp, {"module": "step2.sq", "model": model})
            return resp

        sq_mod.call_llm = _wrap_sq_call
        self._origs.append((sq_mod, "call_llm", orig_sq_call))

        # step2_decompose.semantic._call_llm
        sem_mod = importlib.import_module("step2_decompose.semantic")
        orig_sem_call = sem_mod._call_llm

        def _wrap_sem_call(prompt, model, base_url, api_key, temperature=0.0, max_tokens=10000):
            resp = orig_sem_call(prompt, model, base_url, api_key, temperature, max_tokens)
            self._record(prompt, resp, {"module": "step2.semantic", "model": model})
            return resp

        sem_mod._call_llm = _wrap_sem_call
        self._origs.append((sem_mod, "_call_llm", orig_sem_call))

        # step2_decompose.operator_plan._call_llm
        op_mod = importlib.import_module("step2_decompose.operator_plan")
        orig_op_call = op_mod._call_llm

        def _wrap_op_call(prompt, model, base_url, api_key, temperature=0.0, max_tokens=10000):
            resp = orig_op_call(prompt, model, base_url, api_key, temperature, max_tokens)
            self._record(prompt, resp, {"module": "step2.operator_plan", "model": model})
            return resp

        op_mod._call_llm = _wrap_op_call
        self._origs.append((op_mod, "_call_llm", orig_op_call))

        self._installed = True

    def uninstall(self) -> None:
        for mod, name, fn in self._origs:
            setattr(mod, name, fn)
        self._origs.clear()
        self._installed = False

    def wrap_llm_client(self, llm_client) -> None:
        """Replace LLMClient.query_gpt4o with a proxy-bypassing, retrying
        version that also tees the call into the active trace.

        We replace (not just wrap) the original because the stock
        implementation in step3 doesn't pass proxies={"http":None,"https":None}
        and a system proxy on this host is causing SSL EOF errors.
        """
        if llm_client is None or getattr(llm_client, "_pipeline_v2_wrapped", False):
            return
        import requests as _requests
        from requests.adapters import HTTPAdapter as _HTTPAdapter
        from urllib3.util.retry import Retry as _Retry

        tracer = self

        def _wrapped(prompt: str, max_tokens: int = 10000):
            url = llm_client.base_url.rstrip("/")
            if not url.endswith("/chat/completions"):
                url = url + "/chat/completions"
            headers = {
                "Authorization": f"Bearer {llm_client.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": llm_client.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
            sess = _requests.Session()
            retry = _Retry(
                total=3,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504, 567],
                allowed_methods=["POST"],
                raise_on_status=False,
            )
            sess.mount("https://", _HTTPAdapter(max_retries=retry))
            try:
                resp = sess.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120,
                    proxies={"http": None, "https": None},
                )
                if resp.status_code >= 400:
                    print(f"[!!! step3 LLM HTTP {resp.status_code}] body={resp.text[:300]}")
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tracer._record(prompt, content, {"module": "step3.llm_client",
                                                 "model": llm_client.model})
                return content, usage
            except Exception as e:
                print(f"[LLM Error] step3 query_gpt4o failed: {e}")
                tracer._record(prompt, f"[LLM Error] {e}",
                               {"module": "step3.llm_client", "error": True})
                return "", {}

        llm_client.query_gpt4o = _wrapped
        llm_client._pipeline_v2_wrapped = True


TRACER = LLMTracer()


# =====================================================================
# Stage 1 — fusion bundle
# =====================================================================

def _filter_catalog_by_sources(
    catalog: Dict[str, Any], enabled_sources: List[str],
) -> Dict[str, Any]:
    """Restrict the semantic catalog to the live knowledge base.

    For every concept, keep only the source entries (KG / Table / Doc) that
    belong to `enabled_sources`. Concepts left with no live source are dropped
    entirely, so the semantic router never sees content for a disabled source.
    """
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
    """Load the semantic catalog and filter it to the live knowledge base.

    `enabled_sources` is the subset of {kg, table, doc} the run enabled (each
    source is opt-in). When given, the catalog is trimmed so routing and
    concept-context lookups only see those sources. None means keep the full
    3-source catalog.
    """
    print("\n" + "=" * 70)
    print("STAGE 1: semantic catalog check")
    print("=" * 70)

    if not MERGED_FUSION_PATH.exists():
        raise FileNotFoundError(f"[Stage1] Missing: {MERGED_FUSION_PATH}")
    print(f"  Found: {MERGED_FUSION_PATH}")

    with MERGED_FUSION_PATH.open("r", encoding="utf-8") as f:
        catalog = json.load(f)

    if not isinstance(catalog, dict):
        raise ValueError(f"[Stage1] Expected dict in {MERGED_FUSION_PATH}, got {type(catalog).__name__}")

    if enabled_sources:
        before = len(catalog)
        catalog = _filter_catalog_by_sources(catalog, enabled_sources)
        print(f"  knowledge base   : {', '.join(enabled_sources)}")
        print(f"  catalog filtered : {before} -> {len(catalog)} concepts")

    # `bundle`  = raw catalog, consumed by route.source_routing.
    # `catalog` = per-concept context lookup keyed by concept name, the
    #             structure parse_semantic / operator_plan expect for the
    #             matched-concept context. Routing emits `matched_concept`,
    #             which indexes directly into this lookup.
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
# Executor
# =====================================================================

def build_executor(enabled_sources: Optional[List[str]] = None):
    """Wire the execution engine for the live knowledge base.

    `enabled_sources` is the subset of {kg, table, doc} the run enabled. Each
    source is wired only when listed. None means all three.
    """
    from step3_execute.knowledge_sources.query_llm import LLMClient
    from step3_execute.knowledge_sources.query_table import TableSource
    from step3_execute.knowledge_sources.query_doc import DocSource
    from step3_execute.knowledge_sources.query_kg import KGSource
    from step3_execute.service.KG.kg_retriever import KGRetriever
    from step3_execute.reasoner import MultiSourceReasoner

    try:
        llm_kwargs: Dict[str, Any] = {}
        if EXEC_LLM_API_KEY:
            llm_kwargs["api_key"] = EXEC_LLM_API_KEY
        if EXEC_LLM_BASE_URL:
            llm_kwargs["base_url"] = EXEC_LLM_BASE_URL
        if EXEC_LLM_MODEL:
            llm_kwargs["model"] = EXEC_LLM_MODEL
        llm = LLMClient(**llm_kwargs) if llm_kwargs else LLMClient()
        # Apply overrides explicitly in case LLMClient.__init__ ignores them.
        if EXEC_LLM_API_KEY:
            llm.api_key = EXEC_LLM_API_KEY
        if EXEC_LLM_BASE_URL:
            llm.base_url = EXEC_LLM_BASE_URL
        if EXEC_LLM_MODEL:
            llm.model = EXEC_LLM_MODEL
        TRACER.wrap_llm_client(llm)
        print(f"  LLMClient initialized: model={llm.model} url={llm.base_url}")
    except Exception as e:
        print(f"  LLMClient failed: {e}")
        llm = None

    # Each source (kg / table / doc) is wired only when the knowledge base
    # enables it. A source left as None is invisible to the reasoner.
    sources = {str(s).strip().lower() for s in (enabled_sources or ALL_SOURCES)}
    table_source = None
    doc_source = None
    kg_source = None
    if "table" in sources:
        table_source = TableSource(
            retriever_api_url=TABLE_API_URL,
            k=K_TABLE,
            llm=llm,
            verbose=True,
        )
        print(f"  TableSource initialized: {TABLE_API_URL}")
    if "doc" in sources:
        doc_source = DocSource(
            retriever_api_url=DOC_API_URL,
            k=K_TABLE,
            llm=llm,
            verbose=True,
        )
        print(f"  DocSource initialized: {DOC_API_URL}")
    if "kg" in sources:
        try:
            kg_retriever = KGRetriever(kg_dir=KG_DIR, llm=llm)
            kg_source = KGSource(kg_retriever=kg_retriever, llm=llm, verbose=True)
            print(f"  KGSource initialized: {KG_DIR}")
        except Exception as e:
            print(f"  KGSource failed: {e}")
            kg_source = None

    return MultiSourceReasoner(
        table_source=table_source,
        kg_source=kg_source,
        doc_source=doc_source,
        llm=llm,
    )


# =====================================================================
# Stage 2a — decompose + match + rewrite + route (full question)
# =====================================================================

def stage2a_match_route(
    question: str,
    fusion_bundle: Dict[str, Any],
    trace: Optional[Dict[str, Any]] = None,
    routing_mode: str = "graph",
    profiles_path: Optional[str] = None,
    prompt_version: str = "v2",
    enabled_sources: Optional[List[str]] = None,
    no_decompose: bool = False,
) -> Dict[str, Any]:
    """Run Stage 2a as decomposition followed by per-subquery routing.

    The current step2 API is split in two pieces:
    - `sq.decompose_question(...)` produces `sub_q1` / `sub_q2`
    - `route.source_routing(...)` routes each subquery using the semantic bundle

    routing_mode dispatches to the original graph router or to one of the
    baseline routers ("atomr" / "deepsieve"). prompt_version selects the
    v1 (legacy) or v2 (entity-attribute-aware) prompts for routing_mode="graph".
    """
    from step2_decompose import route as route_mod
    from step2_decompose import route_baselines as route_baseline_mod
    from step2_decompose import sq as sq_mod

    print("\n" + "=" * 70)
    print(f"STAGE 2a: decompose / match / rewrite / route  "
          f"(routing_mode={routing_mode}, prompt_version={prompt_version})")
    print("=" * 70)

    if no_decompose:
        sub_q1 = question
        sub_q2 = ""
        decomposition = {"subqueries": [{"query": question}], "reasoning": "no_decompose_ablation"}
    else:
        TRACER.bind(trace, "step2.sq")
        decomposition = sq_mod.decompose_question(
            question,
            model=LLM_MODEL,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
        TRACER.bind(None, "global")
        subqueries = decomposition.get("subqueries") or []
        sub_q1 = ((subqueries[0] or {}).get("query") if len(subqueries) > 0 else question) or question
        sub_q2 = ((subqueries[1] or {}).get("query") if len(subqueries) > 1 else question) or question

    def _route_subquery(subquery: str) -> Dict[str, Any]:
        if routing_mode == "graph":
            return route_mod.source_routing(
                matched_kind="none",
                matched_item=None,
                bundle=fusion_bundle,
                query=subquery,
                model=LLM_MODEL,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                verbose=False,
                ref_type="entity",
                profiles_path=profiles_path,
                prompt_version=prompt_version,
                enabled_sources=enabled_sources,
            )
        if routing_mode == "atomr":
            return route_baseline_mod.route_atomr_few_shot(
                query=subquery,
                model=LLM_MODEL,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                verbose=False,
                enabled_sources=enabled_sources,
            )
        if routing_mode == "deepsieve":
            if not profiles_path:
                raise ValueError("deepsieve routing requires profiles_path")
            return route_baseline_mod.route_deepsieve(
                query=subquery,
                profiles_path=profiles_path,
                enabled_sources=enabled_sources or ["kg", "table", "doc"],
                model=LLM_MODEL,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                verbose=False,
            )
        raise ValueError(f"Unsupported routing mode: {routing_mode}")

    route1 = _route_subquery(sub_q1)
    route2 = _route_subquery(sub_q2)

    out = {
        "index": -1,
        "type": None,
        "question": question,
        "sub_q1": sub_q1,
        "sub_q2": sub_q2,
        "reasoning": decomposition.get("reasoning", ""),
        "rewrite_reasoning": decomposition.get("reasoning", ""),
        "match1": None,
        "match2": None,
        "route1": route1,
        "route2": route2,
        "decomposition": decomposition,
    }

    print(f"  sub_q1 : {out.get('sub_q1')}")
    print(f"  route1 : primary={out['route1'].get('primary_source')} "
          f"fallback={out['route1'].get('fallback_source')} ")
    print(f"  sub_q2 : {out.get('sub_q2')}")
    print(f"  route2 : primary={out['route2'].get('primary_source')} "
          f"fallback={out['route2'].get('fallback_source')} ")
    return out


# =====================================================================
# Stage 2b — per-SQ semantic + plan
# =====================================================================

def stage2b_semantic_plan(
    sq_id: str,
    sq_text: str,
    route: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    ref_bindings: Dict[str, Any],
    trace: Dict[str, Any],
    include_plan_semlist_metadata: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Returns (semantic, plan_obj). Plan_obj is the dict returned by
    operator_plan.build_plan_for_subquery (has .plan / .bindings)."""
    from step2_decompose.semantic import parse_semantic
    from step2_decompose.operator_plan import build_plan_for_subquery

    matched_concept = (route.get("matched_concept") or "").strip()
    matched_info = catalog.get(matched_concept) if matched_concept else None

    # -- semantic ----------------------------------------------------------
    TRACER.bind(trace, "step2.semantic")
    semantic = parse_semantic(
        question=sq_text,
        route=route,
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        catalog_lookup=catalog,
    )
    print(f"    Semantic -> Entities    : {semantic.get('Entities')}")
    print(f"               Conditions  : {semantic.get('Conditions')}")
    print(f"               Return Type : {semantic.get('Return Type')}")
    print(f"               need_math   : {semantic.get('need_math')}")

    # -- plan --------------------------------------------------------------
    TRACER.bind(trace, "step2.plan_v2")
    sq_for_plan = {
        "id": sq_id,
        "text": sq_text,
        "execution": {
            "primary_source": route.get("primary_source"),
            "fallback_source": route.get("fallback_source"),
            "matched_concept": matched_concept or None,
        },
        "semantic_parse": semantic,
    }
    plan_obj = build_plan_for_subquery(
        sq=sq_for_plan,
        ref_bindings=ref_bindings,
        matched_info=matched_info,
        llm_url=LLM_BASE_URL,
        llm_model=LLM_MODEL,
        api_key=LLM_API_KEY,
        include_semlist_metadata=include_plan_semlist_metadata,
    )
    TRACER.bind(None, "global")

    p = plan_obj.get("plan") or {}
    steps = p.get("steps") if p.get("type") != "map" else p.get("body")
    print(f"    Plan     -> type={p.get('type')} final_ref={p.get('final_ref')} steps={len(steps or [])}")
    for i, st in enumerate((steps or []), 1):
        print(f"      Step {i}: [{st.get('op')}] -> out={st.get('out')}")

    return semantic, plan_obj


# =====================================================================
# Stage 3 — execute one sub-query (with fallback retry)
# =====================================================================

def stage3_execute(
    reasoner,
    sq_id: str,
    sq_text: str,
    route: Dict[str, Any],
    plan_obj: Dict[str, Any],
    prior_results: Dict[str, Any],
    trace: Dict[str, Any],
    catalog: Dict[str, Dict[str, Any]],
    no_fallback: bool = False,
    include_plan_semlist_metadata: bool = True,
):
    """Execute a single sub-query. On empty/verified-empty answers, retry
    with the fallback source and a freshly built plan. Returns (op_result,
    sq_for_reasoner, used_route)."""
    from step3_execute.reasoner import OperatorResult  # noqa: F401
    from step2_decompose.semantic import parse_semantic  # for fallback re-plan
    from step2_decompose.operator_plan import build_plan_for_subquery

    def _build_sq_for_reasoner(plan_obj_local: Dict[str, Any], route_local: Dict[str, Any]):
        plan = plan_obj_local.get("plan") or {}
        bindings = plan_obj_local.get("bindings") or {}
        sq_dict = {
            "id": sq_id,
            "text": sq_text,
            "route": {
                "primary_source": route_local.get("primary_source"),
                "fallback_source": route_local.get("fallback_source"),
                "matched_concept": route_local.get("matched_concept"),
            },
            "plan": plan,
            "bindings": bindings,
        }
        return sq_dict

    sq_for_reasoner = _build_sq_for_reasoner(plan_obj, route)

    print("\n" + "-" * 70)
    print(f"STAGE 3 [{sq_id}]: execute (primary={route.get('primary_source')})")
    print("-" * 70)
    TRACER.bind(trace, f"step3.exec.primary[{route.get('primary_source')}]")
    op_result = reasoner.execute_single_subquery(
        question=sq_text,
        sq=sq_for_reasoner,
        prior_results=prior_results,
        verify_final=True,
    )
    TRACER.bind(None, "global")

    # Drop "polite-refusal" answers (e.g. "the table does not contain ...")
    # so the existing fallback path kicks in for those too.
    op_result = _drop_refusals(sq_text, op_result, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, trace)

    primary_attempt = {
        "source": route.get("primary_source"),
        "answers": list(op_result.answers) if op_result and op_result.answers else [],
        "evidence_count": len(op_result.evidence) if op_result and op_result.evidence else 0,
        "verified": True,
    }
    trace.setdefault("execution", {})["primary_attempt"] = primary_attempt

    # Fallback retry
    used_route = route
    if (not no_fallback) and ((not op_result) or (not op_result.answers)):
        fb_src = route.get("fallback_source")
        prim_src = route.get("primary_source")
        if fb_src and fb_src != prim_src:
            print(f"  [{sq_id}] primary '{prim_src}' empty after verify — retrying with fallback '{fb_src}'")
            used_route = dict(route)
            used_route["primary_source"] = fb_src
            used_route["fallback_source"] = prim_src

            # Re-plan with the fallback as primary so v2 picks the right
            # Search anchor strategy.
            TRACER.bind(trace, "step2.plan_v2.fallback")
            matched_concept = (used_route.get("matched_concept") or "").strip()
            matched_info = catalog.get(matched_concept) if matched_concept else None
            plan_obj_fb = build_plan_for_subquery(
                sq={
                    "id": sq_id,
                    "text": sq_text,
                    "execution": {
                        "primary_source": used_route.get("primary_source"),
                        "fallback_source": used_route.get("fallback_source"),
                        "matched_concept": matched_concept or None,
                    },
                    "semantic_parse": trace.get("semantic_parse") or {},
                },
                ref_bindings=plan_obj.get("bindings") or {},
                matched_info=matched_info,
                llm_url=LLM_BASE_URL,
                llm_model=LLM_MODEL,
                api_key=LLM_API_KEY,
                include_semlist_metadata=include_plan_semlist_metadata,
            )
            TRACER.bind(None, "global")

            sq_for_reasoner = _build_sq_for_reasoner(plan_obj_fb, used_route)
            TRACER.bind(trace, f"step3.exec.fallback[{fb_src}]")
            op_result = reasoner.execute_single_subquery(
                question=sq_text,
                sq=sq_for_reasoner,
                prior_results=prior_results,
                verify_final=True,
            )
            TRACER.bind(None, "global")

            op_result = _drop_refusals(sq_text, op_result, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY, trace)

            trace["execution"]["fallback_attempt"] = {
                "source": fb_src,
                "answers": list(op_result.answers) if op_result and op_result.answers else [],
                "evidence_count": len(op_result.evidence) if op_result and op_result.evidence else 0,
                "verified": True,
                "fallback_plan": plan_obj_fb.get("plan"),
            }
            trace["plan_used"] = plan_obj_fb.get("plan")
        else:
            trace["execution"]["fallback_attempt"] = None
    else:
        trace["execution"]["fallback_attempt"] = None

    return op_result, sq_for_reasoner, used_route


# =====================================================================
# Top-level: run one question
# =====================================================================

def run_pipeline_one(
    question: str,
    fusion_state: Dict[str, Any],
    reasoner,
    ground_truth: Optional[Dict[str, Any]] = None,
    routing_mode: str = "graph",
    profiles_path: Optional[str] = None,
    prompt_version: str = "v2",
    enabled_sources: Optional[List[str]] = None,
    no_decompose: bool = False,
    no_fallback: bool = False,
    include_plan_semlist_metadata: bool = True,
) -> Dict[str, Any]:
    """Run pipeline on a single question. Returns a result dict
    containing per-sq traces, sub-answers, and the final answer."""
    from step3_execute.reasoner import OperatorResult

    print("\n" + "#" * 70)
    print(f"PIPELINE v2: {question}   "
          f"(routing_mode={routing_mode}, prompt_version={prompt_version})")
    print("#" * 70)

    bundle = fusion_state["bundle"]
    catalog = fusion_state["catalog"]

    if not TRACER._installed:
        TRACER.install()

    # ---- Stage 2a (one shared trace bucket for the decompose phase) ----
    decomp_trace: Dict[str, Any] = {"llm_calls": []}
    s2a = stage2a_match_route(
        question,
        bundle,
        trace=decomp_trace,
        routing_mode=routing_mode,
        profiles_path=profiles_path,
        prompt_version=prompt_version,
        enabled_sources=enabled_sources,
        no_decompose=no_decompose,
    )

    sub_qs = [
        {
            "sq_id": "sq1",
            "text": s2a.get("sub_q1") or "",
            "match": s2a.get("match1") or {},
            "route": s2a.get("route1") or {},
        },
        {
            "sq_id": "sq2",
            "text": s2a.get("sub_q2") or "",
            "match": s2a.get("match2") or {},
            "route": s2a.get("route2") or {},
        },
    ]

    sq_traces: List[Dict[str, Any]] = []
    sq_results_by_id: Dict[str, Any] = {}
    ref_bindings: Dict[str, Any] = {}
    sq_for_reasoner_list: List[Dict[str, Any]] = []

    for i, item in enumerate(sub_qs, start=1):
        sq_id = item["sq_id"]
        sq_text = item["text"]
        route = item["route"]
        match = item["match"]

        if not sq_text:
            print(f"\n  [{sq_id}] empty text, skipping")
            continue

        trace: Dict[str, Any] = {
            "sq_id": sq_id,
            "question": question,
            "rewritten_text": sq_text,
            "match": match,
            "route": route,
            "rewrite_reasoning": s2a.get("rewrite_reasoning"),
            "decompose_llm_calls": list(decomp_trace.get("llm_calls", [])) if i == 1 else [],
            "llm_calls": [],
        }

        print("\n" + "-" * 70)
        print(f"STAGE 2b [{sq_id}]: semantic + plan_v2")
        print("-" * 70)

        semantic, plan_obj = stage2b_semantic_plan(
            sq_id=sq_id,
            sq_text=sq_text,
            route=route,
            catalog=catalog,
            ref_bindings=ref_bindings,
            trace=trace,
            include_plan_semlist_metadata=include_plan_semlist_metadata,
        )
        trace["semantic_parse"] = semantic
        trace["plan"] = plan_obj.get("plan")
        trace["bindings"] = plan_obj.get("bindings")

        # ---- Execute ----
        op_result, sq_for_reasoner, used_route = stage3_execute(
            reasoner=reasoner,
            sq_id=sq_id,
            sq_text=sq_text,
            route=route,
            plan_obj=plan_obj,
            prior_results=sq_results_by_id,
            trace=trace,
            catalog=catalog,
            no_fallback=no_fallback,
            include_plan_semlist_metadata=include_plan_semlist_metadata,
        )

        if not op_result or not op_result.answers:
            trace["final_sub_answers"] = []
            trace["status"] = "no_answers_after_fallback"
            sq_traces.append(trace)
            print(f"\n  [{sq_id}] no verified answers after fallback; pipeline halted at this sq.")
            # Still return partial results so the trace files get written.
            return {
                "question": question,
                "sq_traces": sq_traces,
                "decompose": {
                    "sub_q1": s2a.get("sub_q1"),
                    "sub_q2": s2a.get("sub_q2"),
                    "rewrite_reasoning": s2a.get("rewrite_reasoning"),
                    "match1": s2a.get("match1"),
                    "match2": s2a.get("match2"),
                    "route1": s2a.get("route1"),
                    "route2": s2a.get("route2"),
                    "decompose_llm_calls": decomp_trace.get("llm_calls", []),
                },
                "sq_answers": {sid: list(r.answers) for sid, r in sq_results_by_id.items()},
                "final_answer": "",
                "final_synthesis_llm_calls": [],
                "halted_at": sq_id,
            }

        trace["final_sub_answers"] = list(op_result.answers)
        trace["status"] = "ok"
        sq_traces.append(trace)
        sq_results_by_id[sq_id] = op_result
        sq_for_reasoner_list.append(sq_for_reasoner)
        ref_bindings[f"ref_{i}"] = {
            "from_subquery": sq_id,
            "final_ref": (plan_obj.get("plan") or {}).get("final_ref", "f_pre"),
        }

    # ---- Stage 4: final synthesis ----
    print("\n" + "=" * 70)
    print("STAGE 4: final answer")
    print("=" * 70)
    final_answer = ""
    final_trace: Dict[str, Any] = {"llm_calls": []}
    if sq_for_reasoner_list:
        TRACER.bind(final_trace, "step4.final_answer")
        final_answer = reasoner._generate_final_answer(
            question, sq_for_reasoner_list, sq_results_by_id
        ) or ""
        TRACER.bind(None, "global")
    print(f"  final_answer: {final_answer}")

    final_res = OperatorResult([final_answer] if final_answer else [], [])
    sq_results_by_id["__final__"] = final_res

    # Ground-truth comparison
    if ground_truth:
        print("\n" + "-" * 70)
        print("GROUND TRUTH")
        print("-" * 70)
        sq_id_map = {"q1": "sq1", "q2": "sq2"}
        for gt_key in sorted(k for k in ground_truth.keys() if k.startswith("q")):
            exec_id = sq_id_map.get(gt_key)
            if not exec_id:
                continue
            gt_entry = ground_truth.get(gt_key) or {}
            gt_ans = gt_entry.get("answer") or gt_entry.get("answers") or []
            if isinstance(gt_ans, str):
                gt_ans = [gt_ans]
            res = sq_results_by_id.get(exec_id)
            got = list(res.answers) if res else []
            gset, gotset = set(gt_ans), set(got)
            if gset == gotset:
                tag = "EXACT"
            elif gset & gotset:
                tag = f"PARTIAL hits={gset & gotset}"
            else:
                tag = "MISS"
            print(f"  [{exec_id}] expected={gt_ans}  got={got[:10]}  -> {tag}")

    return {
        "question": question,
        "sq_traces": sq_traces,
        "decompose": {
            "sub_q1": s2a.get("sub_q1"),
            "sub_q2": s2a.get("sub_q2"),
            "rewrite_reasoning": s2a.get("rewrite_reasoning"),
            "match1": s2a.get("match1"),
            "match2": s2a.get("match2"),
            "route1": s2a.get("route1"),
            "route2": s2a.get("route2"),
            "decompose_llm_calls": decomp_trace.get("llm_calls", []),
        },
        "sq_answers": {sid: list(r.answers) for sid, r in sq_results_by_id.items()
                       if sid != "__final__"},
        "final_answer": final_answer,
        "final_synthesis_llm_calls": final_trace.get("llm_calls", []),
    }
