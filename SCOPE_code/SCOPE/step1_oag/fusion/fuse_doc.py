"""Fuse the Doc semantic list into the KG+Table semantic catalog.

Pipeline (mirrors the original two-step design — fusion.py merges by name,
generate_concept_descriptions.py rewrites descriptions):

  1. MERGE  : merge `doc_semantic_list.cleaned.json` into `semantic_list.json`
              by normalized concept name (same rule as fusion.py::_merge_by_name).
              * same concept name  -> add a "Doc" entry under that concept's `sources`
              * new concept name   -> add the whole concept as a standalone entry
              Result is the three-source (KG / Table / Doc) semantic catalog.

  2. DESCRIBE : rewrite every concept's descriptions with an LLM:
              * `overall_description`        — broad, source-agnostic
              * `sources.<S>.description`    — grounded in that source's
                                               elements + data_examples, a bit
                                               more concrete than the overall one
              `elements` and `data_examples` are never modified.

Outputs:
  output/semantic_list_3sources.merged.json   (after step 1)
  output/semantic_list_3sources.json          (after step 2, final catalog)

Examples:
    # full run (merge + LLM rewrite)
    python3 fuse_doc.py

    # only merge, skip the LLM
    python3 fuse_doc.py --merge-only

    # rewrite only the first 3 concepts (smoke test)
    python3 fuse_doc.py --limit 3
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ── 默认配置 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent                       # .../fusion
_STEP1 = _HERE.parent                                          # .../step1_oag

DEFAULT_SEMANTIC_LIST = _HERE / "output" / "semantic_list.json"
DEFAULT_DOC_LIST = _STEP1 / "extract" / "output" / "doc_semantic_list.cleaned.json"
DEFAULT_MERGED_OUT = _HERE / "output" / "semantic_list_3sources.merged.json"
DEFAULT_FINAL_OUT = _HERE / "output" / "semantic_list_3sources.json"

DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_KEY = "sk-FGHIXlyPYpUGzovjKzG7UYv7J7vfJYevqKsEf8o3EryiuiCA"


# ── 基础工具 ──────────────────────────────────────────────────────────────
def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(obj).__name__}")
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_key(name: str) -> str:
    """Same normalization fusion.py uses to decide whether two concepts match."""
    name = str(name).strip().lower()
    name = re.sub(r"[\s\-\/]+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "", name)
    return name


def _truncate(text: str, max_len: int = 300) -> str:
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _canonicalize_sources(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Reorder every source object's keys to description -> elements -> data_examples
    (any extra keys kept after), so Doc sources line up with KG/Table."""
    canon = ["description", "elements", "data_examples"]
    for concept in catalog.values():
        if not isinstance(concept, dict):
            continue
        for src_name, src_obj in (concept.get("sources") or {}).items():
            if not isinstance(src_obj, dict):
                continue
            ordered = {k: src_obj[k] for k in canon if k in src_obj}
            for k, v in src_obj.items():
                if k not in ordered:
                    ordered[k] = v
            concept["sources"][src_name] = ordered
    return catalog


# ── 步骤 1：按概念名合并 ──────────────────────────────────────────────────
def merge_doc(semantic_list: Dict[str, Any], doc_list: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the Doc semantic list into the KG+Table catalog by concept name."""
    merged = copy.deepcopy(semantic_list)
    # normalized name -> existing top-level key
    norm_index: Dict[str, str] = {normalize_key(k): k for k in merged}

    matched: List[str] = []
    added: List[str] = []

    for doc_key, doc_concept in doc_list.items():
        if not isinstance(doc_concept, dict):
            continue
        doc_sources = doc_concept.get("sources") or {}
        nk = normalize_key(doc_key)

        if nk in norm_index:
            # 相同概念 -> 在已有概念的 sources 里补充 Doc
            target_key = norm_index[nk]
            target = merged[target_key]
            target.setdefault("sources", {})
            for src_name, src_obj in doc_sources.items():
                target["sources"][src_name] = copy.deepcopy(src_obj)
            matched.append(f"{doc_key} -> {target_key}")
        else:
            # 新概念 -> 单拎一个
            merged[doc_key] = copy.deepcopy(doc_concept)
            norm_index[nk] = doc_key
            added.append(doc_key)

    print(f"[merge] doc concepts: {len(doc_list)}")
    print(f"[merge] matched into existing concepts ({len(matched)}):")
    for m in matched:
        print(f"          {m}")
    print(f"[merge] added as new standalone concepts ({len(added)}):")
    for a in added:
        print(f"          {a}")
    print(f"[merge] final catalog size: {len(merged)} concepts")
    return merged


# ── 步骤 2：用 LLM 重写描述 ───────────────────────────────────────────────
def _call_llm(prompt: str, base_url: str, model: str, api_key: str,
              max_tokens: int = 10000, timeout: int = 90, max_retries: int = 4) -> str:
    """OpenAI-compatible chat call. Bypasses the local proxy (it breaks TLS)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a careful data curation assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers=headers, json=payload, timeout=timeout,
                proxies={"http": None, "https": None},
            )
            resp.raise_for_status()
            return (resp.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2)
    print(f"  [LLM Error] failed after {max_retries} attempts: {last_err}")
    return ""


def _parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start:end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


# 各来源的写作侧重，喂给 LLM 当提示
_SOURCE_HINTS = {
    "KG": "structured knowledge-graph facts: entity IDs, typed attributes, and time/qualifier fields.",
    "Table": "tabular evidence: page/section context, column headers, and row-level records.",
    "Doc": "free-text document passages: titles and natural-language descriptions of the entity.",
}


def _build_prompt(concept_name: str, concept_obj: Dict[str, Any]) -> str:
    overall = concept_obj.get("overall_description") or concept_obj.get("description") or ""
    sources = concept_obj.get("sources") or {}

    source_blocks: List[str] = []
    for src_name, src_obj in sources.items():
        src_obj = src_obj or {}
        elements = src_obj.get("elements") or []
        examples = [_truncate(e) for e in (src_obj.get("data_examples") or [])[:3]]
        hint = _SOURCE_HINTS.get(src_name, "source-specific evidence.")
        source_blocks.append(
            f"Source: {src_name}\n"
            f"  This source carries {hint}\n"
            f"  elements (key fields/columns/slots): {json.dumps(elements, ensure_ascii=False)}\n"
            f"  data_examples: {json.dumps(examples, ensure_ascii=False)}"
        )
    sources_text = "\n\n".join(source_blocks)
    source_names = list(sources.keys())

    return (
        "You are rewriting concept descriptions for a three-source (KG / Table / Doc)\n"
        "fusion catalog. For one concept, write a broad overall description and one\n"
        "description per source.\n\n"
        "Hard rules:\n"
        "1) Do NOT invent facts not supported by the elements / data_examples below.\n"
        "2) `overall_description` must be BROAD and source-agnostic — a general definition\n"
        "   of the concept, wider than any single source.\n"
        "3) Each source description must be GROUNDED in that source's own elements and\n"
        "   data_examples, and be slightly MORE DETAILED and concrete than the overall one\n"
        "   (mention the key elements and how the concept is represented in that source).\n"
        "4) Keep each description to 1-3 sentences. Do not list raw IDs.\n"
        "5) Return STRICT JSON only, no markdown.\n\n"
        f"Concept: {concept_name}\n"
        f"Current overall description: {overall}\n\n"
        f"Sources:\n{sources_text}\n\n"
        "Return exactly this schema (one key under \"sources\" for EACH source listed above, "
        f"i.e. {json.dumps(source_names, ensure_ascii=False)}):\n"
        "{\n"
        '  "overall_description": "...",\n'
        '  "sources": {\n'
        '    "<SourceName>": "<source-specific description>"\n'
        "  }\n"
        "}"
    )


def rewrite_descriptions(merged: Dict[str, Any], base_url: str, model: str, api_key: str,
                         limit: Optional[int] = None, sleep_s: float = 0.0) -> Dict[str, Any]:
    """Rewrite overall + per-source descriptions for every concept via the LLM.

    `elements` and `data_examples` are preserved verbatim; only `description`
    strings change. On any LLM failure the original text is kept.
    """
    out = copy.deepcopy(merged)
    concepts = list(out.items())
    if limit is not None:
        concepts = concepts[:limit]

    for idx, (concept_key, concept_obj) in enumerate(concepts, start=1):
        if not isinstance(concept_obj, dict):
            continue
        concept_name = concept_obj.get("concept") or concept_key
        prompt = _build_prompt(concept_name, concept_obj)
        raw = _call_llm(prompt, base_url=base_url, model=model, api_key=api_key)
        parsed = _parse_json_loose(raw)

        if not parsed:
            print(f"[{idx}/{len(concepts)}] {concept_key}: LLM failed — kept original descriptions")
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue

        new_overall = str(parsed.get("overall_description") or "").strip()
        if new_overall:
            concept_obj["overall_description"] = new_overall

        new_sources = parsed.get("sources") or {}
        rewritten = []
        for src_name, src_obj in (concept_obj.get("sources") or {}).items():
            if not isinstance(src_obj, dict):
                continue
            new_desc = str(new_sources.get(src_name) or "").strip()
            if new_desc:
                src_obj["description"] = new_desc
                rewritten.append(src_name)

        print(f"[{idx}/{len(concepts)}] {concept_key}: rewrote overall + sources {rewritten}")
        if sleep_s > 0:
            time.sleep(sleep_s)

    return out


# ── 主入口 ────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fuse Doc semantic list into the KG+Table catalog and rewrite descriptions.")
    p.add_argument("--semantic-list", default=str(DEFAULT_SEMANTIC_LIST), help="KG+Table semantic_list.json")
    p.add_argument("--doc-list", default=str(DEFAULT_DOC_LIST), help="doc_semantic_list.cleaned.json")
    p.add_argument("--merged-out", default=str(DEFAULT_MERGED_OUT), help="intermediate merged catalog output")
    p.add_argument("--final-out", default=str(DEFAULT_FINAL_OUT), help="final catalog output (after rewrite)")
    p.add_argument("--merge-only", action="store_true", help="only merge, skip the LLM rewrite")
    p.add_argument("--limit", type=int, default=None, help="rewrite only the first N concepts (smoke test)")
    p.add_argument("--sleep-s", type=float, default=0.0, help="optional sleep between LLM calls")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM base URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    p.add_argument("--api-key", default=DEFAULT_API_KEY, help="LLM API key")
    return p


def main() -> None:
    args = build_parser().parse_args()
    print("=" * 80)
    print("FUSE DOC SEMANTIC LIST INTO KG+TABLE CATALOG")
    print("=" * 80)

    semantic_list = _load_json(Path(args.semantic_list))
    doc_list = _load_json(Path(args.doc_list))
    print(f"[load] semantic_list: {len(semantic_list)} concepts ({args.semantic_list})")
    print(f"[load] doc_list:      {len(doc_list)} concepts ({args.doc_list})")
    print("-" * 80)

    # 步骤 1：合并
    merged = merge_doc(semantic_list, doc_list)
    _canonicalize_sources(merged)
    _write_json(Path(args.merged_out), merged)
    print(f"[merge] wrote merged catalog -> {args.merged_out}")
    print("-" * 80)

    if args.merge_only:
        print("[done] --merge-only set, skipping LLM description rewrite.")
        return

    # 步骤 2：重写描述
    print("[describe] rewriting descriptions with LLM...")
    final = rewrite_descriptions(
        merged, base_url=args.base_url, model=args.model, api_key=args.api_key,
        limit=args.limit, sleep_s=args.sleep_s,
    )
    _canonicalize_sources(final)
    _write_json(Path(args.final_out), final)
    print("-" * 80)
    print(f"[done] wrote final three-source catalog -> {args.final_out}")


if __name__ == "__main__":
    main()
