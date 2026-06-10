"""Generate concept descriptions for merged fusion concepts using an LLM.

Given `output/merged_fusion.json`, this script rewrites each concept into a
richer representation with:

- `overall_description`: a broad, source-agnostic description of the concept
- `sources.KG.description`: KG-specific description grounded in KG fields/samples
- `sources.Table.description`: table-specific description grounded in table headers,
  page/section metadata, and sample rows
- `sources.*.elements`: the key fields / attributes / headers used to express the concept
- `sources.*.data_examples`: compact evidence snippets copied from the input

The script is intentionally conservative:
- It never invents new facts.
- It falls back to the original description and source samples when the LLM omits data.
- It writes a JSON file that is easy to consume downstream.

Example:

    export LLM_BASE_URL="https://api.chatanywhere.tech/v1"
    export LLM_MODEL="deepseek-chat"
    export LLM_API_KEY="sk-..."

    python3 generate_concept_descriptions.py \
      --input output/merged_fusion.json \
      --output output/concept_descriptions_llm.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.chatanywhere.tech/v1")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "")


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


def _uniq_keep_order(items: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_source_name(name: str) -> str:
    s = (name or "").strip().lower()
    if s in {"kg", "knowledge graph", "knowledge_graph"}:
        return "KG"
    if s in {"table", "tabular"}:
        return "Table"
    return name.strip() or "Unknown"


def _truncate_text(text: str, max_len: int = 240) -> str:
    text = str(text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _stringify_example(example: Any) -> str:
    if isinstance(example, str):
        return _truncate_text(example)
    return _truncate_text(json.dumps(example, ensure_ascii=False))


def _extract_concept_payload(concept_name: str, concept_obj: Dict[str, Any]) -> Dict[str, Any]:
    sources = concept_obj.get("sources") or {}
    payload_sources: Dict[str, Any] = {}
    for source_name, source_obj in sources.items():
        src = source_obj or {}
        norm = _normalize_source_name(source_name)
        payload_sources[norm] = {
            "description": src.get("description") or "",
            "semantic_parse_guide": src.get("semantic_parse_guide") or "",
            "role": src.get("role") or [],
            "sample": src.get("sample") or [],
        }
    return {
        "concept": concept_name,
        "broad_description": concept_obj.get("description") or "",
        "sources": payload_sources,
    }


def _build_prompt(payload: Dict[str, Any]) -> str:
    concept_name = payload.get("concept", "")
    broad_desc = payload.get("broad_description", "")
    sources = payload.get("sources") or {}
    return (
        "You are rewriting concept descriptions for a KG/Table fusion dataset.\n"
        "Your job is to produce a broad concept description and then a source-specific\n"
        "description for each available source. The broad description should be wider\n"
        "than any source-specific one. The source-specific descriptions must be grounded\n"
        "in the provided roles, metadata, and samples.\n\n"
        "Hard rules:\n"
        "1) Do not invent facts not supported by the input.\n"
        "2) The overall_description should be source-agnostic and broad.\n"
        "3) KG description should focus on KG-style structured facts, entity IDs, and time/qualifier fields.\n"
        "4) Table description should focus on page/section context, headers, rows, and record-level meaning.\n"
        "5) elements should list the key fields/columns/slots that express the concept.\n"
        "6) data_examples should contain short copied evidence snippets.\n"
        "7) Return strict JSON only.\n\n"
        f"Concept: {concept_name}\n"
        f"Existing broad description: {broad_desc}\n\n"
        f"Input sources:\n{json.dumps(sources, ensure_ascii=False, indent=2)}\n\n"
        "Return this schema exactly:\n"
        "{\n"
        '  "concept": "...",\n'
        '  "overall_description": "...",\n'
        '  "sources": {\n'
        '    "KG": {"description": "...", "elements": ["..."], "data_examples": ["..."]},\n'
        '    "Table": {"description": "...", "elements": ["..."], "data_examples": ["..."]}\n'
        "  }\n"
        "}"
    )


def _parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _build_client(base_url: str, api_key: str):
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Please `pip install openai`.")
    if not api_key:
        raise RuntimeError("LLM API key is required. Set LLM_API_KEY or pass --api-key.")
    return OpenAI(api_key=api_key, base_url=base_url)


def _call_llm(client, model: str, prompt: str, max_tokens: int = 10000, temperature: float = 0.1) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful data curation assistant. Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _extract_elements_from_fallback(source_name: str, src_payload: Dict[str, Any]) -> List[str]:
    if source_name == "KG":
        roles = src_payload.get("role") or []
        elems = _uniq_keep_order(roles)
        if elems:
            return elems
        sample = src_payload.get("sample") or []
        if sample and isinstance(sample[0], dict):
            return _uniq_keep_order(sample[0].keys())
        return []

    if source_name == "Table":
        roles = src_payload.get("role") or []
        elems = _uniq_keep_order(roles)
        if elems:
            return elems
        sample = src_payload.get("sample") or []
        if sample and isinstance(sample[0], dict):
            header = sample[0].get("header") or []
            if header:
                return _uniq_keep_order(header)
        return []

    return []


def _build_fallback_examples(source_name: str, src_payload: Dict[str, Any], limit: int = 3) -> List[str]:
    sample = src_payload.get("sample") or []
    if not sample:
        return []
    examples: List[str] = []
    if source_name == "KG":
        for row in sample[:limit]:
            if isinstance(row, dict):
                examples.append(_stringify_example(row))
            else:
                examples.append(_truncate_text(str(row)))
        return examples

    if source_name == "Table":
        for row in sample[:limit]:
            if not isinstance(row, dict):
                examples.append(_truncate_text(str(row)))
                continue
            page = row.get("page_title") or ""
            section = row.get("section_title") or ""
            header = row.get("header") or []
            rows = row.get("sample_rows") or []
            compact = {
                "page_title": page,
                "section_title": section,
                "header": header,
                "sample_row": rows[0] if rows else [],
            }
            examples.append(_stringify_example(compact))
        return examples

    return []


def _fallback_output(payload: Dict[str, Any], parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    concept_name = payload.get("concept", "")
    broad_desc = payload.get("broad_description") or ""
    source_payloads = payload.get("sources") or {}

    out_sources: Dict[str, Any] = {}
    parsed_sources = (parsed or {}).get("sources") or {}
    for source_name in ["KG", "Table"]:
        src_payload = source_payloads.get(source_name) or {}
        src_parsed = parsed_sources.get(source_name) or {}
        desc = src_parsed.get("description") or src_payload.get("description") or broad_desc
        elements = src_parsed.get("elements") or _extract_elements_from_fallback(source_name, src_payload)
        examples = src_parsed.get("data_examples") or _build_fallback_examples(source_name, src_payload)
        out_sources[source_name] = {
            "description": desc,
            "elements": _uniq_keep_order(elements),
            "data_examples": _uniq_keep_order(examples),
        }

    return {
        "concept": concept_name,
        "overall_description": (parsed or {}).get("overall_description") or broad_desc,
        "sources": out_sources,
    }


def generate_descriptions(
    input_path: Path,
    output_path: Path,
    base_url: str,
    model: str,
    api_key: str,
    limit: Optional[int] = None,
    sleep_s: float = 0.0,
) -> Dict[str, Any]:
    fusion = _load_json(input_path)
    client = _build_client(base_url=base_url, api_key=api_key)

    concepts = list(fusion.items())
    if limit is not None:
        concepts = concepts[:limit]

    results: Dict[str, Any] = {}
    for idx, (concept_name, concept_obj) in enumerate(concepts, start=1):
        if not isinstance(concept_obj, dict):
            continue
        payload = _extract_concept_payload(concept_name, concept_obj)
        prompt = _build_prompt(payload)
        raw = _call_llm(client, model=model, prompt=prompt)
        parsed = _parse_json_loose(raw)
        merged = _fallback_output(payload, parsed)
        merged["raw_llm_output"] = raw
        results[concept_name] = merged
        print(f"[{idx}/{len(concepts)}] generated {concept_name}")
        if sleep_s > 0:
            time.sleep(sleep_s)

    _write_json(output_path, results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate broad and source-specific concept descriptions with an LLM.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "output/merged_fusion.json"),
        help="Path to merged_fusion.json",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "output/concept_descriptions_llm.json"),
        help="Output JSON path",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="LLM API key")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N concepts")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Optional sleep between LLM calls")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    generate_descriptions(
        input_path=input_path,
        output_path=output_path,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        limit=args.limit,
        sleep_s=args.sleep_s,
    )
    print(f"Saved concept descriptions to {output_path}")


if __name__ == "__main__":
    main()
