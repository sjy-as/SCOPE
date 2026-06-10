"""Fuse KG and table event/concept/relation metadata into a unified catalog.

Structure:
1. Event fusion
2. Semantic fusion
   2.1 Concept fusion
   2.2 Relation fusion

The semantic part is split into:
- KG concepts + Table concepts
- KG relations + Table relations

Final output contains:
- merged_events
- merged_semantics
  - merged_concepts
  - merged_relations
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


BASE_URL = "https://www.sophnet.com/api/open-apis/v1"
API_KEY = "V8IjyYmmJLK4vniImZ9IpJaowdnjAIR1s84Ch8sDWQQLpqm7TJaGyp2atttRh7hXg54l2H6GYzpBHZQDEQC2wQ"
MODEL = "DeepSeek-V3.2-Fast"


@dataclass
class SourcePayload:
    source: str
    name: str
    kind: str  # event / concept / relation
    description: str
    roles: List[str] = field(default_factory=list)
    time_fields: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    relations: List[Any] = field(default_factory=list)
    sample: List[Any] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "roles": self.roles,
            "time_fields": self.time_fields,
            "attributes": self.attributes,
            "relations": self.relations,
            "sample": self.sample,
            "raw": self.raw,
        }


@dataclass
class MergedItem:
    canonical_name: str
    item_type: str
    description: str
    sources: Dict[str, SourcePayload] = field(default_factory=dict)
    selection_guide: str = ""
    merge_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_name": self.canonical_name,
            "item_type": self.item_type,
            "description": self.description,
            "merge_reason": self.merge_reason,
            "selection_guide": self.selection_guide,
            "sources": {k: v.to_dict() for k, v in self.sources.items()},
        }


def _safe_load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def normalize_key(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[\s\-\/]+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "", name)
    return name


def pretty_title(name: str) -> str:
    name = str(name).replace("_", " ").strip()
    return re.sub(r"\s+", " ", name).title()


def merge_unique_lists(*lists: List[str]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for lst in lists:
        for item in lst or []:
            if item is None:
                continue
            text = str(item)
            if text not in seen:
                seen.add(text)
                merged.append(text)
    return merged


def build_selection_guide(canonical_name: str, description: str, kg: Optional[SourcePayload], table: Optional[SourcePayload]) -> str:
    parts = [f"[Rule] {canonical_name}", f"Definition: {description or 'No description provided.'}"]
    if kg:
        parts.append("KG clues:")
        parts.append(f"- graph-like structured facts from {kg.name}")
        if kg.roles:
            parts.append(f"- roles / slots: {', '.join(kg.roles)}")
        if kg.time_fields:
            parts.append(f"- explicit temporal fields: {', '.join(kg.time_fields)}")
        if kg.attributes:
            parts.append(f"- attributes / concept keys: {', '.join(kg.attributes[:8])}")
    if table:
        parts.append("Table clues:")
        parts.append(f"- evidence grounded in page/row/column samples from {table.name}")
        if table.roles:
            parts.append(f"- table headers / columns: {', '.join(table.roles)}")
        if table.time_fields:
            parts.append(f"- table time columns: {', '.join(table.time_fields)}")
        if table.attributes:
            parts.append(f"- table concepts / entity labels: {', '.join(table.attributes[:8])}")
    return "\n".join(parts)


class EventFusionAnalyzer:
    def __init__(self, kg_events_path: str, kg_semantic_path: Optional[str], table_events_path: str, table_semantic_path: Optional[str]):
        print("[1/6] Loading source files...")
        print(f"  KG events:      {kg_events_path}")
        print(f"  KG semantics:   {kg_semantic_path}")
        print(f"  Table events:   {table_events_path}")
        print(f"  Table semantic: {table_semantic_path}")

        self.kg_events = _safe_load_json(kg_events_path)
        self.table_events = _safe_load_json(table_events_path)
        self.kg_semantic = _safe_load_json(kg_semantic_path) if kg_semantic_path else {}
        self.table_semantic = _safe_load_json(table_semantic_path) if table_semantic_path else {}

        self.merged_events: Dict[str, MergedItem] = {}
        self.merged_concepts: Dict[str, MergedItem] = {}
        self.merged_relations: Dict[str, MergedItem] = {}
        self.client = self._build_client()

        print("[1/6] Source files loaded successfully.")
        print(f"  KG events count:       {len(self.kg_events)}")
        print(f"  KG semantics count:    {len(self.kg_semantic)}")
        print(f"  Table events count:    {len(self.table_events)}")
        print(f"  Table semantics count: {len(self.table_semantic)}")
        print(f"  LLM client available:  {'yes' if self.client else 'no'}")

    def _build_client(self):
        if OpenAI is None:
            return None
        try:
            return OpenAI(api_key=API_KEY, base_url=BASE_URL)
        except Exception:
            return None

    def _extract_kg_event_payload(self, key: str, data: Dict[str, Any]) -> SourcePayload:
        return SourcePayload("kg", key, "event", data.get("description", ""), list(data.get("roles", [])), list(data.get("time_fields", [])), list(data.get("roles", [])), [], list(data.get("sample", [])), data)

    def _extract_table_event_payload(self, key: str, data: Dict[str, Any]) -> SourcePayload:
        return SourcePayload("table", key, "event", data.get("description", ""), list(data.get("roles", [])), list(data.get("time_fields", [])), merge_unique_lists(data.get("semantic_concepts", [])), list(data.get("semantic_relations", [])), list(data.get("sample", [])), data)

    def _extract_kg_concept_payload(self, key: str, data: Dict[str, Any]) -> SourcePayload:
        return SourcePayload("kg", key, "concept", data.get("description", ""), [], [], list(data.get("attributes", [])), [], list(data.get("sample", [])), data)

    def _extract_table_concept_payload(self, key: str, data: Dict[str, Any]) -> SourcePayload:
        return SourcePayload("table", key, "concept", data.get("description", ""), [], [], list(data.get("attributes", [])), [], list(data.get("sample", [])), data)

    def _extract_kg_relation_payload(self, key: str, data: Dict[str, Any]) -> SourcePayload:
        return SourcePayload("kg", key, "relation", data.get("description", ""), [], [], list(data.get("attributes", [])), list(data.get("relations", [])), list(data.get("sample", [])), data)

    def _extract_table_relation_payload(self, key: str, data: Dict[str, Any]) -> SourcePayload:
        return SourcePayload("table", key, "relation", data.get("description", ""), [], [], list(data.get("attributes", [])), list(data.get("relations", [])), list(data.get("sample", [])), data)

    def _merge_by_name(self, kg_map: Dict[str, SourcePayload], table_map: Dict[str, SourcePayload], item_type: str) -> Dict[str, MergedItem]:
        print(f"[{item_type}] merging {len(kg_map)} KG items and {len(table_map)} table items...")
        merged: Dict[str, MergedItem] = {}
        all_keys = sorted(set(kg_map) | set(table_map))
        print(f"  Candidate unique keys: {len(all_keys)}")

        for idx, key in enumerate(all_keys, start=1):
            kg_payload = kg_map.get(key)
            table_payload = table_map.get(key)
            canonical_name = pretty_title(key)
            description = (kg_payload.description if kg_payload and kg_payload.description else "") or (table_payload.description if table_payload else "")

            if kg_payload and table_payload:
                state = "kg + table"
                reason = "Same normalized name exists in both KG and table, merged directly."
            elif kg_payload:
                state = "kg only"
                reason = "Only KG provides this item."
            else:
                state = "table only"
                reason = "Only table provides this item."

            print(f"  [{idx}/{len(all_keys)}] {canonical_name} -> {state}")
            if kg_payload:
                print(f"      KG:    {kg_payload.name}")
            if table_payload:
                print(f"      Table: {table_payload.name}")
            if description:
                print(f"      Desc:  {description[:120]}")

            merged_item = MergedItem(canonical_name=canonical_name, item_type=item_type, description=description, merge_reason=reason)
            if kg_payload:
                merged_item.sources["kg"] = kg_payload
            if table_payload:
                merged_item.sources["table"] = table_payload
            merged_item.selection_guide = build_selection_guide(canonical_name, description, kg_payload, table_payload)
            merged[key] = merged_item

        print(f"[{item_type}] finished. Produced {len(merged)} merged entries.")
        return merged

    def merge_events(self) -> Dict[str, MergedItem]:
        print("[2/6] Preparing event maps...")
        kg_map = {normalize_key(k): self._extract_kg_event_payload(k, v) for k, v in self.kg_events.items()}
        table_map = {normalize_key(k): self._extract_table_event_payload(k, v) for k, v in self.table_events.items()}
        print(f"  Normalized KG events:    {len(kg_map)}")
        print(f"  Normalized table events: {len(table_map)}")
        self.merged_events = self._merge_by_name(kg_map, table_map, "event")
        return self.merged_events

    def merge_concepts(self) -> Dict[str, MergedItem]:
        print("[3/6] Preparing concept maps...")
        kg_entities = self.kg_semantic.get("entities", {}) if isinstance(self.kg_semantic, dict) else {}
        table_entities = self.table_semantic.get("entities", {}) if isinstance(self.table_semantic, dict) else {}
        kg_map = {normalize_key(k): self._extract_kg_concept_payload(k, v) for k, v in kg_entities.items()}
        table_map = {normalize_key(k): self._extract_table_concept_payload(k, v) for k, v in table_entities.items()}
        print(f"  Normalized KG concepts:    {len(kg_map)}")
        print(f"  Normalized table concepts: {len(table_map)}")
        self.merged_concepts = self._merge_by_name(kg_map, table_map, "concept")
        return self.merged_concepts

    def merge_relations(self) -> Dict[str, MergedItem]:
        print("[4/6] Preparing relation maps...")
        kg_relations = self.kg_semantic.get("relations", {}) if isinstance(self.kg_semantic, dict) else {}
        table_relations = self.table_semantic.get("relations", {}) if isinstance(self.table_semantic, dict) else {}
        kg_map = {normalize_key(k): self._extract_kg_relation_payload(k, v) for k, v in kg_relations.items()}
        table_map = {normalize_key(k): self._extract_table_relation_payload(k, v) for k, v in table_relations.items()}
        print(f"  Normalized KG relations:    {len(kg_map)}")
        print(f"  Normalized table relations: {len(table_map)}")
        return self._merge_by_name(kg_map, table_map, "relation")

    def ask_llm_for_rules(self, merged_items: Dict[str, MergedItem], max_items: int = 20) -> Dict[str, str]:
        if not self.client or not merged_items:
            print("[LLM] Skipping rule generation (client unavailable or no items).")
            return {}

        samples = list(merged_items.values())[:max_items]
        print(f"[LLM] Asking for KG/table preference analysis on {len(samples)} sample items...")
        prompt = {
            "task": "Analyze KG and table for each merged item, then write an initial selection suggestion focusing on the differences between the two sources.",
            "requirements": [
                "Do not write a combined_rule.",
                "Only return kg_preference and table_preference.",
                "Explain what is different between KG and table for the same item, to help selection.",
                "Make the output concise and practical for downstream choice.",
            ],
            "items": [
                {
                    "name": item.canonical_name,
                    "type": item.item_type,
                    "description": item.description,
                    "sources": {k: v.to_dict() for k, v in item.sources.items()},
                }
                for item in samples
            ],
        }
        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a data fusion expert. Return valid JSON only."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
                ],
                temperature=0.1,
                max_tokens=10000,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                print(f"[LLM] Returned {len(parsed)} analysis entries.")
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception as exc:
            print(f"[LLM] Rule generation failed, fallback to deterministic rules. Reason: {exc}")
        return {}

    def build_output(self) -> Dict[str, Any]:
        print("[5/6] Building merged output...")
        if not self.merged_events:
            self.merge_events()
        if not self.merged_concepts:
            self.merge_concepts()
        merged_relations = self.merge_relations()

        print("  Applying LLM-guided preference analysis if available...")
        event_guides = self.ask_llm_for_rules(self.merged_events)
        concept_guides = self.ask_llm_for_rules(self.merged_concepts)
        relation_guides = self.ask_llm_for_rules(merged_relations)

        for key, guide in event_guides.items():
            if key in self.merged_events and guide:
                self.merged_events[key].selection_guide = guide
        for key, guide in concept_guides.items():
            if key in self.merged_concepts and guide:
                self.merged_concepts[key].selection_guide = guide
        for key, guide in relation_guides.items():
            if key in merged_relations and guide:
                merged_relations[key].selection_guide = guide

        print(f"  Final merged events:    {len(self.merged_events)}")
        print(f"  Final merged concepts:  {len(self.merged_concepts)}")
        print(f"  Final merged relations: {len(merged_relations)}")
        print("[5/6] Output built successfully.")

        return {
            "summary": {
                "kg_event_count": len(self.kg_events),
                "table_event_count": len(self.table_events),
                "kg_concept_count": len(self.kg_semantic.get("entities", {}) if isinstance(self.kg_semantic, dict) else {}),
                "table_concept_count": len(self.table_semantic.get("entities", {}) if isinstance(self.table_semantic, dict) else {}),
                "kg_relation_count": len(self.kg_semantic.get("relations", {}) if isinstance(self.kg_semantic, dict) else {}),
                "table_relation_count": len(self.table_semantic.get("relations", {}) if isinstance(self.table_semantic, dict) else {}),
                "merged_event_count": len(self.merged_events),
                "merged_concept_count": len(self.merged_concepts),
                "merged_relation_count": len(merged_relations),
            },
            "selection_policy": {
                "kg_preference": "Prefer KG when the item is represented as structured facts, explicit links, triples, or temporal relations.",
                "table_preference": "Prefer table when the item is grounded in page evidence, row/column context, headers, or sample records.",
            },
            "merged_events": {k: v.to_dict() for k, v in self.merged_events.items()},
            "merged_semantics": {
                "merged_concepts": {k: v.to_dict() for k, v in self.merged_concepts.items()},
                "merged_relations": {k: v.to_dict() for k, v in merged_relations.items()},
            },
        }

    def save(self, output_dir: str) -> str:
        print("[6/6] Saving merged output...")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "merged_fusion.json")
        payload = self.build_output()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[6/6] Saved merged output to: {output_path}")
        return output_path

    def validate_coverage(self) -> Dict[str, Any]:
        if not self.merged_events:
            self.merge_events()
        if not self.merged_concepts:
            self.merge_concepts()
        merged_relations = self.merge_relations()

        def _covered(merged: Dict[str, MergedItem], source_name: str) -> set[str]:
            covered = set()
            for item in merged.values():
                for src in item.sources.values():
                    if src.source == source_name:
                        covered.add(normalize_key(src.name))
            return covered

        kg_event_keys = {normalize_key(k) for k in self.kg_events}
        table_event_keys = {normalize_key(k) for k in self.table_events}
        kg_concept_keys = {normalize_key(k) for k in (self.kg_semantic.get("entities", {}) if isinstance(self.kg_semantic, dict) else {})}
        table_concept_keys = {normalize_key(k) for k in (self.table_semantic.get("entities", {}) if isinstance(self.table_semantic, dict) else {})}
        kg_relation_keys = {normalize_key(k) for k in (self.kg_semantic.get("relations", {}) if isinstance(self.kg_semantic, dict) else {})}
        table_relation_keys = {normalize_key(k) for k in (self.table_semantic.get("relations", {}) if isinstance(self.table_semantic, dict) else {})}

        return {
            "events": {
                "kg_total": len(kg_event_keys),
                "table_total": len(table_event_keys),
                "kg_covered": len(_covered(self.merged_events, "kg")),
                "table_covered": len(_covered(self.merged_events, "table")),
                "kg_uncovered": sorted(kg_event_keys - _covered(self.merged_events, "kg")),
                "table_uncovered": sorted(table_event_keys - _covered(self.merged_events, "table")),
            },
            "merged_semantics": {
                "concepts": {
                    "kg_total": len(kg_concept_keys),
                    "table_total": len(table_concept_keys),
                    "kg_covered": len(_covered(self.merged_concepts, "kg")),
                    "table_covered": len(_covered(self.merged_concepts, "table")),
                    "kg_uncovered": sorted(kg_concept_keys - _covered(self.merged_concepts, "kg")),
                    "table_uncovered": sorted(table_concept_keys - _covered(self.merged_concepts, "table")),
                },
                "relations": {
                    "kg_total": len(kg_relation_keys),
                    "table_total": len(table_relation_keys),
                    "kg_covered": len(_covered(merged_relations, "kg")),
                    "table_covered": len(_covered(merged_relations, "table")),
                    "kg_uncovered": sorted(kg_relation_keys - _covered(merged_relations, "kg")),
                    "table_uncovered": sorted(table_relation_keys - _covered(merged_relations, "table")),
                },
            },
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge KG and table event/concept/relation files.")
    parser.add_argument("--kg-events", required=True, help="Path to KG event JSON")
    parser.add_argument("--kg-semantic", required=True, help="Path to KG semantic JSON")
    parser.add_argument("--table-events", required=True, help="Path to table event JSON")
    parser.add_argument("--table-semantic", required=True, help="Path to table semantic JSON")
    parser.add_argument("--output-dir", default="./fusion_output", help="Output directory")
    return parser


def main() -> None:
    print("=" * 80)
    print("START MERGING KG + TABLE EVENT/SEMANTIC FILES")
    print("=" * 80)
    parser = build_arg_parser()
    args = parser.parse_args()

    analyzer = EventFusionAnalyzer(
        kg_events_path=args.kg_events,
        kg_semantic_path=args.kg_semantic,
        table_events_path=args.table_events,
        table_semantic_path=args.table_semantic,
    )

    output_path = analyzer.save(args.output_dir)
    coverage = analyzer.validate_coverage()

    print("[DONE] Merge finished.")
    print(f"[DONE] Output file: {output_path}")
    print("[DONE] Coverage report:")
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
