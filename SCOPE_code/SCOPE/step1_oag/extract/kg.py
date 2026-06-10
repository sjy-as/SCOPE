import os
import json
import csv
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI


# -------------------------
# Minimal heuristics (keep simple)
# -------------------------
_TIME_FIELD_PAT = re.compile(r"(time|start_time|end_timedate|year|season)", re.IGNORECASE)


def _is_time_like_field(field_name: str) -> bool:
    """Very small heuristic: treat fields containing time/date/year/season as time-like."""
    return bool(_TIME_FIELD_PAT.search(field_name or ""))


def _find_time_fields(attributes: list[str]) -> list[str]:
    return [a for a in attributes if _is_time_like_field(a)]


def _sample_rows(rows: list[dict], n: int = 3) -> list[dict]:
    return rows[:n]


def _build_description_prompt(
    item_kind: str,
    item_name: str,
    attributes: list[str] | None = None,
    sample_rows: list[dict] | None = None,
) -> str:
    attributes = attributes or []
    sample_rows = sample_rows or []
    return (
        f"You are writing a description using one sentence for a KG {item_kind}.\n"
        f"Name: {item_name}\n"
        f"Attributes: {json.dumps(attributes, ensure_ascii=False)}\n"
        f"Samples: {json.dumps(_sample_rows(sample_rows, 3), ensure_ascii=False)}\n\n"
        "Return JSON only with a sentence: description."
    )


def _event_roles(attributes: list[str]) -> list[str]:
    roles: list[str] = []
    for item in [*attributes]:
        if item and item not in roles:
            roles.append(item)
    return roles


# -------------------------
# LLM Client (OpenAI-compatible)
# -------------------------
class LLMClient:
    """
    OpenAI-compatible client for Sophnet / DeepSeek-V3.2-Fast.
    """
    def __init__(self, base_url: str = "", api_key: str = "", model: str = "DeepSeek-V3.2-Fast", timeout: int = 60):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = None

        if self.enabled():
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def complete_json(self, prompt: str) -> Optional[dict]:
        """
        Expected to return a JSON object (dict). If anything fails, return None.
        """
        if not self.enabled() or self.client is None:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # 强调不要输出 Markdown 标记
                    {"role": "system", "content": "你是Sophnet智能助手，请只输出纯JSON字符串，不要包含任何Markdown标记（如```json）。"},
                    {"role": "user", "content": prompt},
                ],
                # ⚠️ 很多第三方接口不支持这个参数，容易报错，建议注释掉或移除
                # response_format={"type": "json_object"}, 
                timeout=self.timeout,
            )
            content = response.choices[0].message.content or ""
            
            # --- 新增：清理大模型可能返回的 Markdown 标记 ---
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            # ------------------------------------------------

            return json.loads(content)
            
        except Exception as e:
            # ⚠️ 加上这行打印，如果是网络问题或 API 报错，终端就会显示出来
            print(f"\n[⚠️ LLM 请求或解析失败]: {e}")
            if 'content' in locals():
                print(f"[🔍 LLM 实际返回的内容是]: \n{content}\n")
            return None

def _build_fewshot_prompt(
    relation_type: str,
    connects_from: str,
    connects_to: str,
    relation_attributes: list[str],
    time_fields: list[str],
    sample_rows: list[dict],
) -> str:
    """
    Build a KG-style prompt for event lifting.
    Output should match the storage format requested by the user.
    """
    fewshot_1 = {
        "relation_type": "playsFor",
        "connects": {"from": "Player", "to": "Team"},
        "attributes": ["START_ID(Player)", "END_ID(Team)", "start_time", "end_time", "position"],
        "time_fields": ["start_time", "end_time"],
        "sample_rows": [
            {"START_ID(Player)": "p1", "END_ID(Team)": "t1", "start_time": "1995", "end_time": "1998", "position": "Guard"}
        ],
        "expected": {
            "event_name": "contract_event",
            "description": "A player is under contract / plays for a team during a time interval.",
            "roles": ["START_ID(Player)", "END_ID(Team)", "start_time", "end_time", "position"],
            "time_fields": ["start_time", "end_time"]
        }
    }

    fewshot_2 = {
        "relation_type": "draftedBy",
        "connects": {"from": "Player", "to": "Team"},
        "attributes": ["START_ID(Player)", "END_ID(Team)", "draft_year", "round", "pick"],
        "time_fields": ["draft_year"],
        "sample_rows": [
            {"START_ID(Player)": "p2", "END_ID(Team)": "t2", "draft_year": "2019", "round": "1", "pick": "3"}
        ],
        "expected": {
            "event_name": "draft_event",
            "description": "A team selects a player in a draft at a specific year.",
            "roles": ["START_ID(Player)", "END_ID(Team)", "draft_year", "round", "pick"],
            "time_fields": ["draft_year"]
        }
    }

    task = {
        "relation_type": relation_type,
        "connects": {"from": connects_from, "to": connects_to},
        "attributes": relation_attributes,
        "time_fields_detected": time_fields,
        "sample_rows": _sample_rows(sample_rows, 3),
        "instruction": (
            "You are given a KG relation schema with time-like fields. "
            "Lift it into an event schema. "
            "Return JSON only with keys: event_name, description, roles, time_fields. "
            "roles must be a list of field names, not a dictionary. "
            "Use the relation's head entity, tail entity, and attributes as role names when relevant."
        )
    }

    prompt = (
        "Few-shot examples:\n"
        f"Example1 Input:\n{json.dumps({k: fewshot_1[k] for k in ['relation_type','connects','attributes','time_fields','sample_rows']}, ensure_ascii=False)}\n"
        f"Example1 Output:\n{json.dumps(fewshot_1['expected'], ensure_ascii=False)}\n\n"
        f"Example2 Input:\n{json.dumps({k: fewshot_2[k] for k in ['relation_type','connects','attributes','time_fields','sample_rows']}, ensure_ascii=False)}\n"
        f"Example2 Output:\n{json.dumps(fewshot_2['expected'], ensure_ascii=False)}\n\n"
        "Now do this for the following input and output ONLY the JSON object:\n"
        f"{json.dumps(task, ensure_ascii=False)}"
    )
    return prompt


def extract_semantic_and_event_layers(kg_dir: str) -> tuple[dict, dict]:
    """
    从本地 KG CSV 文件提取语义层和事件层

    语义层：实体类型及其属性 + 关系类型(作为静态事实schema/连接信息)
    事件层：从 KG 关系中筛选“含时间属性”的关系，并上升为事件类型（xx_event）
    """
    semantic_layer = {}  # entity_types + relation_types(schema)
    event_layer = {}     # event_types derived from time-like relations

    llm = LLMClient(
        base_url="https://www.sophnet.com/api/open-apis/v1",   # TODO: fill by yourself
        api_key="V8IjyYmmJLK4vniImZ9IpJaowdnjAIR1s84Ch8sDWQQLpqm7TJaGyp2atttRh7hXg54l2H6GYzpBHZQDEQC2wQ",    # TODO: fill by yourself
        model="DeepSeek-V3.2-Fast",
        timeout=60
    )


    # 获取所有 CSV 文件
    csv_files = sorted(Path(kg_dir).glob("*.csv"))

    # -------------------------
    # 1) Entities -> semantic_layer["entities"]
    # -------------------------
    entity_files = [f for f in csv_files if not f.name.startswith("relation_")]
    entities_schema = {}

    for entity_file in entity_files:
        entity_type = entity_file.stem
        rows = []

        with open(entity_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if rows:
            entity_attributes = list(rows[0].keys())
            description = entity_type
            if llm.enabled():
                desc_prompt = _build_description_prompt(
                    item_kind="concept",
                    item_name=entity_type,
                    attributes=entity_attributes,
                    sample_rows=rows,
                )
                desc_obj = llm.complete_json(desc_prompt)
                if isinstance(desc_obj, dict):
                    description = desc_obj.get("description", description) or description

            entities_schema[entity_type] = {
                "concept_name": entity_type,
                "description": description,
                "attributes": entity_attributes,
                "sample": _sample_rows(rows, 3),
            }

    semantic_layer["entities"] = entities_schema

    # -------------------------
    # 2) Relations -> semantic_layer["relations"] (static fact schema)
    # -------------------------
    relation_files = [f for f in csv_files if f.name.startswith("relation_")]
    relations_schema = {}

    # We'll also prepare candidates for event lifting (time-like relations)
    time_relation_candidates = []

    for relation_file in relation_files:
        relation_type = relation_file.stem.replace("relation_", "")
        rows = []

        with open(relation_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            continue

        first_relation = rows[0]
        # locate START_ID / END_ID fields
        start_keys = [k for k in first_relation.keys() if "START_ID" in k]
        end_keys = [k for k in first_relation.keys() if "END_ID" in k]
        if not start_keys or not end_keys:
            # if schema is unexpected, still store attributes
            start_entity_type = ""
            end_entity_type = ""
            start_id_key = ""
            end_id_key = ""
        else:
            start_id_key = start_keys[0]
            end_id_key = end_keys[0]
            # Extract entity types in parentheses
            start_entity_type = start_id_key.split("(")[1].rstrip(")") if "(" in start_id_key else ""
            end_entity_type = end_id_key.split("(")[1].rstrip(")") if "(" in end_id_key else ""

        attributes = list(first_relation.keys())
        time_fields = _find_time_fields(attributes)

        relation_description = relation_type
        if llm.enabled():
            desc_prompt = _build_description_prompt(
                item_kind="relation",
                item_name=relation_type,
                attributes=attributes,
                sample_rows=rows,
            )
            desc_obj = llm.complete_json(desc_prompt)
            if isinstance(desc_obj, dict):
                relation_description = desc_obj.get("description", relation_description) or relation_description

        relations_schema[relation_type] = {
            "relation_name": relation_type,
            "description": relation_description,
            "attributes": attributes,
            "sample": _sample_rows(rows, 3),
        }

        if time_fields:
            time_relation_candidates.append({
                "relation_type": relation_type,
                "connects": {"from": start_entity_type, "to": end_entity_type},
                "attributes": attributes,
                "time_fields_detected": time_fields,
                "sample": _sample_rows(rows, 3),
            })

    semantic_layer["relations"] = relations_schema

    # -------------------------
    # 3) Event layer: lift time-like relations into event types
    # -------------------------
    # Minimal fallback naming if LLM disabled: "{relation_type}_event"
    for cand in time_relation_candidates:
        relation_type = cand["relation_type"]
        connects_from = cand["connects"]["from"]
        connects_to = cand["connects"]["to"]
        attributes = cand["attributes"]
        time_fields = cand["time_fields_detected"]
        sample_rows = cand["sample"]

        lifted = None
        if llm.enabled():
            prompt = _build_fewshot_prompt(
                relation_type=relation_type,
                connects_from=connects_from,
                connects_to=connects_to,
                relation_attributes=attributes,
                time_fields=time_fields,
                sample_rows=sample_rows
            )
            lifted = llm.complete_json(prompt)

        # Normalize / fallback
        if not isinstance(lifted, dict):
            lifted = {}

        event_name = lifted.get("event_name") or f"{relation_type}_event"
        if not event_name.endswith("_event"):
            event_name = f"{event_name}_event"

        event_description = lifted.get("description", "")
        if not event_description and llm.enabled():
            desc_prompt = _build_description_prompt(
                item_kind="event",
                item_name=event_name,
                attributes=attributes,
                sample_rows=sample_rows,
            )
            desc_obj = llm.complete_json(desc_prompt)
            if isinstance(desc_obj, dict):
                event_description = desc_obj.get("description", event_description) or event_description

        event_layer[event_name] = {
            "event_name": event_name,
            "description": event_description,
            "roles": _event_roles(attributes),
            "time_fields": lifted.get("time_fields", time_fields),
            "sample": _sample_rows(sample_rows, 3),
        }

    return semantic_layer, event_layer


def main():
    import argparse
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kg_dir",
        default="/root/autodl-tmp/new_model/data_sources/KG",
        help="KG 数据源目录"
    )
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/new_model/step1_oag/extract/output",
        help="输出目录"
    )
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 提取语义层和事件层
    semantic_layer, event_layer = extract_semantic_and_event_layers(args.kg_dir)

    # 保存结果（路径不变）
    semantic_output = os.path.join(args.output_dir, "kg_semantic.json")
    event_output = os.path.join(args.output_dir, "kg_event.json")

    with open(semantic_output, "w", encoding="utf-8") as f:
        json.dump(semantic_layer, f, indent=2, ensure_ascii=False)

    with open(event_output, "w", encoding="utf-8") as f:
        json.dump(event_layer, f, indent=2, ensure_ascii=False)

    print(f"✓ 语义层已保存到: {semantic_output}")
    print(f"✓ 事件层已保存到: {event_output}")

    entity_types = list(semantic_layer.get("entities", {}).keys())
    relation_types = list(semantic_layer.get("relations", {}).keys())
    event_types = list(event_layer.keys())

    print(f"\n语义层实体类型: {entity_types}")
    print(f"语义层关系类型: {relation_types}")
    print(f"事件层事件类型: {event_types}")


if __name__ == "__main__":
    main()