import hashlib
from datetime import datetime
import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from openai import OpenAI

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ============================
# Global KG priors
# ============================
def load_kg_vocab_from_json(
    kg_event_path: str,
    kg_semantic_path: str,
) -> tuple[list[dict], list[dict]]:
    """Load KG event / concept vocab from exported JSON files."""
    with open(kg_event_path, "r", encoding="utf-8") as f:
        kg_event_data = json.load(f)
    with open(kg_semantic_path, "r", encoding="utf-8") as f:
        kg_semantic_data = json.load(f)

    kg_event_vocab = []
    for _, item in (kg_event_data or {}).items():
        if not isinstance(item, dict):
            continue
        name = str(item.get("event_name", "") or "").strip()
        desc = str(item.get("description", "") or "").strip()
        if name:
            kg_event_vocab.append({"name": normalize_concept_rule_based(name), "description": desc})

    kg_concept_vocab = []
    for _, item in ((kg_semantic_data or {}).get("entities", {}) or {}).items():
        if not isinstance(item, dict):
            continue
        name = str(item.get("concept_name", "") or "").strip()
        desc = str(item.get("description", "") or "").strip()
        if name:
            kg_concept_vocab.append({"name": normalize_concept_rule_based(name), "description": desc})

    return kg_event_vocab, kg_concept_vocab


KG_EVENT_TYPES: list[dict] = []
KG_CONCEPT_TYPES: list[dict] = []

# ============================
# 0) Parse metadata.sql
# ============================
def parse_metadata_sql(metadata_path: str) -> list[dict]:
    tables = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()

    copy_start = content.find("COPY metadata.nba_context")
    if copy_start == -1:
        raise ValueError("未在 metadata.sql 中找到 'COPY metadata.nba_context' 段落")

    data_start = content.find("\n", copy_start) + 1
    data_end = content.find("\\.", data_start)
    if data_end == -1:
        raise ValueError("未在 metadata.sql 中找到 COPY 数据结束标记 '\\.'")

    data_section = content[data_start:data_end]
    lines = data_section.strip().split("\n")

    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 5:
            try:
                table_id = parts[0]
                page_title = parts[1]
                section_title = parts[2]
                caption = parts[3]
                doc_json = json.loads(parts[4])

                tables.append({
                    "id": table_id,
                    "page_title": page_title,
                    "section_title": section_title,
                    "caption": caption,
                    "header": doc_json.get("header", []),
                    "rows": doc_json.get("rows", [])[:3],
                    "total_rows": len(doc_json.get("rows", []))
                })
            except (json.JSONDecodeError, IndexError):
                continue
    return tables


# ============================
# 1) Split by time signal
# ============================
_TIME_RE = re.compile(
    r"(\b(18|19|20)\d{2}\b|"
    r"\b\d{4}\s*-\s*\d{2}\b|"
    r"\b\d{4}\s*–\s*\d{2}\b|"
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b|"
    r"\b[A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}\b|"
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b|"
    r"\b(season|year|years|date|dates|start_time|end_time|time|season_start_year|from|to)\b"
    r")",
    re.IGNORECASE
)


def table_is_event(table: dict) -> bool:
    blob = f"{table.get('page_title','')} {table.get('section_title','')} {' '.join(table.get('header', []))}"
    return bool(_TIME_RE.search(blob))


def split_tables_by_time(tables: list[dict]) -> tuple[list[dict], list[dict]]:
    event_tables, static_tables = [], []
    for t in tables:
        if table_is_event(t):
            event_tables.append(t)
        else:
            static_tables.append(t)
    return event_tables, static_tables


# ============================
# 2) Clustering
# ============================
def cluster_tables(
    tables: list[dict],
    n_clusters: int = 50,
    method: str = "sentence-transformers"
) -> tuple[list[dict], np.ndarray]:
    if not tables:
        return [], np.array([], dtype=int)

    texts = [f"{t['page_title']} {t['section_title']} {' '.join(t['header'])}" for t in tables]

    if method == "sentence-transformers":
        if not HAS_SENTENCE_TRANSFORMERS:
            print("⚠ sentence-transformers 未安装，回退到 TF-IDF")
            method = "tfidf"
        else:
            model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            embeddings = model.encode(texts, show_progress_bar=True)

    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        embeddings = vectorizer.fit_transform(texts).toarray()

    k = min(n_clusters, len(tables))
    if k <= 1:
        return [tables[0]], np.zeros(len(tables), dtype=int)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    representative_tables = []
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_embeddings = embeddings[cluster_indices]
        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        representative_tables.append(tables[closest_idx])

    return representative_tables, labels


# ============================
# 3) LLM client
# ============================
class LLMClient:
    def __init__(self, base_url: str = "", api_key: str = "", model: str = "DeepSeek-V3.2-Fast", timeout: int = 60):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.enabled() else None

    def enabled(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def complete_json(self, prompt: str, temperature: float = 0.1, max_tokens: int = 10000) -> Optional[dict]:
        if not self.enabled() or self.client is None:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是智能助手，只输出纯JSON，不要Markdown。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            return json.loads(content.strip())
        except Exception as e:
            print(f"[⚠️ LLM失败] {e}")
            return None


# ============================
# 4) Cache utils
# ============================
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _cache_path(cache_dir: str, kind: str) -> str:
    return os.path.join(cache_dir, f"{kind}_cluster_cache.json")


def load_cluster_cache(cache_file: str) -> Optional[dict]:
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cluster_cache(cache_file: str, payload: dict) -> None:
    tmp = cache_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, cache_file)


def build_table_id_index(tables: List[dict]) -> Dict[str, dict]:
    return {str(t["id"]): t for t in tables if "id" in t}


def pick_representatives_from_ids(rep_ids: List[str], table_index: Dict[str, dict]) -> List[dict]:
    reps = []
    for tid in rep_ids:
        t = table_index.get(str(tid))
        if t is None:
            raise KeyError(f"representative table id not found: {tid}")
        reps.append(t)
    return reps


def _table_title_header_blob(table: dict) -> str:
    return f"{table.get('page_title','')} {table.get('section_title','')} {' '.join(table.get('header', []))}"


def _representative_table_digest(table: dict) -> str:
    blob = json.dumps({
        "page_title": table.get("page_title", ""),
        "section_title": table.get("section_title", ""),
        "header": table.get("header", []),
        "rows": table.get("rows", []),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ============================
# 5) normalize
# ============================
_CONCEPT_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9_]+")


def normalize_concept_rule_based(concept: str) -> str:
    if not concept:
        return concept
    c = concept.strip().lower().replace("-", "_").replace(" ", "_")
    c = _CONCEPT_TOKEN_SPLIT_RE.sub("_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def _normalize_name_list(items: list[str]) -> list[str]:
    out, seen = [], set()
    for item in items:
        name = normalize_concept_rule_based(str(item))
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _normalize_vocab_items(items: list[Any]) -> list[dict]:
    out, seen = [], set()
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("event_name") or item.get("concept_name") or item.get("name") or "").strip()
            desc = str(item.get("description", "") or "").strip()
        else:
            name = str(item).strip()
            desc = ""
        norm_name = normalize_concept_rule_based(name)
        if norm_name and norm_name not in seen:
            seen.add(norm_name)
            out.append({"name": norm_name, "description": desc})
    return out


def _parse_json_with_retry(llm: "LLMClient", prompt: str, temperature: float, max_tokens: int, retries: int = 3) -> dict:
    last_obj: dict = {}
    for attempt in range(1, retries + 1):
        obj = llm.complete_json(prompt, temperature=temperature, max_tokens=max_tokens) or {}
        last_obj = obj if isinstance(obj, dict) else {}
        if last_obj:
            return last_obj
        print(f"[⚠️ LLM解析失败] 第 {attempt}/{retries} 次重试")
    return last_obj


def _vocab_names(vocab_items: list[dict]) -> list[str]:
    return [item.get("name", "") for item in vocab_items if item.get("name")]


def _table_min_view(t: dict) -> dict:
    return {
        "id": t.get("id", ""),
        "page_title": t.get("page_title", ""),
        "section_title": t.get("section_title", ""),
        "header": t.get("header", []),
        "sample_rows": t.get("rows", []),
    }


def _stable_entity_description(concept_name: str, raw_desc: str = "", evidence: Optional[dict] = None) -> str:
    raw_desc = str(raw_desc or "").strip()
    if raw_desc:
        return raw_desc

    pretty = concept_name.replace("_", " ").strip()
    if evidence:
        title = f"{evidence.get('page_title', '')} {evidence.get('section_title', '')}".strip()
        caption = str(evidence.get('caption', '') or '').strip()
        header = evidence.get('header', []) or []
        header_part = ", ".join([str(h).strip() for h in header if str(h).strip()][:6])
        evidence_bits = [x for x in [title, caption, header_part] if x]
        if evidence_bits:
            return f"{pretty} described by table evidence: {' | '.join(evidence_bits)}."

    return f"{pretty} entity described from table evidence."


def _table_entity_prompt_context(table: dict) -> dict:
    return {
        "page_title": table.get("page_title", ""),
        "section_title": table.get("section_title", ""),
        "header": table.get("header", []),
        "sample_rows": table.get("rows", []),
    }


def _table_evidence_brief(table: dict, max_rows: int = 2) -> dict:
    rows = table.get("rows", []) or []
    return {
        "page_title": table.get("page_title", ""),
        "section_title": table.get("section_title", ""),
        "caption": table.get("caption", ""),
        "header": table.get("header", []),
        "sample_rows": rows[:max_rows],
    }


def _flatten_attrs(attrs: Any) -> list[str]:
    if attrs is None:
        return []
    if isinstance(attrs, list):
        return [str(x).strip() for x in attrs if str(x).strip()]
    if isinstance(attrs, dict):
        out = []
        for k, v in attrs.items():
            if isinstance(v, list):
                out.extend([str(x).strip() for x in v if str(x).strip()])
            elif isinstance(v, str) and v.strip():
                out.append(v.strip())
            elif isinstance(v, bool) and v:
                out.append(str(k))
        return out
    if isinstance(attrs, str):
        return [attrs.strip()] if attrs.strip() else []
    return [str(attrs).strip()] if str(attrs).strip() else []


def _normalize_entity_payload(name: str, description: str, attributes: Any, sample: list[dict], relations: Any = None) -> dict:
    return {
        "concept_name": name,
        "description": description,
        "attributes": _flatten_attrs(attributes),
        "sample": sample[:3],
        "relations": relations or [],
    }


# ============================
# 6) LLM analysis
# ============================
def analyze_event_representative_tables(
    representative_tables: list[dict],
    kg_event_list: list[Any] | None = None,
    prior_table_event_list: list[Any] | None = None,
    llm: Optional[LLMClient] = None,
) -> tuple[dict, list[dict]]:
    llm = llm or LLMClient()
    cluster_info = {}
    kg_event_vocab = _normalize_vocab_items(list(kg_event_list or []))
    prior_event_vocab = _normalize_vocab_items(list(prior_table_event_list or []))
    event_vocab = kg_event_vocab + [x for x in prior_event_vocab if x.get("name") not in _vocab_names(kg_event_vocab)]

    # 1-fewshot (aligned with KG-side style)
    fewshot_in_1 = {
        "title": "1995–96 Toronto Raptors season - Game log",
        "columns": ["Date", "Opponent", "Result", "PTS", "REB", "AST"]
    }
    fewshot_out_1 = {
        "event_name": "teamgame_event",
        "description": "A broad class of team game records during a season.",
        "roles": ["Season", "Team", "Date", "Opponent", "Result", "PTS", "REB", "AST"],
        "time_fields": ["Season", "Date"]
    }

    print(f"\n[Event] 用 LLM 分析 {len(representative_tables)} 个代表性表...")

    for i, table in enumerate(representative_tables):
        title = f"{table['page_title']} - {table['section_title']}"
        cols = table.get("header", [])
        print(f"[Event] 分析表 {i+1}/{len(representative_tables)}: {title}")

        prompt = f"""You extract an EVENT schema from a sports table cluster center.

Few-shot input/output style must follow the KG-side event lifting format.

Few-shot Input:
{json.dumps(fewshot_in_1, ensure_ascii=False)}
Few-shot Output:
{json.dumps(fewshot_out_1, ensure_ascii=False)}

Known KG events (controlled vocabulary, may be empty):
{json.dumps(kg_event_vocab, ensure_ascii=False)}

Known prior table events (may be empty):
{json.dumps(prior_event_vocab, ensure_ascii=False)}

Task:
- Prefer matching against KG events first, then prior table events.
- Match at the level of a broad event family / superclass, not narrow subtypes.
- If two tables differ only by minor field differences, wording differences, or table layout differences, treat them as the same event family.
- Only create a NEW event if the table clearly represents a genuinely different event family, not just a variant of an existing one.
- Reuse an existing event_name whenever the table is semantically the same large class of event.
- Return event_name, description, roles, and time_fields.

Table title: {title}
Columns: {", ".join(cols)}

Return JSON only:
{{
  "event_name": "..._event",
  "description": "...",
  "roles": {{"...","...",...}},
  "time_fields": ["..."],
  "is_new_event": true
}}"""

        obj = _parse_json_with_retry(llm, prompt, temperature=0.1, max_tokens=10000, retries=3)

        event_name = normalize_concept_rule_based(str(obj.get("event_name", "unknown_event")))
        if not event_name.endswith("_event"):
            event_name += "_event"

        event_desc = str(obj.get("description", "") or "").strip()
        is_new_event = bool(obj.get("is_new_event", False))
        if event_name not in _vocab_names(event_vocab):
            event_vocab.append({"name": event_name, "description": event_desc})

        roles = obj.get("roles", []) or []
        if not isinstance(roles, list):
            roles = [str(roles)]

        cluster_info[i] = {
            "event_name": event_name,
            "event_description": event_desc,
            "is_new_event": is_new_event,
            "description": event_desc,
            "roles": [str(x) for x in roles],
            "time_fields": [str(x) for x in (obj.get("time_fields", []) or [])],
            "semantic_concepts": [normalize_concept_rule_based(str(x)) for x in (obj.get("semantic_concepts", []) or []) if str(x).strip()],
            "semantic_relations": obj.get("semantic_relations", []) or [],
            "sample_table": _table_min_view(table),
        }
        print(f"  → event_name: {event_name} | desc: {event_desc or '[none]'} | new={is_new_event} | roles={cluster_info[i]['roles']} | time_fields={cluster_info[i]['time_fields']}")

    event_vocab_out = []
    seen_event = set()
    for item in event_vocab:
        name = str(item.get("name", "") if isinstance(item, dict) else item).strip()
        norm_name = normalize_concept_rule_based(name)
        if norm_name and norm_name not in seen_event:
            seen_event.add(norm_name)
            desc = str(item.get("description", "") or "").strip() if isinstance(item, dict) else ""
            event_vocab_out.append({"name": norm_name, "description": desc})

    return cluster_info, event_vocab_out


def reanalyze_event_representatives_for_mix(
    representative_tables: list[dict],
    cluster_info: dict,
    event_vocab: list[dict],
    llm: Optional[LLMClient] = None,
) -> tuple[dict, list[dict], dict]:
    llm = llm or LLMClient()
    if not representative_tables or not cluster_info:
        return cluster_info, event_vocab, {"clusters": [], "applied": []}

    event_to_items: Dict[str, list[dict]] = defaultdict(list)
    for cid, info in cluster_info.items():
        ev = normalize_concept_rule_based(str(info.get("event_name", "") or ""))
        if not ev:
            continue
        table = info.get("sample_table") or {}
        event_to_items[ev].append({
            "cluster_id": cid,
            "table_id": str(table.get("id", "")),
            "page_title": table.get("page_title", ""),
            "section_title": table.get("section_title", ""),
            "header": table.get("header", []),
            "attributes": table.get("header", []),
        })

    mix_report = {"clusters": [], "applied": []}
    event_vocab_norm = _normalize_vocab_items(event_vocab)
    vocab_names = _vocab_names(event_vocab_norm)

    for ev_name, items in event_to_items.items():
        if len(items) <= 1:
            continue
        prompt = f"""You are checking whether representative tables under the same event label are actually mixed.
Return JSON only.

Event label: {ev_name}
Representative tables:
{json.dumps(items, ensure_ascii=False)}

Task:
1. Judge whether this set is a significant mix of different event families.
2. If not mixed, return mixed=false.
3. If mixed, split the representative tables into coherent groups.
4. Keep the current event label for the group that truly matches it.
5. For other groups, propose new event labels in snake_case ending with _event.
6. For each group, provide short reason, and list table_ids belonging to it.

Return JSON:
{{
  "mixed": true,
  "reason": "...",
  "groups": [
    {{"event_name":"..._event","table_ids":["..."],"reason":"...","description":"..."}}
  ]
}}"""
        obj = _parse_json_with_retry(llm, prompt, temperature=0.1, max_tokens=10000, retries=3)
        if not bool(obj.get("mixed", False)):
            continue

        groups = obj.get("groups", []) or []
        if not isinstance(groups, list) or len(groups) < 2:
            continue

        mix_report["clusters"].append({"event_name": ev_name, "reason": str(obj.get("reason", "") or ""), "groups": groups})

        keep_group = None
        for g in groups:
            g_name = normalize_concept_rule_based(str(g.get("event_name", "") or ""))
            if g_name == ev_name:
                keep_group = g
                break
        if keep_group is None:
            keep_group = groups[0]

        for g in groups:
            g_name = normalize_concept_rule_based(str(g.get("event_name", "") or ""))
            if not g_name:
                continue
            if not g_name.endswith("_event"):
                g_name += "_event"
            if g_name not in vocab_names:
                event_vocab_norm.append({"name": g_name, "description": str(g.get("description", "") or "").strip()})
                vocab_names.append(g_name)
            mix_report["applied"].append({
                "source_event": ev_name,
                "target_event": g_name,
                "table_ids": [str(x) for x in (g.get("table_ids", []) or [])],
            })
            for cid, info in cluster_info.items():
                sample_table = info.get("sample_table") or {}
                if str(sample_table.get("id", "")) in {str(x) for x in (g.get("table_ids", []) or [])}:
                    cluster_info[cid]["event_name"] = g_name
                    if g.get("description"):
                        cluster_info[cid]["event_description"] = str(g.get("description", "") or "").strip()

    return cluster_info, event_vocab_norm, mix_report




# gemini新增
def refine_event_descriptions_by_group(
    cluster_info: dict,
    event_vocab: list[dict],
    llm: Optional[LLMClient] = None,
) -> tuple[dict, list[dict]]:
    """
    在 mix 拆分完成后，将相同 event_name 的表格聚合，
    利用 LLM 根据组内所有代表性表格的信息重写该事件的全局描述。
    """
    llm = llm or LLMClient()
    event_to_tables = defaultdict(list)

    # 1. 按 event_name 聚合表格
    for cid, info in cluster_info.items():
        ev_name = info.get("event_name")
        if ev_name:
            # 使用 _table_evidence_brief 获取包含标题、列名和少量行的精华信息
            table_view = _table_evidence_brief(info.get("sample_table") or {})
            event_to_tables[ev_name].append(table_view)

    event_desc_map = {}
    print(f"\n[Event] 开始基于组内代表性表格，重写 {len(event_to_tables)} 个事件的全局描述...")

    # 2. 让 LLM 为每个 event_name 生成新描述
    for ev_name, tables in event_to_tables.items():
        prompt = f"""You are an expert in schema integration and knowledge graphs.
You have a set of representative tables that have been clustered into the same event family.
Event Name: {ev_name}

Tables in this event group:
{json.dumps(tables, ensure_ascii=False)}

Task:
Write a concise, comprehensive event description (1-2 sentences) that covers the shared semantic meaning of these tables.
Do NOT blindly copy a prior description. Synthesize a NEW description explicitly based on the table titles, columns, and rows provided.

Return JSON only:
{{
  "description": "..."
}}"""
        obj = _parse_json_with_retry(llm, prompt, temperature=0.1, max_tokens=10000, retries=3)
        desc = str(obj.get("description", "") or "").strip()
        
        if desc:
            event_desc_map[ev_name] = desc
            print(f"  -> {ev_name} 更新描述: {desc}")

    # 3. 更新 cluster_info 中的描述
    for cid, info in cluster_info.items():
        ev_name = info.get("event_name")
        if ev_name in event_desc_map:
            cluster_info[cid]["event_description"] = event_desc_map[ev_name]
            cluster_info[cid]["description"] = event_desc_map[ev_name]

    # 4. 更新 event_vocab 中的描述
    for item in event_vocab:
        name = item.get("name")
        if name in event_desc_map:
            item["description"] = event_desc_map[name]

    return cluster_info, event_vocab


def analyze_static_representative_tables(
    representative_tables: list[dict],
    table_event_list: list[Any] | None = None,
    kg_concept_list: list[Any] | None = None,
    prior_new_concept_list: list[Any] | None = None,
    llm: Optional[LLMClient] = None,
) -> tuple[dict, list[dict], list[dict]]:
    llm = llm or LLMClient()
    cluster_info = {}

    table_event_vocab = _normalize_vocab_items(list(table_event_list or []))
    kg_concept_vocab = _normalize_vocab_items(list(kg_concept_list or []))
    prior_new_concept_vocab = _normalize_vocab_items(list(prior_new_concept_list or []))
    concept_vocab = kg_concept_vocab + [x for x in prior_new_concept_vocab if x.get("name") not in _vocab_names(kg_concept_vocab)]
    kg_concept_name_map = {normalize_concept_rule_based(c.get("name", c)): c.get("name", c) for c in kg_concept_vocab}

    matched_event_list_for_table = []
    concept_list_for_table = []

    for i, table in enumerate(representative_tables):
        title = f"{table['page_title']} - {table['section_title']}"
        cols = table.get("header", [])
        rows = table.get("rows", [])
        evidence = _table_evidence_brief(table)

        prompt = f"""You analyze STATIC tables and must use table evidence to produce semantic concepts.

Prefer broad semantic families over narrow subtypes.
If two tables are clearly close in meaning, it is okay to map them to the same broad label even when their schemas or wording are not identical.
Only separate them when they feel like genuinely different families.

Use this soft priority:
1) event_match for broad table-event families
2) concept_match for broad KG concepts or previously found table concepts
3) new_concept when no broad match feels appropriate

Important rules:
- Be permissive and reuse an existing broad label when the table is reasonably close semantically.
- Do NOT split one broad class into many small variants just because of minor schema or wording differences.
- If a table looks like a loose / partial / imperfect version of an existing event family, still prefer that existing event family.
- If route is concept_match, return a concise, evidence-based description for the matched concept family.
- If route is new_concept, return a new broader conceptual label and description derived from the table evidence.
- Always write descriptions grounded in the table title, columns, captions, and sample rows.

Event vocab (name + description): {json.dumps(table_event_vocab, ensure_ascii=False)}
KG concepts (name + description): {json.dumps(concept_vocab, ensure_ascii=False)}
Table evidence: {json.dumps(evidence, ensure_ascii=False)}

Return JSON:
{{
 "route":"event_match|concept_match|new_concept",
 "matched_event_name":"..._event or empty",
 "matched_event_description":"... or empty",
 "matched_kg_concept":"Award|Coach|Division|Player|Position|Team|Venue or empty",
 "matched_kg_concept_description":"... or empty",
 "new_hyper_concept":"snake_case or empty",
 "new_hyper_concept_description":"... or empty",
 "semantic_relations":[{{"predicate":"...","subject":"...","object":"..."}}]
}}"""
        obj = _parse_json_with_retry(llm, prompt, temperature=0.1, max_tokens=10000, retries=3)

        route = str(obj.get("route", "")).strip().lower()
        matched_event_name = normalize_concept_rule_based(str(obj.get("matched_event_name", "") or ""))
        if matched_event_name and not matched_event_name.endswith("_event"):
            matched_event_name += "_event"
        matched_event_desc = str(obj.get("matched_event_description", "") or "").strip()

        matched_kg_concept_raw = str(obj.get("matched_kg_concept", "") or "").strip()
        matched_kg_concept_norm = normalize_concept_rule_based(matched_kg_concept_raw)
        matched_kg_concept_final = kg_concept_name_map.get(matched_kg_concept_norm, "")
        matched_kg_concept_desc = str(obj.get("matched_kg_concept_description", "") or "").strip()

        new_hyper_concept = normalize_concept_rule_based(str(obj.get("new_hyper_concept", "") or ""))
        new_hyper_concept_desc = str(obj.get("new_hyper_concept_description", "") or "").strip()
        relations = obj.get("semantic_relations", []) or []

        event_name = ""
        event_description = ""
        static_concepts_out = []
        decision = "unknown"

        if route == "event_match" and matched_event_name:
            event_name = matched_event_name
            event_description = matched_event_desc
            static_concepts_out = []
            matched_event_list_for_table.append({"name": event_name, "description": event_description})
            decision = f"event_match -> {event_name}"
        elif route == "concept_match" and matched_kg_concept_final:
            desc = _stable_entity_description(
                matched_kg_concept_final,
                matched_kg_concept_desc or matched_event_desc or "",
                evidence=evidence,
            )
            static_concepts_out.append({"name": matched_kg_concept_final, "description": desc})
            concept_list_for_table.append({"name": matched_kg_concept_final, "description": desc})
            decision = f"concept_match -> {matched_kg_concept_final}"
        else:
            if not new_hyper_concept:
                new_hyper_concept = "unknown_hyper_concept"
            desc = _stable_entity_description(
                new_hyper_concept,
                new_hyper_concept_desc or matched_event_desc or matched_kg_concept_desc or "",
                evidence=evidence,
            )
            static_concepts_out.append({"name": new_hyper_concept, "description": desc})
            concept_list_for_table.append({"name": new_hyper_concept, "description": desc})
            decision = f"new_concept -> {new_hyper_concept}"

        cluster_info[i] = {
            "route": route if route else "new_concept",
            "decision": decision,
            "event_name": event_name,
            "event_description": event_description,
            "time_mode": "relaxed" if event_name else "",
            "matched_kg_concept": matched_kg_concept_final,
            "matched_kg_concept_description": matched_kg_concept_desc,
            "new_hyper_concept": new_hyper_concept,
            "new_hyper_concept_description": new_hyper_concept_desc,
            "semantic_concepts": static_concepts_out,
            "semantic_relations": relations,
            "sample_table": _table_min_view(table),
        }
        print(f"  → route: {cluster_info[i]['route']} | decision: {decision} | concepts: {static_concepts_out}")

    event_vocab_out = []
    seen_event = set()
    for item in matched_event_list_for_table:
        name = normalize_concept_rule_based(str(item.get("name", "") if isinstance(item, dict) else item))
        if name and name not in seen_event:
            seen_event.add(name)
            if isinstance(item, dict):
                event_vocab_out.append({"name": name, "description": str(item.get("description", "") or "").strip()})
            else:
                event_vocab_out.append({"name": name, "description": ""})

    concept_vocab_out = []
    seen_concept = set()
    for item in concept_list_for_table:
        name = normalize_concept_rule_based(str(item.get("name", "") if isinstance(item, dict) else item))
        if name and name not in seen_concept:
            seen_concept.add(name)
            if isinstance(item, dict):
                concept_vocab_out.append({"name": name, "description": str(item.get("description", "") or "").strip()})
            else:
                concept_vocab_out.append({"name": name, "description": ""})

    return cluster_info, event_vocab_out, concept_vocab_out


# ============================
# 7) Aggregation
# ============================
def aggregate_event_results(
    tables: list[dict],
    labels: np.ndarray,
    cluster_info: dict
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    semantic_concepts = {}
    event_layer = {}
    cluster_tables_map = defaultdict(list)

    for table_idx, cluster_id in enumerate(labels):
        cluster_tables_map[cluster_id].append(tables[table_idx])

    for cluster_id, info in cluster_info.items():
        event_type = info.get("event_name", "unknown_event")
        description = info.get("description", "")
        roles = info.get("roles", []) or []
        time_fields = info.get("time_fields", []) or []
        concepts_raw = info.get("semantic_concepts", []) or []
        concepts = [c.get("name", c) if isinstance(c, dict) else c for c in concepts_raw]
        relations = info.get("semantic_relations", []) or []

        if event_type not in event_layer:
            event_layer[event_type] = {
                "event_name": event_type,
                "description": description,
                "roles": roles,
                "time_fields": time_fields,
                "count": 0,
                "semantic_concepts": list(set(concepts)),
                "semantic_relations": relations[:],
                "sample": [],
                "time_mode": "strict",
            }

        cluster_tables = cluster_tables_map.get(cluster_id, [])
        event_layer[event_type]["count"] += len(cluster_tables)
        if len(event_layer[event_type]["sample"]) < 3:
            for table in cluster_tables[:3]:
                event_layer[event_type]["sample"].append(_table_min_view(table))

        for c in concepts:
            cc = normalize_concept_rule_based(str(c))
            if not cc:
                continue
            semantic_concepts.setdefault(cc, {"count": 0, "events": set()})
            semantic_concepts[cc]["count"] += len(cluster_tables)
            semantic_concepts[cc]["events"].add(event_type)

    for c in semantic_concepts:
        semantic_concepts[c]["events"] = sorted(list(semantic_concepts[c]["events"]))

    return {"concepts": semantic_concepts}, event_layer


def aggregate_static_results(
    tables: list[dict],
    labels: np.ndarray,
    cluster_info: dict,
) -> Dict[str, Any]:
    static_event_layer = {}
    static_concept_catalog = {}
    cluster_tables_map = defaultdict(list)

    # Only emit KG concepts when they were explicitly matched by LLM on table evidence.
    kg_prior_norms = {normalize_concept_rule_based(c.get("name", c)) for c in KG_CONCEPT_TYPES}

    for table_idx, cluster_id in enumerate(labels):
        cluster_tables_map[cluster_id].append(tables[table_idx])

    for cluster_id, info in cluster_info.items():
        concepts_raw = info.get("semantic_concepts", []) or []
        concepts = [c.get("name", c) if isinstance(c, dict) else c for c in concepts_raw]
        relations = info.get("semantic_relations", []) or []
        event_type = normalize_concept_rule_based(str(info.get("event_name", "") or ""))
        time_mode = info.get("time_mode", "relaxed") if event_type else ""
        decision = info.get("decision", "")
        matched_kg_concept = info.get("matched_kg_concept", "")
        new_hyper_concept = info.get("new_hyper_concept", "")
        matched_kg_desc = info.get("matched_kg_concept_description", "")
        new_hyper_desc = info.get("new_hyper_concept_description", "")

        cluster_tables = cluster_tables_map.get(cluster_id, [])

        if event_type:
            if not event_type.endswith("_event"):
                event_type += "_event"
            static_event_layer.setdefault(event_type, {
                "event_name": event_type,
                "description": info.get("event_description", ""),
                "count": 0,
                "semantic_concepts": [],
                "semantic_relations": [],
                "sample_tables": [],
                "time_mode": time_mode or "relaxed",
                "source_clusters": [],
            })
            static_event_layer[event_type]["count"] += len(cluster_tables)
            static_event_layer[event_type]["semantic_concepts"].extend(concepts)
            static_event_layer[event_type]["semantic_relations"].extend(relations)
            static_event_layer[event_type]["source_clusters"].append({
                "cluster_id": cluster_id,
                "decision": decision,
                "sample_table": info.get("sample_table", {}),
            })
            if len(static_event_layer[event_type]["sample_tables"]) < 3:
                for t in cluster_tables[:3]:
                    static_event_layer[event_type]["sample_tables"].append(_table_min_view(t))

        matched_kg_norm = normalize_concept_rule_based(str(matched_kg_concept or ""))
        new_hyper_norm = normalize_concept_rule_based(str(new_hyper_concept or ""))

        for c in concepts:
            c_str = str(c).strip()
            c_norm = normalize_concept_rule_based(c_str)
            if not c_norm:
                continue
            if c_norm in kg_prior_norms and c_norm != matched_kg_norm:
                continue
            if c_norm == matched_kg_norm and not matched_kg_desc:
                # matched KG concept should still be emitted, but description must come from LLM/evidence later
                pass

            if matched_kg_norm and c_norm == matched_kg_norm:
                cname = matched_kg_concept or c_norm
                desc = matched_kg_desc
            elif new_hyper_norm and c_norm == new_hyper_norm:
                cname = new_hyper_concept or c_norm
                desc = new_hyper_desc
            else:
                cname = c_norm
                desc = new_hyper_desc or matched_kg_desc

            if cname in kg_prior_norms and cname != matched_kg_concept:
                continue

            resolved_desc = _stable_entity_description(cname, desc, evidence=info.get("sample_table", {}))
            static_concept_catalog.setdefault(cname, {
                "concept_name": cname,
                "description": resolved_desc,
                "attributes": [],
                "sample": [],
                "source_clusters": [],
                "relations": [],
            })
            if len(static_concept_catalog[cname]["sample"]) < 3:
                static_concept_catalog[cname]["sample"].append(info.get("sample_table", {}))
            static_concept_catalog[cname]["source_clusters"].append({
                "cluster_id": cluster_id,
                "decision": decision,
                "matched_kg_concept": matched_kg_concept,
                "new_hyper_concept": new_hyper_concept,
            })
            static_concept_catalog[cname]["attributes"] = list(dict.fromkeys(static_concept_catalog[cname]["attributes"] + _flatten_attrs(info.get("semantic_attributes", []))))

    for ev in static_event_layer:
        static_event_layer[ev]["semantic_concepts"] = _normalize_name_list(static_event_layer[ev]["semantic_concepts"])
        static_event_layer[ev]["semantic_relations"] = [
            {
                "predicate": r.get("predicate", ""),
                "subject": normalize_concept_rule_based(r.get("subject", "")),
                "object": normalize_concept_rule_based(r.get("object", "")),
            }
            for r in static_event_layer[ev]["semantic_relations"]
        ]

    semantic_entities = {}
    for cname, payload in static_concept_catalog.items():
        semantic_entities[cname] = {
            "concept_name": payload.get("concept_name", cname),
            "description": _stable_entity_description(cname, payload.get("description", ""), evidence=payload.get("sample", [{}])[0] if payload.get("sample") else None),
            "attributes": payload.get("attributes", []),
            "sample": payload.get("sample", []),
            "relations": [],
        }

    return {
        "entities": semantic_entities,
        "events": static_event_layer,
    }


def merge_semantic_parts(parts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge table semantic entities only, aligned to KG semantic schema."""
    merged = {"entities": {}}
    for p in parts:
        for c, payload in p.get("entities", {}).items():
            merged.setdefault("entities", {}).setdefault(c, {
                "concept_name": payload.get("concept_name", c),
                "description": payload.get("description", ""),
                "attributes": payload.get("attributes", []),
                "sample": [],
                "relations": [],
            })
            merged["entities"][c]["attributes"] = list(dict.fromkeys(merged["entities"][c]["attributes"] + list(payload.get("attributes", []))))
            for s in payload.get("sample", [])[:3]:
                if len(merged["entities"][c]["sample"]) < 3:
                    merged["entities"][c]["sample"].append(s)
    return merged


# ============================
# 8) NEW: tables index by event/concept
# ============================
def build_event_table_index(
    event_tables: list[dict],
    labels_event: np.ndarray,
    event_cluster_info: dict,
    static_tables: list[dict],
    labels_static: np.ndarray,
    static_cluster_info: dict,
) -> Dict[str, Any]:
    """
    输出结构:
    event_name -> {
      strict: [...tables from dynamic/event path...],
      relaxed: [...tables from static path matched to event...]
    }
    """
    out = {}

    # strict from event path
    cluster_tables_map_event = defaultdict(list)
    for i, cid in enumerate(labels_event):
        cluster_tables_map_event[cid].append(event_tables[i])

    for cid, info in event_cluster_info.items():
        ev = normalize_concept_rule_based(info.get("event_name", "unknown_event"))
        if not ev.endswith("_event"):
            ev += "_event"
        out.setdefault(ev, {"strict": [], "relaxed": []})
        out[ev]["strict"].extend([_table_min_view(t) for t in cluster_tables_map_event.get(cid, [])])

    # relaxed from static path
    cluster_tables_map_static = defaultdict(list)
    for i, cid in enumerate(labels_static):
        cluster_tables_map_static[cid].append(static_tables[i])

    for cid, info in static_cluster_info.items():
        ev = normalize_concept_rule_based(str(info.get("event_name", "") or ""))
        if not ev:
            continue
        if not ev.endswith("_event"):
            ev += "_event"
        out.setdefault(ev, {"strict": [], "relaxed": []})
        out[ev]["relaxed"].extend([_table_min_view(t) for t in cluster_tables_map_static.get(cid, [])])

    # dedup by table id
    for ev in out:
        for mode in ["strict", "relaxed"]:
            dedup, seen = [], set()
            for t in out[ev][mode]:
                tid = str(t.get("id", ""))
                key = tid if tid else f"{t.get('page_title','')}::{t.get('section_title','')}"
                if key not in seen:
                    seen.add(key)
                    dedup.append(t)
            out[ev][mode] = dedup

    return out


def build_concept_table_index(
    event_tables: list[dict],
    labels_event: np.ndarray,
    event_cluster_info: dict,
    static_tables: list[dict],
    labels_static: np.ndarray,
    static_cluster_info: dict,
) -> Dict[str, Any]:
    """
    输出结构:
    {
      "entities": [
        {
          "concept_name": "...",
          "description": "...",
          "tables": [...],
          "sources": ["event_path", "static_path"]
        }
      ]
    }
    """
    out = {}

    cluster_tables_map_event = defaultdict(list)
    for i, cid in enumerate(labels_event):
        cluster_tables_map_event[cid].append(event_tables[i])

    for cid, info in event_cluster_info.items():
        concepts = info.get("semantic_concepts", []) or []
        tables_here = cluster_tables_map_event.get(cid, [])
        for c in concepts:
            cc = normalize_concept_rule_based(str(c))
            if not cc:
                continue
            out.setdefault(cc, {"tables": [], "sources": set(), "description": ""})
            out[cc]["tables"].extend([_table_min_view(t) for t in tables_here])
            out[cc]["sources"].add("event_path")
            if not out[cc]["description"]:
                out[cc]["description"] = str(info.get("event_description", "") or "").strip()

    cluster_tables_map_static = defaultdict(list)
    for i, cid in enumerate(labels_static):
        cluster_tables_map_static[cid].append(static_tables[i])

    for cid, info in static_cluster_info.items():
        concepts = info.get("semantic_concepts", []) or []
        tables_here = cluster_tables_map_static.get(cid, [])
        for c in concepts:
            c_name = str(c).strip()
            cc = normalize_concept_rule_based(c_name)
            key_norm = cc
            out.setdefault(key_norm, {"tables": [], "sources": set(), "description": ""})
            out[key_norm]["tables"].extend([_table_min_view(t) for t in tables_here])
            out[key_norm]["sources"].add("static_path")
            if not out[key_norm]["description"]:
                out[key_norm]["description"] = str(info.get("matched_kg_concept_description", "") or info.get("new_hyper_concept_description", "") or "").strip()

    entities = []
    for c, payload in out.items():
        dedup, seen = [], set()
        for t in payload["tables"]:
            tid = str(t.get("id", ""))
            key = tid if tid else f"{t.get('page_title','')}::{t.get('section_title','')}"
            if key not in seen:
                seen.add(key)
                dedup.append(t)
        entities.append({
            "concept_name": c,
            "description": _stable_entity_description(c, payload.get("description", ""), evidence=dedup[0] if dedup else None),
            "tables": dedup,
            "sources": sorted(list(payload["sources"])),
        })

    return {"entities": entities}


# ============================
# 9) canonicalization (unchanged simplified)
# ============================
def build_concept_context(concept: str, semantic_layer_concepts: Dict[str, Any]) -> Dict[str, Any]:
    item = semantic_layer_concepts.get(concept, {})
    return {"concept": concept, "count": item.get("count", 0), "events": item.get("events", [])}


def llm_canonicalize_concepts(
    semantic_layer_concepts: Dict[str, Any],
    llm: LLMClient,
    batch_size: int = 50,
    temperature: float = 0.1,
    max_tokens: int = 10000,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    normalized_groups: Dict[str, List[str]] = defaultdict(list)
    for c in semantic_layer_concepts.keys():
        normalized_groups[normalize_concept_rule_based(c)].append(c)

    pre_alias_to_rep: Dict[str, str] = {}
    rep_concepts: List[str] = []
    for _, originals in normalized_groups.items():
        rep = sorted(originals, key=lambda x: (len(x), x))[0]
        rep_concepts.append(rep)
        for o in originals:
            pre_alias_to_rep[o] = rep

    rep_concepts = sorted(rep_concepts)

    def chunk(lst: List[str], n: int) -> List[List[str]]:
        return [lst[i:i + n] for i in range(0, len(lst), n)]

    batches = chunk(rep_concepts, batch_size)
    core_concepts: List[str] = []
    core_definitions: Dict[str, str] = {}
    rep_to_core: Dict[str, str] = {}

    for bi, batch in enumerate(batches, start=1):
        ctx = [build_concept_context(c, semantic_layer_concepts) for c in batch]
        prompt = f"""Build canonical concepts. Return JSON only.
Previous core: {json.dumps(core_concepts, ensure_ascii=False)}
Batch: {json.dumps(ctx, ensure_ascii=False)}
Return: {{"core_concepts":[],"alias_to_core":{{}},"core_definitions":{{}}}}"""
        obj = llm.complete_json(prompt, temperature=temperature, max_tokens=max_tokens) or {}
        raw_core = obj.get("core_concepts", []) or []
        raw_a2c = obj.get("alias_to_core", {}) or {}
        raw_defs = obj.get("core_definitions", {}) or {}

        new_core = [normalize_concept_rule_based(str(c)) for c in raw_core if str(c).strip()]
        merged_core = set(core_concepts) | set(new_core)

        a2c = {str(k): normalize_concept_rule_based(str(v)) for k, v in raw_a2c.items()}
        for c in batch:
            if c not in a2c:
                a2c[c] = normalize_concept_rule_based(c)
        for tgt in a2c.values():
            if tgt:
                merged_core.add(tgt)
        for c in batch:
            rep_to_core[c] = a2c.get(c, normalize_concept_rule_based(c))
        for k, v in raw_defs.items():
            nk = normalize_concept_rule_based(str(k))
            if nk and nk not in core_definitions:
                core_definitions[nk] = str(v)

        core_concepts = sorted(list(merged_core))
        print(f"[Canonicalize] {bi}/{len(batches)} done, core={len(core_concepts)}")

    alias_to_canonical = {}
    for old in semantic_layer_concepts.keys():
        rep = pre_alias_to_rep[old]
        core = rep_to_core.get(rep, rep)
        alias_to_canonical[old] = normalize_concept_rule_based(core)

    return alias_to_canonical, core_definitions


def apply_alias_map_to_concepts(semantic_concepts: Dict[str, Any], alias_to_canonical: Dict[str, str]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for old_concept, payload in semantic_concepts.items():
        canonical = alias_to_canonical.get(old_concept, old_concept)
        merged.setdefault(canonical, {"count": 0, "events": set()})
        merged[canonical]["count"] += int(payload.get("count", 0))
        for ev in payload.get("events", []):
            merged[canonical]["events"].add(ev)

    for c in merged:
        merged[c]["events"] = sorted(list(merged[c]["events"]))
    return merged


def _rewrite_relations_with_alias(relations: Dict[str, Any], alias_to_canonical: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for _, payload in relations.items():
        triple = payload.get("triple", {})
        subj = alias_to_canonical.get(triple.get("subject", ""), triple.get("subject", ""))
        pred = triple.get("predicate", "")
        obj = alias_to_canonical.get(triple.get("object", ""), triple.get("object", ""))
        key = f"{subj}::{pred}::{obj}"
        out.setdefault(key, {
            "count": 0, "events": set(),
            "triple": {"subject": subj, "predicate": pred, "object": obj},
            "examples": []
        })
        out[key]["count"] += int(payload.get("count", 0))
        for ev in payload.get("events", []):
            out[key]["events"].add(ev)
        for ex in payload.get("examples", [])[:2]:
            if len(out[key]["examples"]) < 3:
                out[key]["examples"].append(ex)

    for k in out:
        out[k]["events"] = sorted(list(out[k]["events"]))
    return out


# ============================
# 10) Main
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_clusters", default=50, type=int)
    parser.add_argument("--method", default="sentence-transformers", choices=["tfidf", "sentence-transformers"])
    # Canonicalization flags removed.
    parser.add_argument("--llm_base_url", default=os.getenv("LLM_BASE_URL", ""))
    parser.add_argument("--llm_api_key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--llm_model", default=os.getenv("LLM_MODEL", "DeepSeek-V3.2-Fast"))
    args = parser.parse_args()

    _ensure_dir(args.output_dir)
    cache_dir = os.path.join(args.output_dir, "cluster_result")
    _ensure_dir(cache_dir)

    llm = LLMClient(args.llm_base_url, args.llm_api_key, args.llm_model, timeout=60)

    kg_event_path = os.path.join(os.path.dirname(__file__), "output", "kg_event.json")
    kg_semantic_path = os.path.join(os.path.dirname(__file__), "output", "kg_semantic.json")
    kg_event_vocab, kg_concept_vocab = load_kg_vocab_from_json(kg_event_path, kg_semantic_path)

    metadata_digest = _sha256_file(args.metadata_path)
    cache_meta = {
        "version": 6,
        "metadata_path": args.metadata_path,
        "metadata_digest": metadata_digest,
        "method": args.method,
        "n_clusters": args.n_clusters,
        "kg_event_types": _normalize_name_list([x.get("name", "") for x in kg_event_vocab]),
        "kg_concept_types": _normalize_name_list([x.get("name", "") for x in kg_concept_vocab]),
        "reanalysis": "representative_mix_split",
    }

    tables = parse_metadata_sql(args.metadata_path)
    event_tables, static_tables = split_tables_by_time(tables)

    event_vocab = list(kg_event_vocab)
    concept_vocab = list(kg_concept_vocab)
    # Do not seed table_semantic.json with raw KG concepts.
    # Only keep concepts that are actually supported by table evidence / LLM analysis.
    semantic_entities_source = {}

    # Event path
    event_semantic_part, event_layer = {"concepts": {}, "relations": {}}, {}
    event_cluster_info = {}
    labels_event = np.array([], dtype=int)

    if event_tables:
        event_cache_file = _cache_path(cache_dir, "event")
        event_cache = load_cluster_cache(event_cache_file)
        loaded = False
        if event_cache and event_cache.get("meta", {}) == cache_meta:
            try:
                labels_event = np.array(event_cache["labels"], dtype=int)
                _ = pick_representatives_from_ids(event_cache["representative_table_ids"], build_table_id_index(event_tables))
                event_cluster_info = {int(k): v for k, v in event_cache["cluster_info"].items()}
                event_vocab = _normalize_vocab_items(list(event_cache.get("event_list", event_vocab)))
                loaded = True
                print("[Event] cache hit")
            except Exception:
                loaded = False

        if not loaded:
            rep_event, labels_event = cluster_tables(event_tables, args.n_clusters, method=args.method)
            event_cluster_info, event_vocab = analyze_event_representative_tables(
                rep_event,
                kg_event_list=kg_event_vocab,
                prior_table_event_list=event_vocab,
                llm=llm,
            )
            event_cluster_info, event_vocab, mix_report = reanalyze_event_representatives_for_mix(
                rep_event,
                event_cluster_info,
                event_vocab,
                llm=llm,
            )
            
            # gemini新增
            # +++++ 新增：结合组内表格重写全局事件描述 +++++
            event_cluster_info, event_vocab = refine_event_descriptions_by_group(
                event_cluster_info,
                event_vocab,
                llm=llm
            )
            # ++++++++++++++++++++++++++++++++++++++++++++

            
            save_cluster_cache(event_cache_file, {
                "meta": cache_meta,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "labels": labels_event.tolist(),
                "representative_table_ids": [str(t.get("id")) for t in rep_event],
                "cluster_info": event_cluster_info,
                "event_list": event_vocab,
                "mix_report": mix_report,
            })

        event_semantic_part, event_layer = aggregate_event_results(event_tables, labels_event, event_cluster_info)

    # Static path
    static_semantic_part = {"concepts": {}, "relations": {}, "events": {}, "static_concepts": {}}
    static_cluster_info = {}
    static_event_matches = []
    labels_static = np.array([], dtype=int)

    if static_tables:
        static_cache_file = _cache_path(cache_dir, "static")
        static_cache = load_cluster_cache(static_cache_file)
        loaded = False
        if static_cache and static_cache.get("meta", {}) == cache_meta:
            try:
                labels_static = np.array(static_cache["labels"], dtype=int)
                _ = pick_representatives_from_ids(static_cache["representative_table_ids"], build_table_id_index(static_tables))
                static_cluster_info = {int(k): v for k, v in static_cache["cluster_info"].items()}
                static_event_matches = static_cache.get("static_event_matches", [])
                loaded = True
                print("[Static] cache hit")
            except Exception:
                loaded = False

        if not loaded:
            rep_static, labels_static = cluster_tables(static_tables, args.n_clusters, method=args.method)
            static_cluster_info, static_event_matches, new_concept_vocab = analyze_static_representative_tables(
                rep_static,
                table_event_list=event_vocab,
                kg_concept_list=kg_concept_vocab,
                prior_new_concept_list=concept_vocab,
                llm=llm,
            )
            if new_concept_vocab:
                concept_vocab = concept_vocab + [x for x in new_concept_vocab if x.get("name") not in _vocab_names(concept_vocab)]
            save_cluster_cache(static_cache_file, {
                "meta": cache_meta,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "labels": labels_static.tolist(),
                "representative_table_ids": [str(t.get("id")) for t in rep_static],
                "cluster_info": static_cluster_info,
                "static_event_matches": static_event_matches,
            })

        for ev in static_event_matches:
            if ev.get("name") and ev.get("name") not in _vocab_names(event_vocab):
                event_vocab.append(ev)
        static_semantic_part = aggregate_static_results(static_tables, labels_static, static_cluster_info)
        semantic_entities_source.update(static_semantic_part.get("entities", {}))

    semantic_layer = {"entities": semantic_entities_source}

    # ===== NEW indexes =====
    event_tables_index = build_event_table_index(
        event_tables, labels_event, event_cluster_info,
        static_tables, labels_static, static_cluster_info
    )
    concept_tables_index = build_concept_table_index(
        event_tables, labels_event, event_cluster_info,
        static_tables, labels_static, static_cluster_info
    )

    # outputs
    index_output_dir = os.path.join(args.output_dir, "index")
    _ensure_dir(index_output_dir)
    semantic_output = os.path.join(args.output_dir, "table_semantic.json")
    event_output = os.path.join(args.output_dir, "table_event.json")
    event_tables_index_output = os.path.join(index_output_dir, "table_event_tables_index.json")
    concept_tables_index_output = os.path.join(index_output_dir, "table_concept_tables_index.json")

    with open(semantic_output, "w", encoding="utf-8") as f:
        json.dump(semantic_layer, f, indent=2, ensure_ascii=False)
    with open(event_output, "w", encoding="utf-8") as f:
        json.dump(event_layer, f, indent=2, ensure_ascii=False)
    with open(event_tables_index_output, "w", encoding="utf-8") as f:
        json.dump(event_tables_index, f, indent=2, ensure_ascii=False)
    with open(concept_tables_index_output, "w", encoding="utf-8") as f:
        json.dump(concept_tables_index, f, indent=2, ensure_ascii=False)
    with open(semantic_output, "w", encoding="utf-8") as f:
        json.dump(semantic_layer, f, indent=2, ensure_ascii=False)

    print(f"✓ saved: {semantic_output}")
    print(f"✓ saved: {event_output}")
    print(f"✓ saved: {event_tables_index_output}  (per-event strict/relaxed tables)")
    print(f"✓ saved: {concept_tables_index_output} (per-concept tables)")

    print("\n========== 运行汇总 ==========")
    print(f"事件表数量: {len(event_tables)}")
    print(f"静态表数量: {len(static_tables)}")
    print(f"事件层类型数: {len(event_layer)}")
    print(f"语义概念数: {len(semantic_layer.get('concepts', {}))}")
    print(f"事件索引数: {len(event_tables_index)}")
    print(f"语义索引数: {len(concept_tables_index)}")
    print("============================\n")

    # Canonicalization is disabled: keep only the non-merged semantic outputs.


if __name__ == "__main__":
    main()