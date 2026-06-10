"""
utils/data_load.py

Data loaders for QA queries and source profiles.
Supports classic datasets under data/rag and MMQA under data/MMQA.
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple


def _load_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {lineno} in {path}: {e.msg} (col {e.colno})") from e
    return rows


def load_queries(dataset: str, sample_size: int = None, dataset_path: str = None) -> List[Dict[str, str]]:
    """
    Load query/answer pairs.

    Supported modes:
    - dataset == "mmqa": read from `dataset_path` if given, else data/MMQA/kg-table-t2k-65.jsonl
    - dataset == "mmqa_doc": kg-doc benchmark; read from `dataset_path` if given,
      else data/MMQA/kg-doc-160.jsonl. Same q1/q2 layout as mmqa.
    - otherwise: read from `dataset_path` if given, else data/rag/{dataset}_qa.json or data/rag/{dataset}.json
    """
    dataset_l = dataset.lower().strip()

    if dataset_l in ("mmqa", "mmqa_doc"):
        _default_qa = ("data/MMQA/kg-doc-160.jsonl" if dataset_l == "mmqa_doc"
                       else "data/MMQA/kg-table-t2k-65.jsonl")
        answer_path = (dataset_path or "").strip() or _default_qa
        if not os.path.exists(answer_path):
            raise FileNotFoundError(f"Missing MMQA answer file: {answer_path}")

        answer_rows = _load_jsonl(answer_path)
        pairs = []
        for row in answer_rows:
            q = str(row.get("question", "")).strip()
            q1 = row.get("q1") or {}
            q2 = row.get("q2") or {}
            q1_answers = q1.get("answer") or []
            q2_answers = q2.get("answers") or []
            if not q:
                continue
            if isinstance(q2_answers, list) and q2_answers:
                gt = " ; ".join(str(x).strip() for x in q2_answers if str(x).strip())
            else:
                gt = ""
            pairs.append({"query": q, "ground_truth": gt, "q1_answers": q1_answers, "q2_answers": q2_answers, "raw": row})

        if sample_size is None or sample_size >= len(pairs):
            return pairs

        total_size = len(pairs)
        indices = np.linspace(0, total_size - 1, sample_size, dtype=int)
        return [pairs[i] for i in indices]

    if (dataset_path or "").strip():
        file_path = dataset_path.strip()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"--dataset-path not found: {file_path}")
    else:
        qa_path = f"data/rag/{dataset}_qa.json"
        standard_path = f"data/rag/{dataset}.json"

        if os.path.exists(qa_path):
            file_path = qa_path
        elif os.path.exists(standard_path):
            file_path = standard_path
        else:
            raise FileNotFoundError(f"Neither {qa_path} nor {standard_path} exists")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = [
        {"query": item.get("question", ""), "ground_truth": item.get("answer", "")}
        for item in data
    ]

    if sample_size is None or sample_size >= len(pairs):
        return pairs

    total_size = len(pairs)
    indices = np.linspace(0, total_size - 1, sample_size, dtype=int)
    return [pairs[i] for i in indices]


# Per-source profile descriptions for the MMQA-family knowledge base, shared
# with new_model's router. Keys: table / kg / doc.
NEW_MODEL_SOURCE_PROFILES = "/root/autodl-tmp/new_model/data_sources/source_profiles.json"


def load_source_profiles(enabled_sources: List[str]) -> Dict[str, str]:
    """Load per-source profile text for the live knowledge base.

    Profiles come from new_model's source_profiles.json (keys: table/kg/doc).
    Returns {source_name: profile_text} for each enabled source; a missing
    profile degrades to an empty string rather than failing the run.
    """
    try:
        with open(NEW_MODEL_SOURCE_PROFILES, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️ Could not load source profiles ({NEW_MODEL_SOURCE_PROFILES}): {e}")
        raw = {}

    # Accept a few key aliases so older profile files still resolve.
    alias = {"kg": ["kg", "graph"], "table": ["table"], "doc": ["doc", "text"]}
    out: Dict[str, str] = {}
    for name in enabled_sources:
        prof = ""
        for key in alias.get(name, [name]):
            if raw.get(key):
                prof = raw[key]
                break
        out[name] = prof
    return out


def load_corpus_and_profiles(dataset: str) -> Tuple[List[str], List[str], str, str]:
    """
    Backward-compatible loader.

    Returns:
    - source_a_docs (used as table docs placeholder in MMQA mode)
    - source_b_docs (used as kg docs placeholder in MMQA mode)
    - source_a_profile
    - source_b_profile
    """
    dataset_l = dataset.lower().strip()

    if dataset_l in ("mmqa", "mmqa_doc"):
        summary_path = "data/data_sources/source_summary_complex.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)

        kg_profile = profiles.get("graph", profiles.get("kg", ""))

        if dataset_l == "mmqa_doc":
            # kg-doc: source A = Doc passages, source B = KG.
            doc_profile = profiles.get("text", profiles.get("doc", ""))
            doc_docs = ["MMQA_DOC_SOURCE_PLACEHOLDER"]
            kg_docs = ["MMQA_KG_SOURCE_PLACEHOLDER"]
            return doc_docs, kg_docs, doc_profile, kg_profile

        table_profile = profiles.get("table", "")
        # placeholders only; retrieval comes from source-specific retrievers
        table_docs = ["MMQA_TABLE_SOURCE_PLACEHOLDER"]
        kg_docs = ["MMQA_KG_SOURCE_PLACEHOLDER"]
        return table_docs, kg_docs, table_profile, kg_profile

    with open(f"data/rag/{dataset}_corpus_local.json", "r", encoding="utf-8") as f:
        local = [f"{x['title']}. {x['text']}" for x in json.load(f)]
    with open(f"data/rag/{dataset}_corpus_global.json", "r", encoding="utf-8") as f:
        global_ = [f"{x['title']}. {x['text']}" for x in json.load(f)]
    with open(f"data/rag/{dataset}_corpus_profiles.json", "r", encoding="utf-8") as f:
        profiles = json.load(f)
    return local, global_, profiles["local_profile"], profiles["global_profile"]
