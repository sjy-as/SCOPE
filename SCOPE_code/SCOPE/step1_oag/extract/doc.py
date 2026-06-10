import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


# ============================
# Helpers
# ============================
_CONCEPT_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9_]+")


def normalize_concept_rule_based(concept: str) -> str:
    if not concept:
        return concept
    c = concept.strip().lower().replace("-", "_").replace(" ", "_")
    c = _CONCEPT_TOKEN_SPLIT_RE.sub("_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_get(d: dict, key: str, default: Any = "") -> Any:
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def _dedup_list(items: List[Any]) -> List[Any]:
    out, seen = [], set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ============================
# Input loading
# ============================
def load_document_corpus(doc_path: str) -> List[dict]:
    """Load documents from json/jsonl/txt. Each item should minimally contain text fields."""
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Document path not found: {doc_path}")

    def _clean_wiki_text(text: Any) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\n", " ").strip()
        if "Section::::" in text:
            text = text.split("Section::::", 1)[0].strip()
        return text

    docs: List[dict] = []
    if path.is_dir():
        for p in sorted(path.glob("*.jsonl")):
            docs.extend(load_document_corpus(str(p)))
        for p in sorted(path.glob("*.json")):
            docs.extend(load_document_corpus(str(p)))
        for p in sorted(path.glob("*.txt")):
            docs.extend(load_document_corpus(str(p)))
        return docs

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    text = obj.get("text", "")
                    if isinstance(text, list):
                        text = " ".join(str(x) for x in text)
                    docs.append({
                        "id": str(obj.get("_id", obj.get("id", ""))),
                        "title": str(obj.get("wikipedia_title", obj.get("title", "")) or ""),
                        "text": _clean_wiki_text(text),
                    })
                else:
                    docs.append({"text": _clean_wiki_text(str(obj))})
        return docs

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = raw
            cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
            cleaned = re.sub(r"([}\]])(\s*\")", r"\1,\2", cleaned)
            obj = json.loads(cleaned)

        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if isinstance(text, list):
                        text = " ".join(str(x) for x in text)
                    docs.append({
                        "id": str(item.get("_id", item.get("id", ""))),
                        "title": str(item.get("wikipedia_title", item.get("title", "")) or ""),
                        "text": _clean_wiki_text(text),
                    })
                else:
                    docs.append({"text": _clean_wiki_text(str(item))})
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    text = v.get("text", "")
                    if isinstance(text, list):
                        text = " ".join(str(x) for x in text)
                    docs.append({
                        "id": str(v.get("_id", k)),
                        "title": str(v.get("wikipedia_title", v.get("title", "")) or ""),
                        "text": _clean_wiki_text(text),
                    })
                else:
                    docs.append({"id": str(k), "text": _clean_wiki_text(v)})
        else:
            docs.append({"text": _clean_wiki_text(obj)})
        return docs

    if path.suffix.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                text = line.strip()
                if text:
                    docs.append({"id": str(i), "text": _clean_wiki_text(text)})
        return docs

    raise ValueError(f"Unsupported document format: {path.suffix}")


def _doc_text(doc: dict) -> str:
    parts = []
    for key in ["title", "headline", "section_title", "summary", "text", "content"]:
        value = _safe_get(doc, key, "")
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    if not parts:
        parts.append(json.dumps(doc, ensure_ascii=False, sort_keys=True))
    return " ".join(parts)


# ============================
# Semantic list loading
# ============================
def load_semantic_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("semantic list must be a JSON object")

    vocab = []
    for _, item in data.items():
        if not isinstance(item, dict):
            continue
        concept = str(item.get("concept", "") or "").strip()
        overall_description = str(item.get("overall_description", "") or "").strip()
        sources = item.get("sources", {}) or {}
        vocab.append({
            "name": normalize_concept_rule_based(concept),
            "description": overall_description,
            "raw_concept": concept,
            "sources": sources,
        })
    return vocab


# ============================
# Embedding + clustering
# ============================
def cluster_documents(
    docs: List[dict],
    n_clusters: int = 50,
    method: str = "sentence-transformers",
) -> Tuple[List[dict], np.ndarray]:
    if not docs:
        return [], np.array([], dtype=int)

    texts = [_doc_text(d) for d in docs]

    if method == "sentence-transformers":
        if not HAS_SENTENCE_TRANSFORMERS:
            print("⚠ sentence-transformers not installed; falling back to TF-IDF")
            method = "tfidf"
        else:
            model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            embeddings = model.encode(texts, show_progress_bar=True)

    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        embeddings = vectorizer.fit_transform(texts).toarray()

    k = min(n_clusters, len(docs))
    if k <= 1:
        return [docs[0]], np.zeros(len(docs), dtype=int)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    representative_docs = []
    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        cluster_embeddings = embeddings[cluster_indices]
        center = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        representative_docs.append(docs[closest_idx])

    return representative_docs, labels


# ============================
# LLM client
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


def _parse_json_with_retry(llm: LLMClient, prompt: str, temperature: float = 0.1, max_tokens: int = 10000, retries: int = 3) -> dict:
    for attempt in range(1, retries + 1):
        obj = llm.complete_json(prompt, temperature=temperature, max_tokens=max_tokens) or {}
        if isinstance(obj, dict) and obj:
            return obj
        print(f"[⚠️ LLM解析失败] 第 {attempt}/{retries} 次重试")
    return {}


# ============================
# Clustering + semantic labeling
# ============================
def _build_doc_brief(doc: dict, max_chars: int = 1200) -> dict:
    text = _doc_text(doc)
    text = text[:max_chars]
    return {
        "id": _safe_get(doc, "id", ""),
        "title": _safe_get(doc, "title", ""),
        "section_title": _safe_get(doc, "section_title", ""),
        "text": text,
    }


def _doc_has_match_signal(doc: dict, semantic_vocab: List[dict]) -> bool:
    text = _doc_text(doc).lower()
    vocab_tokens = []
    for item in semantic_vocab:
        name = str(item.get("name", "") or "").replace("_", " ").lower().strip()
        if name:
            vocab_tokens.extend(name.split())
    vocab_tokens = [t for t in vocab_tokens if len(t) >= 4]
    if not vocab_tokens:
        return True
    hits = sum(1 for tok in set(vocab_tokens) if tok in text)
    return hits > 0


def _cluster_document_views(docs: List[dict], labels: np.ndarray) -> Dict[int, List[dict]]:
    out: Dict[int, List[dict]] = defaultdict(list)
    for i, cluster_id in enumerate(labels):
        out[int(cluster_id)].append(docs[i])
    return out


def label_document_clusters(
    representative_docs: List[dict],
    semantic_vocab: List[dict],
    llm: Optional[LLMClient] = None,
) -> Tuple[Dict[int, dict], List[dict]]:
    llm = llm or LLMClient()
    cluster_info: Dict[int, dict] = {}
    vocab = [
        {"name": item["name"], "description": item.get("description", "")}
        for item in semantic_vocab
        if item.get("name")
    ]
    vocab_names = {item["name"] for item in vocab}

    print(f"\n[Doc] 用 LLM 分析 {len(representative_docs)} 个代表性文档...")

    for i, doc in enumerate(representative_docs):
        brief = _build_doc_brief(doc)

        if not _doc_has_match_signal(doc, semantic_vocab):
            cluster_info[i] = {
                "route": "skip",
                "concept_name": "",
                "concept_description": "",
                "semantic_relations": [],
                "sample_doc": brief,
            }
            print(f"  → skip: no strong semantic signal")
            continue

        prompt = f"""You are clustering documents into broad semantic concepts.

Known semantic vocabulary (name + description):
{json.dumps(vocab, ensure_ascii=False)}

Task:
- Judge whether this representative document matches an existing broad semantic concept.
- If it matches, reuse that concept name.
- If no existing concept fits well, create a new concept.
- Keep concepts broad and reusable; avoid unnecessary micro-splitting.
- If the document is only weakly related or not actually about a semantic concept in the vocabulary, return route=skip.
- Return a concise concept description grounded in the document evidence.

Representative document:
{json.dumps(brief, ensure_ascii=False)}

Return JSON only:
{{
  "route": "concept_match|new_concept|skip",
  "matched_concept": "... or empty",
  "matched_description": "... or empty",
  "new_concept": "snake_case or empty",
  "new_concept_description": "... or empty",
  "semantic_relations": [{{"predicate":"...","subject":"...","object":"..."}}]
}}"""
        obj = _parse_json_with_retry(llm, prompt, temperature=0.1, max_tokens=10000, retries=3)

        route = str(obj.get("route", "")).strip().lower() or "skip"
        matched_concept = normalize_concept_rule_based(str(obj.get("matched_concept", "") or ""))
        new_concept = normalize_concept_rule_based(str(obj.get("new_concept", "") or ""))
        matched_desc = str(obj.get("matched_description", "") or "").strip()
        new_desc = str(obj.get("new_concept_description", "") or "").strip()
        relations = obj.get("semantic_relations", []) or []

        if route == "concept_match" and matched_concept:
            concept_name = matched_concept
            concept_desc = matched_desc
            if concept_name not in vocab_names:
                vocab.append({"name": concept_name, "description": concept_desc})
                vocab_names.add(concept_name)
        elif route == "new_concept" and new_concept:
            concept_name = new_concept
            concept_desc = new_desc
            if concept_name not in vocab_names:
                vocab.append({"name": concept_name, "description": concept_desc})
                vocab_names.add(concept_name)
        else:
            cluster_info[i] = {
                "route": "skip",
                "concept_name": "",
                "concept_description": "",
                "semantic_relations": [],
                "sample_doc": brief,
            }
            print(f"  → skip: route={route} or empty concept")
            continue

        cluster_info[i] = {
            "route": route,
            "concept_name": concept_name,
            "concept_description": concept_desc,
            "semantic_relations": relations,
            "sample_doc": brief,
        }
        print(f"  → concept: {concept_name} | route={route} | desc={concept_desc or '[none]'}")

    return cluster_info, vocab


# ============================
# Final aggregation to semantic list format
# ============================
def _stable_doc_description(concept_name: str, raw_desc: str = "", evidence: Optional[dict] = None) -> str:
    raw_desc = str(raw_desc or "").strip()
    if raw_desc:
        return raw_desc

    pretty = concept_name.replace("_", " ").strip()
    if evidence:
        title = str(evidence.get("title", "") or "").strip()
        section_title = str(evidence.get("section_title", "") or "").strip()
        text = str(evidence.get("text", "") or "").strip()
        bits = [x for x in [title, section_title, text[:220]] if x]
        if bits:
            return f"{pretty} described from document evidence: {' | '.join(bits)}."
    return f"{pretty} concept described from document evidence."


def _collect_source_payload(docs: List[dict], limit_examples: int = 3) -> dict:
    examples = []
    for doc in docs[:limit_examples]:
        examples.append({
            "doc_id": _safe_get(doc, "id", ""),
            "title": _safe_get(doc, "title", ""),
            "section_title": _safe_get(doc, "section_title", ""),
            "text": _doc_text(doc)[:500],
        })
    return {
        "description": "Document cluster evidence extracted from representative documents.",
        "elements": ["title", "section_title", "text"],
        "data_examples": [json.dumps(x, ensure_ascii=False) for x in examples],
    }


def build_document_semantic_list(
    docs: List[dict],
    semantic_vocab: List[dict],
    n_clusters: int = 50,
    cluster_method: str = "sentence-transformers",
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    representative_docs, labels = cluster_documents(docs, n_clusters=n_clusters, method=cluster_method)
    cluster_info, vocab_out = label_document_clusters(representative_docs, semantic_vocab, llm=llm)
    cluster_docs_map = _cluster_document_views(docs, labels)

    semantic_list: Dict[str, Any] = {}
    for cluster_id, info in cluster_info.items():
        concept_name = normalize_concept_rule_based(str(info.get("concept_name", "") or ""))
        if not concept_name:
            continue
        docs_here = cluster_docs_map.get(cluster_id, [])
        if not docs_here:
            continue
        desc = _stable_doc_description(concept_name, info.get("concept_description", ""), evidence=info.get("sample_doc", {}))

        semantic_list[concept_name] = {
            "concept": concept_name,
            "overall_description": desc,
            "sources": {
                "Doc": _collect_source_payload(docs_here),
            },
        }

        if info.get("semantic_relations"):
            semantic_list[concept_name]["sources"]["Doc"]["semantic_relations"] = info.get("semantic_relations", [])

    # Ensure new vocab items are also reflected if cluster naming produced unseen concepts
    for item in vocab_out:
        name = normalize_concept_rule_based(str(item.get("name", "") or ""))
        if not name:
            continue
        if name not in semantic_list:
            continue

    return semantic_list


# ============================
# I/O
# ============================
def save_semantic_list(payload: Dict[str, Any], output_path: str) -> None:
    _ensure_dir(str(Path(output_path).parent))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build document semantic list by clustering and LLM labeling.")
    parser.add_argument("--omp_num_threads", type=int, default=None, help="Optional override for OMP_NUM_THREADS")
    parser.add_argument("--doc_path", type=str, required=True, help="Path to documents json/jsonl/txt or directory")
    parser.add_argument("--semantic_list_path", type=str, required=True, help="Path to existing semantic_list.json")
    parser.add_argument("--output_path", type=str, required=True, help="Path to write new semantic list")
    parser.add_argument("--n_clusters", type=int, default=50)
    parser.add_argument("--cluster_method", type=str, default="sentence-transformers", choices=["sentence-transformers", "tfidf"])
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "DeepSeek-V3.2-Fast"))
    args = parser.parse_args()

    if args.omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    docs = load_document_corpus(args.doc_path)
    semantic_vocab = load_semantic_list(args.semantic_list_path)
    llm = LLMClient(base_url=args.base_url, api_key=args.api_key, model=args.model)

    result = build_document_semantic_list(
        docs=docs,
        semantic_vocab=semantic_vocab,
        n_clusters=args.n_clusters,
        cluster_method=args.cluster_method,
        llm=llm,
    )
    save_semantic_list(result, args.output_path)
    print(f"[Doc] saved semantic list to {args.output_path}")


if __name__ == "__main__":
    main()
