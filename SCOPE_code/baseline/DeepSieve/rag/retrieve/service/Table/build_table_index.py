import argparse
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _WORD_RE.findall(text)]


def _table_to_text(page_title: str, section_title: str, caption: str, header: list, rows_preview: list) -> str:
    header_str = " | ".join([str(h) for h in (header or []) if h is not None and str(h) != ""])
    rows_str = " ; ".join(
        [
            " | ".join([str(c) for c in (row or []) if c is not None and str(c) != ""])
            for row in (rows_preview or [])
            if row
        ]
    )
    parts = [page_title, section_title, caption, header_str, rows_str]
    parts = [p for p in parts if p and str(p).strip()]
    return "\n".join(parts)


def _iter_metadata_copy_rows(metadata_sql_path: str) -> Iterable[Tuple[str, str, str, str, str]]:
    in_copy = False
    with open(metadata_sql_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not in_copy:
                if line.startswith("COPY metadata.nba_context"):
                    in_copy = True
                continue

            line = line.rstrip("\n")
            if line == "\\.":
                break
            if not line:
                continue

            # Fields are tab-separated in pg_dump COPY
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            table_id, page_title, section_title, caption = parts[0], parts[1], parts[2], parts[3]
            doc_json = "\t".join(parts[4:])
            yield table_id, page_title, section_title, caption, doc_json


def _parse_doc_json(doc_json: str) -> dict:
    # doc_json is already JSON in the dump. It might contain escaped sequences.
    return json.loads(doc_json)


def load_table_docs(metadata_sql_path: str, preview_rows: int = 3) -> List[dict]:
    docs: List[dict] = []
    pid = 1
    for table_id, page_title, section_title, caption, doc_json in _iter_metadata_copy_rows(metadata_sql_path):
        try:
            doc = _parse_doc_json(doc_json)
        except Exception:
            continue

        header = doc.get("header") or []
        rows = doc.get("rows") or []
        rows_preview = rows[:preview_rows] if isinstance(rows, list) else []

        text = _table_to_text(page_title, section_title, caption, header, rows_preview)

        docs.append(
            {
                "pid": pid,
                "table_id": table_id,
                "page_title": page_title,
                "section_title": section_title,
                "caption": caption,
                "header": header,
                "rows_preview": rows_preview,
                "text": text,
            }
        )
        pid += 1

    return docs


def build_bm25_index(docs: List[dict], k1: float = 1.2, b: float = 0.75):
    # Standard BM25 over docs' concatenated text
    doc_tokens: List[List[str]] = []
    df: Dict[str, int] = defaultdict(int)

    for d in docs:
        tokens = _tokenize(d.get("text", ""))
        doc_tokens.append(tokens)
        for t in set(tokens):
            df[t] += 1

    N = len(docs)
    if N == 0:
        raise Exception("No documents loaded from metadata.sql")

    idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    doc_len = [len(toks) for toks in doc_tokens]
    avgdl = sum(doc_len) / max(1, N)

    inv_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for i, toks in enumerate(doc_tokens):
        tf = Counter(toks)
        for t, f in tf.items():
            inv_index[t].append((i, f))

    payload = {
        "k1": k1,
        "b": b,
        "idf": idf,
        "avgdl": avgdl,
        "doc_len": doc_len,
        "inv_index": dict(inv_index),
        "docs": docs,
    }
    return payload


def save_index(index: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(index, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_sql",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir,
            "datasets",
            "data_source",
            "table",
            "metadata.sql",
        ),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "table_bm25_index.pkl"),
    )
    parser.add_argument("--preview_rows", type=int, default=3)
    args = parser.parse_args()

    docs = load_table_docs(args.metadata_sql, preview_rows=args.preview_rows)
    index = build_bm25_index(docs)
    save_index(index, args.out)
    print(f"Built BM25 index. docs={len(docs)} saved_to={args.out}")


if __name__ == "__main__":
    main()
