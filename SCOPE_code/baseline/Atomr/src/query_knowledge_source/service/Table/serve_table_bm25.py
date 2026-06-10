import argparse
import math
import os
import pickle
from functools import lru_cache

from flask import Flask, request


app = Flask(__name__)


def _bm25_score(query_tokens, inv_index, idf, doc_len, avgdl, k1, b):
    scores = {}
    for t in query_tokens:
        postings = inv_index.get(t)
        if not postings:
            continue
        t_idf = idf.get(t, 0.0)
        for doc_i, tf in postings:
            dl = doc_len[doc_i]
            denom = tf + k1 * (1 - b + b * (dl / avgdl))
            s = t_idf * (tf * (k1 + 1)) / (denom if denom != 0 else 1.0)
            scores[doc_i] = scores.get(doc_i, 0.0) + s
    return scores


@lru_cache(maxsize=100000)
def _tokenize(text: str):
    import re

    return [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", text or "")]


INDEX = None


def load_index(path: str):
    global INDEX
    with open(path, "rb") as f:
        INDEX = pickle.load(f)


def _result_to_topk(doc, score, rank):
    # Return all fields needed by TableSource._format_table_for_prompt
    text = doc.get("text", "")
    prob = 1.0 / (1.0 + math.exp(-score)) if score is not None else 0.0
    return {
        "text": text,
        "pid": doc.get("pid"),
        "rank": rank,
        "score": float(score),
        "prob": float(prob),
        "table_id": doc.get("table_id"),
        "page_title": doc.get("page_title"),
        "section_title": doc.get("section_title"),
        "caption": doc.get("caption"),
        "header": doc.get("header") or [],
        "rows_preview": doc.get("rows_preview") or [],
    }


@app.route("/api/search", methods=["GET"])
def api_search():
    if INDEX is None:
        return {"error": "index_not_loaded"}, 500

    query = request.args.get("query", "")
    k = request.args.get("k", "10")
    try:
        k = min(int(k), 50)
    except Exception:
        k = 10

    tokens = _tokenize(query)

    scores = _bm25_score(
        tokens,
        INDEX["inv_index"],
        INDEX["idf"],
        INDEX["doc_len"],
        INDEX["avgdl"],
        INDEX["k1"],
        INDEX["b"],
    )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

    topk = []
    for r, (doc_i, s) in enumerate(ranked, start=1):
        doc = INDEX["docs"][doc_i]
        topk.append(_result_to_topk(doc, s, r))

    return {"query": query, "topk": topk}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        default="table_bm25_index.pkl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1216)
    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(
            f"Table index not found: {args.index}. Run build_table_index.py first."
        )

    load_index(args.index)
    app.run(args.host, args.port, debug=False)


if __name__ == "__main__":
    main()
