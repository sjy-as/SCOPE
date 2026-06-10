import os
import math
from functools import lru_cache

# Force CPU mode for service startup/testing.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from flask import Flask, request

from colbert import Searcher


app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(base_dir, os.pardir))

experiment = 'nba_datalake'
index_name = 'nba_datalake.nbits=2'

candidate_index_roots = [
    os.path.join(repo_root, 'experiments', experiment, 'indexes'),
    os.path.join(base_dir, 'experiments', experiment, 'indexes'),
    os.path.join(repo_root, 'ColBERT', 'experiments', experiment, 'indexes'),
]

index_root = None
for root in candidate_index_roots:
    if os.path.exists(os.path.join(root, index_name, 'metadata.json')) or os.path.exists(
        os.path.join(root, index_name, 'plan.json')
    ):
        index_root = root
        break

if index_root is None:
    raise FileNotFoundError(f"Could not find index '{index_name}' under any of: {candidate_index_roots}")

searcher = Searcher(index=index_name, index_root=index_root)

counter = {"api": 0}


@lru_cache(maxsize=1000000)
def api_search_query(query, k):
    print(f"Query={query}")
    if k is None:
        k = 10
    k = min(int(k), 100)

    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]

    probs = [math.exp(score) for score in scores]
    probs_sum = sum(probs) or 1.0
    probs = [prob / probs_sum for prob in probs]

    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob}
        topk.append(d)

    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query": query, "topk": topk}


@app.route('/api/search', methods=['GET'])
def api_search():
    if request.method == 'GET':
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(request.args.get('query'), request.args.get('k'))
    return ('', 405)


if __name__ == '__main__':
    # GPU for serving (optional). If you want CPU-only serving, set CUDA_VISIBLE_DEVICES="" before running.
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    app.run('127.0.0.1', 1215, debug=False)
