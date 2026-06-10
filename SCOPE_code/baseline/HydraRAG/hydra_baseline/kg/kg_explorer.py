"""HydraRAG 的 KG 头：从 topic 实体出发做子图 BFS，再枚举 KG 路径。

输出的每条路径都是统一的 "KG 边" 表示：
    {"path_text": "{A} -[rel]-> {B} <-[rel2]- {C}",
     "triples": [(head, rel, tail, props), ...],
     "source": "kg", "doc_id": "kg",
     "entities": [...], "qids": [...], "n_seeds": int}

Table 头转出来的边也走同一套 schema，从而能和 KG 边一起融合排序。
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

try:
    from kg.kg_retriever import KGRetriever, EvidenceEntity
except ImportError:  # 包内导入
    from .kg_retriever import KGRetriever, EvidenceEntity


_PROP_KEYS = ("time", "year", "season", "start_time", "end_time",
              "position", "sport_number", "round", "pick")


def _fmt_props(props: dict) -> str:
    """把关系上的属性（如 time=2009）压成简短字符串，挂到 relation 后面。"""
    if not props:
        return ""
    bits = []
    for k in _PROP_KEYS:
        v = props.get(k)
        if v:
            bits.append(f"{k}={v}")
    if not bits:
        bits = [f"{k}={v}" for k, v in list(props.items())[:2] if v]
    return (" | " + ", ".join(bits)) if bits else ""


class KGExplorer:
    """对本地 NBA KG 做实体链接 + 子图探索 + 路径枚举。无 LLM 调用，确定性。"""

    def __init__(
        self,
        kg_retriever: KGRetriever,
        max_hop: int = 3,
        max_degree: int = 60,
        max_paths: int = 400,
        max_nodes: int = 4000,
        branch_cap: int = 14,
    ):
        self.kg = kg_retriever
        self.index = kg_retriever.kg          # LocalKGIndex
        self.max_hop = max_hop
        self.max_degree = max_degree
        self.max_paths = max_paths
        self.max_nodes = max_nodes
        self.branch_cap = branch_cap
        self._expand_cache: Dict[str, List[tuple]] = {}

    # ------------------------------------------------------------------ #
    #  实体链接                                                           #
    # ------------------------------------------------------------------ #
    def link_entities(self, labels: List[str]) -> List[EvidenceEntity]:
        """把 topic 实体文本链接到 KG（每个文本取最优一个有 qid 的命中）。"""
        linked: List[EvidenceEntity] = []
        seen: Set[str] = set()
        for lab in labels:
            lab = (lab or "").strip()
            if not lab:
                continue
            for e in self.kg.search_entity(lab):
                if e.qid and e.qid not in seen:
                    seen.add(e.qid)
                    linked.append(e)
                    break
        return linked

    # ------------------------------------------------------------------ #
    #  子图 BFS                                                           #
    # ------------------------------------------------------------------ #
    def _expand(self, qid: str) -> List[tuple]:
        """返回 qid 的所有边：(relation, neighbor_qid, direction, props)。"""
        if qid in self._expand_cache:
            return self._expand_cache[qid]
        edges: List[tuple] = []
        for rel in self.index.relations:
            for t in self.index.relate(qid, rel, inverse=False):
                edges.append((rel, t.tail_qid, "out", t.props))
            for t in self.index.relate(qid, rel, inverse=True):
                edges.append((rel, t.head_qid, "in", t.props))
        if len(edges) > self.max_degree:
            edges = edges[: self.max_degree]
        self._expand_cache[qid] = edges
        return edges

    def build_subgraph(self, seed_qids: List[str]) -> Dict[str, List[tuple]]:
        """从 seed 出发 BFS 扩展 max_hop 层，返回 graph[qid] = 边列表。"""
        graph: Dict[str, List[tuple]] = {}
        visited: Set[str] = set(seed_qids)
        frontier = list(seed_qids)
        for _ in range(self.max_hop):
            nxt: List[str] = []
            for qid in frontier:
                if qid in graph:
                    continue
                edges = self._expand(qid)
                graph[qid] = edges
                if len(graph) >= self.max_nodes:
                    return graph
                for (_, nbr, _, _) in edges:
                    if nbr not in visited:
                        visited.add(nbr)
                        nxt.append(nbr)
            frontier = nxt
            if not frontier:
                break
        return graph

    # ------------------------------------------------------------------ #
    #  路径枚举                                                           #
    # ------------------------------------------------------------------ #
    def find_paths(self, graph: Dict[str, List[tuple]], seed_qids: List[str]) -> List[dict]:
        """BFS 枚举从各 seed 出发的路径（短路径优先），收集成统一 path 对象。"""
        paths: List[dict] = []
        seed_set = set(seed_qids)
        # queue 元素: (node_qids, edge_records)；edge_record = (rel, dir, props, nbr)
        queue: deque = deque((q, [q], []) for q in seed_qids)
        while queue and len(paths) < self.max_paths:
            cur, nodes, edges = queue.popleft()
            if edges:
                paths.append(self._make_path(nodes, edges, seed_set))
            if len(edges) >= self.max_hop:
                continue
            branched = 0
            for (rel, nbr, direction, props) in graph.get(cur, []):
                if nbr in nodes:
                    continue                       # 不走回头路
                queue.append((nbr, nodes + [nbr], edges + [(rel, direction, props, nbr)]))
                branched += 1
                if branched >= self.branch_cap:
                    break
        return paths

    def _make_path(self, nodes: List[str], edges: List[tuple], seed_set: Set[str]) -> dict:
        lbl = lambda q: self.index.qid_to_label(q) or q
        text = "{" + lbl(nodes[0]) + "}"
        triples: List[tuple] = []
        for i, (rel, direction, props, _nbr) in enumerate(edges):
            tag = f"{rel}{_fmt_props(props)}"
            if direction == "out":
                text += f" -[{tag}]-> " + "{" + lbl(nodes[i + 1]) + "}"
                triples.append((lbl(nodes[i]), rel, lbl(nodes[i + 1]), dict(props)))
            else:
                text += f" <-[{tag}]- " + "{" + lbl(nodes[i + 1]) + "}"
                triples.append((lbl(nodes[i + 1]), rel, lbl(nodes[i]), dict(props)))
        return {
            "path_text": text,
            "triples": triples,
            "source": "kg",
            "doc_id": "kg",
            "entities": [lbl(q) for q in nodes],
            "qids": list(nodes),
            "n_seeds": sum(1 for q in nodes if q in seed_set),
        }

    # ------------------------------------------------------------------ #
    #  对外主入口                                                         #
    # ------------------------------------------------------------------ #
    def explore(
        self,
        topic_labels: List[str],
        extra_seed_qids: Optional[List[str]] = None,
    ) -> dict:
        """链接实体 -> 建子图 -> 枚举路径。返回 dict。"""
        linked = self.link_entities(topic_labels)
        seed_qids: List[str] = [e.qid for e in linked if e.qid]
        for q in (extra_seed_qids or []):
            if q and q not in seed_qids:
                seed_qids.append(q)

        if not seed_qids:
            return {"linked_entities": [], "paths": [], "reached": [], "subgraph_size": 0}

        graph = self.build_subgraph(seed_qids)
        paths = self.find_paths(graph, seed_qids)
        # 连接 >=2 个 seed 的路径价值最高，排前面
        paths.sort(key=lambda p: (-p["n_seeds"], len(p["triples"])))
        reached = [(self.index.qid_to_label(q) or q, q) for q in graph]
        return {
            "linked_entities": [{"label": e.label, "qid": e.qid, "meta": e.meta} for e in linked],
            "paths": paths,
            "reached": reached,
            "subgraph_size": len(graph),
        }
