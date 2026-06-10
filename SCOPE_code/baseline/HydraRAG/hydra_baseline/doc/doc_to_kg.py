"""HydraRAG 的 Doc 头（核心机制，与 table/table_to_kg.py 对偶）。

职责：
  1. 用 ColBERT 段落检索服务取回与（子）问题相关的文档段落；
  2. 用 LLM 把每段文档 **转换成 KG 边**，使文档证据与 KG 证据共用同一套
     "{Head} -[relation]-> {Tail}" 表示，从而能一起做多源融合排序。

这对应原版 HydraRAG 的 document_path_generation —— 原版就是把网页/维基段落转成
KG 路径。kg-doc 任务下，本文件取代 table/table_to_kg.py 充当「外部补充头」。
"""
from __future__ import annotations

import re
from typing import List, Optional


# 关系 token：  -[rel]->  或  <-[rel]-
_REL_TOKEN = re.compile(r"(<-\[[^\]]+\]-|-\[[^\]]+\]->)")
_ENT = re.compile(r"\{([^{}]*)\}")


def _parse_edge_line(line: str) -> Optional[dict]:
    """从一行形如 [{A} -[r]-> {B}] 的文本里解析出 triples + 规范文本。"""
    if line.startswith("[") and line.endswith("]"):
        inner = line[1:-1].strip()
    else:
        inner = line.strip()

    parts = _REL_TOKEN.split(inner)
    if len(parts) < 3:
        return None

    ents: List[str] = []
    rels: List[tuple] = []   # (rel_text, direction)
    for i, seg in enumerate(parts):
        if i % 2 == 0:
            m = _ENT.search(seg)
            ent = (m.group(1) if m else seg).strip().strip("{}").strip()
            ents.append(ent)
        else:
            if seg.startswith("<-"):
                rels.append((seg[3:-2].strip(), "in"))    # <-[ rel ]-
            else:
                rels.append((seg[2:-3].strip(), "out"))   # -[ rel ]->

    if len(ents) < 2 or len(rels) != len(ents) - 1:
        return None
    if any(not e for e in ents):
        return None

    triples: List[tuple] = []
    text = "{" + ents[0] + "}"
    for j, (rel, direction) in enumerate(rels):
        if direction == "out":
            text += f" -[{rel}]-> " + "{" + ents[j + 1] + "}"
            triples.append((ents[j], rel, ents[j + 1], {}))
        else:
            text += f" <-[{rel}]- " + "{" + ents[j + 1] + "}"
            triples.append((ents[j + 1], rel, ents[j], {}))
    return {"text": text, "triples": triples, "entities": ents}


def _parse_edges(resp: str) -> List[dict]:
    """从 LLM 回复里抽出所有形如 [{A} -[r]-> {B}] 的边。"""
    if not resp:
        return []
    if "edges:" in resp:
        resp = resp.split("edges:", 1)[1]
    out: List[dict] = []
    for raw in resp.splitlines():
        line = raw.strip()
        if "{" not in line or "}" not in line:
            continue
        parsed = _parse_edge_line(line)
        if parsed:
            out.append(parsed)
    return out


class DocToKG:
    """Doc 头：检索段落 + 用 LLM 把段落转成 KG 边。

    暴露与 `TableToKG` 一致的接口（`retrieve` / `to_kg_edges` /
    `retrieve_and_convert`），从而能直接作为 pipeline 的「外部补充头」插入，
    无需改 pipeline 的检索循环。
    """

    def __init__(self, doc_retriever, llm, k: int = 5, verbose: bool = False):
        self.retriever = doc_retriever
        self.llm = llm
        self.k = k
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    #  检索                                                               #
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: Optional[int] = None) -> List[dict]:
        """ColBERT 取回 top-k 候选段落。"""
        try:
            passages = self.retriever.retrieve_topk_passages(query, k or self.k) or []
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"[DocToKG] retrieve failed: {e}")
            passages = []
        if self.verbose:
            print(f"[DocToKG] query='{query[:60]}' -> {len(passages)} passages")
        return passages

    @staticmethod
    def unit_id(unit: dict) -> str:
        """给一段文档一个稳定 id，供 pipeline 跨 query 去重用。"""
        return str(unit.get("doc_id", "")) or str(id(unit))

    # ------------------------------------------------------------------ #
    #  段落 -> KG 边                                                       #
    # ------------------------------------------------------------------ #
    def to_kg_edges(self, question: str, passages: List[dict]) -> List[dict]:
        """把若干段文档逐一交给 LLM，转成统一格式的 KG 边路径。"""
        # 延迟导入：包内/脚本两种运行方式都能拿到 prompts。
        try:
            import prompts
        except ImportError:  # pragma: no cover
            from .. import prompts  # type: ignore

        paths: List[dict] = []
        for p in passages:
            doc_id = str(p.get("doc_id", ""))
            passage_text = self.retriever.format_passage(p)
            if not passage_text.strip():
                continue

            prompt = (
                prompts.DOC_TO_KG
                + f"Question: {question}\nDocument:\n{passage_text}\nA:\n"
            )
            try:
                resp, _ = self.llm.query_gpt4o(prompt, max_tokens=900, stage="doc_to_kg")
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    print(f"[DocToKG] LLM failed on doc {doc_id}: {e}")
                resp = ""

            edges = _parse_edges(resp)
            meta = {"doc_id": doc_id, "title": p.get("title", "")}
            if edges:
                for e in edges:
                    paths.append({
                        "path_text": e["text"],
                        "triples": e["triples"],
                        "source": "doc",
                        "doc_id": doc_id or "doc",
                        "entities": e["entities"],
                        "qids": [],
                        "n_seeds": 0,
                        "doc_meta": meta,
                    })
            else:
                # 兜底：LLM 没转出边时，至少把段落本身留作原始证据，不丢信息。
                snippet = (p.get("text") or "")[:600]
                paths.append({
                    "path_text": f"[Doc {doc_id}] {meta['title']}: {snippet}",
                    "triples": [],
                    "source": "doc_raw",
                    "doc_id": doc_id or "doc",
                    "entities": [],
                    "qids": [],
                    "n_seeds": 0,
                    "doc_meta": meta,
                })
        if self.verbose:
            print(f"[DocToKG] {len(passages)} passages -> {len(paths)} KG-style edges")
        return paths

    # 与 TableToKG.tables_to_kg_edges 同名的别名，pipeline 旧路径也能调用。
    def tables_to_kg_edges(self, question: str, passages: List[dict]) -> List[dict]:
        return self.to_kg_edges(question, passages)

    def retrieve_and_convert(self, question: str, query: str, k: Optional[int] = None) -> List[dict]:
        """检索 + 转边 一步到位。"""
        passages = self.retrieve(query, k)
        return self.to_kg_edges(question, passages)
