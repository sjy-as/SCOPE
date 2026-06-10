"""HydraRAG 的 Table 头（核心机制）。

职责：
  1. 用 BM25 服务检索与（子）问题相关的表格；
  2. 用 LLM 把每张表的行 **转换成 KG 边**，使表格证据与 KG 证据共用
     同一套 "{Head} -[relation]-> {Tail}" 表示，从而能一起做多源融合排序。

这一步对应原版 HydraRAG 的 document_path_generation —— 原版把网页/维基段落转成
KG 路径，这里改成把检索到的 Table 转成 KG 边。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

try:
    import prompts
except ImportError:  # 包内导入
    from .. import prompts  # type: ignore


# 关系 token：  -[rel]->  或  <-[rel]-
_REL_TOKEN = re.compile(r"(<-\[[^\]]+\]-|-\[[^\]]+\]->)")
_ENT = re.compile(r"\{([^{}]*)\}")


class TableToKG:
    def __init__(self, table_retriever, llm, k: int = 5, verbose: bool = False):
        self.retriever = table_retriever
        self.llm = llm
        self.k = k
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    #  检索                                                               #
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: Optional[int] = None) -> List[dict]:
        """BM25 取回 top-k 候选表。"""
        try:
            tables = self.retriever.retrieve_topk_tables(query, k or self.k) or []
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"[TableToKG] retrieve failed: {e}")
            tables = []
        if self.verbose:
            print(f"[TableToKG] query='{query[:60]}' -> {len(tables)} tables")
        return tables

    # ------------------------------------------------------------------ #
    #  Table 行 -> KG 边                                                   #
    # ------------------------------------------------------------------ #
    def tables_to_kg_edges(self, question: str, tables: List[dict]) -> List[dict]:
        """把若干张表逐一交给 LLM，转成统一格式的 KG 边路径。"""
        paths: List[dict] = []
        for t in tables:
            table_id = t.get("table_id", "")
            table_text = self.retriever.format_table_full(t)
            if not table_text.strip():
                continue

            prompt = (
                prompts.TABLE_TO_KG
                + f"Question: {question}\nTable:\n{table_text}\nA:\n"
            )
            try:
                resp, _ = self.llm.query_gpt4o(prompt, max_tokens=900, stage="table_to_kg")
            except Exception as e:  # noqa: BLE001
                if self.verbose:
                    print(f"[TableToKG] LLM failed on {table_id}: {e}")
                resp = ""

            edges = self._parse_edges(resp)
            meta = {
                "table_id": table_id,
                "page_title": t.get("page_title", ""),
                "section_title": t.get("section_title", ""),
            }
            if edges:
                for e in edges:
                    paths.append({
                        "path_text": e["text"],
                        "triples": e["triples"],
                        "source": "table",
                        "doc_id": table_id or "table",
                        "entities": e["entities"],
                        "qids": [],
                        "n_seeds": 0,
                        "table_meta": meta,
                    })
            else:
                # 兜底：LLM 没转出边时，至少把表本身留作原始证据，不丢信息
                snippet = table_text[:600]
                paths.append({
                    "path_text": f"[Table {table_id}] {meta['page_title']} / {meta['section_title']}: {snippet}",
                    "triples": [],
                    "source": "table_raw",
                    "doc_id": table_id or "table",
                    "entities": [],
                    "qids": [],
                    "n_seeds": 0,
                    "table_meta": meta,
                })
        if self.verbose:
            print(f"[TableToKG] {len(tables)} tables -> {len(paths)} KG-style edges")
        return paths

    def to_kg_edges(self, question: str, tables: List[dict]) -> List[dict]:
        """与 DocToKG.to_kg_edges 同名的别名，让 pipeline 的外部头与源无关。"""
        return self.tables_to_kg_edges(question, tables)

    @staticmethod
    def unit_id(unit: dict) -> str:
        """给一张表一个稳定 id，供 pipeline 跨 query 去重用。"""
        return str(unit.get("table_id", "")) or str(id(unit))

    def retrieve_and_convert(self, question: str, query: str, k: Optional[int] = None) -> List[dict]:
        """检索 + 转边 一步到位。"""
        tables = self.retrieve(query, k)
        return self.tables_to_kg_edges(question, tables)

    # ------------------------------------------------------------------ #
    #  解析 LLM 输出的边                                                   #
    # ------------------------------------------------------------------ #
    def _parse_edges(self, resp: str) -> List[dict]:
        """从 LLM 回复里抽出形如 [{A} -[r]-> {B}] 的边，解析成 triples + 规范文本。"""
        if not resp:
            return []
        # 取 "edges:" 之后的部分（若有）
        if "edges:" in resp:
            resp = resp.split("edges:", 1)[1]

        out: List[dict] = []
        for raw in resp.splitlines():
            line = raw.strip()
            if "{" not in line or "}" not in line:
                continue
            parsed = self._parse_edge_line(line)
            if parsed:
                out.append(parsed)
        return out

    def _parse_edge_line(self, line: str) -> Optional[dict]:
        # 去掉单层外包的 [ ]
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
