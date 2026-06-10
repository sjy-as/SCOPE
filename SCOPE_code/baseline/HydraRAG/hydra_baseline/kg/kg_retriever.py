from __future__ import annotations
import json
import re
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
try:
    from rapidfuzz import fuzz  # type: ignore
except ImportError:  # pragma: no cover
    fuzz = None

try:
    from kg.localkg_index import LocalKGIndex
except ImportError:  # 支持以包内模块方式导入
    from .localkg_index import LocalKGIndex

try:
    import sys as _sys
    if "/root/autodl-tmp" not in _sys.path:
        _sys.path.insert(0, "/root/autodl-tmp")
    from _common.cost_counter import bump_retrieval as _bump_retrieval
except Exception:  # pragma: no cover
    def _bump_retrieval(*_a, **_kw): pass


def _kg_relation_mapping_prompt(question: str, relation_text: str, available_relations: list) -> str:
    """把自然语言关系映射到 KG 中真实的 relation key（内联，原 new_model Prompt.KG_relation_mapping）。"""
    return f"""You are a knowledge graph expert. Your task is to map a natural language relation description to the most appropriate relation name in the knowledge graph.

Question: {question}
Natural Language Relation: {relation_text}
Available Relations in KG: {available_relations}

Based on semantic similarity and the context of the question, output ONLY the exact relation name from the available list that best matches the natural language relation.
If no relation is a good match, output 'None'.

Examples:
Question: Which team did John Havlicek play for?
Natural Language Relation: team played for
Available Relations: ['playsFor', 'draftedBy', 'coachedBy', 'hasHomeVenue']
Answer: playsFor

Question: Who was the coach of the team?
Natural Language Relation: has_role
Available Relations: ['playsFor', 'draftedBy', 'coachedBy', 'hasHomeVenue', 'isSameAs']
Answer: coachedBy

Your Turn.
Question: {question}
Natural Language Relation: {relation_text}
Available Relations: {available_relations}
Answer: """


@dataclass(frozen=True)
class EvidenceEntity:
    label: str
    qid: Optional[str] = None
    meta: Optional[dict] = None

    def __str__(self) -> str:
        return self.label

class KGRetriever:
    """提供带有业务逻辑的图谱操作服务层。"""

    def __init__(self, kg_dir: str, llm: Optional[Any] = None):
        self.kg = LocalKGIndex(kg_dir)
        self.llm = llm

    @staticmethod
    def _token_set_ratio(left: str, right: str) -> float:
        if fuzz is not None:
            return float(fuzz.token_set_ratio(left, right))
        left_tokens = sorted(set(str(left).split()))
        right_tokens = sorted(set(str(right).split()))
        left_norm = " ".join(left_tokens)
        right_norm = " ".join(right_tokens)
        return difflib.SequenceMatcher(a=left_norm, b=right_norm).ratio() * 100.0

    def search_entity(self, label: str) -> List[EvidenceEntity]:
        _bump_retrieval("kg")
        raw = str(label or "").strip()
        if not raw:
            return [EvidenceEntity(label=label, qid=None, meta={"not_found": True})]

        # --- Stage 1: 精确匹配 (Exact/Normalized Lookups) ---
        norm_ws = self._normalize_label(raw)
        qids: List[str] = []

        # 尝试各种归一化后的精确匹配
        qids = self.kg.resolve_label(norm_ws)
        if not qids and norm_ws != raw:
            qids = self.kg.resolve_label(raw)
        if not qids:
            qids = self.kg.resolve_label_normalized(raw)
        
        # 处理括号，比如 "Apple (Company)" -> "Apple"
        if not qids:
            stripped = self._strip_parenthetical(norm_ws)
            if stripped and stripped != norm_ws:
                qids = self.kg.resolve_label(stripped) or self.kg.resolve_label_normalized(stripped)

        # 如果搜到了精确结果，直接返回
        if qids:
            out: List[EvidenceEntity] = []
            for qid in qids[:5]:
                out.append(EvidenceEntity(
                    label=self.kg.qid_to_label(qid) or raw, 
                    qid=qid, 
                    meta={"matched_by": "exact"}
                ))
            return out

        # --- Stage 2: 模糊匹配 (Fuzzy Recall + Re-ranking) ---
        tokens = self._expanded_tokens(raw)
        tokens = [t for t in tokens if len(t) >= 2] # 过滤掉单字，减少噪音

        cand_qids = set()
        # 按照 Token 长度降序搜索，优先匹配长词
        tokens_sorted = sorted(tokens, key=len, reverse=True)
        for t in tokens_sorted:
            # 过滤高频无意义词
            if t in {"la", "ny", "dc"}: continue
            cand_qids |= set(self.kg.resolve_tokens([t]))
            if len(cand_qids) >= 1000: # 限制候选集大小
                break
        
        # 如果模糊匹配也没搜到候选 QID，直接宣告失败（删除了原有的 Stage 3 属性搜索）
        if not cand_qids:
            return [EvidenceEntity(label=label, qid=None, meta={"not_found": True, "stage": "fuzzy_recall"})]

        # 对候选 QID 进行相似度打分重排
        scored: List[Tuple[float, str]] = []
        for qid in cand_qids:
            cand_label = self.kg.qid_to_label(qid) or ""
            s = self._label_similarity(raw, cand_label)
            scored.append((s, qid))

        scored.sort(key=lambda x: x[0], reverse=True)
        
        # 设定阈值，过滤掉相关性太低的
        threshold = 0.45
        keep = [(s, qid) for (s, qid) in scored if s >= threshold][:5]
        
        if not keep:
            return [EvidenceEntity(label=label, qid=None, meta={"not_found": True, "top_score": scored[0][0] if scored else 0.0})]

        # 返回模糊匹配的结果
        out2: List[EvidenceEntity] = []
        for s, qid in keep:
            out2.append(
                EvidenceEntity(
                    label=self.kg.qid_to_label(qid) or raw,
                    qid=qid,
                    meta={"matched_by": "fuzzy", "score": round(s, 4)},
                )
            )
        return out2

    # ---正向与反向的关系游走封装 ---
    def _resolve_relation(self, relation_text: str, question: str = "") -> str:
        """将自然语言关系描述映射到图谱中实际存储的 relation key。
        
        优先级：
        1. 精确匹配
        2. LLM 映射（如果配置了 LLM）
        3. 模糊匹配（fallback）
        """
        available = list(self.kg.relations)
        if not available:
            return relation_text
        
        # 先尝试精确匹配
        if relation_text in available:
            return relation_text
        
        # 如果配置了 LLM，使用 LLM 进行关系映射
        if self.llm is not None:
            try:
                prompt = _kg_relation_mapping_prompt(question, relation_text, available)
                # print(f"prompt: {prompt}")
                response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=20)
                mapped_rel = response.strip()
                # print(f"available: {available}")
                # print(f"mapped_rel: {mapped_rel}")
                
                # 清理响应（移除 "Answer:" 前缀等）
                if mapped_rel.startswith("Answer:"):
                    mapped_rel = mapped_rel.replace("Answer:", "").strip()
                
                # 验证映射结果是否在可用关系列表中
                if mapped_rel in available and mapped_rel.lower() != "none":
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"  ✓ LLM mapped '{relation_text}' -> '{mapped_rel}'")
                    return mapped_rel
                else:
                    if hasattr(self, '_verbose') and self._verbose:
                        print(f"  ⚠️ LLM returned '{mapped_rel}' which is not in available relations")
            except Exception as e:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"[Warning] LLM relation mapping failed: {e}")
                # 降级到模糊匹配
        
        # 用 fuzzy token_set_ratio 找最近的（fallback）
        best_key, best_score = relation_text, 0.0
        for rel in available:
            score = self._token_set_ratio(
                relation_text.lower().replace("_", " "),
                rel.lower().replace("_", " "),
            )
            if score > best_score:
                best_score, best_key = score, rel
        
        if best_score >= 40.0:
            if hasattr(self, '_verbose') and self._verbose:
                print(f"  ✓ Fuzzy matched '{relation_text}' -> '{best_key}' (score: {best_score})")
            return best_key
        
        if hasattr(self, '_verbose') and self._verbose:
            print(f"  ✗ Could not map '{relation_text}' to any available relation")
            print(f"     Available relations: {available}")
        return relation_text

    def relate(self, entity: EvidenceEntity, relation_text: str, question: str = "") -> Tuple[List[EvidenceEntity], List[dict]]:
        if not entity.qid: return [], []
        resolved = self._resolve_relation(relation_text, question=question)
        triples = self.kg.relate(head_qid=entity.qid, relation=resolved, inverse=False)
        out_entities = [EvidenceEntity(label=self.kg.qid_to_label(t.tail_qid) or t.tail_qid, qid=t.tail_qid, meta=t.props) for t in triples]
        return out_entities, []

    def find_incoming_entities(self, target_entity: EvidenceEntity, relation_text: str, question: str = "") -> List[EvidenceEntity]:
        if not target_entity.qid: return []
        resolved = self._resolve_relation(relation_text, question=question)
        triples = self.kg.relate(head_qid=target_entity.qid, relation=resolved, inverse=True)
        # inverse=True时，t.head_qid 是入边起点节点
        return [EvidenceEntity(label=self.kg.qid_to_label(t.head_qid) or t.head_qid, qid=t.head_qid, meta=t.props) for t in triples]

    # --- Schema 结构化表查询支持 ---
    def get_schema_columns(self, target_type: str) -> List[str]:
        return self.kg.get_columns_for_type(target_type)

    def schema_reverse_search(self, target_type: str, column_name: str, match_value: str) -> List[EvidenceEntity]:
        """利用底层的属性检索，返回组装好的实体"""
        qids = self.kg.search_by_property_value(target_type, column_name, match_value)
        out_entities = []
        for qid in qids:
            label = self.kg.qid_to_label(qid) or qid
            props = self.kg.get_entity_props(qid)
            out_entities.append(EvidenceEntity(
                label=label, 
                qid=qid, 
                meta={
                    "source_type": target_type, 
                    "matched_column": column_name, 
                    "matched_value": props.get(column_name)
                }
            ))
        return out_entities

    def get_entity_attribute(self, entity_qid: str, column_name: str):
        """
        通过实体 qid 和列名获取属性值
        """
        props = self.kg.get_entity_props(entity_qid)
        return props.get(column_name) if props else None

    def verify_entity_type(self, entity_qid: str, expected_type: str) -> bool:
        """
        验证实体是否属于预期的类型
        """
        actual_type = self.kg.qid2type.get(entity_qid)
        return actual_type and actual_type.lower() == expected_type.lower()

    def get_entity_type(self, entity_qid: str) -> Optional[str]:
        """
        获取实体的类型
        """
        return self.kg.qid2type.get(entity_qid)

    def get_entity_types(self, entity_qid: str) -> List[str]:
        if hasattr(self.kg, "get_entity_types"):
            return self.kg.get_entity_types(entity_qid)
        entity_type = self.get_entity_type(entity_qid)
        return [entity_type] if entity_type else []

    def get_entity_props(self, qid: str) -> Dict[str, str]:
        """获取实体的属性，代理到 LocalKGIndex"""
        return self.kg.get_entity_props(qid) if hasattr(self.kg, 'get_entity_props') else {}


    def _label_similarity(self, query: str, cand: str) -> float:
        qn, cn = self.kg.normalize_label_for_index(query), self.kg.normalize_label_for_index(cand)
        if not qn or not cn: return 0.0
        try: return self._token_set_ratio(qn, cn) / 100.0
        except Exception: return difflib.SequenceMatcher(a=qn, b=cn).ratio()

    def _expanded_tokens(self, label: str) -> List[str]:
        """将输入字符串分词，复用 LocalKGIndex 的标准化分词逻辑。"""
        return self.kg.tokenize_label(label)

    @staticmethod
    def _normalize_label(label: str) -> str: return " ".join(str(label).strip().split())
    @staticmethod
    def _strip_parenthetical(label: str) -> str: return re.sub(r"\s*\([^)]*\)\s*$", "", str(label)).strip()

