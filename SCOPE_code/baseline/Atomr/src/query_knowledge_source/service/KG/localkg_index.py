# import csv
# import os
# import re
# from collections import defaultdict
# from dataclasses import dataclass
# from typing import Dict, Iterable, List, Optional, Set

# @dataclass(frozen=True)
# class KGTriple:
#     head_qid: str
#     relation: str
#     tail_qid: str
#     props: Dict[str, str]

# class LocalKGIndex:
#     def __init__(self, kg_dir: str):
#         self.kg_dir = os.path.abspath(kg_dir)
#         if not os.path.isdir(self.kg_dir):
#             raise FileNotFoundError(f"LocalKGIndex init failed: kg_dir not found: {self.kg_dir}")

#         self.qid2label: Dict[str, str] = {}
#         self.qid2props: Dict[str, Dict[str, str]] = {}
#         self.qid2type: Dict[str, str] = {}
#         self.type2columns: Dict[str, Set[str]] = defaultdict(set)
        
#         self.label2qids: Dict[str, Set[str]] = defaultdict(set)
#         self.norm_label2qids: Dict[str, Set[str]] = defaultdict(set)
#         self.token2qids: Dict[str, Set[str]] = defaultdict(set)
#         self.prop_token2qids: Dict[str, Set[str]] = defaultdict(set)

#         # Preserve every relation row as an individual triple so repeated
#         # head-relation-tail rows with different metadata remain available.
#         self.head_rel2triples: Dict[str, Dict[str, List[KGTriple]]] = defaultdict(lambda: defaultdict(list))
#         self.tail_rel2triples: Dict[str, Dict[str, List[KGTriple]]] = defaultdict(lambda: defaultdict(list))
#         self.relations: Set[str] = set()

#         self._load_entities()
#         self._load_relations()

#     @staticmethod
#     def normalize_label_for_index(label: str) -> str:
#         s = str(label or "").strip().lower()
#         if not s: return ""
#         s = re.sub(r"[^a-z0-9]+", " ", s)
#         return " ".join(s.split())

#     @staticmethod
#     def tokenize_label(label: str) -> List[str]:
#         s = LocalKGIndex.normalize_label_for_index(label)
#         return [t for t in s.split() if t] if s else []

#     def _iter_csv_files(self) -> Iterable[str]:
#         for fn in os.listdir(self.kg_dir):
#             if fn.lower().endswith(".csv"):
#                 yield os.path.join(self.kg_dir, fn)

#     def _load_entities(self) -> None:
#         for path in self._iter_csv_files():
#             base = os.path.basename(path)
#             if base.startswith("relation_"): continue

#             # 提取实体类型（例如 "City.csv" -> "city"）
#             entity_type = os.path.splitext(base)[0].lower()

#             with open(path, "r", encoding="utf-8", newline="") as f:
#                 reader = csv.DictReader(f)
#                 if not reader.fieldnames: continue

#                 # 记录该表的 Schema 列名
#                 self.type2columns[entity_type].update(reader.fieldnames)

#                 id_field = next((c for c in reader.fieldnames if ":ID(" in c), None)
#                 if not id_field: continue
#                 name_field = "name" if "name" in reader.fieldnames else None
#                 if not name_field: continue

#                 for row in reader:
#                     qid = (row.get(id_field) or "").strip()
#                     label = (row.get(name_field) or "").strip()
#                     if not qid or not label: continue
                    
#                     self.qid2label[qid] = label
#                     self.qid2type[qid] = entity_type # 记录该 QID 属于哪个文件/类型
#                     self.label2qids[label.lower()].add(qid)

#                     norm = self.normalize_label_for_index(label)
#                     if norm:
#                         self.norm_label2qids[norm].add(qid)
#                         for tok in self.tokenize_label(norm):
#                             self.token2qids[tok].add(qid)

#                     props: Dict[str, str] = {}
#                     for k, v in row.items():
#                         if not k or k in {id_field, name_field} or v is None: continue
#                         sv = str(v).strip()
#                         if sv:
#                             props[str(k)] = sv
#                             norm_v = self.normalize_label_for_index(sv)
#                             if norm_v:
#                                 for tok in self.tokenize_label(norm_v):
#                                     if tok: self.prop_token2qids[tok].add(qid)
                                    
#                     if props and qid not in self.qid2props:
#                         self.qid2props[qid] = props

#     def _relation_name_from_filename(self, filename: str) -> str:
#         base = os.path.basename(filename)
#         if base.startswith("relation_") and base.lower().endswith(".csv"):
#             return base[len("relation_") : -len(".csv")]
#         return os.path.splitext(base)[0]

#     def _load_relations(self) -> None:
#         for path in self._iter_csv_files():
#             base = os.path.basename(path)
#             if not base.startswith("relation_"): continue

#             relation = self._relation_name_from_filename(base)
#             self.relations.add(relation)
#             with open(path, "r", encoding="utf-8", newline="") as f:
#                 reader = csv.DictReader(f)
#                 if not reader.fieldnames: continue

#                 start_field = next((c for c in reader.fieldnames if c.startswith(":START_ID(")), None)
#                 end_field = next((c for c in reader.fieldnames if c.startswith(":END_ID(")), None)
#                 prop_fields = [c for c in reader.fieldnames if not c.startswith(":START_ID(") and not c.startswith(":END_ID(")]
                
#                 if not start_field or not end_field: continue

#                 for row in reader:
#                     head = (row.get(start_field) or "").strip()
#                     tail = (row.get(end_field) or "").strip()
#                     if not head or not tail: continue

#                     props = {pf: str(v).strip() for pf in prop_fields if (v := row.get(pf)) is not None and str(v).strip() != ""}
#                     triple = KGTriple(head_qid=head, relation=relation, tail_qid=tail, props=props)
#                     self.head_rel2triples[head][relation].append(triple)
#                     self.tail_rel2triples[tail][relation].append(triple)

#     # --- 基础检索方法保持不变 ---
#     def resolve_label(self, label: str) -> List[str]: 
#         return list(self.label2qids.get(label.strip().lower(), set()))
#     def resolve_label_normalized(self, label: str) -> List[str]: 
#         return list(self.norm_label2qids.get(self.normalize_label_for_index(label), set()))
#     def resolve_tokens(self, tokens: List[str]) -> List[str]:
#         """根据输入的 token 列表，召回包含这些 token 的所有 QID"""
#         result = set()
#         for t in tokens:
#             result |= self.token2qids.get(t, set())
#         return list(result)
#     def qid_to_label(self, qid: str) -> Optional[str]: 
#         return self.qid2label.get(qid)
#     def get_entity_props(self, qid: str) -> Dict[str, str]: 
#         return dict(self.qid2props.get(qid, {}) or {})

#     def relate(self, head_qid: str, relation: str, inverse: bool = False) -> List[KGTriple]:
#         if not inverse:
#             return list(self.head_rel2triples.get(head_qid, {}).get(relation, []))
#         return list(self.tail_rel2triples.get(head_qid, {}).get(relation, []))

#     def get_columns_for_type(self, entity_type: str) -> List[str]:
#         """获取某张实体表的所有列名"""
#         return list(self.type2columns.get(entity_type.lower(), set()))

#     def search_by_property_value(self, target_type: str, column_name: str, match_value: str) -> List[str]:
#         """在指定的实体类型(表)中，寻找某列包含特定值的 QID 列表"""
#         matched_qids = []
#         target_type = target_type.lower()
#         match_value = match_value.lower()
        
#         # 遍历所有符合该类型的实体
#         for qid, e_type in self.qid2type.items():
#             if e_type != target_type: continue
            
#             props = self.get_entity_props(qid)
#             # 检查属性列是否存在且包含目标值 (模糊包含)
#             if column_name in props and match_value in props[column_name].lower():
#                 matched_qids.append(qid)
                
#         return matched_qids
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

@dataclass(frozen=True)
class KGTriple:
    head_qid: str
    relation: str
    tail_qid: str
    props: Dict[str, str]

class LocalKGIndex:
    def __init__(self, kg_dir: str):
        self.kg_dir = os.path.abspath(kg_dir)
        if not os.path.isdir(self.kg_dir):
            raise FileNotFoundError(f"LocalKGIndex init failed: kg_dir not found: {self.kg_dir}")

        self.qid2label: Dict[str, str] = {}
        self.qid2props: Dict[str, Dict[str, str]] = {}
        self.qid2type: Dict[str, str] = {}
        self.qid2types: Dict[str, Set[str]] = defaultdict(set)
        self.type2columns: Dict[str, Set[str]] = defaultdict(set)
        
        self.label2qids: Dict[str, Set[str]] = defaultdict(set)
        self.norm_label2qids: Dict[str, Set[str]] = defaultdict(set)
        self.token2qids: Dict[str, Set[str]] = defaultdict(set)
        self.prop_token2qids: Dict[str, Set[str]] = defaultdict(set)

        # Preserve every relation row as an individual triple so repeated
        # head-relation-tail rows with different metadata remain available.
        self.head_rel2triples: Dict[str, Dict[str, List[KGTriple]]] = defaultdict(lambda: defaultdict(list))
        self.tail_rel2triples: Dict[str, Dict[str, List[KGTriple]]] = defaultdict(lambda: defaultdict(list))
        self.relations: Set[str] = set()

        self._load_entities()
        self._load_relations()

    @staticmethod
    def normalize_label_for_index(label: str) -> str:
        s = str(label or "").strip().lower()
        if not s: return ""
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return " ".join(s.split())

    @staticmethod
    def tokenize_label(label: str) -> List[str]:
        s = LocalKGIndex.normalize_label_for_index(label)
        return [t for t in s.split() if t] if s else []

    def _iter_csv_files(self) -> Iterable[str]:
        for fn in os.listdir(self.kg_dir):
            if fn.lower().endswith(".csv"):
                yield os.path.join(self.kg_dir, fn)

    def _load_entities(self) -> None:
        for path in self._iter_csv_files():
            base = os.path.basename(path)
            if base.startswith("relation_"): continue

            # 提取实体类型（例如 "City.csv" -> "city"）
            entity_type = os.path.splitext(base)[0].lower()

            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: continue

                # 记录该表的 Schema 列名
                self.type2columns[entity_type].update(reader.fieldnames)

                id_field = next((c for c in reader.fieldnames if ":ID(" in c), None)
                if not id_field: continue
                name_field = "name" if "name" in reader.fieldnames else None
                if not name_field: continue

                for row in reader:
                    qid = (row.get(id_field) or "").strip()
                    label = (row.get(name_field) or "").strip()
                    if not qid or not label: continue
                    
                    self.qid2label[qid] = label
                    self.qid2types[qid].add(entity_type)
                    self.qid2type[qid] = entity_type # 记录该 QID 属于哪个文件/类型
                    self.label2qids[label.lower()].add(qid)

                    norm = self.normalize_label_for_index(label)
                    if norm:
                        self.norm_label2qids[norm].add(qid)
                        for tok in self.tokenize_label(norm):
                            self.token2qids[tok].add(qid)

                    props: Dict[str, str] = {}
                    for k, v in row.items():
                        if not k or k in {id_field, name_field} or v is None: continue
                        sv = str(v).strip()
                        if sv:
                            props[str(k)] = sv
                            norm_v = self.normalize_label_for_index(sv)
                            if norm_v:
                                for tok in self.tokenize_label(norm_v):
                                    if tok: self.prop_token2qids[tok].add(qid)
                                    
                    if props and qid not in self.qid2props:
                        self.qid2props[qid] = props

    def _relation_name_from_filename(self, filename: str) -> str:
        base = os.path.basename(filename)
        if base.startswith("relation_") and base.lower().endswith(".csv"):
            return base[len("relation_") : -len(".csv")]
        return os.path.splitext(base)[0]

    def _load_relations(self) -> None:
        for path in self._iter_csv_files():
            base = os.path.basename(path)
            if not base.startswith("relation_"): continue

            relation = self._relation_name_from_filename(base)
            self.relations.add(relation)
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: continue

                start_field = next((c for c in reader.fieldnames if c.startswith(":START_ID(")), None)
                end_field = next((c for c in reader.fieldnames if c.startswith(":END_ID(")), None)
                prop_fields = [c for c in reader.fieldnames if not c.startswith(":START_ID(") and not c.startswith(":END_ID(")]
                
                if not start_field or not end_field: continue

                for row in reader:
                    head = (row.get(start_field) or "").strip()
                    tail = (row.get(end_field) or "").strip()
                    if not head or not tail: continue

                    props = {pf: str(v).strip() for pf in prop_fields if (v := row.get(pf)) is not None and str(v).strip() != ""}
                    triple = KGTriple(head_qid=head, relation=relation, tail_qid=tail, props=props)
                    self.head_rel2triples[head][relation].append(triple)
                    self.tail_rel2triples[tail][relation].append(triple)

    # --- 基础检索方法保持不变 ---
    def resolve_label(self, label: str) -> List[str]: 
        return list(self.label2qids.get(label.strip().lower(), set()))
    def resolve_label_normalized(self, label: str) -> List[str]: 
        return list(self.norm_label2qids.get(self.normalize_label_for_index(label), set()))
    def resolve_tokens(self, tokens: List[str]) -> List[str]:
        """根据输入的 token 列表，召回包含这些 token 的所有 QID"""
        result = set()
        for t in tokens:
            result |= self.token2qids.get(t, set())
        return list(result)
    def qid_to_label(self, qid: str) -> Optional[str]: 
        return self.qid2label.get(qid)
    def get_entity_props(self, qid: str) -> Dict[str, str]: 
        return dict(self.qid2props.get(qid, {}) or {})
    def get_entity_types(self, qid: str) -> List[str]:
        return sorted(self.qid2types.get(qid, set()))

    def relate(self, head_qid: str, relation: str, inverse: bool = False) -> List[KGTriple]:
        if not inverse:
            return list(self.head_rel2triples.get(head_qid, {}).get(relation, []))
        return list(self.tail_rel2triples.get(head_qid, {}).get(relation, []))

    def get_columns_for_type(self, entity_type: str) -> List[str]:
        """获取某张实体表的所有列名"""
        return list(self.type2columns.get(entity_type.lower(), set()))

    def search_by_property_value(self, target_type: str, column_name: str, match_value: str) -> List[str]:
        """在指定的实体类型(表)中，寻找某列包含特定值的 QID 列表"""
        matched_qids = []
        target_type = target_type.lower()
        match_value = match_value.lower()
        
        # 遍历所有符合该类型的实体
        for qid, e_type in self.qid2type.items():
            if e_type != target_type: continue
            
            props = self.get_entity_props(qid)
            # 检查属性列是否存在且包含目标值 (模糊包含)
            if column_name in props and match_value in props[column_name].lower():
                matched_qids.append(qid)
                
        return matched_qids
