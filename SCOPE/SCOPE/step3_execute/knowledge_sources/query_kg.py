from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from step3_execute.prompts.kg_prompt import Prompt
import json
import re
import ast

from step3_execute.service.KG.kg_retriever import EvidenceEntity

class KGSource:
    """Knowledge Graph operator layer with unified LLM interaction."""
    
    def __init__(self, kg_retriever, llm=None, verbose: bool = True):
        if kg_retriever is None: raise ValueError("KGSource requires kg_retriever")
        self.kg = kg_retriever
        self.llm = llm
        self.verbose = verbose
        if self.verbose: print("✓ KGSource operators initialized successfully\n")

    @staticmethod
    def _clean_answer_text(text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"^[\*\-\u2022\s]+", "", text)
        text = re.sub(r"[\*\s]+$", "", text)
        text = text.strip("[]'\"`")
        return text.strip()

    @staticmethod
    def _dedupe_labels(labels: List[str]) -> List[str]:
        deduped = []
        seen = set()
        for label in labels:
            clean = KGSource._clean_answer_text(label)
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(clean)
        return deduped

    @staticmethod
    def _extract_years_from_value(value: Any) -> List[int]:
        years: List[int] = []
        if value is None:
            return years
        if isinstance(value, dict):
            for nested in value.values():
                years.extend(KGSource._extract_years_from_value(nested))
            return years
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                years.extend(KGSource._extract_years_from_value(nested))
            return years

        for match in re.findall(r"\b(1[0-9]{3}|20[0-9]{2}|2100)\b", str(value)):
            try:
                years.append(int(match))
            except ValueError:
                continue
        return years

    def _extract_years_from_evidence(self, evidence_item: dict) -> List[int]:
        meta = evidence_item.get("meta") or {}
        if not isinstance(meta, dict):
            return []

        years: List[int] = []
        preferred_keys = ("time", "year", "date", "season")
        for key, value in meta.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in preferred_keys):
                years.extend(self._extract_years_from_value(value))

        if not years:
            years.extend(self._extract_years_from_value(meta))

        return sorted(set(years))

    @staticmethod
    def _has_consecutive_years(years: List[int]) -> bool:
        if len(years) < 2:
            return False
        ordered = sorted(set(years))
        return any(curr - prev == 1 for prev, curr in zip(ordered, ordered[1:]))

    def _group_evidence_by_entity(self, evidence: List[dict]) -> List[dict]:
        grouped: Dict[str, dict] = {}
        ordered_keys: List[str] = []

        for ev in evidence:
            label = str(ev.get("label", "")).strip()
            qid = ev.get("qid")
            key = f"qid:{qid}" if qid else f"label:{label.lower()}"
            if key not in grouped:
                grouped[key] = {"qid": qid, "items": []}
                ordered_keys.append(key)
            grouped[key]["items"].append(ev)

        return [grouped[key] for key in ordered_keys]

    def _collect_matching_evidence(
        self,
        answers: List[str],
        evidence: List[dict],
        reasoning_text: str,
        reasoning_key: str,
        fuzzy: bool = False,
    ) -> Tuple[List[str], List[dict]]:
        matched_labels: List[str] = []
        matched_evidence: List[dict] = []
        seen_labels = set()
        seen_indices = set()

        for answer in self._dedupe_labels(answers):
            answer_lower = answer.lower()
            for idx, ev in enumerate(evidence):
                ev_label = str(ev.get("label", "")).strip()
                ev_label_lower = ev_label.lower()
                is_match = ev_label_lower == answer_lower
                if fuzzy:
                    is_match = is_match or answer_lower in ev_label_lower or ev_label_lower in answer_lower
                if not is_match or idx in seen_indices:
                    continue

                seen_indices.add(idx)
                ev_copy = ev.copy()
                suffix = " (fuzzy matched)" if fuzzy else ""
                ev_copy[reasoning_key] = reasoning_text + suffix
                matched_evidence.append(ev_copy)

                if ev_label_lower not in seen_labels:
                    seen_labels.add(ev_label_lower)
                    matched_labels.append(ev_label)

        return matched_labels, matched_evidence

    @staticmethod
    def _dedupe_evidence_rows(evidence: List[dict]) -> List[dict]:
        deduped: List[dict] = []
        seen = set()

        for ev in evidence or []:
            if not isinstance(ev, dict):
                continue

            meta = ev.get("meta") or {}
            try:
                meta_key = json.dumps(meta, ensure_ascii=False, sort_keys=True)
            except TypeError:
                meta_key = str(meta)

            key = (
                str(ev.get("label", "")).strip().lower(),
                str(ev.get("qid", "")).strip(),
                str(ev.get("head_entity", "")).strip().lower(),
                str(ev.get("relation", "")).strip().lower(),
                str(ev.get("matched_col", "")).strip().lower(),
                meta_key,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ev)

        return deduped

    @staticmethod
    def _format_final_verify_evidence_row(index: int, evidence_item: dict) -> str:
        label = str(evidence_item.get("label", "")).strip() or "Unknown"
        qid = str(evidence_item.get("qid", "")).strip() or "Unknown"
        head_entity = str(evidence_item.get("head_entity", "")).strip()
        relation = str(evidence_item.get("relation", "")).strip()
        source = str(evidence_item.get("source", "")).strip()
        meta = evidence_item.get("meta") or {}

        context_bits: List[str] = [f"Label: {label}"]
        if head_entity:
            context_bits.append(f"Source Entity: {head_entity}")
        if relation:
            context_bits.append(f"Relation: {relation}")
        if source:
            context_bits.append(f"Evidence Source: {source}")

        if isinstance(meta, dict) and meta:
            intersect_ref = str(meta.get("intersect_ref", "")).strip()
            intersect_group = str(meta.get("intersect_group", "")).strip()
            if intersect_ref:
                context_bits.append(f"Intersect Ref: {intersect_ref}")
            if intersect_group:
                context_bits.append(f"Intersect Group: {intersect_group}")
            meta_str = ", ".join(f"{k}: {v}" for k, v in meta.items())
        else:
            meta_str = "No specific metadata"

        context_text = f" | {' | '.join(context_bits)}" if context_bits else ""
        return f"[Evidence #{index}] QID: {qid}{context_text} | Metadata: {meta_str}"

    def _structured_filter(self, condition: str, evidence: List[dict]) -> Optional[Tuple[List[str], List[dict], str]]:
        condition_lower = str(condition or "").lower()
        expects_consecutive = "consecutive" in condition_lower
        count_match = re.search(r"count\s*[=:]\s*(\d+)", condition_lower)
        expected_count = int(count_match.group(1)) if count_match else None

        if not expects_consecutive and expected_count is None:
            return None

        selected_labels: List[str] = []
        selected_evidence: List[dict] = []
        reasoning_bits: List[str] = []

        for group in self._group_evidence_by_entity(evidence):
            group_label = group["label"]
            group_items = group["items"]
            representative_meta = group_items[0].get("meta") or {}

            if len(group_items) == 1 and isinstance(representative_meta, dict) and "count" in representative_meta:
                try:
                    count = int(representative_meta.get("count"))
                except (TypeError, ValueError):
                    count = len(group_items)
                years = self._extract_years_from_value(representative_meta.get("years"))
                if not years:
                    years = self._extract_years_from_evidence(group_items[0])
                has_consecutive = bool(representative_meta.get("has_consecutive_years")) or self._has_consecutive_years(years)
            else:
                count = len(group_items)
                years = sorted({year for item in group_items for year in self._extract_years_from_evidence(item)})
                has_consecutive = self._has_consecutive_years(years)

            if expected_count is not None and count != expected_count:
                continue
            if expects_consecutive and not has_consecutive:
                continue

            selected_labels.append(group_label)
            selected_evidence.extend(group_items)
            if years:
                reasoning_bits.append(f"{group_label}: years={years}, count={count}")
            else:
                reasoning_bits.append(f"{group_label}: count={count}")

        if not selected_labels:
            return None

        reasoning = "Structured filter matched from evidence metadata: " + "; ".join(reasoning_bits)
        return self._dedupe_labels(selected_labels), selected_evidence, reasoning

    def _compute_group_count(self, evidence: List[dict]) -> Tuple[List[str], List[dict]]:
        results: List[str] = []
        result_evidence: List[dict] = []

        for group in self._group_evidence_by_entity(evidence):
            label = group["label"]
            qid = group["qid"]
            items = group["items"]
            years = sorted({year for item in items for year in self._extract_years_from_evidence(item)})

            results.append(label)
            result_evidence.append({
                "label": label,
                "qid": qid,
                "meta": {
                    "count": len(items),
                    "years": years,
                    "has_consecutive_years": self._has_consecutive_years(years),
                },
                "grouped_evidence": items,
            })

        return results, result_evidence

    @staticmethod
    def _is_type_compatible(actual_type: Optional[str], target_type: Optional[str]) -> bool:
        if not actual_type or not target_type:
            return False

        actual = str(actual_type).strip().lower()
        target = str(target_type).strip().lower()
        if actual == target:
            return True

        equivalence_groups = [
            {"person", "player", "coach"},
        ]
        return any(actual in group and target in group for group in equivalence_groups)

    @staticmethod
    def _normalize_entity_type(entity_type: Optional[str]) -> str:
        return str(entity_type or "").strip().lower()

    def _call_llm(self, prompt: str, max_tokens: int = 10000):
        """统一 LLM 调用入口，失败时抛出异常由调用方处理。"""
        response, meta = self.llm.query_gpt4o(prompt=prompt, max_tokens=max_tokens)
        return response, meta

    #########################################################################################
    # KG Search
    #########################################################################################
    def Search(self, question: str, entity_name: str, descriptors: str = "") -> Tuple[List[str], List[dict]]:
        """Search for entity in knowledge graph with mandatory LLM verification and fallback hint."""
        if self.verbose:
            print("-" * 70)
            print("Search Operation")
            print("-" * 70)
            print(f"Searching for entity: '{entity_name}'")
            if descriptors:
                print(f"Descriptors: '{descriptors}'")
            print()

        query = f"{str(entity_name).strip()} {str(descriptors).strip()}".strip()
        entities = self.kg.search_entity(query)
        print(entities)
        # 初始检索结果 (保留有 QID 的实体)
        labels = [e.label for e in entities if e.qid is not None]
        evidence = [{
            "label": e.label,
            "qid": e.qid,
            "meta": e.meta
        } for e in entities if e.qid is not None]
        
        if self.verbose:
            print(f"Initial Results: {len(labels)} entities found via BM25/Fuzzy Match\n")
            if labels:
                print(labels)
            print()
            
        # 🚀 只要初始有结果，且配置了 LLM-->则经过 LLM 的筛选
        if self.llm is not None and evidence:
            if self.verbose:
                print(">>> Triggering LLM for MMQA_search verification...\n")
                
            supporting_knowledge = {"KG": evidence}
            prompt = Prompt.KG_search(question, entity_name, supporting_knowledge)
            
            try:
                response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=10000)
                verified_labels, verified_evidence, reasoning_text = self._parse_search_response(response, evidence)
                
                if self.verbose:
                    print(f"Verified Results: Filtered down to {len(verified_labels)} actual matches\n")
                    if reasoning_text:
                        print(f"🧠 LLM Reasoning: {reasoning_text}\n")
                        
                    if verified_labels:
                        for i, (label, ev) in enumerate(zip(verified_labels, verified_evidence), 1):
                            print(f"{i}. Label: {label}")
                            print(f"   QID: {ev.get('qid')}")
                            print(f"   Meta: {ev.get('meta')}")
                            print()
                            
                # 将最终结果更新为 LLM 筛选后的结果
                labels = verified_labels
                evidence = verified_evidence
                
            except Exception as e:
                print(f"[Warning] MMQA_search LLM verification failed: {e}")
                # 报错时优雅降级，保持使用初始结果
        
        # 针对没有配置 LLM 时的原始结果打印
        elif self.llm is None and self.verbose and labels:
            print("LLM未配置，输出原始结果")
            for i, (label, ev) in enumerate(zip(labels, evidence), 1):
                print(f"{i}. Label: {label}")
                print(f"   QID: {ev.get('qid')}")
                print(f"   Meta: {ev.get('meta')}")
                print()

        # 🚀 统一收口：只要最终结果为空（无论是因为初始没搜到，还是被 LLM 全否决了：需合并 search 与 relate 操作，基于目标实体的属性值倒查。）
        if not labels:
            if self.verbose:
                print("⚠️ LLM determined no entities matched the target, or search returned empty.\n")
                print(">>> 💡 提示：目标实体在图中无独立节点，可能作为属性值存在。")
                print(">>> 💡 建议：需合并 search 与 relate 操作，基于目标实体的属性值倒查。")
                print(f">>> 💡 动作：发起针对 '{entity_name}' 的 ReverseSearch。\n")
            
            # 返回空结果 + 错误信息，通过 evidence 传递
            return [], [{
                "error": "ENTITY_NOT_FOUND",
                "missing_entity": entity_name,
                "suggestion": "Call ReverseSearch with this missing_entity"
            }]
            
        return labels, evidence

    
    #########################################################################################
    # KG Relate
    #########################################################################################
    def Relate(self, question: str, entities: List[dict], relation: str, target_table: str = None) -> Tuple[List[str], List[dict]]:
        if self.verbose:
            print("-" * 70)
            print("Relate Operation")
            print("-" * 70)
            print(f"Relation: '{relation}' | Target Entity Type: '{target_table}'\n")

        if not entities: 
            return [], []
        
        all_answers, all_evidence = [], []
        
        for entity_data in entities:
            if entity_data.get("error") == "ENTITY_NOT_FOUND":
                missing_entity = entity_data.get("missing_entity")  
                if self.verbose:
                    print(f"⚠️ Detected missing entity: '{missing_entity}', triggering ReverseSearch directly...\n")
                
                fb_labels, fb_evidence = self.ReverseSearch(
                    question=question,
                    missing_entity=missing_entity,
                    relation=relation,
                    target_entity=target_table
                )
                all_answers.extend(fb_labels)
                all_evidence.extend(fb_evidence)
                continue
            
            evidence_entity = EvidenceEntity(
                label=entity_data.get("label", ""),
                qid=entity_data.get("qid"),
                meta=entity_data.get("meta")
            )
            
            # 1. 尝试正向查询 (Out-edges) 
            # 🚀 修改点：提取 LLM 在底层的关系映射结果 (mapped_relation)
            related_entities, mapped_relation = self.kg.relate(entity=evidence_entity, relation_text=relation, question=question)
            
            # 判断关系映射是否成功
            mapping_success = bool(mapped_relation and str(mapped_relation).strip().lower() not in ["none", "", "null"])

            if self.verbose:
                print(f"正向查询结束，得到 {len(related_entities)} 个相关实体")
                if mapping_success:
                    print(f"  ✓ 关系映射成功: '{mapped_relation}'")
                if related_entities:
                    for ent in related_entities[:5]:
                        print(f"  - {ent.label} (qid: {ent.qid})")

            # 2. 尝试反向查询 (In-edges)
            if not related_entities:
                if self.verbose: 
                    print(f"  (No outgoing edges for '{relation}', searching incoming edges...)")
                related_entities = self.kg.find_incoming_entities(target_entity=evidence_entity, relation_text=relation, question=question)
                if self.verbose:
                    print(f"反向查询结束，得到 {len(related_entities)} 个相关实体")
                    if related_entities:
                        for ent in related_entities[:5]:
                            print(f"  - {ent.label} (qid: {ent.qid})")

            # ================= 核心分支判断 =================
            
            # 分支 A：找到了图谱中的关系边，需要验证类型
            if related_entities:
                if self.verbose:
                    print(f"\n🔍 验证返回的实体类型是否匹配目标类型 '{target_table}'...")
                
                # 收集实体信息和 QID
                entities_info = []
                for ent in related_entities:
                    ent_type = self.kg.get_entity_type(ent.qid) if ent.qid else None
                    entities_info.append({
                        "label": ent.label,
                        "qid": ent.qid,
                        "type": ent_type
                    })
                
                # # 先使用 KG 中的真实实体类型做确定性校验。
                type_check_passed = True
                if target_table:
                    typed_entities = [e for e in entities_info if e["type"]]
                    compatible_entities = [
                        e for e in entities_info
                        if self._is_type_compatible(e["type"], target_table)
                    ]

                    if self.llm is not None:
                        prompt = Prompt.KG_relation_type_check(
                            question=question,
                            relation=relation,
                            target_type=target_table,
                            retrieved_entities=[e["label"] for e in entities_info]
                        )
                        try:
                            response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=10000)
                            if self.verbose:
                                print(f"LLM 类型验证响应: {response}\n")

                            if "FAIL" in response:
                            # if "Verification: FAIL" in response or "FAIL" in response.split("\n")[0]:
                                type_check_passed = False
                                if self.verbose:
                                    print(f"⚠️ LLM 类型验证失败: 返回的实体类型不是 '{target_table}'，尝试属性查找...\n")
                            else:
                                if self.verbose:
                                    print(f"✓ LLM 类型验证通过: 返回的实体类型匹配 '{target_table}'\n")
                        except Exception as e:
                            if self.verbose:
                                print(f"[Warning] LLM type verification failed: {e}")
                            type_check_passed = False
                
                # 如果类型验证通过，直接使用结果
                if type_check_passed:
                    for ent in related_entities:
                        all_answers.append(ent.label)
                        all_evidence.append({
                            "label": ent.label, 
                            "qid": ent.qid, 
                            "meta": ent.meta,
                            "head_entity": evidence_entity.label, 
                            "relation": relation,
                            "source": "graph_edge"
                        })
                else:
                    # 类型验证失败，触发属性查找
                    if self.verbose:
                        print(f"🔍 触发属性查找: 从 '{evidence_entity.label}' 的属性中查找 '{target_table}' 信息...")
                    
                    # 从源实体的属性中查找目标信息
                    attr_labels, attr_evidence = self._search_entity_attributes(
                        question=question,
                        source_entity=evidence_entity,
                        target_type=target_table,
                        relation=relation
                    )
                    
                    if attr_labels:
                        all_answers.extend(attr_labels)
                        all_evidence.extend(attr_evidence)
                        if self.verbose:
                            print(f"✓ 属性查找成功: 找到 {len(attr_labels)} 个结果\n")
                    else:
                        # 如果属性查找也失败，再尝试 SchemaFallback
                        if self.verbose:
                            print(f"⚠️ 属性查找无结果，尝试 SchemaFallback...\n")
                        fb_labels, fb_evidence = self.SchemaFallback(
                            question=question, 
                            missing_entity=evidence_entity.label, 
                            relation=relation, 
                            target_entity=target_table 
                        )
                        all_answers.extend(fb_labels)
                        all_evidence.extend(fb_evidence)
            
            # 分支 B：正查反查都没有结果
            else:
                # 🚀 修改点：判断如果关系映射成功，且没有查到结果，则直接结束
                if mapping_success:
                    if self.verbose: 
                        print(f"⚠️ 大模型在关系映射上成功 ('{mapped_relation}')，但是正反向均没查到结果，直接结束 relate 分支。\n")
                    continue  # 跳过当前实体，不进入属性查找
                
                # 🚀 修改点：如果没有关系映射成功，才进入属性查找
                if self.verbose: 
                    print(f"⚠️ 关系映射未成功 (且无图谱边)，进入属性查找: 从 '{evidence_entity.label}' 的属性中查找 '{target_table}' 信息...\n")
                
                # 先尝试从源实体属性中查找
                attr_labels, attr_evidence = self._search_entity_attributes(
                    question=question,
                    source_entity=evidence_entity,
                    target_type=target_table,
                    relation=relation
                )
                
                if attr_labels:
                    all_answers.extend(attr_labels)
                    all_evidence.extend(attr_evidence)
                    if self.verbose:
                        print(f"✓ 属性查找成功: 找到 {len(attr_labels)} 个结果\n")
                else:
                    # 属性查找失败，再尝试 SchemaFallback
                    fb_labels, fb_evidence = self.SchemaFallback(
                        question=question, 
                        missing_entity=evidence_entity.label, 
                        relation=relation, 
                        target_entity=target_table 
                    )
                    all_answers.extend(fb_labels)
                    all_evidence.extend(fb_evidence)

        if self.verbose:
            print(f"Results: {len(all_answers)} related entities found\n")
                
        return all_answers, all_evidence
    
    def _search_entity_attributes(self, question: str, source_entity: EvidenceEntity, 
                                target_type: str, relation: str) -> Tuple[List[str], List[dict]]:
        """
        从源实体的属性中查找目标信息。
        例如：从 Hakim Warrick 的属性中查找 draft_year 来获取 season 信息。
        """
        if self.verbose:
            print(f"  🔍 _search_entity_attributes: 从 '{source_entity.label}' 的属性中查找 '{target_type}'")
        
        if not source_entity.qid:
            if self.verbose:
                print(f"    ✗ 源实体没有 QID，无法获取属性")
            return [], []
        
        # 1. 获取源实体的所有属性
        entity_props = self.kg.get_entity_props(source_entity.qid)
        if not entity_props:
            if self.verbose:
                print(f"    ✗ 源实体没有属性数据")
            return [], []
        
        if self.verbose:
            print(f"    ✓ 获取到 {len(entity_props)} 个属性: {list(entity_props.keys())[:10]}...")
        
        # 2. 使用 LLM 判断哪个属性包含目标信息
        mapped_property = None
        if self.llm is not None:
            prompt = Prompt.KG_property_extraction(
                question=question,
                source_entity=source_entity.label,
                target_type=target_type,
                relation=relation,
                entity_props=entity_props
            )
            # print(f"prompt: {prompt}")
            try:
                response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=10000)
                mapped_property = response.strip()
                
                # print(f"response: {mapped_property}")

                # 清理响应
                if "Property Name:" in mapped_property:
                    mapped_property = mapped_property.split("Property Name:")[-1].strip()
                if mapped_property.startswith("Answer:"):
                    mapped_property = mapped_property.replace("Answer:", "").strip()
                
                if self.verbose:
                    print(f"    LLM 映射结果: {mapped_property}")
                
                # 🚀 修改点：如果返回的是 None，表示没有这个对应的属性，直接返回空值
                if mapped_property.lower() == "none":
                    if self.verbose:
                        print(f"    ✗ LLM 判断该实体没有对应的属性 (返回 None)，直接返回空值。")
                    return [], []

            except Exception as e:
                if self.verbose:
                    print(f"    [Warning] LLM property extraction failed: {e}")
        
        # 3. 尝试在属性中查找
        candidates = []
        property_to_check = []
        
        # 如果 LLM 有映射结果 (且上面拦截了 none)，优先使用
        if mapped_property:
            property_to_check = [mapped_property]
        else:
            # 否则 (例如未配置 LLM) 尝试语义相关的属性
            semantic_keys = [
                "draft_year", "draft_season", "drafted_in", "draft", "season",
                "year", "date", "time", "start_time", "end_time",
                "drafted_in_season", "draft_round", "draft_pick"
            ]
            property_to_check = semantic_keys
        
        # 查找匹配的属性值
        for prop_key in property_to_check:
            if prop_key in entity_props:
                prop_value = entity_props[prop_key]
                if prop_value and str(prop_value).strip():
                    candidates.append({
                        "value": str(prop_value),
                        "property": prop_key,
                        "source": "attribute"
                    })
                    if self.verbose:
                        print(f"    ✓ 找到属性 '{prop_key}': '{prop_value}'")
        
        # 如果没有找到语义匹配，尝试模糊匹配属性名 (被注释分支逻辑保持原样中断)
        if not candidates and self.llm is not None:
            if self.verbose:
                print(f"    未找到匹配的属性，该分支中断！！！")
        
        # 4. 构建结果
        all_answers = []
        all_evidence = []
        
        for cand in candidates:
            all_answers.append(cand["value"])
            all_evidence.append({
                "label": cand["value"],
                "qid": None,  # 属性值通常没有 QID
                "meta": {
                    "source_entity": source_entity.label,
                    "source_qid": source_entity.qid,
                    "property": cand["property"],
                    "value": cand["value"],
                    "match_type": cand["source"],
                    "relation": relation,
                    "target_type": target_type
                },
                "head_entity": source_entity.label,
                "relation": relation
            })
        
        return all_answers, all_evidence

    def ReverseSearch(self, question: str, missing_entity: str, relation: str, target_entity: str) -> Tuple[List[str], List[dict]]:
        """
        独立算子：基于属性值的实体倒查 (Reverse Search by Property Value)。
        当 Search 算子找不到目标实体时，LLM 可调用此算子，直接去目标类型 (target_entity) 的表里，
        寻找哪个属性列 (relation) 包含该缺失的实体名 (missing_entity)。
        """
        if self.verbose:
            print("-" * 70)
            print("🔄 ReverseSearch Operation (Property-based Fallback)")
            print("-" * 70)
            print(f"Target Table: '{target_entity}' | Relation: '{relation}' | Finding: '{missing_entity}'\n")

        # 1. 向底层请求这张表所有的列名
        columns = self.kg.get_schema_columns(target_entity)
        if not columns:
            if self.verbose: print(f"  ✗ Reverse Search failed: Schema for type '{target_entity}' not found.\n")
            return [], []

        # 2. 调用 LLM 进行列映射
        mapped_column = None
        if self.llm is not None:
            # 注意传参顺序与你的 def MMQA_schema_mapping 一致
            prompt = Prompt.KG_schema_mapping(question, relation, target_entity, columns)
            try:
                response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=10000)
                mapped_column = response.strip()
                if mapped_column.startswith("Answer:"):
                    mapped_column = mapped_column.replace("Answer:", "").strip()
            except Exception as e:
                print(f"[Warning] Schema mapping failed: {e}")
        
        # 4. 调用中间引擎去底层进行属性查询
        matched_entities = self.kg.schema_reverse_search(
            target_type=target_entity, 
            column_name=mapped_column, 
            match_value=missing_entity
        )

        all_answers = [e.label for e in matched_entities]
        all_evidence = [{"label": e.label, "qid": e.qid, "meta": e.meta} for e in matched_entities]

        if self.verbose:
            print(f"Results: {len(all_answers)} matching targets found.\n")
            # 顺手打印一下查出来的前几个结果，方便在控制台确认找得对不对
            for i, (ans, ev) in enumerate(zip(all_answers[:5], all_evidence[:5]), 1):
                print(f"    {i}. {ans} (Matched Value: {ev['meta'].get('matched_value')})")
            print()

        return all_answers, all_evidence

    def SchemaFallback(self, question: str, missing_entity: str, relation: str, target_entity: str) -> Tuple[List[str], List[dict]]:
        """
        利用现有的 KGRetriever 结构化表查询方法进行兜底。
        """
        # 1. 调用已有的 get_schema_columns 获取目标表的所有列
        columns = self.kg.get_schema_columns(target_entity)
        if not columns:
            if self.verbose: print(f"  ✗ Target Table '{target_entity}' has no available columns.\n")
            return [], []

        mapped_column = None
        
        # 2. 使用 Prompt 让 LLM 寻找关系和列名的映射
        if self.llm is not None:
            # 🔧 修改：按照正确的参数顺序传入，并添加 entity_type
            prompt = Prompt.KG_schema_mapping(
                question=question,
                relation=relation,
                target_entity=target_entity,
                columns=columns,
                missing_entity=missing_entity,      # 新增：缺失的实体名称
                entity_type=target_entity            # 新增：实体类型（这里用 target_entity 作为类型）
            )
            try:
                response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=10000)
                mapped_column = response.strip()
                if mapped_column.startswith("Answer:"):
                    mapped_column = mapped_column.replace("Answer:", "").strip()
            except Exception as e:
                print(f"[Warning] Schema mapping failed: {e}")

        # 3. 如果映射成功，调用已有的 schema_reverse_search 去底层查数据
        if mapped_column and mapped_column in columns and mapped_column.lower() != "none":
            if self.verbose:
                print(f"  ✓ LLM mapped relation '{relation}' -> column '{mapped_column}'.")
                
            # 直接传入映射好的列名和我们要找的实体字符串
            matched_entities = self.kg.schema_reverse_search(
                target_type=target_entity, 
                column_name=mapped_column, 
                match_value=missing_entity
            )
            
            all_answers = [e.label for e in matched_entities]
            all_evidence = [{"label": e.label, "qid": e.qid, "meta": e.meta} for e in matched_entities]
            
            return all_answers, all_evidence
            
        else:
            if self.verbose:
                print(f"  ✗ LLM could not map relation '{relation}' to any column in '{target_entity}'.\n")
            return [], []
    
    #########################################################################################
    # KG Filter
    #########################################################################################

    def Filter(self, question: str, answers: List[str], evidence: List[dict], condition: str) -> Tuple[List[str], List[dict]]:
        """Filter answers based on condition."""
        if self.verbose:
            print("-" * 70)
            print("Filter Operation")
            print("-" * 70)
            print(f"Condition: '{condition}'")
            print(f"Input answers count: {len(answers)}")
            print()

        if not evidence:
            if self.verbose:
                print("Cannot proceed with Filter - no input evidence.\n")
            return [], []

        supporting_knowledge = {"KG": evidence}

        if self.llm is not None:
            prompt = Prompt.KG_filter(question, condition, supporting_knowledge)
            try:
                response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=10000)

                filtered_answers, filtered_evidence, reasoning_text = self._parse_filter_response(response)

                if self.verbose:
                    print(f"Results: Filtered down to {len(filtered_answers)} answers\n")

                    if reasoning_text:
                        print(f"🧠 LLM Reasoning: {reasoning_text}\n")

                    for i, ans in enumerate(filtered_answers[:5], 1):
                        print(f"{i}. {ans}")
                    if len(filtered_answers) > 5:
                        print(f"... and {len(filtered_answers) - 5} more results\n")

                return filtered_answers, filtered_evidence
            except Exception as e:
                print(f"[Warning] MMQA_filter failed: {e}")
                return [], []

        return [], []
    

    #########################################################################################
    # KG Math
    #########################################################################################
    def Math(self, question: str, answers: List[str], evidence: List[dict], operation: str) -> Tuple[List[str], List[dict]]:
        """Perform mathematical operations on answers."""
        if self.verbose:
            print("-" * 70)
            print("Math Operation")
            print("-" * 70)
            print(f"Operation: '{operation}'")
            print(f"Input answers count: {len(answers)}")
            print()

        if not evidence:
            if self.verbose:
                print("Cannot proceed with Math - no input evidence.\n")
            return [], []

        supporting_knowledge = {"KG": evidence}

        if self.llm is not None:
            prompt = Prompt.KG_math(question, operation, supporting_knowledge)
            try:
                response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=10000)
                results, result_evidence = self._parse_math_response(response)

                if self.verbose:
                    print(f"Results: Computed {len(results)} answers\n")
                    for i, ans in enumerate(results[:5], 1):
                        print(f"{i}. {ans}")
                    if len(results) > 5:
                        print(f"... and {len(results) - 5} more results\n")

                return results, result_evidence
            except Exception as e:
                print(f"[Warning] MMQA_math failed: {e}")
                return [], []

        return [], []

    #########################################################################################
    # KG Final Verification
    #########################################################################################
    # def FinalVerify(
    #     self,
    #     question: str,
    #     answers: List[str],
    #     evidence: List[dict]
    # ) -> Tuple[List[str], List[dict]]:
    #     """
    #     对着原问题进行最后一次全量验证，利用 KG 边上的 Meta 属性剔除不符合时间/条件的脏数据，并在输出前完成去重。
    #     """
    #     if self.verbose:
    #         print("-" * 70)
    #         print("KG Final Verify (Ultimate Patch & Dedup)")
    #         print("-" * 70)
    #         print(f"Inputs: {len(answers)} candidates\n")

    #     # 内部封装的去重函数（连带 evidence 一起去重）
    #     def _deduplicate(ans_list: List[str], ev_list: List[dict]) -> Tuple[List[str], List[dict]]:
    #         # 保持原序去重 labels
    #         deduped_ans = list(dict.fromkeys(ans_list))
    #         deduped_ev = []
    #         seen_labels = set()
    #         for e in ev_list:
    #             lbl = e.get('label', '')
    #             if lbl and lbl not in seen_labels:
    #                 seen_labels.add(lbl)
    #                 deduped_ev.append(e)
    #         return deduped_ans, deduped_ev

    #     if not answers or not evidence:
    #         return [], []
        
    #     if self.llm is None:
    #         if self.verbose:
    #             print("  [FinalVerify] No LLM, skipping verification, applying dedup only.\n")
    #         return _deduplicate(answers, evidence)

    #     # 1. 拼装 Key-Value 形式的清晰 Evidence 给 LLM 检查
    #     lines = []
    #     for e in evidence:
    #         lbl = e.get('label', '')
    #         meta = e.get('meta', {})
            
    #         # 将字典形式的 meta 转换为清晰的字符串，例如 "start_time: 2010, end_time: 2014"
    #         if isinstance(meta, dict) and meta:
    #             meta_str = ", ".join([f"{k}: {v}" for k, v in meta.items()])
    #         else:
    #             meta_str = "No specific metadata"
                
    #         lines.append(f"[Candidate: {lbl}] Metadata: {meta_str}")
        
    #     candidate_evidence = "\n".join(lines)
    #     prompt = Prompt.KG_final_verification(question, candidate_evidence)

    #     try:
    #         # 2. 调用 LLM 进行质检
    #         if self.verbose:
    #             print(f">>> Triggering LLM for Final Verification...\n")
                
    #         response, _ = self._call_llm(prompt, max_tokens=10000)
            
    #         # 3. 解析 LLM 返回的结果
    #         verified_labels = []
    #         reasoning_text = ""
            
    #         ans_match = re.search(r"Answer:\s*(.*?)(?:\n|$|Reasoning:)", response, re.IGNORECASE | re.DOTALL)
    #         reason_match = re.search(r"Reasoning:\s*(.*)", response, re.IGNORECASE | re.DOTALL)

    #         if reason_match:
    #             reasoning_text = reason_match.group(1).strip()
            
    #         if ans_match:
    #             answer_part = ans_match.group(1).strip()
    #             if answer_part.lower() not in ["none", "null", ""]:
    #                 raw_answers = answer_part.split(",")
    #                 # 剥离多余空格或可能存在的括号
    #                 verified_labels = [ans.strip().strip("[]'\"") for ans in raw_answers if ans.strip()]

    #         if not verified_labels:
    #             if self.verbose:
    #                 print(f" 🧠 LLM Reasoning: {reasoning_text}")
    #                 print(" ⚠️ [FinalVerify] LLM rejected ALL candidates. Returning empty list.\n")
    #             return [], []
                
    #         # 4. 从原始 evidence 中挑出被 LLM 盖章通过的
    #         verified_set = set(label.lower() for label in verified_labels)
    #         verified_evidence = []
            
    #         for e in evidence:
    #             # 为了防止同名实体覆盖，我们只看 label 是不是在 verified_set 里
    #             if e.get('label', '').lower() in verified_set:
    #                 e_copy = e.copy()
    #                 e_copy["final_verify_reasoning"] = reasoning_text
    #                 verified_evidence.append(e_copy)
            
    #         if self.verbose:
    #             print(f"🧠 LLM Reasoning: {reasoning_text}\n")
    #             print(f"Results: LLM verified {len(verified_labels)} candidates")
    #             for i, lbl in enumerate(verified_labels[:5], 1):
    #                 print(f"    {i}. '{lbl}'")
    #             print()
                
    #         # 5. 🌟 执行最终的去重逻辑！
    #         return _deduplicate(verified_labels, verified_evidence)

    #     except Exception as ex:
    #         if self.verbose:
    #             print(f"[Warning] FinalVerify LLM check failed: {ex}")
    #             print("Fallback: Applying simple deduplication only.\n")
    #         # 万一 LLM 抽风，安全兜底：只去重，不丢失原有的候选答案
    #         return _deduplicate(answers, evidence)

    def FinalVerify(
        self,
        question: str,
        answers: List[str],
        evidence: List[dict]
    ) -> Tuple[List[str], List[dict]]:
        """Use LLM to directly return the final verified evidence rows."""
        if self.verbose:
            print("-" * 70)
            print("KG Final Verify (LLM Evidence Direct)")
            print("-" * 70)
            print(f"Inputs: {len(answers)} candidates\n")

        if not answers or not evidence:
            return [], []

        if self.llm is None:
            if self.verbose:
                print("  [FinalVerify] No LLM, returning empty result.\n")
            return [], []

        candidate_evidence = "\n".join(
            self._format_final_verify_evidence_row(idx, ev)
            for idx, ev in enumerate(evidence, 1)
        )
        prompt = Prompt.KG_final_verification(question, candidate_evidence)

        try:
            if self.verbose:
                print(">>> Triggering LLM for Final Verification...\n")

            response, _ = self._call_llm(prompt, max_tokens=10000)
            reasoning_text = ""
            verified_labels: List[str] = []
            verified_evidence: List[dict] = []

            reason_match = re.search(r"Reasoning:\s*(.*?)(?:\nAnswer:|\nEvidence:|$)", response, re.IGNORECASE | re.DOTALL)
            if reason_match:
                reasoning_text = reason_match.group(1).strip()

            ans_match = re.search(r"Answer:\s*(.*?)(?:\nEvidence:|\nReasoning:|$)", response, re.IGNORECASE | re.DOTALL)
            if ans_match:
                answer_part = ans_match.group(1).strip()
                if answer_part.lower() not in ["none", "null", ""]:
                    raw_answers = answer_part.split(",")
                    verified_labels = [ans.strip().strip("[]'\"") for ans in raw_answers if ans.strip()]

            evidence_block = ""
            evidence_match = re.search(r"Evidence:\s*(.*)", response, re.IGNORECASE | re.DOTALL)
            if evidence_match:
                evidence_block = evidence_match.group(1).strip()

            if evidence_block:
                for line in evidence_block.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
                    if not line:
                        continue
                    try:
                        parsed = ast.literal_eval(line)
                    except Exception:
                        parsed = {"label": line}
                    if isinstance(parsed, dict):
                        if reasoning_text:
                            parsed = dict(parsed)
                            parsed["final_verify_reasoning"] = reasoning_text
                        verified_evidence.append(parsed)

            if not verified_evidence:
                if self.verbose:
                    print(f"LLM Reasoning: {reasoning_text}")
                    print("  [FinalVerify] LLM returned no usable evidence rows. Returning empty list.\n")
                return [], []

            if not verified_labels:
                verified_labels = self._dedupe_labels([
                    str(ev.get("label", "")).strip()
                    for ev in verified_evidence
                    if str(ev.get("label", "")).strip()
                ])

            if self.verbose:
                print(f"LLM Reasoning: {reasoning_text}\n")
                print(f"Results: LLM kept {len(verified_evidence)} evidence rows")
                for i, ev in enumerate(verified_evidence[:5], 1):
                    print(f"    {i}. {ev.get('label', 'Unknown')}")
                print()

            return verified_labels, verified_evidence

        except Exception as ex:
            if self.verbose:
                print(f"[Warning] FinalVerify LLM check failed: {ex}")
                print("Fallback: Returning empty result.\n")
            return [], []


    #########################################################################################
    # Parse LLM Response
    #########################################################################################

    def _parse_search_response(self, response: str, evidence: List[dict]) -> Tuple[List[str], List[dict], str]:
        """Parse MMQA_search response to extract verified answers and matching evidence."""
        answers = []
        verified_evidence = []
        reasoning_text = ""
        
        try:
            # 使用正则分别提取 Answer 和 Reasoning，忽略大小写和换行干扰
            ans_match = re.search(r"Answer:\s*(.*?)(?:\n|$|Reasoning:)", response, re.IGNORECASE | re.DOTALL)
            reason_match = re.search(r"Reasoning:\s*(.*)", response, re.IGNORECASE | re.DOTALL)

            if reason_match:
                reasoning_text = reason_match.group(1).strip()

            if ans_match:
                answer_part = ans_match.group(1).strip()
                if answer_part.lower() != "none":
                    raw_answers = answer_part.split(",")
                    answers = [self._clean_answer_text(ans) for ans in raw_answers if self._clean_answer_text(ans)]
            
            # 🚀 核心修改点：匹配 evidence，防止同名实体被重复匹配
            if answers:
                used_indices = set()  # 记录已经被匹配过的 evidence 索引
                for ans in answers:
                    for i, ev in enumerate(evidence):
                        if i in used_indices:
                            continue  # 已经被认领的实体，直接跳过
                        
                        if ev.get("label", "").lower() == ans.lower():
                            ev_copy = ev.copy() 
                            ev_copy["search_reasoning"] = reasoning_text
                            verified_evidence.append(ev_copy)
                            used_indices.add(i)  # 记录该索引已被使用
                            break 
                            
        except Exception as e:
            print(f"[Parser Warning] Failed to parse search response: {e}")
            pass
        
        return answers, verified_evidence, reasoning_text

    def _parse_relate_response(self, response: str, evidence: List[dict]) -> Tuple[List[str], List[dict]]:
        """Parse MMQA_relate response to extract verified answers and matching evidence."""
        answers = []
        verified_evidence = []

        try:
            # MMQA_relate 格式：答案在最后一行 "So, the next step can be carried out from these sources: ..."
            # 同时也尝试匹配 "Paraphrase Answer:" 段落中的实体名
            # 策略：先尝试从 evidence 标签列表中找到 response 里出现的 label
            for ev in evidence:
                label = ev.get("label", "")
                if label and label.lower() in response.lower():
                    answers.append(label)
                    ev_copy = ev.copy()
                    ev_copy["relate_reasoning"] = response
                    verified_evidence.append(ev_copy)

            # 如果上述模糊匹配一个都没命中，则降级保留全部原始 evidence
            if not answers:
                answers = [ev.get("label", "") for ev in evidence]
                verified_evidence = evidence

        except Exception as e:
            print(f"[Parser Warning] Failed to parse relate response: {e}")
            answers = [ev.get("label", "") for ev in evidence]
            verified_evidence = evidence

        return answers, verified_evidence

    def _parse_filter_response(self, response: str) -> Tuple[List[str], List[dict], str]:
        """Parse KG_filter response to extract answers and explicit evidence rows."""
        answers: List[str] = []
        filtered_evidence: List[dict] = []
        reasoning_text = ""

        def _extract_block(text: str, start_marker: str, end_markers: List[str]) -> str:
            start_idx = text.find(start_marker)
            if start_idx == -1:
                return ""
            block = text[start_idx + len(start_marker):]
            end_idx = len(block)
            for marker in end_markers:
                marker_idx = block.find(marker)
                if marker_idx != -1:
                    end_idx = min(end_idx, marker_idx)
            return block[:end_idx].strip()

        try:
            if "Reasoning:" in response:
                reasoning_text = _extract_block(response, "Reasoning:", ["Answer:"])

            answer_block = _extract_block(response, "Answer:", ["Evidence:", "Evidence List:", "Evidence Source List:", "Reasoning:"])
            if answer_block:
                raw_answers = [a.strip() for a in answer_block.split(",")]
                answers = [self._clean_answer_text(ans) for ans in raw_answers if self._clean_answer_text(ans)]
                answers = [ans for ans in answers if ans.lower() not in ("none", "")]

            evidence_block = ""
            for marker in ["Evidence:", "Evidence List:", "Evidence Source List:"]:
                evidence_block = _extract_block(response, marker, ["Reasoning:", "Answer:"])
                if evidence_block:
                    break

            if evidence_block:
                for line in evidence_block.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
                    if not line:
                        continue
                    parsed = None
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            parsed = ast.literal_eval(line)
                        except Exception:
                            parsed = None
                    elif "|" in line:
                        parts = [p.strip() for p in line.split("|") if p.strip()]
                        if parts:
                            parsed = {"label": parts[0]}
                            if len(parts) > 1:
                                parsed["evidence"] = " | ".join(parts[1:])
                    if parsed is None:
                        parsed = {"label": line}
                    if isinstance(parsed, dict):
                        filtered_evidence.append(parsed)

            if not answers and filtered_evidence:
                answers = self._dedupe_labels([str(ev.get("label", "")).strip() for ev in filtered_evidence if str(ev.get("label", "")).strip()])

            if answers and not filtered_evidence:
                filtered_evidence = [{"label": ans} for ans in answers]

        except Exception as e:
            print(f"[Parser Warning] Failed to parse filter response: {e}")
            if answers:
                filtered_evidence = [{"label": ans} for ans in answers]

        return answers, filtered_evidence, reasoning_text

    def _parse_math_response(self, response: str) -> Tuple[List[str], List[dict]]:
        results: List[str] = []
        result_evidence: List[dict] = []

        def _extract_block(text: str, start_marker: str, end_markers: List[str]) -> str:
            start_idx = text.find(start_marker)
            if start_idx == -1:
                return ""
            block = text[start_idx + len(start_marker):]
            end_idx = len(block)
            for marker in end_markers:
                marker_idx = block.find(marker)
                if marker_idx != -1:
                    end_idx = min(end_idx, marker_idx)
            return block[:end_idx].strip()

        try:
            answer_block = _extract_block(response, "Answer:", ["Evidence:", "Evidence Source List:", "Reasoning:"])
            if answer_block:
                results = [self._clean_answer_text(ans) for ans in answer_block.split(",")]
                results = [ans for ans in results if ans and ans.lower() not in ("none", "null")]

            evidence_block = ""
            for marker in ["Evidence:", "Evidence Source List:"]:
                evidence_block = _extract_block(response, marker, ["Reasoning:", "Answer:"])
                if evidence_block:
                    break

            if evidence_block:
                for line in evidence_block.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    line = re.sub(r"^[\-\*\d\.\)\s]+", "", line)
                    if not line:
                        continue
                    parsed = None
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            parsed = ast.literal_eval(line)
                        except Exception:
                            parsed = None
                    elif "|" in line:
                        parts = [p.strip() for p in line.split("|") if p.strip()]
                        if parts:
                            parsed = {"label": parts[0]}
                            if len(parts) > 1:
                                parsed["evidence"] = " | ".join(parts[1:])
                    if parsed is None:
                        parsed = {"label": line}
                    if isinstance(parsed, dict):
                        result_evidence.append(parsed)

        except Exception:
            pass

        return results, result_evidence
