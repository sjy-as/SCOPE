
"""
Global reasoner for executing operator trees across multiple knowledge sources.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from step3_execute.knowledge_sources.query_table import TableSource
from step3_execute.knowledge_sources.query_kg import KGSource
from step3_execute.knowledge_sources.query_doc import DocSource
from step3_execute.prompts.table_prompt import Prompt


class OperatorResult:
    """Result from executing a single operator."""
    
    def __init__(self, answers: List[str], evidence: List[dict]):
        self.answers = answers
        self.evidence = evidence


class MultiSourceReasoner:
    """Reasoner for executing operator trees across KG / Table / Doc sources.

    The knowledge base is whichever pair of sources is wired in: kg-table
    (table_source + kg_source) or kg-doc (doc_source + kg_source).
    """

    def __init__(
        self,
        table_source: Optional[TableSource] = None,
        kg_source: Optional[KGSource] = None,
        doc_source: Optional[DocSource] = None,
        llm=None,
    ):
        """Initialize reasoner with knowledge sources.

        Args:
            table_source: TableSource instance
            kg_source: KGSource instance
            doc_source: DocSource instance
            llm: LLM instance for prompt-based operations
        """
        self.table_source = table_source
        self.kg_source = kg_source
        self.doc_source = doc_source
        self.llm = llm

        # Inject LLM into sources if provided
        if llm:
            if table_source:
                table_source.llm = llm
            if kg_source:
                kg_source.llm = llm
            if doc_source:
                doc_source.llm = llm
        # Cross-source fallback should happen at the subquery level in pipeline.py,
        # not in the middle of a partially executed operator tree.
        self.allow_operator_fallback = False

    def _copy_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(plan, dict):
            return {}
        try:
            return json.loads(json.dumps(plan, ensure_ascii=False))
        except Exception:
            return dict(plan)

    @staticmethod
    def _subst_in_obj(obj: Any, replacements: List[Tuple[str, str]]) -> Any:
        """Recursively substitute placeholder tokens in the string VALUES of a
        plan structure.

        Replaces the old ``json.dumps`` -> ``str.replace`` -> ``json.loads``
        round-trip, which produced invalid JSON whenever a substituted answer
        contained a double quote, backslash or newline (e.g. an answer extracted
        from a document passage). Walking the structure touches only string
        values, so the result is always well-formed.
        """
        if isinstance(obj, str):
            for placeholder, ref_str in replacements:
                if placeholder:
                    obj = obj.replace(placeholder, ref_str)
            return obj
        if isinstance(obj, list):
            return [MultiSourceReasoner._subst_in_obj(x, replacements) for x in obj]
        if isinstance(obj, dict):
            return {k: MultiSourceReasoner._subst_in_obj(v, replacements) for k, v in obj.items()}
        return obj

    def _infer_source_from_evidence_list(self, evidence: List[dict]) -> Optional[str]:
        if not evidence:
            return None
        first = evidence[0]
        if not isinstance(first, dict):
            return None
        # Doc evidence is checked first: it carries a `doc_id` / source tag and
        # never `pid`, so it would otherwise be misread as KG evidence.
        if first.get("source") == "doc" or first.get("doc_id"):
            return "doc"
        if first.get("table_id") or first.get("matched_col") or first.get("matched_row") or first.get("header"):
            return "table"
        if first.get("qid") or first.get("wikidata_id") or first.get("pid") or first.get("meta") is not None:
            return "kg"
        return None

    def _deduplicate_result(self, answers: List[str], evidence: List[dict]) -> OperatorResult:
        dedup_answers: List[str] = []
        seen_answers = set()
        for ans in answers or []:
            text = str(ans).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen_answers:
                continue
            seen_answers.add(key)
            dedup_answers.append(text)

        dedup_evidence: List[dict] = []
        seen_evidence = set()
        for ev in evidence or []:
            if not isinstance(ev, dict):
                continue
            label = str(ev.get("label", "")).strip()
            key = (
                label.lower(),
                str(ev.get("qid", "")),
                str(ev.get("table_id", "")),
                str(ev.get("matched_col", "")),
            )
            if key in seen_evidence:
                continue
            seen_evidence.add(key)
            dedup_evidence.append(ev)

        return OperatorResult(dedup_answers, dedup_evidence)

    def _intersect_results(self, refs: List[str], context: Dict[str, "OperatorResult"]) -> OperatorResult:
        valid_refs = [ref for ref in refs if ref in context]
        if len(valid_refs) < 2:
            return OperatorResult([], [])

        label_maps = []
        for ref in valid_refs:
            result = context[ref]
            current = {}
            for ev in result.evidence or []:
                if not isinstance(ev, dict):
                    continue
                label = str(ev.get("label", "")).strip()
                if not label:
                    continue
                current.setdefault(label.lower(), {"label": label, "evidence": []})
                current[label.lower()]["evidence"].append(ev)
            label_maps.append(current)

        if not label_maps:
            return OperatorResult([], [])

        common_keys = set(label_maps[0].keys())
        for current in label_maps[1:]:
            common_keys &= set(current.keys())

        merged_answers: List[str] = []
        merged_evidence: List[dict] = []
        for key in common_keys:
            representative = label_maps[0][key]["label"]
            merged_answers.append(representative)

            for ref, current in zip(valid_refs, label_maps):
                evs = current[key]["evidence"]
                for idx, ev in enumerate(evs):
                    if not isinstance(ev, dict):
                        continue
                    ev_copy = dict(ev)
                    ev_meta = dict(ev_copy.get("meta") or {})
                    ev_meta.update({
                        "condition_type": "intersect_member",
                        "intersect_ref": ref,
                        "intersect_group": "|".join(valid_refs),
                    })
                    ev_copy["meta"] = ev_meta
                    # Reuse an existing dedupe field so evidence from r1/r2 is preserved
                    # as separate rows through final verification.
                    ev_copy["matched_col"] = f"{ref}:{idx}"
                    ev_copy["source"] = "intersect_filter"
                    merged_evidence.append(ev_copy)

        return OperatorResult(list(dict.fromkeys(merged_answers)), merged_evidence)

    def final_verify(
        self,
        question: str,
        result: Optional[OperatorResult],
        route: Optional[Dict[str, Any]] = None,
    ) -> OperatorResult:
        if result is None:
            return OperatorResult([], [])

        answers = list(result.answers or [])
        evidence = list(result.evidence or [])
        if not answers or not evidence:
            return OperatorResult(answers, evidence)

        route = route or {}
        source = (
            self._infer_source_from_evidence_list(evidence)
            or route.get("primary_source")
            or route.get("fallback_source")
            or ""
        )

        try:
            if source == "table" and self.table_source and hasattr(self.table_source, "FinalVerify"):
                verified_answers, verified_evidence = self.table_source.FinalVerify(question, answers, evidence)
                return OperatorResult(verified_answers, verified_evidence)
            if source == "doc" and self.doc_source and hasattr(self.doc_source, "FinalVerify"):
                verified_answers, verified_evidence = self.doc_source.FinalVerify(question, answers, evidence)
                return OperatorResult(verified_answers, verified_evidence)
            if source == "kg" and self.kg_source and hasattr(self.kg_source, "FinalVerify"):
                verified_answers, verified_evidence = self.kg_source.FinalVerify(question, answers, evidence)
                return OperatorResult(verified_answers, verified_evidence)
        except Exception as e:
            print(f"[Warning] Final verification failed on source={source}: {e}")

        return self._deduplicate_result(answers, evidence)

    def _resolve_bindings_for_sq(
        self,
        sq: Dict[str, Any],
        prior_results: Optional[Dict[str, OperatorResult]] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, OperatorResult]]:
        sq_text = sq.get("text", "") or ""
        plan = self._copy_plan(sq.get("plan", {}))
        bindings = sq.get("bindings", {}) or {}
        prior_results = prior_results or {}
        bound_values: Dict[str, OperatorResult] = {}

        for var_name, binding in bindings.items():
            from_sq = binding.get("from_subquery")
            if not from_sq or from_sq not in prior_results:
                continue

            prev_result = prior_results[from_sq]
            bound_values[var_name] = prev_result
            ref_str = ", ".join(prev_result.answers) if prev_result.answers else ""

            placeholder = "${" + var_name + "}"
            sq_text = sq_text.replace(placeholder, ref_str)
            numeric_placeholder = None
            if var_name.startswith("ref_") and var_name[4:].isdigit():
                numeric_placeholder = f"[{var_name[4:]}]"
                sq_text = sq_text.replace(numeric_placeholder, ref_str)

            # Substitute placeholders by walking the plan structure directly,
            # touching only string values. A previous json.dumps -> replace ->
            # json.loads round-trip crashed whenever ref_str contained a JSON
            # special char (e.g. a double quote in a doc-extracted answer).
            replacements = [(placeholder, ref_str)]
            if numeric_placeholder:
                replacements.append((numeric_placeholder, ref_str))
            plan = self._subst_in_obj(plan, replacements)

        return sq_text, plan, bound_values

    def execute_single_subquery(
        self,
        question: str,
        sq: Dict[str, Any],
        prior_results: Optional[Dict[str, OperatorResult]] = None,
        verify_final: bool = False,
    ) -> OperatorResult:
        sq_id = sq.get("id", "")
        sq_text, plan, bound_values = self._resolve_bindings_for_sq(
            sq=sq,
            prior_results=prior_results,
        )
        plan_type = plan.get("type", "single_path")
        sq_primary_source = sq.get("route", {}).get("primary_source", "")

        print(f"\n{'='*60}")
        print(f"[{sq_id}] {sq_text or sq.get('text', '')}")
        print(f"{'='*60}")

        if plan_type == "single_path":
            result = self.execute_plan(plan, sq_text or question, sq_primary_source)
        elif plan_type == "map":
            result = self._execute_map_plan(plan, sq_text or question, bound_values, sq_primary_source)
        else:
            print(f"[Warning] Unknown plan type: {plan_type}")
            result = OperatorResult([], [])

        if verify_final:
            result = self.final_verify(
                question=sq_text or question,
                result=result,
                route=sq.get("route", {}),
            )

        print(f"  [{sq_id}] answers  : {result.answers[:5]}{'...' if len(result.answers) > 5 else ''}")
        print(f"  [{sq_id}] evidence : {len(result.evidence)} items")
        return result
    
    # ------------------------------------------------------------------ #
    #  Public: execute a full multi-subquery record                        #
    # ------------------------------------------------------------------ #

    def execute_subqueries(self, record: Dict[str, Any]) -> Dict[str, OperatorResult]:
        """
        按顺序执行 record 中所有 sub_queries，处理 bindings 变量替换。
        返回 {sq_id -> OperatorResult}
        """
        question = record.get("question", "")
        sub_queries = record.get("sub_queries", [])
        # prior_results: 来自 pipeline 的前序 sq 已有结果，避免重复执行
        prior_results: Dict[str, OperatorResult] = record.get("prior_results", {})
        sq_results: Dict[str, OperatorResult] = {}

        # 合并 prior + 当前批次，供 bindings 解析使用
        def _all_results() -> Dict[str, OperatorResult]:
            merged = dict(prior_results)
            merged.update(sq_results)
            return merged

        for sq in sub_queries:
            sq_id    = sq.get("id", "")
            sq_text  = sq.get("text", "")
            plan     = sq.get("plan", {})
            bindings = sq.get("bindings", {})

            print(f"\n{'='*60}")
            print(f"[{sq_id}] {sq_text}")
            print(f"{'='*60}")

            # 解析 bindings：将前序 sq 的结果替换占位符
            bound_values: Dict[str, OperatorResult] = {}
            all_prev = _all_results()
            for var_name, binding in bindings.items():
                from_sq = binding.get("from_subquery")
                if from_sq and from_sq in all_prev:
                    prev_result = all_prev[from_sq]
                    bound_values[var_name] = prev_result
                    ref_str = ", ".join(prev_result.answers) if prev_result.answers else ""
                    placeholder = "${" + var_name + "}"
                    sq_text = sq_text.replace(placeholder, ref_str)
                    plan_str = json.dumps(plan, ensure_ascii=False)
                    plan_str = plan_str.replace("${" + var_name + "}", ref_str)
                    plan = json.loads(plan_str)

            plan_type = plan.get("type", "single_path")
            # 取 sq 级别的 primary_source，用于整个 sq 的所有算子
            sq_primary_source = sq.get("route", {}).get("primary_source", "")
            print(f"  [{sq_id}] primary_source: {sq_primary_source or 'auto'}")
            if plan_type == "single_path":
                result = self.execute_plan(plan, sq_text or question, sq_primary_source)
            elif plan_type == "map":
                result = self._execute_map_plan(plan, sq_text or question, bound_values, sq_primary_source)
            else:
                print(f"[Warning] Unknown plan type: {plan_type}")
                result = OperatorResult([], [])

            sq_results[sq_id] = result
            print(f"  [{sq_id}] answers  : {result.answers[:5]}{'...' if len(result.answers) > 5 else ''}")
            print(f"  [{sq_id}] evidence : {len(result.evidence)} items")

        # ── 最终 LLM 综合回答 ───────────────────────────────────────────────
        final_answer = self._generate_final_answer(question, sub_queries, sq_results)
        if final_answer:
            # 存入特殊 key 供调用方使用
            sq_results["__final__"] = OperatorResult([final_answer], [])
            print(final_answer)

        return sq_results

    def _execute_map_plan(
        self,
        plan: Dict[str, Any],
        question: str,
        bound_values: Dict[str, OperatorResult],
        sq_primary_source: str = "",
    ) -> OperatorResult:
        """
        map plan: 对 bound_values['ref_1'] 里的每个答案单独执行 plan body，收集所有结果。
        """
        over_var   = plan.get("over", "ref_1")
        item_var   = plan.get("var", "item")
        body       = plan.get("body", [])
        final_ref  = plan.get("final_ref", "")

        source_result = bound_values.get(over_var)
        if not source_result or not source_result.answers:
            return OperatorResult([], [])

        all_answers: List[str] = []
        all_evidence: List[dict] = []

        for item_val in source_result.answers:
            body_str = json.dumps(body, ensure_ascii=False)
            body_str = body_str.replace("${" + item_var + "}", item_val)
            resolved_body = json.loads(body_str)
            sub_plan = {"type": "single_path", "steps": resolved_body, "final_ref": final_ref}
            sub_q = question.replace("${" + item_var + "}", item_val)
            res = self.execute_plan(sub_plan, sub_q, sq_primary_source)
            for ans in res.answers:
                if ans not in all_answers:
                    all_answers.append(ans)
            all_evidence.extend(res.evidence)

        return OperatorResult(all_answers, all_evidence)

    def _generate_final_answer(
        self,
        question: str,
        sub_queries: List[dict],
        sq_results: Dict[str, OperatorResult],
    ) -> str:
        """基于所有 subquery 的中间结果，调用 LLM 生成最终自然语言回答。"""
        if not self.llm:
            return ""
        # 构建 sq_summaries 字符串
        lines = []
        for sq in sub_queries:
            sq_id = sq.get("id", "")
            sq_text = sq.get("text", "")
            result = sq_results.get(sq_id)
            answers = result.answers if result else []
            lines.append(f"[{sq_id}] Question: {sq_text}")
            lines.append(f"[{sq_id}] Answers: {', '.join(answers[:20]) if answers else '(none)'}")
            lines.append("")
        sq_summaries = "\n".join(lines).strip()

        prompt = Prompt.Table_final_answer(
            question=question,
            sq_summaries=sq_summaries,
        )
        try:
            response, _ = self.llm.query_gpt4o(prompt=prompt, max_tokens=10000)
            import re as _re
            ans_m = _re.search(r'Answer:\s*(.+?)(?:\n|Reasoning:|$)', response, _re.IGNORECASE | _re.DOTALL)
            if ans_m:
                final = ans_m.group(1).strip()
                print(f"\n{'='*70}")
                print(f"FINAL ANSWER (LLM synthesized)")
                print(f"{'='*70}")
                print(f"  {final}")
                return final
        except Exception as e:
            print(f"[Warning] Final answer generation failed: {e}")
        return ""

    def execute_plan(self, plan: Dict[str, Any], question: str, sq_primary_source: str = "") -> OperatorResult:
        """Execute a complete operator plan.
        
        Args:
            plan: Operator plan dict
            question: Sub-query text
            sq_primary_source: sq 级别的优先 source（来自 sq.route.primary_source），
                               覆盖每个 step 的 route.primary_source
        """
        steps = plan.get("steps", [])
        context = {}  # Store intermediate results

        for step in steps:
            op_type = step.get("op")
            out_ref = step.get("out")
            
            if op_type == "Search":
                result = self._execute_search(step, question, context, sq_primary_source)
            elif op_type == "Relate":
                result = self._execute_relate(step, question, context, sq_primary_source)
            elif op_type == "Filter":
                result = self._execute_filter(step, question, context, sq_primary_source)
            elif op_type == "Math":
                result = self._execute_math(step, question, context, sq_primary_source)
            else:
                print(f"[Warning] Unknown operator: {op_type}")
                continue
            
            # Store result in context
            if out_ref:
                context[out_ref] = result
        
        # Return final result
        final_ref = plan.get("final_ref")
        if final_ref and final_ref in context:
            return context[final_ref]
        # 兜底：返回最后一步的结果
        if context:
            return list(context.values())[-1]
        return OperatorResult([], [])
    
    def _execute_search(self, step: Dict[str, Any], question: str, context: Dict, sq_primary_source: str = "") -> OperatorResult:
        """Execute Search operator."""
        entity_name = step.get("entity_name", "")
        entity_name = self._resolve_context_ref(entity_name, context)
        route = step.get("route", {})
        knowledge_sources = route.get("knowledge_sources", ["kg", "table"])
        # sq 级别的 primary_source 优先
        primary_source = sq_primary_source or route.get("primary_source", "kg")
        fallback_source = route.get("fallback_source", "")
        print(f"  [Search] source={primary_source} entity='{entity_name}'")

        all_answers = []
        all_evidence = []

        # Try primary source first
        if primary_source in knowledge_sources or primary_source:
            result = self._search_source(primary_source, question, entity_name)
            if result and result.answers:
                all_answers.extend(result.answers)
                all_evidence.extend(result.evidence)
                return OperatorResult(all_answers, all_evidence)
            # KG Search may return answers=[] with evidence containing an
            # ENTITY_NOT_FOUND sentinel. Propagate that evidence so the next
            # Relate step can detect it and fire its in-source recovery
            # (ReverseSearch). Without this, the sentinel is dropped and
            # ReverseSearch never gets a chance.
            if result and result.evidence and any(
                isinstance(e, dict) and e.get("error") == "ENTITY_NOT_FOUND"
                for e in result.evidence
            ):
                print(f"  [Search] no direct match — propagating ENTITY_NOT_FOUND "
                      f"sentinel so Relate can ReverseSearch.")
                return OperatorResult([], list(result.evidence))

        if not self.allow_operator_fallback:
            return OperatorResult(all_answers, all_evidence)

        # Fallback
        fallback = fallback_source or next((s for s in knowledge_sources if s != primary_source), "")
        if fallback and fallback != primary_source:
            print(f"  [Search] fallback to source={fallback}")
            result = self._search_source(fallback, question, entity_name)
            if result and result.answers:
                all_answers.extend(result.answers)
                all_evidence.extend(result.evidence)

        return OperatorResult(all_answers, all_evidence)
    
    def _search_source(self, source: str, question: str, entity_name: str) -> Optional[OperatorResult]:
        """Search in a specific source."""
        try:
            if source == "table" and self.table_source:
                answers, evidence = self.table_source.Search(question, entity_name)
                return OperatorResult(answers, evidence)
            elif source == "doc" and self.doc_source:
                answers, evidence = self.doc_source.Search(question, entity_name)
                return OperatorResult(answers, evidence)
            elif source == "kg" and self.kg_source:
                answers, evidence = self.kg_source.Search(question, entity_name)
                return OperatorResult(answers, evidence)
        except Exception as e:
            print(f"[Warning] Search in {source} failed: {e}")
        
        return None
    
    def _resolve_context_ref(self, value: str, context: Dict) -> str:
        """Replace ${ref} placeholders in a string with context values."""
        if not value:
            return value
        import re as _re
        def _replacer(m):
            key = m.group(1)
            if key in context:
                answers = context[key].answers
                return ", ".join(answers) if answers else m.group(0)
            return m.group(0)
        return _re.sub(r'\$\{([^}]+)\}', _replacer, value)

    def _infer_source_from_context(self, ref: str, context: Dict) -> Optional[str]:
        """根据 context 里某个 ref 的 evidence 来源推断 knowledge source。"""
        if ref not in context:
            return None
        ev = context[ref].evidence
        if not ev:
            return None
        first = ev[0]
        # Doc evidence 有 doc_id / source 标签
        if first.get("source") == "doc" or first.get("doc_id"):
            return "doc"
        # Table evidence 有 table_id
        if first.get("table_id") or first.get("matched_col"):
            return "table"
        # KG evidence 有 wikidata_id 或 relation
        if first.get("wikidata_id") or first.get("pid"):
            return "kg"
        return None

    def _execute_relate(self, step: Dict[str, Any], question: str, context: Dict, sq_primary_source: str = "") -> OperatorResult:
        """Execute Relate operator."""
        entity_ref = step.get("entity")
        relation = step.get("relation", "")
        target_entity = step.get("target_entity", step.get("target", ""))
        route = step.get("route", {})
        knowledge_sources = route.get("knowledge_sources", ["kg", "table"])
        # sq 级别的 primary_source 优先
        primary_source = sq_primary_source or route.get("primary_source", "kg")
        fallback_source = route.get("fallback_source", "")

        # 解析 entity 引用：支持 "s1"、"${s1}"、字面量
        resolved_ref = None
        if entity_ref:
            if entity_ref.startswith("${") and entity_ref.endswith("}"):
                # ${s1} 格式
                resolved_ref = entity_ref[2:-1]
            elif entity_ref in context:
                # 直接是 context key，如 "s1"
                resolved_ref = entity_ref

        subject_entity = ""
        if resolved_ref and resolved_ref in context:
            entities = context[resolved_ref].evidence
            subject_entity = ", ".join(context[resolved_ref].answers or [])
            # 闸门：上游 Search 步骤已经判定没有相关表/实体（evidence 与 answers 都为空），
            # 此时直接返回空，避免 table_source.Relate 内部再用 BM25 把刚被 judge 拒掉的
            # 次优表请回来，从而保证 pipeline 能切到 fallback 源（KG）。
            if not entities and not (context[resolved_ref].answers or []):
                print(f"  [Relate] upstream '{resolved_ref}' returned empty — short-circuit to let pipeline fallback to '{fallback_source or 'KG'}'.")
                return OperatorResult([], [])
        elif not resolved_ref and entity_ref:
            # entity_ref 是字面量字符串，尝试从 context 中找最近的 Search 结果
            # 避免因计划生成填了实体名而丢失 QID
            search_result = context.get("s1")
            if search_result and search_result.evidence:
                entities = search_result.evidence
                print(f"  [Relate] entity='{entity_ref}' is a literal, using context['s1'] evidence instead")
                subject_entity = str(entity_ref)
            else:
                entities = [{"label": entity_ref}] if entity_ref else []
                subject_entity = str(entity_ref or "")
        else:
            entities = [{"label": entity_ref}] if entity_ref else []
            subject_entity = str(entity_ref or "")

        print(f"  [Relate] source={primary_source} relation='{relation}' target='{target_entity}' entities={len(entities)}")

        # 若 primary_source=table 但 entities 里没有 table dict（无 header/table_id），
        # 说明 entity 引用的是 KG evidence，此时改为用 context 里最近的 table evidence
        def _is_table_evidence(ev_list):
            return any(e.get('table_id') or e.get('header') or e.get('rows_preview') for e in ev_list)

        if primary_source == "table" and entities and not _is_table_evidence(entities):
            # 找 context 里最近的含 table_id 的 result
            table_entities = None
            for ref_key in reversed(list(context.keys())):
                cand = context[ref_key].evidence
                if _is_table_evidence(cand):
                    table_entities = cand
                    print(f"  [Relate] entities has no table_id, using context['{ref_key}'] as tables")
                    break
            if table_entities:
                entities = table_entities

        all_answers = []
        all_evidence = []

        # Try primary source first
        result = self._relate_source(primary_source, question, entities, relation, target_entity, subject_entity)
        if result and result.answers:
            return result

        if not self.allow_operator_fallback:
            return OperatorResult(all_answers, all_evidence)

        # Fallback
        fallback = fallback_source or next((s for s in knowledge_sources if s != primary_source), "")
        if fallback and fallback != primary_source:
            print(f"  [Relate] fallback to source={fallback}")
            result = self._relate_source(fallback, question, entities, relation, target_entity, subject_entity)
            if result and result.answers:
                return result

        return OperatorResult(all_answers, all_evidence)
    
    def _relate_source(self, source: str, question: str, entities: List[dict], relation: str, target_entity: str = "", subject_entity: str = "") -> Optional[OperatorResult]:
        """Relate in a specific source."""
        try:
            if source == "table" and self.table_source:
                answers, evidence = self.table_source.Relate(
                    question=question,
                    tables=entities,
                    relation=relation,
                    target_table=target_entity or None,
                    subject_entity=subject_entity,
                )
                return OperatorResult(answers, evidence)
            elif source == "doc" and self.doc_source:
                answers, evidence = self.doc_source.Relate(
                    question=question,
                    passages=entities,
                    relation=relation,
                    target_entity=target_entity or "",
                    subject_entity=subject_entity,
                )
                return OperatorResult(answers, evidence)
            elif source == "kg" and self.kg_source:
                answers, evidence = self.kg_source.Relate(
                    question=question,
                    entities=entities,
                    relation=relation,
                    target_table=target_entity or None,
                )
                return OperatorResult(answers, evidence)
        except Exception as e:
            print(f"[Warning] Relate in {source} failed: {e}")
        
        return None
    
    def _execute_filter(self, step: Dict[str, Any], question: str, context: Dict, sq_primary_source: str = "") -> OperatorResult:
        """Execute Filter operator."""
        entities_ref = step.get("entities", {})
        condition = step.get("condition", "")
        condition_struct = step.get("condition_struct", {}) or {}
        route = step.get("route", {})

        refs = [ref for ref in (entities_ref.get("refs") or condition_struct.get("refs") or []) if ref]
        if condition_struct.get("type") == "intersect" and refs:
            print(f"  [Filter] executing intersect over refs={refs}")
            return self._intersect_results(refs, context)

        # Get entities from context
        ref = entities_ref.get("ref")
        if ref and ref in context:
            prev_result = context[ref]
            answers = prev_result.answers
            evidence = prev_result.evidence
        else:
            answers = []
            evidence = []

        # sq 级别的 primary_source 优先；无则从上游 evidence 推断
        primary_source = sq_primary_source
        if not primary_source and ref:
            primary_source = self._infer_source_from_context(ref, context) or ""
        if not primary_source:
            primary_source = route.get("primary_source", "kg")
        fallback_source = route.get("fallback_source", "")
        knowledge_sources = route.get("knowledge_sources") or [primary_source]
        print(f"  [Filter] source={primary_source} condition='{condition}'")

        # Try primary source first
        result = self._filter_source(primary_source, question, answers, evidence, condition)
        if result and result.answers:
            return result

        if not self.allow_operator_fallback:
            return OperatorResult(answers, evidence)

        # Fallback
        fallback = fallback_source or next((s for s in knowledge_sources if s != primary_source), "")
        if fallback and fallback != primary_source:
            print(f"  [Filter] fallback to source={fallback}")
            result = self._filter_source(fallback, question, answers, evidence, condition)
            if result and result.answers:
                return result

        # 所有 source 都没过滤出结果时，返回原始输入
        return OperatorResult(answers, evidence)
    
    def _filter_source(self, source: str, question: str, answers: List[str], evidence: List[dict], condition: str) -> Optional[OperatorResult]:
        """Filter in a specific source."""
        try:
            if source == "table" and self.table_source:
                filtered_answers, filtered_evidence = self.table_source.Filter(question, answers, evidence, condition)
                return OperatorResult(filtered_answers, filtered_evidence)
            elif source == "doc" and self.doc_source:
                filtered_answers, filtered_evidence = self.doc_source.Filter(question, answers, evidence, condition)
                return OperatorResult(filtered_answers, filtered_evidence)
            elif source == "kg" and self.kg_source:
                filtered_answers, filtered_evidence = self.kg_source.Filter(question, answers, evidence, condition)
                return OperatorResult(filtered_answers, filtered_evidence)
        except Exception as e:
            print(f"[Warning] Filter in {source} failed: {e}")
        
        return None
    
    def _execute_math(self, step: Dict[str, Any], question: str, context: Dict, sq_primary_source: str = "") -> OperatorResult:
        """Execute Math operator."""
        data_ref = step.get("data", {})
        operation = step.get("operation", "")
        route = step.get("route", {})

        # Get data from context
        ref = data_ref.get("ref")
        if ref and ref in context:
            prev_result = context[ref]
            answers = prev_result.answers
            evidence = prev_result.evidence
        else:
            answers = []
            evidence = []

        # sq 级别的 primary_source 优先；无则从上游 evidence 推断
        primary_source = sq_primary_source
        if not primary_source and ref:
            primary_source = self._infer_source_from_context(ref, context) or ""
        if not primary_source:
            primary_source = route.get("primary_source", "table")
        fallback_source = route.get("fallback_source", "")
        knowledge_sources = route.get("knowledge_sources") or [primary_source]
        print(f"  [Math] source={primary_source} operation='{operation}'")

        # Try primary source first
        result = self._math_source(primary_source, question, answers, evidence, operation)
        if result and result.answers:
            return result

        if not self.allow_operator_fallback:
            return OperatorResult(answers, evidence)

        # Fallback
        fallback = fallback_source or next((s for s in knowledge_sources if s != primary_source), "")
        if fallback and fallback != primary_source:
            print(f"  [Math] fallback to source={fallback}")
            result = self._math_source(fallback, question, answers, evidence, operation)
            if result and result.answers:
                return result

        # Math 失败时返回原始输入
        return OperatorResult(answers, evidence)
        
        return OperatorResult([], [])
    
    def _math_source(self, source: str, question: str, answers: List[str], evidence: List[dict], operation: str) -> Optional[OperatorResult]:
        """Math operation in a specific source."""
        try:
            if source == "table" and self.table_source:
                results, result_evidence = self.table_source.Math(question, answers, evidence, operation)
                return OperatorResult(results, result_evidence)
            elif source == "doc" and self.doc_source:
                results, result_evidence = self.doc_source.Math(question, answers, evidence, operation)
                return OperatorResult(results, result_evidence)
            elif source == "kg" and self.kg_source:
                results, result_evidence = self.kg_source.Math(question, answers, evidence, operation)
                return OperatorResult(results, result_evidence)
        except Exception as e:
            print(f"[Warning] Math in {source} failed: {e}")
        
        return None
