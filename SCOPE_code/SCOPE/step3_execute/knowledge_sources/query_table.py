from typing import List, Optional, Tuple
import re
import json

from step3_execute.prompts.table_prompt import Prompt
from step3_execute.service.Table.table_retriever import TableRetriever


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """小写 + 压缩空格，用于宽松字符串匹配。"""
    text = str(text).strip().lower()
    # Normalize common Unicode dashes so '1995-96' and '1995–96' can match.
    text = re.sub(r'[\u2010-\u2015\u2212]+', '-', text)
    return re.sub(r'\s+', ' ', text)


def _clean_cell(val: str) -> str:
    """去除 Wikipedia markup 残留（Category:xxx、#xxx 等）。"""
    val = re.split(r'Category:', str(val))[0].strip()
    val = re.sub(r'\s*#.*$', '', val).strip()
    return val


def _find_col_idx(header: List[str], target: str) -> int:
    """在 header 里找与 target 最匹配的列索引，找不到返回 -1。
    只做字符串层面的匹配（精确 > 包含 > 去复数），语义匹配交给 LLM。
    """
    norm_target = _norm(target)
    stem_target = norm_target.rstrip('s')   # 简单去复数 players -> player
    # 1. 精确
    for i, h in enumerate(header):
        if _norm(h) == norm_target:
            return i
    # 2. 任一方向包含
    for i, h in enumerate(header):
        nh = _norm(h)
        if norm_target in nh or nh in norm_target:
            return i
    # 3. 词级（去复数后）
    for i, h in enumerate(header):
        nh = _norm(h)
        if stem_target and (stem_target in nh or nh in stem_target):
            return i
    return -1


def _build_evidence_context(evidence_item: dict) -> str:
    """Build a searchable text blob from row-level and table-level context."""
    row = evidence_item.get('matched_row', {}) or {}
    parts = [
        evidence_item.get('page_title', ''),
        evidence_item.get('section_title', ''),
        evidence_item.get('label', ''),
        evidence_item.get('row_text', ''),
        ' | '.join(str(k) for k in row.keys()),
        ' | '.join(str(v) for v in row.values()),
    ]
    return _norm(' '.join(str(p) for p in parts if p))


def _condition_keywords(condition: str) -> List[str]:
    stopwords = {
        'a', 'an', 'the', 'of', 'by', 'in', 'on', 'at', 'to', 'for',
        'is', 'was', 'are', 'were', 'be', 'been', 'that', 'which',
        'who', 'with', 'and', 'or', 'not', 'player', 'team', 'games',
        'game', 'during', 'time', 'selected', 'chosen', 'from',
    }
    return [
        w for w in re.split(r'[\s,]+', _norm(condition))
        if w and w not in stopwords and len(w) > 2
    ]


def _subject_variants(subject_entity: str) -> List[str]:
    text = str(subject_entity or "").strip()
    if not text:
        return []

    variants: List[str] = []
    for part in re.split(r',|\band\b', text, flags=re.IGNORECASE):
        cleaned = _norm(part)
        if cleaned and cleaned not in variants:
            variants.append(cleaned)
    return variants


def _row_matches_subject(row: List[str], header: List[str], subject_variants: List[str]) -> bool:
    if not subject_variants:
        return True

    row_dict = {str(h): _clean_cell(str(v)) for h, v in zip(header, row)}
    row_blob = _norm(" | ".join(str(c) for c in row))
    row_values_blob = _norm(" | ".join(str(v) for v in row_dict.values()))
    searchable = f"{row_blob} | {row_values_blob}"
    return any(subject in searchable for subject in subject_variants)


class TableSource:
    """Table knowledge source operator layer."""

    def __init__(self, retriever_api_url: str, k: int = 5, llm=None, verbose: bool = True):
        if not retriever_api_url:
            raise ValueError("TableSource requires retriever_api_url")
        self.retriever_api_url = retriever_api_url
        self.k = k
        self.llm = llm
        self.verbose = verbose
        self._retriever = TableRetriever(api_url=retriever_api_url)
        if self.verbose:
            print("\u2713 TableSource initialized\n")

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _call_llm(self, prompt: str, max_tokens: int = 10000):
        response, meta = self.llm.query_gpt4o(prompt=prompt, max_tokens=max_tokens)
        return response, meta

    def _expand_table_to_evidence_rows(self, table: dict) -> List[dict]:
        """Expand a BM25 table dict into per-row evidence dicts.

        Used as a fallback when the LLM's Relate JSON is truncated/unparseable:
        downstream Filter relies on `row` being present, but the raw BM25 table
        dict stores rows under `rows_preview` (list of lists) which Filter's
        `_render_evidence` ignores. This helper flattens the full SQL rows (or
        rows_preview as fallback) into the {page_title, section_title, header,
        row} schema.
        """
        page_title = table.get("page_title", "") or ""
        section_title = table.get("section_title", "") or ""
        table_id = table.get("table_id", "") or ""
        header = table.get("header") or []

        rows: list = []
        if table_id:
            try:
                rows = self._retriever._load_full_rows(table_id) or []
            except Exception:
                rows = []
        if not rows:
            rows = table.get("rows_preview") or []

        if not header and table_id:
            try:
                header = self._retriever._load_header_from_sql(table_id) or []
            except Exception:
                header = []
        if not header:
            matched_row = table.get("matched_row") or {}
            if isinstance(matched_row, dict) and matched_row:
                header = list(matched_row.keys())

        header_str = [str(h) for h in header]

        expanded: List[dict] = []
        for row in rows:
            if isinstance(row, dict):
                row_dict = {str(k): str(v) for k, v in row.items()}
            elif isinstance(row, (list, tuple)):
                row_dict = {str(h): str(c) for h, c in zip(header_str, row)}
            else:
                continue
            if not row_dict:
                continue
            expanded.append({
                "page_title": page_title,
                "section_title": section_title,
                "table_id": table_id,
                "header": header_str,
                "row": row_dict,
            })

        if not expanded:
            expanded.append({
                "page_title": page_title,
                "section_title": section_title,
                "table_id": table_id,
                "header": header_str,
                "row": {},
            })
        return expanded

    def _parse_answer_list(self, response: str) -> List[str]:
        """Parse comma/newline-separated values from a plain-text LLM response.

        This is a fallback for non-JSON responses ONLY. A JSON-shaped response is
        handled by _parse_json_answer_evidence; if we still land here on such a
        response the JSON was malformed, and we must NOT mine its braces/keys as
        if they were answers (this used to leak a stray "{" as the answer).
        """
        text = (response or "").strip()
        if not text:
            return []
        if text.startswith(('{', '[')) and text.endswith(('}', ']')):
            return []

        def _clean_tokens(block: str) -> List[str]:
            if not block or _norm(block) == 'none':
                return []
            tokens = []
            for part in re.split(r'[,\n]', block):
                part = part.strip().lstrip('-').lstrip('*').strip().strip('"').strip("'")
                if not part or _norm(part) == 'none':
                    continue
                # Drop pure JSON/markup punctuation tokens (e.g. '{', '[]', '```').
                if re.fullmatch(r'[\s{}\[\]()`:;.\-]+', part):
                    continue
                if part.endswith(('.', '!', '?')) and len(part) > 60:
                    continue
                if len(part) > 150:
                    continue
                # 清洗 'Name: count' 或 'Name: N' 格式（Math 输出），只保留名字部分
                name_count_m = re.match(r'^(.+?)\s*:\s*\d+$', part)
                if name_count_m:
                    part = name_count_m.group(1).strip()
                tokens.append(part)
            return tokens

        # Pass 1: explicit Answer: block
        m = re.search(r'Answer:\s*(.*?)(?:Reasoning:|$)', text, re.IGNORECASE | re.DOTALL)
        if m:
            cleaned = _clean_tokens(m.group(1).strip())
            if cleaned:
                return cleaned

        # Pass 2: first plausible short line
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(('{', '}', '[', ']', '`')):
                continue
            if any(kw in line for kw in [
                'Evidence Source', 'So the next', 'Table:', 'Page:', 'Section:',
                'Row ', 'Header:', 'http', 'Answer:', 'Reasoning:', '"Answer"', '"Evidence"'
            ]):
                continue
            parts = [p.strip().strip('"').strip("'") for p in line.split(',')]
            short = [p for p in parts if p and len(p) <= 80
                     and not p.endswith(('.', '!', '?'))
                     and _norm(p) not in ('none', '')]
            if short and len(line) <= 200:
                return list(dict.fromkeys(short))

        return []

    def _parse_json_answer_evidence(self, response: str) -> Tuple[List[str], List[dict], bool]:
        """Best-effort extraction of structured JSON with Answer and Evidence.

        Returns (answers, evidence, json_ok). ``json_ok`` is True when a JSON
        object was successfully parsed — in that case the result is authoritative
        even if Answer is empty or "None", and callers must NOT fall back to the
        plain-text parser.
        """
        text = (response or "").strip()
        if not text:
            return [], [], False

        def _normalize_answer(values) -> List[str]:
            if values is None:
                return []
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                return []
            cleaned: List[str] = []
            for val in values:
                sval = str(val).strip().strip('"').strip("'")
                if sval and _norm(sval) != 'none':
                    cleaned.append(sval)
            return cleaned

        def _normalize_evidence(values) -> List[dict]:
            if values is None:
                return []
            if not isinstance(values, list):
                return []
            cleaned: List[dict] = []
            for item in values:
                if not isinstance(item, dict):
                    continue
                page_title = str(item.get('page_title', '')).strip()
                section_title = str(item.get('section_title', '')).strip()
                header = item.get('header', [])
                row = item.get('row', {})
                if isinstance(header, list):
                    header = [str(h) for h in header]
                else:
                    header = []
                if isinstance(row, dict):
                    row = {str(k): str(v) for k, v in row.items()}
                else:
                    row = {}
                cleaned.append({
                    'page_title': page_title,
                    'section_title': section_title,
                    'header': header,
                    'row': row,
                })
            return cleaned

        candidates = [text]
        obj_m = re.search(r'\{[\s\S]*\}', text)
        if obj_m and obj_m.group(0) not in candidates:
            candidates.append(obj_m.group(0))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return (
                    _normalize_answer(parsed.get('Answer') or parsed.get('answer')),
                    _normalize_evidence(parsed.get('Evidence') or parsed.get('evidence')),
                    True,
                )
        return [], [], False

    def _extract_json_answer_array(self, response: str) -> Optional[List[str]]:
        answers, _, _ = self._parse_json_answer_evidence(response)
        return answers or None

    #########################################################################################
    # Table Search
    #########################################################################################   
    
    def Search(
        self,
        question: str,
        entity_name: str,
        descriptors: str = "",
    ) -> Tuple[List[str], List[dict]]:
        """BM25 retrieval + LLM verification (Streamlined)."""
        self._log("-" * 70)
        self._log("Table Search")
        self._log(f"  entity: '{entity_name}'")
        if descriptors:
            self._log(f"  descriptors: '{descriptors}'")
        self._log("")

        query = f"{entity_name.strip()} {descriptors.strip()}".strip()
        topk = self._retriever.retrieve_topk_tables(query, self.k)
        if not topk:
            self._log("  [Search] BM25 returned empty.")
            return [], []

        self._log(f"  BM25 retrieved {len(topk)} tables")
        for i, t in enumerate(topk[:3], 1):
            sc = t.get('score', '')
            self._log(f"    {i}. '{t.get('page_title','')}' / '{t.get('section_title','')}' score={sc}")

        if self.llm is None:
            labels = [f"{t.get('page_title', '')} {t.get('section_title', '')}".strip() for t in topk]
            return labels, topk

        # LLM verification for TABLE
        self._log("  [Search] Table-only LLM verification...")
        supporting_knowledge = {"Table": self._retriever.format_tables(topk, full=False)}
        prompt = Prompt.Table_search(question, query, supporting_knowledge)

        try:
            response, _ = self._call_llm(prompt, max_tokens=10000)
            parsed = None

            # 优先尝试纯 JSON 解析
            try:
                parsed = json.loads(response)
            except Exception:
                json_m = re.search(r'\{[\s\S]*\}', response)
                if json_m:
                    parsed = json.loads(json_m.group(0))

            if not isinstance(parsed, dict) or not parsed.get('selected'):
                # Judge 明确表示没有相关表 → 不再用 BM25 Top-1 塞次优表（会触发下游幻觉）。
                # 返回空，让 pipeline 级别的 fallback 切到 KG 源。
                self._log("  [Search] LLM judge selected nothing — returning empty so pipeline can fallback to KG.")
                return [], []

            selected = parsed.get('selected', [])
            reasoning = str(parsed.get('reasoning', '')).strip()

            # 💡 新增一条 Debug 打印，看看大模型到底输出了什么奇葩 ID
            self._log(f"  [Debug] LLM selected raw: {selected}")

            # 建立 table_id 映射
            topk_by_tid = {str(t.get('table_id', t.get('pid', id(t)))): t for t in topk}

            evidence: List[dict] = []
            seen: set = set()
            for item in selected:
                if not isinstance(item, dict):
                    continue
                
                tid = str(item.get('table_id', '')).strip()
                matched_table = None
                
                # 🛡️ 策略 1：优先尝试精确匹配 table_id
                if tid in topk_by_tid:
                    matched_table = topk_by_tid[tid]
                else:
                    # 🛡️ 策略 2（双重保险）：如果大模型瞎编了 ID，用 page_title + section_title 兜底找回
                    pt = str(item.get('page_title', '')).strip().lower()
                    st = str(item.get('section_title', '')).strip().lower()
                    for t in topk:
                        t_pt = str(t.get('page_title', '')).strip().lower()
                        t_st = str(t.get('section_title', '')).strip().lower()
                        if pt == t_pt and st == t_st:
                            matched_table = t
                            # 找回它真实的 table_id
                            tid = str(t.get('table_id', t.get('pid', id(t))))
                            break

                # 提取并保存匹配到的表格
                if matched_table and tid not in seen:
                    seen.add(tid)
                    tc = dict(matched_table)
                    tc['search_reasoning'] = reasoning
                    tc['search_decision'] = item
                    evidence.append(tc)

            self._log(f"  [Search] LLM verified {len(evidence)} tables")

            if reasoning:
                self._log(f"  Reasoning: {reasoning}")

            if not evidence:
                # 与上面同理：判定结果对不上任何已知表，就直接返回空，等 pipeline 切到 KG。
                self._log("  [Search] LLM judge rejected all candidates — returning empty so pipeline can fallback to KG.")
                return [], []

            # 生成组合 Label: Page Title + Section Title
            labels = list(dict.fromkeys(
                f"{e.get('page_title', '')} {e.get('section_title', '')}".strip() for e in evidence
            ))
            return labels, evidence

        except Exception as e:
            self._log(f"  [Search] LLM failed: {e}")
            fallback_t = topk[:1]
            labels = [f"{t.get('page_title', '')} {t.get('section_title', '')}".strip() for t in fallback_t]
            return labels, fallback_t

    #########################################################################################
    # Table Relation
    #########################################################################################
    
    def Relate(
        self,
        question: str,
        tables: List[dict],
        relation: str,
        target_table: str = "",
        subject_entity: str = "",
    ) -> Tuple[List[str], List[dict]]:
        """
        从表格中提取 entity -[relation]-> target_entity 的目标实体值。

        现在直接走 LLM 提取，不再做第一阶段的程序化抽取。
        """
        self._log("-" * 70)
        self._log("Table Relate")
        self._log(f"  relation: '{relation}'  target: '{target_table}'  tables: {len(tables)}")
        if str(subject_entity).strip():
            self._log(f"  subject entity: '{subject_entity}'")
        self._log("")

        if not tables:
            self._log("  [Relate] No input tables — retrying with BM25...")
            query = f"{question} {relation}".strip()
            tables = self._retriever.retrieve_topk_tables(query, self.k) or []
            if not tables:
                self._log("  [Relate] BM25 also empty.")
                return [], []

        if self.llm is None:
            labels = [t.get('page_title', '') for t in tables]
            return labels, tables

        self._log("  [Relate] LLM extraction...")

        all_answers: List[str] = []
        all_evidence: List[dict] = []
        saw_any_response = False

        for idx, table in enumerate(tables, 1):
            supporting_knowledge = {"Table": self._retriever.format_tables([table], full=True)}
            prompt = Prompt.Table_relate(
                question=question,
                supporting_knowledge=supporting_knowledge,
                relation=relation,
                target_entity=target_table or "",
            )
            try:
                response, _ = self._call_llm(prompt, max_tokens=10000)
                saw_any_response = True
                print(f"Response: {response}")
                answers, evidence, json_ok = self._parse_json_answer_evidence(response)
                if not answers and not json_ok:
                    answers = self._parse_answer_list(response)

                self._log(f"  [Relate] table {idx}/{len(tables)} extracted {len(answers)} values")
                for i, a in enumerate(answers[:6], 1):
                    self._log(f"    {i}. '{a}'")

                # 解析失败兜底：LLM 给了 answers 但 evidence 解析不出来
                # (常见原因：JSON 被 max_tokens 截断)。此时把整张表的全量行展开
                # 成 evidence，下游 Filter 才能看到 row 数据。
                if answers and not evidence:
                    evidence = self._expand_table_to_evidence_rows(table)
                    self._log(
                        f"  [Relate] table {idx}/{len(tables)} evidence parse failed "
                        f"— fallback to {len(evidence)} full rows"
                    )

                for ans in answers:
                    if ans not in all_answers:
                        all_answers.append(ans)
                if evidence:
                    all_evidence.extend(evidence)
            except Exception as ex:
                self._log(f"  [Relate] table {idx}/{len(tables)} failed: {ex}")
                continue

        if all_answers:
            if all_evidence:
                return all_answers, all_evidence
            return all_answers, tables

        if saw_any_response:
            self._log("  [Relate] LLM extracted nothing from all candidate tables")
            return [], []

        self._log("  [Relate] LLM failed for all candidate tables")
        return [t.get('page_title', '') for t in tables], tables

    #########################################################################################
    # Table Filter
    #########################################################################################

    def Filter(
        self,
        question: str,
        answers: List[str],
        evidence: List[dict],
        condition: str,
    ) -> Tuple[List[str], List[dict]]:
        """Filter answers using Evidence-only candidate descriptions.

        This method intentionally stops depending on `matched_row` / `row`
        and instead treats Relate-style evidence as the source of truth.
        """
        self._log("-" * 70)
        self._log("Table Filter")
        self._log(f"  condition: '{condition}'")
        self._log(f"  inputs: answers={len(answers)} evidence={len(evidence)}")
        self._log("")

        if not evidence:
            self._log("  [Filter] No evidence.")
            return [], []

        def _infer_label(e: dict) -> str:
            label = str(e.get('label', '')).strip()
            if label:
                return label
            for key in ('name', 'player', 'entity', 'value', 'answer'):
                val = e.get(key)
                if val not in (None, ''):
                    return str(val).strip()
            page = str(e.get('page_title', '')).strip()
            section = str(e.get('section_title', '')).strip()
            if page or section:
                return f"{page} {section}".strip()
            return ''

        def _render_evidence(e: dict) -> str:
            if not isinstance(e, dict):
                return "[Candidate]"
            parts = []
            lbl = _infer_label(e)
            parts.append(f"[Candidate: {lbl}]" if lbl else "[Candidate]")
            for key in ('page_title', 'section_title', 'header', 'row', 'row_text', 'matched_row', 'meta'):
                val = e.get(key)
                if val in (None, '', [], {}):
                    continue
                if isinstance(val, (dict, list)):
                    val_str = json.dumps(val, ensure_ascii=False)
                else:
                    val_str = str(val)
                parts.append(f"{key}: {val_str}")
            return "\n".join(parts)

        supporting_knowledge = {"Table": ["\n\n".join(_render_evidence(e) for e in evidence)]}
        prompt = Prompt.Table_filter(question, condition, supporting_knowledge)
        try:
            response, _ = self._call_llm(prompt, max_tokens=10000)
            print(f"Response: {response}")
            filtered_labels, filtered_evidence, json_ok = self._parse_json_answer_evidence(response)
            if not filtered_labels and not json_ok:
                filtered_labels = self._parse_answer_list(response)
                filtered_evidence = []

            if not filtered_labels:
                self._log("  [Filter] LLM explicitly returned [] — filtering ALL candidates.")
                return [], []

            if len(filtered_labels) == 1 and str(filtered_labels[0]).lower() == 'none':
                self._log("  [Filter] LLM returned None/Empty string — keeping all.")
                return answers if answers else [_infer_label(e) for e in evidence], evidence

            if filtered_evidence:
                self._log(f"  [Filter] LLM: {len(filtered_labels)} passed with evidence")
                return filtered_labels, filtered_evidence

            filtered_ev = [e for e in evidence if _infer_label(e) in set(filtered_labels)]
            if not filtered_ev:
                filtered_ev = evidence

            self._log(f"  [Filter] LLM: {len(filtered_labels)} passed")
            return filtered_labels, filtered_ev

        except Exception as ex:
            self._log(f"  [Filter] LLM failed: {ex}")
            return answers, evidence


    #########################################################################################
    # Table Math
    #########################################################################################
    
    def Math(
        self,
        question: str,
        answers: List[str],
        evidence: List[dict],
        operation: str,) -> Tuple[List[str], List[dict]]:
        """Mathematical/statistical operation on answers via LLM."""
        self._log("-" * 70)
        self._log("Table Math")
        self._log(f"  operation: '{operation}'  inputs: {len(answers)}")
        self._log("")

        if not evidence:
            self._log("  [Math] No evidence.")
            return [], []
        if self.llm is None:
            return [], []

        def _render_evidence(e: dict) -> str:
            if not isinstance(e, dict):
                return "[Candidate]"
            parts = []
            lbl = str(e.get('label', '')).strip()
            parts.append(f"[Candidate: {lbl}]" if lbl else "[Candidate]")
            for key in ('page_title', 'section_title', 'header', 'row', 'row_text', 'meta'):
                val = e.get(key)
                if val in (None, '', [], {}):
                    continue
                if isinstance(val, (dict, list)):
                    val_str = json.dumps(val, ensure_ascii=False)
                else:
                    val_str = str(val)
                parts.append(f"{key}: {val_str}")
            return "\n".join(parts)

        supporting_knowledge = {"Table": ["\n\n".join(_render_evidence(e) for e in evidence)]}
        prompt = Prompt.Table_math(question, operation, supporting_knowledge)

        def _synth_math_evidence(results_: List[str]) -> List[dict]:
            # 为每个 Math 结果各产出一行 self-describing evidence dict。
            # 多实体场景（"For A, B, how many ..."）会返回多个 result；只取 [0]
            # 会把后面实体的答案直接吞掉，所以这里改成 1 result -> 1 synth dict。
            # 同时按 page_title 分桶上游证据，当 result 数与不同 page 数相等时
            # 做 1:1 归属，让 FinalVerify 能把每个 result 对回它的源表。
            if not results_:
                return []

            by_page: "Dict[str, List[dict]]" = {}
            page_order: List[str] = []
            for ev in evidence or []:
                if not isinstance(ev, dict):
                    continue
                pt = str(ev.get("page_title", "")).strip()
                if pt not in by_page:
                    by_page[pt] = []
                    page_order.append(pt)
                by_page[pt].append(ev)

            # 多实体 1:1 归属：result 数 == 不同 page 数（>1）。
            use_page_attribution = (
                len(results_) > 1
                and len(page_order) > 1
                and len(results_) == len(page_order)
            )

            fallback_upstream = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
            fallback_page = str(fallback_upstream.get("page_title", "")).strip()
            fallback_section = str(fallback_upstream.get("section_title", "")).strip()
            total_rows = len(evidence or [])

            synth: List[dict] = []
            for i, res in enumerate(results_):
                label = str(res).strip()
                if use_page_attribution:
                    page_title = page_order[i]
                    src_rows = by_page[page_title]
                    section_title = str(
                        (src_rows[0] if src_rows else {}).get("section_title", "")
                    ).strip()
                    n_rows = len(src_rows)
                else:
                    page_title = fallback_page
                    section_title = fallback_section
                    n_rows = total_rows
                synth.append({
                    "kind": "math_result",
                    "label": label,
                    "computation": operation,
                    "source_rows": n_rows,
                    "page_title": page_title,
                    "section_title": section_title,
                    "header": ["computation", "result", "source_rows"],
                    "row": {
                        "computation": operation,
                        "result": label,
                        "source_rows": str(n_rows),
                    },
                })
            return synth

        try:
            response, _ = self._call_llm(prompt, max_tokens=10000)
            print(f"Response: {response}")
            results, math_evidence, json_ok = self._parse_json_answer_evidence(response)
            if not results and not json_ok:
                results = self._parse_answer_list(response)
            self._log(f"  [Math] result: {results}")
            return results, _synth_math_evidence(results)
        except Exception as ex:
            self._log(f"  [Math] LLM failed: {ex}")
            return [], []

    #########################################################################################
    # Table Final Verification
    #########################################################################################
    def FinalVerify(
        self,
        question: str,
        answers: List[str],
        evidence: List[dict]
    ) -> Tuple[List[str], List[dict]]:
        """Final verification that consumes Evidence directly."""
        self._log("-" * 70)
        self._log("Table Final Verify (Evidence-only)")
        self._log(f"  Inputs: {len(answers)} candidates")
        self._log("")

        def _infer_label(e: dict) -> str:
            label = str(e.get('label', '')).strip()
            if label:
                return label
            for key in ('name', 'player', 'entity', 'value', 'answer'):
                val = e.get(key)
                if val not in (None, ''):
                    return str(val).strip()
            page = str(e.get('page_title', '')).strip()
            section = str(e.get('section_title', '')).strip()
            if page or section:
                return f"{page} {section}".strip()
            return ''

        def _deduplicate(ans_list: List[str], ev_list: List[dict]) -> Tuple[List[str], List[dict]]:
            deduped_ans = list(dict.fromkeys(ans_list))
            deduped_ev = []
            seen = set()
            for e in ev_list:
                key = (
                    e.get('page_title', ''),
                    e.get('section_title', ''),
                    json.dumps(e, sort_keys=True, ensure_ascii=False),
                )
                if key not in seen:
                    seen.add(key)
                    deduped_ev.append(e)
            return deduped_ans, deduped_ev

        if not answers or not evidence:
            return [], []

        if self.llm is None:
            self._log("  [FinalVerify] No LLM, skipping verification, applying dedup only.")
            return _deduplicate(answers, evidence)

        def _is_aggregation_question() -> bool:
            q = _norm(question)
            agg_markers = (
                'consecutive', 'most', 'least', 'maximum', 'minimum',
                'highest', 'lowest', 'times', 'count', 'counts',
            )
            if any(marker in q for marker in agg_markers):
                return True
            labels = [
                _norm(_infer_label(e))
                for e in evidence
                if isinstance(e, dict) and _infer_label(e)
            ]
            return len(labels) != len(set(labels))

        candidate_evidence = []
        for e in evidence:
            if not isinstance(e, dict):
                continue
            parts = [f"[Candidate: {_infer_label(e)}]" if _infer_label(e) else "[Candidate]"]
            for key in ('page_title', 'section_title', 'header', 'row', 'row_text', 'meta'):
                val = e.get(key)
                if val in (None, '', [], {}):
                    continue
                if isinstance(val, (dict, list)):
                    val_str = json.dumps(val, ensure_ascii=False)
                else:
                    val_str = str(val)
                parts.append(f"{key}: {val_str}")
            candidate_evidence.append("\n".join(parts))

        candidate_evidence_text = "\n\n".join(candidate_evidence)
        print(f"Candidate evidence: {candidate_evidence_text}")
        prompt = Prompt.Table_final_verification(question, candidate_evidence_text)

        try:
            response, _ = self._call_llm(prompt, max_tokens=10000)
            print(response)
            verified_labels, verified_evidence, json_ok = self._parse_json_answer_evidence(response)
            if not verified_labels and not json_ok:
                raw_labels = self._parse_answer_list(response)
                verified_labels = [str(lbl).strip() for lbl in raw_labels if str(lbl).strip()]
                verified_evidence = []

            if not verified_labels:
                if _is_aggregation_question():
                    self._log("  [FinalVerify] Empty verification result on an aggregation-style question. Keeping pre-verified candidates.")
                    return _deduplicate(answers, evidence)
                self._log("  [FinalVerify] LLM rejected ALL candidates.")
                return [], []

            if verified_evidence:
                self._log(f"  [FinalVerify] LLM verified {len(verified_labels)} unique candidates")
                return _deduplicate(verified_labels, verified_evidence)

            verified_set = set(verified_labels)
            verified_evidence = [e for e in evidence if _infer_label(e) in verified_set]
            if not verified_evidence and verified_labels:
                self._log("  [FinalVerify] Synthesis mode detected. Retaining original evidence sources for the new answer.")
                for ans in verified_labels:
                    base_ev = evidence[0].copy() if evidence else {}
                    base_ev['label'] = ans
                    verified_evidence.append(base_ev)

            self._log(f"  [FinalVerify] LLM verified {len(verified_set)} unique candidates")
            return _deduplicate(verified_labels, verified_evidence)

        except Exception as ex:
            self._log(f"  [FinalVerify] LLM failed: {ex}")
            return _deduplicate(answers, evidence)



    # ------------------------------------------------------------------ #
    #  Helper: format evidence for prompt consumption                     #
    # ------------------------------------------------------------------ #

    def _format_evidence_items(self, evidence: List[dict]) -> List[dict]:
        formatted: List[dict] = []
        for e in evidence:
            if not isinstance(e, dict):
                continue
            row = e.get('matched_row', {}) or e.get('row', {}) or {}
            if not isinstance(row, dict):
                row = {}
            header = list(row.keys())
            formatted.append({
                'page_title': e.get('page_title', ''),
                'section_title': e.get('section_title', ''),
                'header': header,
                'row': {str(k): str(v) for k, v in row.items()},
            })
        return formatted


