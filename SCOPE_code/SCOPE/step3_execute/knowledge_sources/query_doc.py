from typing import List, Optional, Tuple
import re
import json

from step3_execute.prompts.doc_prompt import Prompt
from step3_execute.service.Doc.doc_retriever import DocRetriever


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    """小写 + 压缩空格，用于宽松字符串匹配。"""
    text = str(text).strip().lower()
    # Normalize common Unicode dashes so '1995-96' and '1995–96' can match.
    text = re.sub(r'[‐-―−]+', '-', text)
    return re.sub(r'\s+', ' ', text)


class DocSource:
    """Document (passage) knowledge source operator layer.

    Mirrors knowledge_sources/query_table.py: it exposes Search / Relate /
    Filter / Math / FinalVerify with the same signatures the reasoner expects,
    so a kg-doc knowledge base behaves exactly like a kg-table one.
    """

    def __init__(self, retriever_api_url: str, k: int = 5, llm=None, verbose: bool = True):
        if not retriever_api_url:
            raise ValueError("DocSource requires retriever_api_url")
        self.retriever_api_url = retriever_api_url
        self.k = k
        self.llm = llm
        self.verbose = verbose
        self._retriever = DocRetriever(api_url=retriever_api_url)
        if self.verbose:
            print("✓ DocSource initialized\n")

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _call_llm(self, prompt: str, max_tokens: int = 10000):
        response, meta = self.llm.query_gpt4o(prompt=prompt, max_tokens=max_tokens)
        return response, meta

    @staticmethod
    def _is_doc_passage(item: dict) -> bool:
        """A doc passage carries actual passage text we can feed to the LLM."""
        if not isinstance(item, dict):
            return False
        return bool(item.get("text") or item.get("passage") or item.get("snippet"))

    def _parse_answer_list(self, response: str) -> List[str]:
        """Parse comma/newline-separated values from a plain-text 'Answer:' block.

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
                if len(part) > 150:
                    continue
                tokens.append(part)
            return tokens

        m = re.search(r'Answer:\s*(.*?)(?:Reasoning:|Evidence:|$)', text, re.IGNORECASE | re.DOTALL)
        if m:
            cleaned = _clean_tokens(m.group(1).strip())
            if cleaned:
                return list(dict.fromkeys(cleaned))

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith(('{', '}', '[', ']', '`')):
                continue
            if any(kw in line for kw in ['Evidence', 'Reasoning', 'Title:', 'Doc ID:', 'http', '"Answer"']):
                continue
            cleaned = _clean_tokens(line)
            if cleaned:
                return list(dict.fromkeys(cleaned))
        return []

    @staticmethod
    def _normalize_evidence(values) -> List[dict]:
        """Normalize LLM-returned evidence rows into doc evidence dicts."""
        if not isinstance(values, list):
            return []
        cleaned: List[dict] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            title = str(item.get('title') or item.get('label') or '').strip()
            doc_id = str(item.get('doc_id') or item.get('id') or '').strip()
            snippet = str(item.get('snippet') or item.get('text') or item.get('passage') or '').strip()
            cleaned.append({
                'label': title,
                'title': title,
                'doc_id': doc_id,
                'text': snippet,
                'snippet': snippet,
                'source': 'doc',
            })
        return cleaned

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
                answers = _normalize_answer(parsed.get('Answer') or parsed.get('answer'))
                evidence = self._normalize_evidence(parsed.get('Evidence') or parsed.get('evidence') or [])
                return answers, evidence, True
        return [], [], False

    def _render_evidence(self, e: dict, as_passage: bool = False) -> str:
        """Render a doc evidence dict as a readable block for prompts.

        With ``as_passage=True`` the block is headed "[Passage]" instead of
        "[Candidate: <title>]". The passage title is NOT a candidate answer, and
        labelling it as one makes the verifier echo the title (e.g. an award
        name) instead of the real extracted answer.
        """
        if not isinstance(e, dict):
            return "[Passage]" if as_passage else "[Candidate]"
        label = str(e.get('label') or e.get('title') or '').strip()
        if as_passage:
            parts = ["[Passage]"]
        else:
            parts = [f"[Candidate: {label}]" if label else "[Candidate]"]
        doc_id = str(e.get('doc_id') or '').strip()
        if doc_id:
            parts.append(f"doc_id: {doc_id}")
        title = str(e.get('title') or '').strip()
        if title:
            parts.append(f"title: {title}")
        snippet = str(e.get('snippet') or e.get('text') or e.get('passage') or '').strip()
        if snippet:
            parts.append(f"snippet: {self._retriever._truncate(snippet, 10000)}")
        return "\n".join(parts)

    @staticmethod
    def _infer_label(e: dict) -> str:
        for key in ('label', 'title', 'name', 'entity', 'answer'):
            val = e.get(key)
            if val not in (None, ''):
                return str(val).strip()
        return ''

    # ------------------------------------------------------------------ #
    #  Doc Search                                                         #
    # ------------------------------------------------------------------ #

    def Search(
        self,
        question: str,
        entity_name: str,
        descriptors: str = "",
    ) -> Tuple[List[str], List[dict]]:
        """ColBERT retrieval + LLM verification of the most relevant passages."""
        self._log("-" * 70)
        self._log("Doc Search")
        self._log(f"  entity: '{entity_name}'")
        if descriptors:
            self._log(f"  descriptors: '{descriptors}'")
        self._log("")

        query = f"{str(entity_name).strip()} {str(descriptors).strip()}".strip()
        passages = self._retriever.retrieve_topk_passages(query, self.k)
        if not passages:
            self._log("  [Search] retrieval returned empty.")
            return [], []

        self._log(f"  retrieved {len(passages)} passages")
        for i, p in enumerate(passages[:3], 1):
            self._log(f"    {i}. '{p.get('title','')}' (doc_id={p.get('doc_id','')}) score={p.get('score','')}")

        if self.llm is None:
            labels = [p.get("label", "") for p in passages]
            return labels, passages

        self._log("  [Search] LLM verification...")
        supporting_knowledge = {"Doc": self._retriever.format_passages(passages)}
        prompt = Prompt.Doc_search(question, query, supporting_knowledge)

        try:
            response, _ = self._call_llm(prompt, max_tokens=10000)
            parsed = None
            try:
                parsed = json.loads(response)
            except Exception:
                json_m = re.search(r'\{[\s\S]*\}', response)
                if json_m:
                    parsed = json.loads(json_m.group(0))

            if not isinstance(parsed, dict) or not parsed.get('selected'):
                # Judge rejected everything → return empty so the pipeline can
                # fall back to KG, instead of forcing a noisy top-1 passage.
                self._log("  [Search] LLM judge selected nothing — returning empty for fallback.")
                return [], []

            selected = parsed.get('selected', [])
            reasoning = str(parsed.get('reasoning', '')).strip()
            self._log(f"  [Debug] LLM selected raw: {selected}")

            passages_by_id = {str(p.get('doc_id', '')): p for p in passages}

            evidence: List[dict] = []
            seen: set = set()
            for item in selected:
                if not isinstance(item, dict):
                    continue
                did = str(item.get('doc_id', '')).strip()
                matched = passages_by_id.get(did)
                if matched is None:
                    # Fall back to title matching if the LLM mangled the id.
                    title = _norm(item.get('title', ''))
                    for p in passages:
                        if title and _norm(p.get('title', '')) == title:
                            matched = p
                            did = str(p.get('doc_id', ''))
                            break
                if matched and did not in seen:
                    seen.add(did)
                    pc = dict(matched)
                    pc['search_reasoning'] = reasoning
                    evidence.append(pc)

            self._log(f"  [Search] LLM verified {len(evidence)} passages")
            if reasoning:
                self._log(f"  Reasoning: {reasoning}")

            if not evidence:
                self._log("  [Search] LLM judge matched no known passage — returning empty for fallback.")
                return [], []

            labels = list(dict.fromkeys(e.get('label', '') for e in evidence))
            return labels, evidence

        except Exception as e:
            self._log(f"  [Search] LLM failed: {e}")
            fallback = passages[:1]
            return [p.get("label", "") for p in fallback], fallback

    # ------------------------------------------------------------------ #
    #  Doc Relate                                                         #
    # ------------------------------------------------------------------ #

    def Relate(
        self,
        question: str,
        passages: List[dict],
        relation: str,
        target_entity: str = "",
        subject_entity: str = "",
    ) -> Tuple[List[str], List[dict]]:
        """Extract entity -[relation]-> target values from retrieved passages."""
        self._log("-" * 70)
        self._log("Doc Relate")
        self._log(f"  relation: '{relation}'  target: '{target_entity}'  passages: {len(passages or [])}")
        if str(subject_entity).strip():
            self._log(f"  subject entity: '{subject_entity}'")
        self._log("")

        # If the upstream evidence is not doc passages (e.g. came from a KG
        # Search), re-retrieve passages from the doc service for this relation.
        doc_passages = [p for p in (passages or []) if self._is_doc_passage(p)]
        if not doc_passages:
            query = f"{str(subject_entity).strip()} {relation} {question}".strip()
            self._log(f"  [Relate] no input passages — retrieving with '{query}'")
            doc_passages = self._retriever.retrieve_topk_passages(query, self.k)
            if not doc_passages:
                self._log("  [Relate] retrieval also empty.")
                return [], []

        if self.llm is None:
            return [p.get("label", "") for p in doc_passages], doc_passages

        self._log("  [Relate] LLM extraction...")
        supporting_knowledge = {"Doc": self._retriever.format_passages(doc_passages)}
        prompt = Prompt.Doc_relate(
            question=question,
            supporting_knowledge=supporting_knowledge,
            relation=relation,
            target_entity=target_entity or "",
        )
        try:
            response, _ = self._call_llm(prompt, max_tokens=10000)
            print(f"Response: {response}")
            answers, evidence, json_ok = self._parse_json_answer_evidence(response)
            if not answers and not json_ok:
                answers = self._parse_answer_list(response)

            self._log(f"  [Relate] extracted {len(answers)} values")
            for i, a in enumerate(answers[:6], 1):
                self._log(f"    {i}. '{a}'")

            if not answers:
                self._log("  [Relate] LLM extracted nothing.")
                return [], []
            return answers, (evidence or doc_passages)

        except Exception as ex:
            self._log(f"  [Relate] LLM failed: {ex}")
            return [p.get("label", "") for p in doc_passages], doc_passages

    # ------------------------------------------------------------------ #
    #  Doc Filter                                                         #
    # ------------------------------------------------------------------ #

    def Filter(
        self,
        question: str,
        answers: List[str],
        evidence: List[dict],
        condition: str,
    ) -> Tuple[List[str], List[dict]]:
        """Filter candidate answers using passage-level evidence."""
        self._log("-" * 70)
        self._log("Doc Filter")
        self._log(f"  condition: '{condition}'")
        self._log(f"  inputs: answers={len(answers or [])} evidence={len(evidence or [])}")
        self._log("")

        if not evidence:
            self._log("  [Filter] No evidence.")
            return [], []
        if self.llm is None:
            return answers, evidence

        # Render evidence as [Passage] blocks, NOT [Candidate: <title>] — the
        # candidate answers are the extracted `answers`, and are passed to the
        # prompt separately. Labelling a passage title as a candidate makes the
        # filter echo the title (player name) instead of the real answer.
        supporting_knowledge = {"Doc": ["\n\n".join(self._render_evidence(e, as_passage=True) for e in evidence)]}
        prompt = Prompt.Doc_filter(question, condition, answers, supporting_knowledge)
        try:
            response, _ = self._call_llm(prompt, max_tokens=10000)
            print(f"Response: {response}")
            filtered_labels, filtered_evidence, json_ok = self._parse_json_answer_evidence(response)
            if not filtered_labels and not json_ok:
                filtered_labels = self._parse_answer_list(response)
                filtered_evidence = []

            if not filtered_labels:
                self._log("  [Filter] LLM returned [] — filtering ALL candidates.")
                return [], []

            if len(filtered_labels) == 1 and _norm(filtered_labels[0]) == 'none':
                self._log("  [Filter] LLM returned None — keeping all.")
                return (answers or [self._infer_label(e) for e in evidence]), evidence

            if filtered_evidence:
                self._log(f"  [Filter] LLM: {len(filtered_labels)} passed with evidence")
                return filtered_labels, filtered_evidence

            kept = set(_norm(x) for x in filtered_labels)
            filtered_ev = [e for e in evidence if _norm(self._infer_label(e)) in kept]
            if not filtered_ev:
                filtered_ev = evidence
            self._log(f"  [Filter] LLM: {len(filtered_labels)} passed")
            return filtered_labels, filtered_ev

        except Exception as ex:
            self._log(f"  [Filter] LLM failed: {ex}")
            return answers, evidence

    # ------------------------------------------------------------------ #
    #  Doc Math                                                           #
    # ------------------------------------------------------------------ #

    def Math(
        self,
        question: str,
        answers: List[str],
        evidence: List[dict],
        operation: str,
    ) -> Tuple[List[str], List[dict]]:
        """Mathematical/statistical operation over passage evidence via LLM."""
        self._log("-" * 70)
        self._log("Doc Math")
        self._log(f"  operation: '{operation}'  inputs: {len(answers or [])}")
        self._log("")

        if not evidence:
            self._log("  [Math] No evidence.")
            return [], []
        if self.llm is None:
            return [], []

        # Render evidence as [Passage] blocks; the data to operate on is the
        # extracted `answers`, passed to the prompt separately.
        supporting_knowledge = {"Doc": ["\n\n".join(self._render_evidence(e, as_passage=True) for e in evidence)]}
        prompt = Prompt.Doc_math(question, operation, answers, supporting_knowledge)

        def _synth_math_evidence(results_: List[str]) -> List[dict]:
            # Build a single self-describing evidence dict so downstream verifiers
            # do not mistake the input passages for Math's output. We keep doc
            # source markers (source/doc_id) so _infer_source_from_evidence_list
            # still recognises this as "doc".
            label = str(results_[0]).strip() if results_ else ""
            upstream = evidence[0] if evidence and isinstance(evidence[0], dict) else {}
            return [{
                "kind": "math_result",
                "label": label,
                "computation": operation,
                "source_passages": len(evidence),
                "source": "doc",
                "doc_id": str(upstream.get("doc_id", "")).strip(),
                "row_text": f"{operation}({len(evidence)} passages) = {label}",
            }]

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

    # ------------------------------------------------------------------ #
    #  Doc Final Verification                                             #
    # ------------------------------------------------------------------ #

    def FinalVerify(
        self,
        question: str,
        answers: List[str],
        evidence: List[dict],
    ) -> Tuple[List[str], List[dict]]:
        """Final verification / synthesis over passage-grounded candidates."""
        self._log("-" * 70)
        self._log("Doc Final Verify")
        self._log(f"  Inputs: {len(answers or [])} candidates")
        self._log("")

        def _deduplicate(ans_list: List[str], ev_list: List[dict]) -> Tuple[List[str], List[dict]]:
            deduped_ans = list(dict.fromkeys(ans_list))
            deduped_ev = []
            seen = set()
            for e in ev_list:
                key = json.dumps(e, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    deduped_ev.append(e)
            return deduped_ans, deduped_ev

        if not answers or not evidence:
            return [], []
        if self.llm is None:
            self._log("  [FinalVerify] No LLM, applying dedup only.")
            return _deduplicate(answers, evidence)

        passages_text = "\n\n".join(self._render_evidence(e, as_passage=True) for e in evidence)
        print(f"Candidate answers: {answers}")
        print(f"Supporting passages: {passages_text}")
        prompt = Prompt.Doc_final_verification(question, answers, passages_text)
        try:
            response, _ = self._call_llm(prompt, max_tokens=10000)
            print(response)
            verified_labels, verified_evidence, json_ok = self._parse_json_answer_evidence(response)
            if not verified_labels and not json_ok:
                verified_labels = self._parse_answer_list(response)
                verified_evidence = []

            if not verified_labels or (len(verified_labels) == 1 and _norm(verified_labels[0]) == 'none'):
                self._log("  [FinalVerify] LLM rejected ALL candidates.")
                return [], []

            if verified_evidence:
                self._log(f"  [FinalVerify] LLM verified {len(verified_labels)} candidates")
                return _deduplicate(verified_labels, verified_evidence)

            kept = set(_norm(x) for x in verified_labels)
            verified_evidence = [e for e in evidence if _norm(self._infer_label(e)) in kept]
            if not verified_evidence:
                # Synthesis mode: keep the original passages as support.
                for ans in verified_labels:
                    base = dict(evidence[0]) if evidence else {"source": "doc"}
                    base['label'] = ans
                    verified_evidence.append(base)
            self._log(f"  [FinalVerify] LLM verified {len(verified_labels)} candidates")
            return _deduplicate(verified_labels, verified_evidence)

        except Exception as ex:
            self._log(f"  [FinalVerify] LLM failed: {ex}")
            return _deduplicate(answers, evidence)
