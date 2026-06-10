"""HydraRAG baseline 主流程。

保留原版 HydraRAG 的算法灵魂，落到「本地 KG + Table」两个知识源上：

  Stage A  问题理解   —— LLM 分解出 Thinking CoT + 子问题，抽取 topic 实体
  Stage B  实体定位   —— topic 实体链接到本地 NBA KG（拿 qid）
  Stage C  多源检索   —— KG 头：子图 BFS 找路径；Table 头：BM25 检索 + 把表行转成 KG 边
  Stage D  证据融合   —— 多源 beam search 三步排序（KG 边与表转边一起排）
  Stage E  答案+迭代  —— LLM 判 {Yes}/{No}，不足则重写 query / 预测桥接实体再检索，最多 3 轮

「把检索到的 Table 证据改成 KG 边」由 table/table_to_kg.py 完成，是本 baseline 的核心。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import config
import prompts
from fusion.beam_search import multi_source_beam_search


# --------------------------------------------------------------------------
# 解析 helper
# --------------------------------------------------------------------------
def _braced(text: str) -> List[str]:
    """取出所有 {...} 内容，逗号再切分，清洗。"""
    out: List[str] = []
    for blk in re.findall(r"\{([^{}]*)\}", text or ""):
        for piece in blk.split(","):
            piece = piece.strip().strip("'\"").strip()
            if piece and piece.lower() not in ("xxx", "none", "answer"):
                out.append(piece)
    return out


def _line_after(text: str, key: str) -> str:
    """取 'key: 内容' 这一行的内容。"""
    m = re.search(rf"{re.escape(key)}\s*:\s*(.+)", text or "", re.IGNORECASE)
    return m.group(1).strip() if m else ""


class HydraPipeline:
    def __init__(self, kg_explorer, external_heads, llm, verbose: bool = True,
                 kb: str = "kg"):
        """external_heads: 外部补充头列表。知识库 kg,table 传 [TableToKG]，
        kg,doc 传 [DocToKG]，三源全开传 [TableToKG, DocToKG]，只有 KG 时传 []。

        每个 head 都暴露同一套接口（retrieve / to_kg_edges / unit_id），所以
        pipeline 的多源检索循环逐个 head 检索、再把转出的 KG 边合并，与外部
        补充源具体是表还是文档、有几个都无关。
        """
        self.kg_explorer = kg_explorer
        self.external_heads = list(external_heads or [])
        # 向后兼容旧属性名（取第一个外部头，没有则 None）
        self.external_head = self.external_heads[0] if self.external_heads else None
        self.table_to_kg = self.external_head
        self.llm = llm
        self.verbose = verbose
        self.kb = kb

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _call(self, prompt: str, max_tokens: int, stage: str) -> str:
        try:
            resp, _ = self.llm.query_gpt4o(prompt, max_tokens=max_tokens, stage=stage)
            return resp or ""
        except Exception as e:  # noqa: BLE001
            self._log(f"  [LLM] {stage} failed: {e}")
            return ""

    # ====================================================================
    #  Stage A —— 问题理解
    # ====================================================================
    def _split_question(self, question: str) -> Tuple[str, List[str]]:
        resp = self._call(prompts.SPLIT_QUESTION + f"Question: {question}\nA:\n",
                           max_tokens=300, stage="split_question")
        cot = _line_after(resp, "Thinking CoT")
        sqs: List[str] = []
        for i in (1, 2, 3):
            sq = _line_after(resp, f"split_question{i}")
            if sq:
                sqs.append(sq)
        if not sqs:
            sqs = [question]
        return cot, sqs

    def _extract_topic_entities(self, question: str) -> List[str]:
        resp = self._call(prompts.EXTRACT_TOPIC_ENTITY + f"Question: {question}\nA:\n",
                           max_tokens=200, stage="extract_topic")
        kw_line = ""
        m = re.search(r"keywords\s*:\s*(.+)", resp, re.IGNORECASE)
        if m:
            kw_line = m.group(1)
        topics = _braced(kw_line) if kw_line else _braced(resp)
        return topics[:3]

    # ====================================================================
    #  Stage C —— 多源检索
    # ====================================================================
    def _kg_retrieve(self, topic_labels: List[str],
                     extra_seed_qids: Optional[List[str]] = None,
                     _counts: Optional[dict] = None) -> dict:
        if _counts is not None:
            _counts["kg"] = _counts.get("kg", 0) + 1
        try:
            return self.kg_explorer.explore(topic_labels, extra_seed_qids=extra_seed_qids)
        except Exception as e:  # noqa: BLE001
            self._log(f"  [KG] explore failed: {e}")
            return {"linked_entities": [], "paths": [], "reached": [], "subgraph_size": 0}

    @staticmethod
    def _bridge_candidates(kg_result: dict, topic_labels: List[str]) -> List[str]:
        """从 KG 路径里取出桥接实体（非 topic 本身），供 Table 检索复用。

        kg_result['paths'] 已按 (n_seeds 降序, 跳数升序) 排好，所以连接多个 seed 的
        路径上的实体自然排前；同时也收 1-seed 路径上的实体（如 hop1 解出的桥接实体，
        典型如 award <-receivesAward- player 这种 n_seeds=1 的路径上的 player）——
        因为你的题里两个 topic 常常一个在 KG、一个只在 Table，根本凑不出 2-seed 路径。
        """
        topics_low = {t.lower() for t in topic_labels}
        cands: List[str] = []
        seen = set()
        for p in kg_result.get("paths", []):
            for ent in p.get("entities", []):
                el = str(ent).strip()
                ell = el.lower()
                if not el or ell in topics_low or ell in seen:
                    continue
                seen.add(ell)
                cands.append(el)
        return cands[:5]

    @staticmethod
    def _external_edge_entities(ext_paths: List[dict]) -> List[str]:
        """从外部源（表/文档）转出的 KG 边里取出实体，作为回探 KG 的 seed（外部→KG 第二跳）。"""
        ents: List[str] = []
        seen = set()
        for p in ext_paths:
            if p.get("source") not in ("table", "doc"):   # *_raw 没结构化实体
                continue
            for e in p.get("entities", []):
                el = str(e).strip()
                ell = el.lower()
                if not el or ell in seen or el.isdigit():
                    continue
                seen.add(ell)
                ents.append(el)
        return ents[:6]

    def _external_retrieve(self, question: str, queries: List[str],
                            _counts: Optional[dict] = None) -> List[dict]:
        """对若干 query 做外部源检索，再把各外部头转出的 KG 边合并返回。

        逐个外部头（Table BM25 / Doc ColBERT）独立检索：每个头内部用
        round-robin 汇总取 top-N 个单元后转成 KG 边。round-robin 保证桥接
        实体查询也能贡献证据——否则问题原文的检索结果会占光名额。

        没有外部头（知识库只有 KG）时返回空，pipeline 自然退化成纯 KG。
        """
        all_edges: List[dict] = []
        for head in self.external_heads:
            # Identify source type from the head class name for counting.
            src = "doc" if "Doc" in type(head).__name__ else "table"
            per_query: List[List[dict]] = []
            for q in queries:
                q = (q or "").strip()
                if q:
                    if _counts is not None:
                        _counts[src] = _counts.get(src, 0) + 1
                    per_query.append(head.retrieve(q) or [])

            units: List[dict] = []
            seen_ids = set()
            rank = 0
            while len(units) < config.TABLE_CONVERT_TOPN and any(rank < len(pq) for pq in per_query):
                for pq in per_query:
                    if rank >= len(pq):
                        continue
                    u = pq[rank]
                    uid = head.unit_id(u)
                    if uid in seen_ids:
                        continue
                    seen_ids.add(uid)
                    units.append(u)
                    if len(units) >= config.TABLE_CONVERT_TOPN:
                        break
                rank += 1

            if units:
                all_edges += head.to_kg_edges(question, units)
        return all_edges

    @staticmethod
    def _merge_paths(pool: List[dict], new: List[dict]) -> List[dict]:
        """按 path_text 去重合并。"""
        seen = {p.get("path_text", "") for p in pool}
        for p in new:
            txt = p.get("path_text", "")
            if txt and txt not in seen:
                seen.add(txt)
                pool.append(p)
        return pool

    # ====================================================================
    #  Stage E —— 答案验证 / 迭代检索
    # ====================================================================
    def _verify(self, question: str, cot: str, sqs: List[str],
                selected: List[dict]) -> Tuple[str, List[str], str]:
        prompt = (
            prompts.ANSWER_VERIFY
            + prompts.question_block(question, cot, sqs)
            + "\n\nEdges:\n" + prompts.edges_block(selected, with_index=False)
            + "\nA:\n"
        )
        resp = self._call(prompt, max_tokens=400, stage="answer_verify")
        verdict = "No"
        if re.search(r"\{?\s*yes\s*\}?", _line_after(resp, "response"), re.IGNORECASE):
            verdict = "Yes"
        answer = _braced(_line_after(resp, "answer"))
        reason = _line_after(resp, "reason")
        return verdict, answer, reason

    def _answer_direct(self, question: str, cot: str, sqs: List[str],
                       selected: List[dict]) -> Tuple[List[str], str]:
        prompt = (
            prompts.ANSWER_DIRECT
            + prompts.question_block(question, cot, sqs)
            + "\n\nEdges:\n" + prompts.edges_block(selected, with_index=False)
            + "\nA:\n"
        )
        resp = self._call(prompt, max_tokens=400, stage="answer_direct")
        return _braced(_line_after(resp, "answer")), _line_after(resp, "reason")

    def _regen_query(self, question: str, cot: str, sqs: List[str],
                     selected: List[dict]) -> str:
        prompt = (
            prompts.REGEN_QUERY
            + prompts.question_block(question, cot, sqs)
            + "\n\nEdges so far:\n" + prompts.edges_block(selected, with_index=False)
            + "\nA:\n"
        )
        resp = self._call(prompt, max_tokens=200, stage="regen_query")
        q = _line_after(resp, "query")
        return _braced(q)[0] if _braced(q) else (q or question)

    def _predict_entities(self, question: str, cot: str, sqs: List[str],
                          selected: List[dict]) -> List[str]:
        prompt = (
            prompts.PREDICT_ENTITY
            + prompts.question_block(question, cot, sqs)
            + "\n\nEdges so far:\n" + prompts.edges_block(selected, with_index=False)
            + "\nA:\n"
        )
        resp = self._call(prompt, max_tokens=200, stage="predict_entity")
        return _braced(_line_after(resp, "Predicted"))[:3]

    def _final_synthesis(self, question: str, sqs: List[str],
                         selected: List[dict], answer: List[str]) -> str:
        prompt = (
            prompts.FINAL_SYNTHESIS
            + prompts.question_block(question, "", sqs)
            + "\n\nCandidate answer entities/values: " + ", ".join(answer or [])
            + "\nSupporting edges:\n" + prompts.edges_block(selected, with_index=False)
            + "\nA:\n"
        )
        resp = self._call(prompt, max_tokens=200, stage="final_synthesis")
        final = _line_after(resp, "Answer")
        if not final:
            final = ", ".join(answer) if answer else ""
        return final.strip()

    # ====================================================================
    #  主入口
    # ====================================================================
    def run(self, question: str, index: Optional[int] = None) -> dict:
        if self.llm is not None:
            self.llm.start_trace()

        # ---- Stage A ----
        cot, sqs = self._split_question(question)
        topics = self._extract_topic_entities(question)
        self._log(f"  [A] CoT={cot[:80]}")
        self._log(f"  [A] split={sqs}")
        self._log(f"  [A] topics={topics}")

        evidence_pool: List[dict] = []
        selected: List[dict] = []
        answer: List[str] = []
        verdict = "No"
        iterations: List[dict] = []
        predicted: List[str] = []
        _counts: dict = {}

        for it in range(1, config.MAX_ITERATIONS + 1):
            round_paths: List[dict] = []
            new_q: Optional[str] = None
            ext_queries: List[str] = []

            # ---- KG 头：基础知识 ----
            if it == 1:
                kg_res = self._kg_retrieve(topics, _counts=_counts)
                round_paths += kg_res.get("paths", [])
                bridges = self._bridge_candidates(kg_res, topics)
                ext_queries = [question] + sqs + bridges
                self._log(f"  [C] KG subgraph={kg_res.get('subgraph_size')} "
                          f"kg_paths={len(kg_res.get('paths', []))} bridges={bridges}")
            elif it == 2:
                # 迭代 2：重写检索 query
                new_q = self._regen_query(question, cot, sqs, selected)
                self._log(f"  [iter2] regen query: {new_q}")
                round_paths += self._kg_retrieve(topics + [new_q], _counts=_counts).get("paths", [])
                ext_queries = [new_q] + sqs
            else:
                # 迭代 3：预测桥接实体，重新锚定检索
                predicted = self._predict_entities(question, cot, sqs, selected)
                self._log(f"  [iter3] predicted bridge entities: {predicted}")
                if predicted:
                    round_paths += self._kg_retrieve(topics + predicted, _counts=_counts).get("paths", [])
                ext_queries = (predicted or []) + sqs

            # ---- 外部补充头：kg-table 用 Table 头，kg-doc 用 Doc 头（KG 先、外部补）----
            ext_paths = self._external_retrieve(question, ext_queries, _counts=_counts)
            round_paths += ext_paths

            # ---- 外部→KG 回探：拿外部源查到的实体回 KG 继续探索下一跳 ----
            # （第二跳在 KG 的题靠它接力；其它题也能借此补全 KG 侧证据）
            handoff_ents = self._external_edge_entities(ext_paths)
            if handoff_ents:
                handoff_paths = self._kg_retrieve(handoff_ents, _counts=_counts).get("paths", [])
                for p in handoff_paths:
                    p["handoff"] = True          # 标记为第二跳边，融合召回门会优先保留
                round_paths += handoff_paths
                self._log(f"  [iter{it}] 外部→KG 回探 seeds={handoff_ents} "
                          f"-> {len(handoff_paths)} KG paths")

            evidence_pool = self._merge_paths(evidence_pool, round_paths)
            selected = multi_source_beam_search(
                self.llm, question, cot, sqs, list(evidence_pool)
            )
            verdict, answer, reason = self._verify(question, cot, sqs, selected)
            self._log(f"  [iter{it}] pool={len(evidence_pool)} selected={len(selected)} "
                      f"verdict={verdict} answer={answer}")

            iterations.append({
                "iteration": it,
                "new_paths": len(round_paths),
                "pool_size": len(evidence_pool),
                "handoff_entities": handoff_ents,
                "selected": [p.get("path_text", "") for p in selected],
                "verdict": verdict,
                "answer": answer,
                "reason": reason,
                "regen_query": new_q,
                "predicted_entities": predicted if it == 3 else None,
            })

            if verdict == "Yes" and answer:
                break

        # ---- 最终答案 ----
        if not (verdict == "Yes" and answer):
            answer, direct_reason = self._answer_direct(question, cot, sqs, selected)
            self._log(f"  [direct] forced answer={answer}")

        final = self._final_synthesis(question, sqs, selected, answer)
        self._log(f"  [final] {final}")

        # 线程局部 trace —— 准确的本题 LLM 调用数（不受其它并发线程干扰）
        llm_calls = self.llm.pop_trace() if self.llm is not None else []

        return {
            "index": index,
            "question": question,
            "kb": self.kb,
            "thinking_cot": cot,
            "split_questions": sqs,
            "topic_entities": topics,
            "answer_entities": answer,
            "final": final,
            "iterations": iterations,
            "n_iterations": len(iterations),
            "selected_evidence": [
                {"source": p.get("source"), "doc_id": p.get("doc_id"),
                 "path_text": p.get("path_text"), "confidence": p.get("confidence")}
                for p in selected
            ],
            "n_llm_calls": len(llm_calls),
            "llm_calls": llm_calls,
            "retrieval_counts": _counts,
            "error": None,
        }
