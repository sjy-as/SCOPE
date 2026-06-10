import time
from typing import Optional

from query_knowledge_source.query_text import retrieve_topk
from query_knowledge_source.query_table import execute_table_query
from query_knowledge_source.query_kg import semantic_parsing_api, engine_exec_api
from trace_recorder import get_recorder


def overlap_coefficient(set1: set, set2: set):
    intersection_size = len(set1.intersection(set2))
    smaller_set_size = min(len(set1), len(set2))
    if smaller_set_size == 0:
        return 0
    return intersection_size / smaller_set_size


class Text:
    def __init__(self, retriever_api_url: Optional[str] = None, k: int = 3):
        if retriever_api_url is None:
            raise Exception("Text() instance initialization failed: no retriever_api_url provided.")
        self.retriever_api_url = retriever_api_url
        self.k = k
        print(f"Text instance initialized. retriever_api_url = '{self.retriever_api_url}', k = {self.k}")

    def retrieve_topk_passages(self, query: str, k: int, prob_threshold: float = 1.0):
        retrieved = retrieve_topk(query, k, self.retriever_api_url)
        retry = 0
        while retrieved is None:
            retry += 1
            print(f"Retry {retry}.")
            if retry > 5:
                print("More than 5 retries, exiting.")
                return None, None
            retrieved = retrieve_topk(query, k, self.retriever_api_url)
            time.sleep(5)

        clean_entity_list = []
        entity_and_passage_list = []
        for entry in retrieved:
            text = str(entry.get("text", ""))
            split_index = text.find("|")
            if split_index == -1:
                entity = text.strip()
                passage = ""
            else:
                entity = text[:split_index].strip()
                passage = text[split_index + 1 :].strip()

            prob = entry.get("prob", None)
            rank = entry.get("rank", None)

            clean_entity_list.append(entity)
            entity_and_passage_list.append({"entity": entity, "passage": passage, "rank": rank, "prob": prob})

            if prob is not None and prob > prob_threshold:
                break

        return clean_entity_list, entity_and_passage_list

    def Search(self, question: str, entity_name: str, descriptors: Optional[str] = ""):
        get_recorder().record_retrieval("doc")
        query = f"{entity_name.strip()} {descriptors.strip()}".strip()
        return self.retrieve_topk_passages(query, self.k, prob_threshold=0.75)

    def Relate(self, question: str, entity: str, relation: str):
        get_recorder().record_retrieval("doc")
        query = f"{entity.strip()} {relation.strip()}".strip()
        return self.retrieve_topk_passages(query, self.k, prob_threshold=0.85)

    def Filter(self, question: str, entities: list, condition: str):
        filtered_entity_score_dict = {}
        filtered_entity_passage_dict = {}
        for entity in entities:
            get_recorder().record_retrieval("doc")
            query = f"{entity} {condition}".strip()
            _, top1_full = self.retrieve_topk_passages(query, 1)
            if not top1_full:
                continue
            top1 = top1_full[0]
            passage_with_title = f"{top1['entity']} {top1['passage']}"
            passage_with_title_wordset = set(passage_with_title.lower().split())

            entity_and_condition_wordset = set(query.lower().split())
            overlap_coeff = overlap_coefficient(entity_and_condition_wordset, passage_with_title_wordset)
            if overlap_coeff >= 0.5:
                ent = top1["entity"]
                filtered_entity_score_dict[ent] = overlap_coeff
                filtered_entity_passage_dict[ent] = top1["passage"]

        if len(filtered_entity_score_dict) == 0:
            return None, None

        final_entity_and_score_list = sorted(filtered_entity_score_dict.items(), key=lambda item: item[1], reverse=True)
        if final_entity_and_score_list[0][1] >= 0.85:
            discard_index = 0
            for entity in final_entity_and_score_list:
                if entity[1] < 0.85:
                    final_entity_and_score_list = final_entity_and_score_list[:discard_index]
                    break
                discard_index += 1

        final_clean_entity_list = []
        final_entity_and_passage_list = []
        rank = 1
        for entry in final_entity_and_score_list:
            ent = entry[0]
            final_clean_entity_list.append(ent)
            final_entity_and_passage_list.append(
                {"entity": ent, "passage": filtered_entity_passage_dict[ent], "overlap_coeff": filtered_entity_score_dict[ent], "rank": rank}
            )
            rank += 1

        return final_clean_entity_list, final_entity_and_passage_list


class Table:
    def __init__(self, table_api_url: Optional[str] = None, k: int = 3):
        if table_api_url is None:
            raise Exception("Table() instance initialization failed: no table_api_url provided.")
        self.table_api_url = table_api_url
        self.k = k

    def retrieve_topk_tables(self, query: str, k: int):
        try:
            clean_title_list, full_results = execute_table_query(query, self.table_api_url, k)
        except Exception as e:
            retry = 1
            full_results = None
            clean_title_list = None
            while full_results is None:
                if retry > 3:
                    return None, None
                time.sleep(5)
                try:
                    print(f"Retry {retry}")
                    clean_title_list, full_results = execute_table_query(query, self.table_api_url, k)
                except Exception as ex:
                    print("Exception:", ex)
                    retry += 1
        return clean_title_list, full_results

    def Search(self, question: str, entity_name: str, descriptors: Optional[str] = ""):
        get_recorder().record_retrieval("table")
        query = f"{entity_name.strip()} {descriptors.strip()}".strip()
        clean_title_list, full_results = self.retrieve_topk_tables(query, self.k)
        if not full_results or "organic_results" not in full_results:
            return None, None
        return clean_title_list, full_results

    def Relate(self, question: str, entity: str, relation: str):
        get_recorder().record_retrieval("table")
        query = f"{entity.strip()} {relation.strip()}".strip()
        clean_title_list, full_results = self.retrieve_topk_tables(query, self.k)
        if not full_results or "organic_results" not in full_results:
            return None, None
        return clean_title_list, full_results

    def Filter(self, question: str, entities: list, condition: str):
        filtered_entity_score_dict = {}
        filtered_entity_results_dict = {}
        for entity in entities:
            get_recorder().record_retrieval("table")
            query = f"{entity} {condition}".strip()
            _, top1_full_results = self.retrieve_topk_tables(query, 1)
            if not top1_full_results or "organic_results" not in top1_full_results:
                continue

            organic = top1_full_results.get("organic_results", [])
            if len(organic) == 0:
                continue

            filtered_entity_results_dict[entity] = top1_full_results
            entity_condition_wordset = set(query.lower().split())

            organic_result_top1 = organic[0]
            snippets_and_passages_string = f"{organic_result_top1.get('title', '')} | {organic_result_top1.get('snippet', '')}".strip()
            snippets_and_passages_wordset = set(snippets_and_passages_string.lower().split())

            overlap_coeff = overlap_coefficient(entity_condition_wordset, snippets_and_passages_wordset)
            if overlap_coeff >= 0.5:
                filtered_entity_score_dict[entity] = overlap_coeff

        if len(filtered_entity_score_dict) == 0:
            return None, None

        final_entity_and_score_list = sorted(filtered_entity_score_dict.items(), key=lambda item: item[1], reverse=True)
        if final_entity_and_score_list[0][1] >= 0.85:
            discard_index = 0
            for entity in final_entity_and_score_list:
                if entity[1] < 0.85:
                    final_entity_and_score_list = final_entity_and_score_list[:discard_index]
                    break
                discard_index += 1

        final_entity_list = []
        final_entity_results_dict = {}
        for entry in final_entity_and_score_list:
            ent = entry[0]
            final_entity_list.append(ent)
            final_entity_results_dict[ent] = filtered_entity_results_dict[ent]

        return final_entity_list, final_entity_results_dict


class KG:
    def __init__(
        self,
        kg_query_language: Optional[str] = None,
        kopl_parser_url: Optional[str] = None,
        kopl_engine_url: Optional[str] = None,
    ):
        self.kg_query_language = (kg_query_language or "local").strip()
        self.kopl_parser_url = kopl_parser_url or "local://kg-parser"
        self.kopl_engine_url = kopl_engine_url or "local://kg-engine"

    def obtain_kopl_results(self, question: str):
        program = semantic_parsing_api(question, self.kopl_parser_url)
        print("program:", program)
        result = engine_exec_api(program, self.kopl_engine_url)

        # KG evidence retrieval is fixed to top-100 by product requirement.
        kg_top_k = 100

        docs: List[str] = []
        scores: List[float] = []

        _ATTR_SKIP_KEYS = {"enwiki_title", "wikidata_id"}

        def _fmt_entity_attrs(snap):
            if not snap:
                return ""
            attrs = snap.get("attrs") or {}
            kept = {k: v for k, v in attrs.items() if k not in _ATTR_SKIP_KEYS and v}
            return str(kept) if kept else ""

        evidence = result.get("evidence") or []
        for ev in evidence[:kg_top_k]:
            head = ev.get("head_entity", "")
            rel = ev.get("relation", "")
            tail = ev.get("label", "")
            direction = ev.get("direction", "")
            meta = ev.get("meta", {}) or {}
            line = f"{head} --[{rel}/{direction}]--> {tail}"
            if meta:
                line += f" ; rel_meta={meta}"
            head_attrs_str = _fmt_entity_attrs(ev.get("head_entity_snapshot"))
            if head_attrs_str:
                line += f" ; head_attrs={head_attrs_str}"
            tail_attrs_str = _fmt_entity_attrs(ev.get("tail_entity_snapshot"))
            if tail_attrs_str:
                line += f" ; tail_attrs={tail_attrs_str}"
            docs.append(line)
            scores.append(1.0 / (1 + len(scores)))

        # 如果常规 evidence 全是 entity_only（self 边、meta 空），上面循环输出的 doc
        # 信息密度极低，这里追加一份纯实体属性 doc 作补充。
        for ev in evidence[:kg_top_k]:
            if (ev.get("direction") or "") != "self":
                continue
            head_attrs_str = _fmt_entity_attrs(ev.get("head_entity_snapshot"))
            if head_attrs_str:
                docs.append(f"KG entity: {ev.get('head_entity', '')} ; attrs={head_attrs_str}")
                scores.append(1.0 / (1 + len(scores)))

        inner_content = (((result.get("inner_content") or [{}])[0]).get("content") or [])
        if not docs and inner_content:
            for ans in inner_content[:kg_top_k]:
                docs.append(f"KG candidate answer: {ans}")
                scores.append(1.0 / (1 + len(scores)))

        if not docs:
            ans = result.get("answer", "")
            docs = [f"KG answer: {ans}" if ans else ""]
            scores = [0.0 if not ans else 1.0]

        return {"docs": docs,"doc_scores": scores,
            "metrics": {
                "avg_similarity": float(sum(scores) / len(scores)) if scores else 0.0,
                "max_similarity": float(max(scores)) if scores else 0.0,
                "total_docs_searched": len(evidence) if evidence else len(docs),},}

    def Search(self, question: str, entity_name: str, descriptors: Optional[str] = ""):
        get_recorder().record_retrieval("kg")
        if descriptors:
            query = f"{entity_name} {descriptors}".strip()
        else:
            query = entity_name

        return self.obtain_kopl_results(query)

    def Relate(self, question: str, entity: str, relation: str):
        get_recorder().record_retrieval("kg")
        # Prefer full natural-language question for KG semantic parsing.
        # Fallback to "entity + relation" only when question is empty.
        query = (question or "").strip()
        print("query:", query)
        if not query:
            query = f"{entity} {relation}".strip()
        return self.obtain_kopl_results(query)

    def Filter(self, question: str, entities: list, condition: str):
        filtered = []
        details = {}
        for e in entities:
            get_recorder().record_retrieval("kg")
            q = f"{e} {condition}".strip()
            kg_result = self.obtain_kopl_results(q)
            if kg_result is not None and len(kg_result) > 0:
                filtered.append(e)
                details[e] = kg_result
        if len(filtered) == 0:
            return None, None
        return filtered[:10], {"filtered": details}
