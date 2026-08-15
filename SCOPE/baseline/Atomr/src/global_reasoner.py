from typing import Any, Dict, List, Optional

from knowledge_sources import Text, Table, KG
from query_knowledge_source.query_llm import OpenAICaller
from prompts.answer_formulation_prompts import (
    format_direct_rag_prompt_mmqa,
    format_relate_prompt_mmqa,
    format_filter_prompt_mmqa,
    format_search_prompt_mmqa,
    format_answer_from_child_qa_pairs_prompt_mmqa,
    format_direct_answer_prompt_mmqa,
)
from utils import extract_llm_answers


direct_rag_prompts = {
    "mmqa": format_direct_rag_prompt_mmqa,
}

direct_answer_prompts = {
    "mmqa": format_direct_answer_prompt_mmqa,
}

search_prompts = {
    "mmqa": format_search_prompt_mmqa,
}

relate_prompts = {
    "mmqa": format_relate_prompt_mmqa,
}

filter_prompts = {
    "mmqa": format_filter_prompt_mmqa,
}

answer_from_child_qa_prompts = {
    "mmqa": format_answer_from_child_qa_pairs_prompt_mmqa,
}


def format_text_retrieval_answers(full_answers):
    return " ... ".join(f"[{entry['rank']}] {entry['entity']} | {entry['passage']}" for entry in full_answers)


def format_table_retrieval_answers(full_answers, k, print_article_text=True):
    result_index = 1
    str_results_list = []
    for result_type, result in full_answers.items():
        if result_type != "organic_results":
            title = ""
            contents = []
            for attribute, value in result.items():
                if value != "":
                    if attribute == "title":
                        title = f"{value} | "
                    if attribute == "snippet":
                        contents.append(value)
                    if attribute == "article_text" and print_article_text:
                        contents.append(value)
            str_result = ""
            if len(title) > 0:
                str_result += title
            if len(contents) > 0:
                str_result += " ".join(contents)
            if len(str_result) > 0:
                str_results_list.append(f"[{str(result_index)}] {str_result}".strip())
                result_index += 1
        else:
            for organic_result in result:
                if result_index > k:
                    break
                title = ""
                contents = []
                for attribute, value in organic_result.items():
                    if value != "":
                        if attribute == "title":
                            title = f"{value} | "
                        if attribute == "snippet":
                            contents.append(value)
                        if attribute == "article_text" and print_article_text:
                            contents.append(value)
                str_result = ""
                if len(title) > 0:
                    str_result += title
                if len(contents) > 0:
                    str_result += " ".join(contents)
                if len(str_result) > 0:
                    str_results_list.append(f"[{str(result_index)}] {str_result}".strip())
                    result_index += 1
    return (" ".join(str_results_list)).strip()


def print_supporting_knowledge(supporting_knowledge):
    knowledge = ""
    if "Text" in supporting_knowledge and len(supporting_knowledge["Text"]) > 0:
        knowledge += f"\nText Passages: {supporting_knowledge['Text']}"
        print("supporting_knowledge: Text")
    if "Table" in supporting_knowledge and len(supporting_knowledge["Table"]) > 0:
        knowledge += f"\nTable Rows: {supporting_knowledge['Table']}"
        print("supporting_knowledge: Table")
    if "KG" in supporting_knowledge and len(supporting_knowledge["KG"]) > 0:
        knowledge += f"\nKG Triples: {supporting_knowledge['KG']}"
        print("supporting_knowledge: KG")
    # print("=== Supporting knowledge: ")
    # print(knowledge)


def _preview_text(value: Any, max_chars: int = 200) -> str:
    s = str(value or "").replace("\n", " ").strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "...(truncated)"


def _split_kg_items(value: str) -> List[str]:
    if not value:
        return []
    text = str(value)
    if " ; " in text:
        return [v.strip() for v in text.split(" ; ") if v.strip()]
    return [v.strip() for v in text.split(",") if v.strip()]


def summarize_source_outputs(raw_outputs: Dict[str, Any], supporting_knowledge: Dict[str, str]) -> Dict[str, Any]:
    source_details: Dict[str, Any] = {}
    for src in ["Text", "Table", "KG"]:
        raw = raw_outputs.get(src)
        support_str = supporting_knowledge.get(src, "")

        detail = {
            "hit_count": 0,
            "raw_answers_preview": [],
            "supporting_knowledge_preview": _preview_text(support_str, 300),
            "raw_type": type(raw).__name__ if raw is not None else "NoneType",
        }

        if src == "KG":
            kg_items = _split_kg_items(support_str)
            detail["hit_count"] = len(kg_items)
            detail["raw_answers_preview"] = kg_items[:8]
        elif src == "Text":
            if isinstance(raw, list):
                detail["hit_count"] = len(raw)
                detail["raw_answers_preview"] = [
                    _preview_text(f"{entry.get('entity', '')} | {entry.get('passage', '')}", 180)
                    for entry in raw[:5]
                ]
            elif raw is not None:
                detail["hit_count"] = 1
                detail["raw_answers_preview"] = [_preview_text(raw, 180)]
        elif src == "Table":
            if isinstance(raw, dict):
                organic = raw.get("organic_results", []) or []
                detail["hit_count"] = len(organic)
                detail["raw_answers_preview"] = [
                    _preview_text(f"{item.get('title', '')} | {item.get('snippet', '')}", 180)
                    for item in organic[:5]
                ]
            elif raw is not None:
                detail["hit_count"] = 1
                detail["raw_answers_preview"] = [_preview_text(raw, 180)]

        source_details[src] = detail

    return source_details


class GlobalReasoner:
    def _set_last_call_trace(
        self,
        *,
        operation: str,
        question: str,
        selected_sources: set,
        raw_source_outputs: Dict[str, Any],
        supporting_knowledge: Dict[str, str],
        clean_answer_list: List[str],
        paraphrase_answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.last_call_trace = {
            "operation": operation,
            "question": question,
            "selected_sources": sorted(list(selected_sources or set())),
            "source_details": summarize_source_outputs(raw_source_outputs, supporting_knowledge),
            "clean_answer_list": clean_answer_list or [],
            "paraphrase_answer": paraphrase_answer or "",
            "metadata": metadata or {},
        }

    def __init__(
        self,
        openai_caller: OpenAICaller,
        available_knowledge_sources: set,
        text_retriever_url: Optional[str] = None,
        table_retriever_url: Optional[str] = None,
        kg_query_language: Optional[str] = "local",
        kopl_parser_url: Optional[str] = None,
        kopl_engine_url: Optional[str] = None,
        k: Optional[int] = 3,
    ):
        if len(available_knowledge_sources) == 0:
            raise Exception("please provide at least one knowledge source in [Text, Table, KG].")

        self.openai_caller = openai_caller
        self.available_knowledge_sources = available_knowledge_sources
        self.k = k

        if "Text" in available_knowledge_sources:
            self.text = Text(text_retriever_url, k)
        if "Table" in available_knowledge_sources:
            self.table = Table(table_retriever_url, k)
        if "KG" in available_knowledge_sources:
            self.kg = KG(kg_query_language=kg_query_language, kopl_parser_url=kopl_parser_url, kopl_engine_url=kopl_engine_url)

        self.last_call_trace: Dict[str, Any] = {}

    def direct_Answer(self, dataset_name, question: str):
        prompt = direct_answer_prompts[dataset_name](question)
        llm_response, finish_reason = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        if finish_reason == "length":
            llm_response, _ = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        paraphrase_answer, clean_answer_list = extract_llm_answers(llm_response)
        self._set_last_call_trace(
            operation="direct_Answer",
            question=question,
            selected_sources=set(),
            raw_source_outputs={},
            supporting_knowledge={},
            clean_answer_list=clean_answer_list,
            paraphrase_answer=paraphrase_answer,
            metadata={"finish_reason": finish_reason},
        )
        return clean_answer_list, paraphrase_answer

    def direct_RAG(self, dataset_name: str, knowledge_sources: set, question: str):
        supporting_knowledge = {}
        raw_source_outputs: Dict[str, Any] = {}
        for source in knowledge_sources:
            retrieved_knowledge = ""
            if source == "Text":
                _, full_answers = self.text.retrieve_topk_passages(question, self.text.k)
                if full_answers is not None:
                    raw_source_outputs["Text"] = full_answers
                    retrieved_knowledge = format_text_retrieval_answers(full_answers)
            elif source == "Table":
                _, full_answers = self.table.retrieve_topk_tables(question, self.table.k)
                if full_answers is not None:
                    raw_source_outputs["Table"] = full_answers
                    retrieved_knowledge = format_table_retrieval_answers(full_answers, self.table.k, print_article_text=True)
            elif source == "KG":
                kg_result = self.kg.obtain_kopl_results(question)
                if kg_result is not None:
                    supporting_knowledge[source] = kg_result

        print_supporting_knowledge(supporting_knowledge)
        prompt = direct_rag_prompts[dataset_name](question, supporting_knowledge)
        llm_response, finish_reason = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        if finish_reason == "length":
            llm_response, _ = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        paraphrase_answer, clean_answer_list = extract_llm_answers(llm_response)
        self._set_last_call_trace(
            operation="direct_RAG",
            question=question,
            selected_sources=knowledge_sources,
            raw_source_outputs=raw_source_outputs,
            supporting_knowledge=supporting_knowledge,
            clean_answer_list=clean_answer_list,
            paraphrase_answer=paraphrase_answer,
            metadata={"finish_reason": finish_reason},
        )
        return clean_answer_list, paraphrase_answer, supporting_knowledge

    def Search(self, dataset_name: str, knowledge_sources: set, question: str, entity_name: str, descriptors: Optional[str] = ""):
        supporting_knowledge = {}
        raw_source_outputs: Dict[str, Any] = {}
        for source in knowledge_sources:
            if source == "Text":
                _, full_answers = self.text.Search(question, entity_name, descriptors)
                if full_answers is not None:
                    raw_source_outputs["Text"] = full_answers
                    supporting_knowledge["Text"] = format_text_retrieval_answers(full_answers)
            elif source == "Table":
                _, full_answers = self.table.Search(question, entity_name, descriptors)
                if full_answers is not None:
                    raw_source_outputs["Table"] = full_answers
                    supporting_knowledge["Table"] = format_table_retrieval_answers(full_answers, self.table.k, print_article_text=True)
            elif source == "KG":
                print("search kg entity_name:", entity_name)
                kg_result = self.kg.Search(question, entity_name, descriptors)
                if kg_result is not None:
                    supporting_knowledge["KG"] = kg_result

        if supporting_knowledge == {}:
            raise Exception("Search() function execution failed: obtained empty supporting knowledge.")

        print_supporting_knowledge(supporting_knowledge)
        if descriptors != "":
            question = f"{question} ({descriptors})"
        prompt = search_prompts[dataset_name](question, supporting_knowledge)
        llm_response, finish_reason = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        if finish_reason == "length":
            llm_response, _ = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        paraphrase_answer, clean_answer_list = extract_llm_answers(llm_response)
        self._set_last_call_trace(
            operation="Search",
            question=question,
            selected_sources=knowledge_sources,
            raw_source_outputs=raw_source_outputs,
            supporting_knowledge=supporting_knowledge,
            clean_answer_list=clean_answer_list,
            paraphrase_answer=paraphrase_answer,
            metadata={
                "entity_name": entity_name,
                "descriptors": descriptors,
                "finish_reason": finish_reason,
            },
        )
        return clean_answer_list, paraphrase_answer, supporting_knowledge

    def Relate(self, dataset_name: str, knowledge_sources: set, question: str, entity: str, relation: str):
        supporting_knowledge = {}
        raw_source_outputs: Dict[str, Any] = {}
        for source in knowledge_sources:
            if source == "Text":
                clean_answers, full_answers = self.text.Relate(question, entity, relation)
                if clean_answers is not None and full_answers is not None:
                    raw_source_outputs["Text"] = full_answers
                    supporting_knowledge["Text"] = format_text_retrieval_answers(full_answers)
            elif source == "Table":
                _, full_answers = self.table.Relate(question, entity, relation)
                if full_answers is not None:
                    raw_source_outputs["Table"] = full_answers
                    supporting_knowledge["Table"] = format_table_retrieval_answers(full_answers, k=self.table.k, print_article_text=True)
            elif source == "KG":
                kg_result = self.kg.Relate(question, entity, relation)
                if kg_result is not None:
                    supporting_knowledge["KG"] = kg_result

        if supporting_knowledge == {}:
            raise Exception("Relate() function execution failed: obtained empty supporting knowledge.")

        print_supporting_knowledge(supporting_knowledge)
        # print("relate supporting_knowledge:", supporting_knowledge)
        prompt = relate_prompts[dataset_name](question, supporting_knowledge)
        # print("relate prompt:", prompt)
        llm_response, finish_reason = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        print("relate llm_response:", llm_response)
        # print("relate finish_reason:", finish_reason)
        if finish_reason == "length":
            llm_response, _ = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        paraphrase_answer, clean_answer_list = extract_llm_answers(llm_response)
        self._set_last_call_trace(
            operation="Relate",
            question=question,
            selected_sources=knowledge_sources,
            raw_source_outputs=raw_source_outputs,
            supporting_knowledge=supporting_knowledge,
            clean_answer_list=clean_answer_list,
            paraphrase_answer=paraphrase_answer,
            metadata={
                "entity": entity,
                "relation": relation,
                "finish_reason": finish_reason,
            },
        )
        return clean_answer_list, paraphrase_answer, supporting_knowledge

    def Filter(self, dataset_name: str, knowledge_sources: set, question: str, entities: list, condition: str):
        supporting_knowledge = {}
        raw_source_outputs: Dict[str, Any] = {}
        for source in knowledge_sources:
            if source == "Text":
                _, full_answers = self.text.Filter(question, entities, condition)
                if full_answers is not None:
                    raw_source_outputs["Text"] = full_answers
                    supporting_knowledge["Text"] = format_text_retrieval_answers(full_answers)
            elif source == "Table":
                _, full_answer_results_dict = self.table.Filter(question, entities, condition)
                if full_answer_results_dict is not None:
                    raw_source_outputs["Table"] = full_answer_results_dict
                    filter_string_answer = ""
                    for entity, results in full_answer_results_dict.items():
                        filter_string_answer += f"'{entity}' related results: \n"
                        filter_string_answer += format_table_retrieval_answers(results, k=self.table.k, print_article_text=True)
                    supporting_knowledge["Table"] = filter_string_answer
            elif source == "KG":
                kg_result = self.kg.obtain_kopl_results(question)
                if kg_result is not None:
                    supporting_knowledge[source] = kg_result

        if supporting_knowledge == {}:
            raise Exception("Filter() function execution failed: obtained empty supporting knowledge.")

        print_supporting_knowledge(supporting_knowledge)
        # print("filter supporting_knowledge:", supporting_knowledge)
        prompt = filter_prompts[dataset_name](question, condition, supporting_knowledge)
        llm_response, finish_reason = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        print("filter llm_response:", llm_response)
        if finish_reason == "length":
            llm_response, _ = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        paraphrase_answer, clean_answer_list = extract_llm_answers(llm_response)
        self._set_last_call_trace(
            operation="Filter",
            question=question,
            selected_sources=knowledge_sources,
            raw_source_outputs=raw_source_outputs,
            supporting_knowledge=supporting_knowledge,
            clean_answer_list=clean_answer_list,
            paraphrase_answer=paraphrase_answer,
            metadata={
                "entities": entities,
                "condition": condition,
                "finish_reason": finish_reason,
            },
        )
        return clean_answer_list, paraphrase_answer, supporting_knowledge

    def AnswerFromQAPairs(self, dataset_name: str, question: str, child_qa_pairs: str):
        prompt = answer_from_child_qa_prompts[dataset_name](question, child_qa_pairs)
        # print("answer from child qa pairs prompt:", prompt)
        # print("开始融合子答案进行做答")
        llm_response, finish_reason = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        # print("首次做答完成")
        # print("answer from child qa pairs llm_response:", llm_response)
        # print("answer from child qa pairs finish_reason:", finish_reason)
        if finish_reason == "length":
            llm_response, _ = self.openai_caller.query_deepseek(prompt=prompt, max_tokens=10000)
        paraphrase_answer, clean_answer_list = extract_llm_answers(llm_response)
        supporting_knowledge = {"child_qa_pairs": child_qa_pairs}
        self._set_last_call_trace(
            operation="AnswerFromQAPairs",
            question=question,
            selected_sources=set(),
            raw_source_outputs={"child_qa_pairs": child_qa_pairs},
            supporting_knowledge={},
            clean_answer_list=clean_answer_list,
            paraphrase_answer=paraphrase_answer,
            metadata={
                "child_qa_pair_count": len(child_qa_pairs) if isinstance(child_qa_pairs, list) else 0,
                "finish_reason": finish_reason,
            },
        )
        return clean_answer_list, paraphrase_answer, supporting_knowledge
