"""
pipeline/subquery_executor.py

Execute one subquery with routing, retrieval, answer generation, and reflection.
"""

import json
import sys
import time
from utils.llm_call import call_openai_chat
from utils.metrics import count_tokens
from pipeline.reasoning_pipeline import route_query_with_llm, substitute_variables

try:
    if "/root/autodl-tmp" not in sys.path:
        sys.path.insert(0, "/root/autodl-tmp")
    from _common.cost_counter import bump_retrieval as _bump_retrieval
except Exception:  # pragma: no cover
    def _bump_retrieval(*_a, **_kw): pass


def execute_subquery(
    subquery_info: dict,
    variable_values: dict,
    sources: list,
    merged_rag,
    use_routing,
    use_reflection,
    max_reflexion_times,
    openai_api_key,
    openai_model,
    openai_base_url,
) -> dict:
    """`sources` is the live knowledge base: a list of
    {"name", "rag", "profile"} dicts. Routing picks one of them; on failure
    the fail-history is fed back so the next attempt re-routes elsewhere.
    `merged_rag` is used only when routing is disabled."""
    answer = ""
    reason = ""
    success = 0
    route = "merged"
    retrieved = {"docs": [], "doc_scores": []}
    subquery_metrics = {
        "subquery_id": subquery_info.get("id", ""),
        "retrieval_time": 0,
        "docs_searched": 0,
        "avg_similarity": 0,
        "max_similarity": 0,
    }
    token_count = 0
    current_variables = {}

    subquery_id = subquery_info["id"]
    original_query = subquery_info["query"]

    if subquery_info.get("depends_on"):
        print(f"\n⏳ Processing dependencies for query {subquery_id}: {subquery_info['depends_on']}")

    for var in subquery_info.get("variables", []):
        var_name = var["name"]
        source_query = var["source_query"]
        if source_query not in variable_values:
            print(f"❌ Error: Query {subquery_id} depends on an incomplete query {source_query}")
            continue
        current_variables[var_name] = variable_values[source_query]

    actual_query = substitute_variables(original_query, current_variables)
    print(f"\n🔍 Processing query {subquery_id}: {actual_query}")
    print(f"Original query: {original_query}")
    if current_variables:
        print(f"Variable substitution: {current_variables}")

    fail_history = ""
    left_reflexion_times = max_reflexion_times
    attempts = []  # 每轮 reflection 的快照（route / docs / answer / success ...）

    while left_reflexion_times > 0:
        left_reflexion_times -= 1
        success = 0
        fail_history_in = fail_history  # 该轮 routing prompt 输入的失败历史

        if use_routing:
            route = route_query_with_llm(
                actual_query,
                [{"name": s["name"], "profile": s.get("profile", "")} for s in sources],
                api_key=openai_api_key,
                model=openai_model,
                base_url=openai_base_url,
                fail_history=fail_history,
            )
            rag = next(
                (s["rag"] for s in sources
                 if str(s["name"]).strip().lower() == route),
                None,
            )
            if rag is None:
                # Routing returned an unusable name; fall back to first source.
                rag = sources[0]["rag"]
                route = str(sources[0]["name"]).strip().lower()
            print(f"Routing to {route.upper()} source")
        else:
            rag = merged_rag
            route = "merged"
            print("Using merged source")

        try:
            _bump_retrieval(route)
            retrieved = rag.rag_qa(actual_query, k=5)
            # print(f"Retrieved: {retrieved}")
            metrics = retrieved.get("metrics", {})

            subquery_metrics = {
                "subquery_id": subquery_id,
                "retrieval_time": metrics.get("retrieval_time", 0),
                "docs_searched": metrics.get("total_docs_searched", 0),
                "avg_similarity": metrics.get("avg_similarity", 0),
                "max_similarity": metrics.get("max_similarity", 0),
            }

            prompt = f"""Answer the following question based on the provided evidence documents.

Please respond strictly in JSON format with the following fields:
- answer: the direct, concise answer (just the value/entity/fact, no explanation). Leave it empty (\"\") if the answer is not found.
- reason: a brief explanation of how you arrived at this answer.
- success: 1 if the answer is confidently found from the evidence, 0 otherwise.

Format:
{{
"answer": "...",
"reason": "...",
"success": 1
}}

If the answer is not mentioned or cannot be inferred from the evidence, return:
{{
"answer": "",
"reason": "no relevant information found",
"success": 0
}}

Question: {actual_query}

Evidence Documents:
"""
            for d in retrieved.get("docs", []):
                prompt += f"- {d}\n"
            prompt += "\nOnly output valid JSON. Do not add any explanation or markdown code block markers."

            token_count = count_tokens(prompt, openai_model)
            print(f"🧮 Prompt Token Count: {token_count}")

            response = call_openai_chat(prompt, openai_api_key, openai_model, openai_base_url)

            try:
                cleaned_response = response.strip()
                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[7:]
                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[:-3]
                cleaned_response = cleaned_response.strip()

                parsed_response = json.loads(cleaned_response)
                answer = str(parsed_response.get("answer", "")).strip()
                reason = str(parsed_response.get("reason", "")).strip()
                success = int(parsed_response.get("success", 0))

                if success == 1:
                    variable_values[subquery_id] = answer
                    print(f"Extracted answer: {answer}")
                    print(f"Reasoning: {reason}")
                    print(f"Success: {success}")
                else:
                    variable_values[subquery_id] = ""
                    fail_history += (
                        f"Fail History: Last routing failed because {reason}. "
                        f"Last routing result is {route}. "
                        f"So please try another routing choice, don't choose {route} again."
                    )

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"⚠️ Failed to parse answer: {str(e)}")
                print(f"Raw response: {response}")
                answer = f"Error: {str(e)}"
                reason = ""
                success = 0

        except Exception as e:
            print(f"⚠️ Error occurred while processing query: {str(e)}")
            answer = f"Error: {str(e)}"
            reason = ""
            success = 0
            retrieved = {"docs": [], "doc_scores": []}
            subquery_metrics = {
                "subquery_id": subquery_id,
                "retrieval_time": 0,
                "docs_searched": 0,
                "avg_similarity": 0,
                "max_similarity": 0,
            }
            token_count = 0

        # 每轮迭代结束后留一份快照，方便事后区分初始路由 vs 最终（成功的）路由
        attempts.append({
            "attempt_index": len(attempts),
            "route": route,
            "fail_history_in": fail_history_in,
            "success": success,
            "answer": answer,
            "reason": reason,
            "docs_preview": [
                {
                    "score": (retrieved.get("doc_scores")
                              or [None] * len(retrieved.get("docs", [])))[i],
                    "text": (d if len(str(d)) <= 1200 else str(d)[:1200] + "...<truncated>"),
                }
                for i, d in enumerate(retrieved.get("docs", []))
            ],
            "retrieval_time": subquery_metrics.get("retrieval_time", 0),
            "docs_searched": subquery_metrics.get("docs_searched", 0),
            "avg_similarity": subquery_metrics.get("avg_similarity", 0),
            "max_similarity": subquery_metrics.get("max_similarity", 0),
            "prompt_token_count": token_count,
        })

        if success == 1 or (not use_routing) or (not use_reflection) or left_reflexion_times <= 0:
            break

    return {
        "answer": answer,
        "reason": reason,
        "success": success,
        "docs": retrieved.get("docs", []),
        "doc_scores": retrieved.get("doc_scores", []),
        "variables_used": current_variables,
        "metrics": subquery_metrics,
        "prompt_token_count": token_count,
        "subquery_id": subquery_id,
        "original_query": original_query,
        "actual_query": actual_query,
        "routing": route,                                    # 向后兼容：=最终一次路由
        "initial_routing": attempts[0]["route"] if attempts else route,
        "final_routing": route,
        "num_attempts": len(attempts),
        "attempts": attempts,                                # 每轮迭代的完整快照
        "retrieval_time": subquery_metrics["retrieval_time"],
        "docs_searched": subquery_metrics["docs_searched"],
        "avg_similarity": subquery_metrics["avg_similarity"],
        "max_similarity": subquery_metrics["max_similarity"],
    }
