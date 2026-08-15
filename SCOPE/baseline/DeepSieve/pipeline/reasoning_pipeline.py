"""
pipeline/reasoning_pipeline.py

Multi-hop reasoning control pipeline:
- Subquestion decomposition
- Knowledge routing (table vs kg, backward-compatible with local vs global)
- Final answer fusion
- Variable substitution
"""

import json
from typing import List, Dict
from utils.llm_call import call_openai_chat
from utils.metrics import count_tokens


__all__ = [
    "plan_subqueries_with_llm",
    "route_query_with_llm",
    "get_fused_final_answer",
    "substitute_variables"
]


def plan_subqueries_with_llm(decompose: bool, query: str, api_key: str, model: str, base_url: str) -> dict:
    if decompose is False:
        return {
            "subqueries": [
                {
                    "id": "q1",
                    "query": query,
                    "depends_on": [],
                    "variables": []
                }
            ]
        }

    prompt = f"""You are a reasoning planner. Your task is to decompose a multi-hop question into a sequence of dependent sub-questions.
For each sub-question, you should:
1. Identify any variables that need to be filled from previous sub-questions' answers
2. Specify the dependency relationship between sub-questions
3. Use consistent variable names in square brackets (e.g. [birthplace]) to show dependencies

Question: {query}

Please output in JSON format as follows:
{{
    "subqueries": [
        {{
            "id": "q1",
            "query": "First sub-question without dependencies",
            "depends_on": [],
            "variables": []
        }},
        {{
            "id": "q2",
            "query": "Second sub-question that may contain [variable_from_q1]",
            "depends_on": ["q1"],
            "variables": [
                {{
                    "name": "variable_from_q1",
                    "source_query": "q1"
                }}
            ]
        }}
    ]
}}
Only output valid JSON. Do not add any explanation or markdown code block markers."""

    response = call_openai_chat(prompt, api_key, model, base_url)
    try:
        cleaned_response = str(response).strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        result = json.loads(cleaned_response)
        if "subqueries" not in result:
            print("⚠️ Missing subqueries field in response:")
            print(result)
            return {"subqueries": []}
        return result
    except json.JSONDecodeError as e:
        print("⚠️ Failed to parse JSON from LLM response:")
        print(response)
        print(f"Error: {str(e)}")
        return {"subqueries": []}


def substitute_variables(query: str, variable_values: dict) -> str:
    result = query
    for var_name, value in variable_values.items():
        result = result.replace(f"[{var_name}]", value)
    return result


def route_query_with_llm(
    query: str,
    sources: List[Dict],
    api_key: str,
    model: str,
    base_url: str,
    fail_history: str,
) -> str:
    """
    Route a query to ONE of the live knowledge sources.

    `sources` is the live knowledge base: a list of {"name", "profile"} dicts
    (e.g. kg / table / doc, or classic local / global). Returns the chosen
    source name. With a single source, returns it directly without an LLM
    call. Defaults to the first source on any failure. `fail_history` carries
    the previous failed routing so the LLM can re-route to a different source.
    """
    names = [str(s.get("name", "")).strip().lower() for s in (sources or [])]
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]

    blocks = []
    for i, s in enumerate(sources):
        nm = str(s.get("name", "")).strip().lower()
        blocks.append(
            f"SOURCE {i + 1} NAME: {nm}\n"
            f"SOURCE {i + 1} PROFILE:\n{s.get('profile', '')}"
        )
    sources_block = "\n\n".join(blocks)
    names_str = " / ".join(f'"{n}"' for n in names)

    prompt = f"""You are a routing assistant. Your task is to decide which ONE knowledge source should answer a query.

{sources_block}

QUERY:
{query}

{fail_history}

Please output only one word — exactly one of: {names_str} — based on which source profile is most relevant to the query.
Do not add any explanation or extra words."""

    try:
        response = call_openai_chat(prompt, api_key, model, base_url)
        if not response:
            print(f"⚠️ Routing response is empty, defaulting to {names[0]}")
            return names[0]

        route = response.strip().lower()
        if route in names:
            return route
        # Tolerate the model wrapping the name in quotes / extra tokens.
        hit = next((n for n in names if n in route), None)
        if hit:
            return hit
        print(f"⚠️ Unexpected routing output: {route}, defaulting to {names[0]}")
        return names[0]
    except Exception as e:
        print(f"⚠️ Routing error: {str(e)}, defaulting to {names[0]}")
        return names[0]


def get_fused_final_answer(original_question: str, subquery_results: List[Dict], api_key: str, model: str, base_url: str) -> tuple:
    prompt = f"""You are a multi-hop reasoning assistant. Your task is to generate the final answer to a multi-hop question based on the following reasoning steps.

Original Question: {original_question}

Subquestion Reasoning Steps:
"""
    for r in subquery_results:
        prompt += f"{r['subquery_id']}: {r['actual_query']} → {r['answer']}\n"
        prompt += f"Reason: {r['reason']}\n\n"

    prompt += """\nBased on the above reasoning steps, what is the final answer to the original question?

Please respond in JSON format:
{
  "answer": "final_answer",
  "reason": "final_reasoning"
}
Only output valid JSON. Do not add any explanation or markdown code block markers."""

    token_count = count_tokens(prompt, model)
    print(f"🧠 Fusion Prompt Token Count: {token_count}")

    response = call_openai_chat(prompt, api_key, model, base_url)
    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()

        parsed = json.loads(cleaned_response)
        answer = parsed.get("answer", "").strip()
        reason = parsed.get("reason", "").strip()
        print(f"✅ Final fused answer: {answer}")
        print(f"🔎 Final reasoning: {reason}")
        return answer, reason, token_count, prompt
    except Exception as e:
        print(f"⚠️ Failed to parse fused answer: {e}")
        return "", "", token_count, prompt
