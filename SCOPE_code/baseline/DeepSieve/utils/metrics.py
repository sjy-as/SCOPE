"""
utils/metrics.py

This module contains functions for evaluating the performance of the RAG system.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

import tiktoken

from utils.llm_call import call_openai_chat


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    from string import punctuation

    s = str(s or "").lower()
    s = s.translate(str.maketrans("", "", punctuation))
    s = " ".join(s.split())
    stop_words = {"a", "an", "the", "is", "are", "was", "were"}
    s = " ".join([w for w in s.split() if w not in stop_words])
    return s.strip()


def _split_answers(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        items = re.split(r"\s*[;|/]\s*|\n+", value)
    else:
        items = [value]
    return [str(x).strip() for x in items if str(x).strip()]


def _token_overlap_score(prediction: str, answers: Sequence[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    best = 0.0
    for gt in answers:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        gt_counter = Counter(gt_tokens)
        common = sum((pred_counter & gt_counter).values())
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(prediction: str, ground_truth: str) -> dict:
    return {"exact_match": compute_exact_match(prediction, ground_truth), "f1": compute_f1(prediction, ground_truth)}


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def is_descriptive_answer(text: str) -> bool:
    normalized = normalize_answer(text)
    if not normalized:
        return False
    if any(ch.isdigit() for ch in text):
        return False
    return len(normalized.split()) >= 4 or len(normalized) >= 25


def judge_descriptive_answer(prediction: str, answers: Sequence[str], api_key: str, model: str, base_url: str) -> bool:
    prompt = f"""You are grading whether a predicted answer sufficiently matches the reference answers.
Be lenient with paraphrases, semantically equivalent statements, and concise descriptions.
Return only valid JSON: {{"correct": true}} or {{"correct": false}}.

Predicted answer:
{prediction}

Reference answers:
{json.dumps(list(answers), ensure_ascii=False)}
"""
    response = call_openai_chat(prompt, api_key, model, base_url)
    try:
        cleaned = str(response).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        parsed = json.loads(cleaned.strip())
        return bool(parsed.get("correct", False))
    except Exception:
        return False


def score_answer_prediction(
    prediction: str,
    references: Any,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> Dict[str, Any]:
    answers = _split_answers(references)
    if not answers:
        return {"score": 0.0, "hit_count": 0, "ref_count": 0, "matched_answers": []}

    normalized_prediction = normalize_answer(prediction)
    matched_answers: List[str] = []
    hit_count = 0

    for answer in answers:
        if not answer:
            continue
        if normalize_answer(answer) in normalized_prediction or normalized_prediction in normalize_answer(answer):
            matched_answers.append(answer)
            hit_count += 1
            continue
        if _token_overlap_score(prediction, [answer]) >= 0.6:
            matched_answers.append(answer)
            hit_count += 1

    if hit_count == 0 and len(answers) == 1 and api_key and model and base_url and is_descriptive_answer(answers[0]):
        if judge_descriptive_answer(prediction, answers, api_key, model, base_url):
            return {"score": 1.0, "hit_count": 1, "ref_count": 1, "matched_answers": answers, "judged_by_llm": True}

    ref_count = len(answers)
    score = hit_count / ref_count if ref_count else 0.0
    return {
        "score": score,
        "hit_count": hit_count,
        "ref_count": ref_count,
        "matched_answers": matched_answers,
    }


def calculate_overall_metrics(all_metrics):
    total_queries = len(all_metrics)
    if total_queries == 0:
        return {}

    overall = {
        "avg_exact_match": sum(m["evaluation_metrics"]["exact_match"] for m in all_metrics) / total_queries,
        "avg_f1": sum(m["evaluation_metrics"]["f1"] for m in all_metrics) / total_queries,
        "avg_retrieval_time": sum(m.get("total_retrieval_time", 0) for m in all_metrics) / total_queries,
        "avg_docs_searched": sum(m.get("total_docs_searched", 0) for m in all_metrics) / total_queries,
        "avg_similarity": sum(m.get("avg_similarity", 0) for m in all_metrics) / total_queries,
        "avg_prompt_tokens_per_subquery": sum(m.get("avg_prompt_tokens", 0) for m in all_metrics) / total_queries,
        "avg_total_tokens_per_query": sum(m.get("total_prompt_tokens", 0) for m in all_metrics) / total_queries,
        "avg_execution_time": sum(m.get("execution_time", 0) for m in all_metrics) / total_queries,
        "avg_sq1_score": sum(m.get("sq1_score", 0) for m in all_metrics) / total_queries,
        "avg_sq2_score": sum(m.get("sq2_score", 0) for m in all_metrics) / total_queries,
        "avg_total_score": sum(m.get("total_score", 0) for m in all_metrics) / total_queries,
    }
    overall["sq1_hit_rate"] = overall["avg_sq1_score"] / 0.5 if overall["avg_sq1_score"] else 0.0
    overall["sq2_hit_rate"] = overall["avg_sq2_score"]
    return overall
