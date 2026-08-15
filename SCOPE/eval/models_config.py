"""
Shared configuration for the eval runners.

Centralises:
  * which models each experiment runs / evaluates
  * how each model slug maps to the evaluator's CLI flag and predictions path
  * display names used in the aggregated Excel tables

Edit this file (NOT the runners) when adding a new baseline or ablation model.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional


# ---------------------------------------------------------------------------
# Per-experiment model lists
# ---------------------------------------------------------------------------

# Main experiment: SCOPE + 9 baselines.
# These are the slugs run by run_ex_main.py and shown in the main table.
MAIN_EXPERIMENT_MODELS: List[str] = [
    "SCOPE",
    "atomr",
    "deepsieve",
    "hydrarag",
    "standard_prompt", "cot", "self_ask",
    "standard_rag",    "ircot", "cok", "tog2",
]

# Ablation experiment: ONLY the models that get RUN by run_ex_abla.py.
# The full SCOPE used as a reference is reused from the main experiment
# output (eval/result/ex_main/...), so it isn't listed here.
ABLATION_EXPERIMENT_MODELS: List[str] = [
    "SCOPE_wo_semantic_directory_few-shot",
    "SCOPE_wo_semantic_directory_summary",
    "SCOPE_wo_decomposition",
    "SCOPE_wo_reflection",
    "SCOPE_wo_semantic_guide_on_planning",
    "SCOPE_wo_consolidation",
    "SCOPE_wo_operator_planning",
]

# Models that appear in the ablation TABLE — full SCOPE + ablation variants.
ABLATION_TABLE_MODELS: List[str] = ["SCOPE"] + ABLATION_EXPERIMENT_MODELS


# ---------------------------------------------------------------------------
# Evaluator CLI flag mapping
# ---------------------------------------------------------------------------

class EvalFlag(NamedTuple):
    flag: str           # CLI flag name passed to evaluate_answer_*.py
    pred_pattern: str   # predictions path under <out_root>/<slug>/
                        #   may contain "{dataset}" for dataset-specific names
                        #   empty string -> directory (deepsieve)
    is_dir: bool = False


# Map model slug -> how the evaluator script expects to receive it.
MODEL_EVAL_FLAGS: Dict[str, EvalFlag] = {
    "SCOPE":       EvalFlag("--new-model",      "predictions.jsonl"),
    "hydrarag":        EvalFlag("--hyprarag",       "predictions.jsonl"),
    "atomr":           EvalFlag("--atomr",          "{dataset}_pred.jsonl"),
    "deepsieve":       EvalFlag("--deepserice-dir", "",                       is_dir=True),
    "standard_prompt": EvalFlag("--standard-prompt", "predictions.jsonl"),
    "cot":             EvalFlag("--cot",             "predictions.jsonl"),
    "self_ask":        EvalFlag("--self-ask",        "predictions.jsonl"),
    "standard_rag":    EvalFlag("--standard-rag",    "predictions.jsonl"),
    "ircot":           EvalFlag("--ircot",           "predictions.jsonl"),
    "cok":             EvalFlag("--cok",             "predictions.jsonl"),
    "tog2":            EvalFlag("--tog2",            "predictions.jsonl"),
    "SCOPE_wo_semantic_directory_few-shot": EvalFlag("--wo-semantic-directory-few-shot", "predictions.jsonl"),
    "SCOPE_wo_semantic_directory_summary":  EvalFlag("--wo-semantic-directory-summary",  "predictions.jsonl"),
    "SCOPE_wo_decomposition":       EvalFlag("--wo-decomp",       "predictions.jsonl"),
    "SCOPE_wo_reflection":     EvalFlag("--wo-reflection",     "predictions.jsonl"),
    "SCOPE_wo_semantic_guide_on_planning": EvalFlag("--wo-semantic-guide-on-planning", "predictions.jsonl"),
    "SCOPE_wo_consolidation": EvalFlag("--wo-consolidation", "predictions.jsonl"),
    "SCOPE_wo_operator_planning":       EvalFlag("--wo-opplan",       "predictions.jsonl"),
}


# Evaluator-side source name for each slug (some evaluators spell deepsieve /
# hydrarag oddly: "deepserive" / "hyprarag"). Used when reading summary.json.
MODEL_TO_SOURCE: Dict[str, str] = {
    "SCOPE": "SCOPE",
    "atomr":     "atomr",
    "deepsieve": "deepserive",
    "hydrarag":  "hyprarag",
    "standard_prompt": "standard_prompt",
    "cot":             "cot",
    "self_ask":        "self_ask",
    "standard_rag":    "standard_rag",
    "ircot":           "ircot",
    "cok":             "cok",
    "tog2":            "tog2",
    "SCOPE_wo_semantic_directory_few-shot": "SCOPE_wo_semantic_directory_few-shot",
    "SCOPE_wo_semantic_directory_summary":  "SCOPE_wo_semantic_directory_summary",
    "SCOPE_wo_decomposition":       "SCOPE_wo_decomposition",
    "SCOPE_wo_reflection":     "SCOPE_wo_reflection",
    "SCOPE_wo_semantic_guide_on_planning": "SCOPE_wo_semantic_guide_on_planning",
    "SCOPE_wo_consolidation": "SCOPE_wo_consolidation",
    "SCOPE_wo_operator_planning":       "SCOPE_wo_operator_planning",
}


# ---------------------------------------------------------------------------
# Display names (used in Excel tables)
# ---------------------------------------------------------------------------

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "standard_prompt": "Standard Prompt",
    "cot":             "CoT",
    "self_ask":        "Self-ASK",
    "standard_rag":    "Standard RAG",
    "ircot":           "IRCoT",
    "cok":             "CoK",
    "tog2":            "ToG2",
    "hydrarag":        "HydraRAG",
    "deepsieve":       "DeepSieve",
    "atomr":           "Atomr",
    "SCOPE":       "SCOPE (Ours)",
    "SCOPE_wo_semantic_directory_few-shot": "w/o Semantic Directory (Few-shot)",
    "SCOPE_wo_semantic_directory_summary":  "w/o Semantic Directory Summary",
    "SCOPE_wo_decomposition":       "w/o Decomposition",
    "SCOPE_wo_reflection":     "w/o Reflection",
    "SCOPE_wo_semantic_guide_on_planning": "w/o Semantic Guide on Planning",
    "SCOPE_wo_consolidation": "w/o Consolidation",
    "SCOPE_wo_operator_planning":       "w/o Operator Planning",
}


# ---------------------------------------------------------------------------
# Main-experiment Excel grouping
# ---------------------------------------------------------------------------

MAIN_WITHOUT_RETRIEVE: List[str] = ["standard_prompt", "cot", "self_ask"]
MAIN_WITH_RETRIEVE:    List[str] = ["standard_rag", "ircot", "cok", "tog2",
                                    "hydrarag", "deepsieve", "atomr"]
MAIN_OUR_METHOD:       List[str] = ["SCOPE"]


def predictions_path(out_root, dataset: str, slug: str):
    """Return the predictions path/dir for <out_root>/<slug> for the given dataset.

    `out_root` should already include the (dataset, llm) layer — i.e. it's
    <RESULT_ROOT>/<dataset>/<llm>/. Returns a Path-like (caller's Path type).
    """
    flag = MODEL_EVAL_FLAGS.get(slug)
    if flag is None:
        raise KeyError(f"unknown model slug: {slug}")
    base = out_root / slug
    if flag.is_dir:
        return base
    name = flag.pred_pattern.format(dataset=dataset)
    return base / name
