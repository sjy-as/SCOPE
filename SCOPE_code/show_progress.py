#!/usr/bin/env python3
"""
Progress viewer for run_ex_main.py and run_ex_abla.py experiments.

Usage:
    python3 show_progress.py                      # main experiment, one-shot
    python3 show_progress.py --mode abla          # ablation experiment, one-shot
    python3 show_progress.py --watch              # main, refresh every 30s
    python3 show_progress.py --mode abla --watch 10
    python3 show_progress.py --llm deepseek-chat --watch 60  # filter by LLM
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── shared paths ──────────────────────────────────────────────────────────────

RESULT_ROOT_MAIN = Path("/root/autodl-tmp/eval/result/ex_main")
RESULT_ROOT_ABLA = Path("/root/autodl-tmp/eval/result/ex_abla")
QA_BENCH         = Path("/root/autodl-tmp/new_model/qa_bench")

DATASETS = {
    "kg_doc":    {"file": QA_BENCH / "kg-doc-1154.jsonl",            "label": "KG-Doc"},
    "kg_table":  {"file": QA_BENCH / "kg-table-1147.jsonl",           "label": "KG-Table"},
    "table_doc": {"file": QA_BENCH / "table-doc-1120.jsonl", "label": "Table-Doc"},
}
DATASET_ORDER = ["kg_doc", "kg_table", "table_doc"]

# ── main experiment config ────────────────────────────────────────────────────

MAIN_MODEL_ORDER = [
    "new_model", "atomr", "deepsieve", "hydrarag", "aop",
    "standard_prompt", "cot", "self_ask",
    "standard_rag", "ircot", "cok", "tog2",
]
MAIN_MODEL_SHORT = {
    "new_model":       "new_mdl",
    "atomr":           "atomr  ",
    "deepsieve":       "dpsieve",
    "hydrarag":        "hydra  ",
    "aop":             "aop    ",
    "standard_prompt": "std_pmt",
    "cot":             "cot    ",
    "self_ask":        "selfask",
    "standard_rag":    "std_rag",
    "ircot":           "ircot  ",
    "cok":             "cok    ",
    "tog2":            "tog2   ",
}
MAIN_LLM_ORDER = ["gpt-4o-mini", "gpt-4o", "qwen3-8b", "qwen3-32b", "qwen3-max-2026-01-23", "qwen3.5-397b-a17b", "gemini-2.5-flash", "deepseek-r1", "deepseek-chat"]
MAIN_LLM_SHORT = {
    "gpt-4o-mini":   "gpt-4o-mini",
    "gpt-4o":        "gpt-4o     ",
    "qwen3-8b":      "qwen3-8b   ",
    "qwen3-32b":     "qwen3-32b  ",
    "qwen3-max-2026-01-23": "qwen3max2601",
    "qwen3.5-397b-a17b":    "qwen3.5-397",
    "gemini-2.5-flash": "gemini-2.5 ",
    "deepseek-r1":   "deepseek-r1",
    "deepseek-chat": "dpseekchat ",
}
MAIN_SIMPLE_BASELINES = {
    "standard_prompt", "cot", "self_ask",
    "standard_rag", "ircot", "cok", "tog2",
}

# ── ablation experiment config ────────────────────────────────────────────────

ABLA_MODEL_ORDER = [
    "new_modl_wo_semlist_atomr",
    "new_modl_wo_semlist_deepsieve",
    "new_modl_wo_decomp",
    "new_modl_wo_fallback",
    "new_modl_wo_semlist_plan",
    "new_modl_wo_opplan",
]
ABLA_MODEL_SHORT = {
    "new_modl_wo_semlist_atomr":     "wo_sl_atomr",
    "new_modl_wo_semlist_deepsieve": "wo_sl_dpsv ",
    "new_modl_wo_decomp":            "wo_decomp  ",
    "new_modl_wo_fallback":          "wo_fallback",
    "new_modl_wo_semlist_plan":      "wo_sl_plan ",
    "new_modl_wo_opplan":            "wo_opplan  ",
}
ABLA_MODEL_DISPLAY = {
    "new_modl_wo_semlist_atomr":     "w/o SemList (AtomR)",
    "new_modl_wo_semlist_deepsieve": "w/o SemList (DeepSieve)",
    "new_modl_wo_decomp":            "w/o Decomposition",
    "new_modl_wo_fallback":          "w/o Fallback",
    "new_modl_wo_semlist_plan":      "w/o SemList Metadata in Plan",
    "new_modl_wo_opplan":            "w/o Operator Planning",
}
ABLA_LLM_ORDER = ["gpt-4o-mini", "gpt-4o", "qwen3-32b", "deepseek-r1", "deepseek-chat", "chatanywhere_deepseek-chat"]
ABLA_LLM_SHORT = {
    "gpt-4o-mini":                "gpt-4o-mini  ",
    "gpt-4o":                     "gpt-4o       ",
    "qwen3-32b":                  "qwen3-32b    ",
    "deepseek-r1":                "deepseek-r1  ",
    "deepseek-chat":              "dpseekchat   ",
    "chatanywhere_deepseek-chat": "ca-dpseekchat",
}

# ── ANSI colours ──────────────────────────────────────────────────────────────

RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
CYAN    = "\033[36m"
RED     = "\033[31m"
BLUE    = "\033[34m"
MAGENTA = "\033[35m"

def c(text, *codes):
    return "".join(codes) + str(text) + RESET

# ── shared helpers ────────────────────────────────────────────────────────────

def _dataset_size(dataset: str) -> int:
    p = DATASETS[dataset]["file"]
    if not p.exists():
        return 160
    with p.open("rb") as f:
        return sum(1 for _ in f)

def _count_jsonl(p: Path) -> int:
    if not p.exists() or not p.is_file():
        return 0
    with p.open("rb") as f:
        return sum(1 for _ in f)

BAR_WIDTH = 14

def bar(done: int, total: int) -> str:
    if total == 0:
        return "[" + "-" * BAR_WIDTH + "]"
    frac = min(done / total, 1.0)
    filled = round(frac * BAR_WIDTH)
    if done >= total:
        inner = c("█" * BAR_WIDTH, GREEN)
    elif done == 0:
        inner = c("░" * BAR_WIDTH, DIM)
    else:
        inner = c("█" * filled, YELLOW) + c("░" * (BAR_WIDTH - filled), DIM)
    return "[" + inner + "]"

def status_icon(done: int, total: int) -> str:
    if done >= total > 0:
        return c("✓", GREEN, BOLD)
    if done == 0:
        return c("·", DIM)
    return c("~", YELLOW)

# ── main experiment helpers ───────────────────────────────────────────────────

def main_model_progress(dataset: str, llm: str, model: str) -> tuple[int, int]:
    d = RESULT_ROOT_MAIN / dataset / llm / model
    total = _dataset_size(dataset)
    if model in ("new_model", "hydrarag", "aop") or model in MAIN_SIMPLE_BASELINES:
        return _count_jsonl(d / "predictions.jsonl"), total
    if model == "atomr":
        pred = _count_jsonl(d / f"{dataset}_pred.jsonl")
        if dataset == "kg_table":
            trace = d / "trace"
            traces = len(list(trace.glob("idx_*"))) if trace.exists() else 0
            return max(pred, traces), total
        return pred, total
    if model == "deepsieve":
        files = list(d.glob("query_*_results.jsonl")) if d.exists() else []
        return len(files), total
    return 0, total

def main_eval_done(dataset: str, llm: str) -> bool:
    p = RESULT_ROOT_MAIN / dataset / llm / "eval_summary" / "summary.json"
    return p.exists() and p.is_file() and p.stat().st_size > 0

# ── ablation experiment helpers ───────────────────────────────────────────────

def abla_model_progress(dataset: str, llm: str, model: str) -> tuple[int, int]:
    d = RESULT_ROOT_ABLA / dataset / llm / model
    total = _dataset_size(dataset)
    return _count_jsonl(d / "predictions.jsonl"), total

def abla_eval_done(dataset: str, llm: str) -> bool:
    p = RESULT_ROOT_ABLA / dataset / llm / "eval_summary" / "summary.json"
    return p.exists() and p.is_file() and p.stat().st_size > 0

# ── render: main experiment ───────────────────────────────────────────────────

def render_main(llm_filter: list[str] | None = None) -> str:
    lines: list[str] = []
    W = 88
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    active_llms = llm_filter if llm_filter else MAIN_LLM_ORDER

    lines.append(c("═" * W, CYAN))
    title = f"  Main Experiment Progress   {now}"
    if llm_filter:
        title += f"   [LLM: {', '.join(llm_filter)}]"
    lines.append(c(title, CYAN, BOLD))
    lines.append(c("═" * W, CYAN))

    for dataset in DATASET_ORDER:
        total = _dataset_size(dataset)
        label = DATASETS[dataset]["label"]

        all_cells: dict[tuple[str, str], tuple[int, int]] = {}
        for llm in active_llms:
            for model in MAIN_MODEL_ORDER:
                all_cells[(llm, model)] = main_model_progress(dataset, llm, model)

        finished_runs = sum(1 for (d, t) in all_cells.values() if d >= t > 0)
        total_runs    = len(active_llms) * len(MAIN_MODEL_ORDER)
        evals_done    = sum(1 for llm in active_llms if main_eval_done(dataset, llm))

        lines.append("")
        lines.append(c(f"  ▌ {label}  ({total} questions)", BOLD, MAGENTA))
        lines.append(c(f"    Inference: {finished_runs}/{total_runs} model-runs done  │  "
                       f"Eval: {evals_done}/{len(MAIN_LLM_ORDER)} LLMs evaluated", DIM))
        lines.append(c("  " + "─" * (W - 2), DIM))

        col_w    = 11
        llm_col  = 13
        header_cells = "  ".join(MAIN_MODEL_SHORT[m] for m in MAIN_MODEL_ORDER)
        lines.append(f"  {' ' * llm_col}  {c(header_cells, BOLD)}")

        for llm in active_llms:
            llm_label = MAIN_LLM_SHORT.get(llm, llm[:13])
            cells = []
            for model in MAIN_MODEL_ORDER:
                d, t = all_cells[(llm, model)]
                icon = status_icon(d, t)
                num  = f"{d:3d}/{t}"
                cells.append(f"{icon} {num}")
            row = "  ".join(cells)
            ev = c(" [eval✓]", GREEN) if main_eval_done(dataset, llm) else c(" [eval·]", DIM)
            lines.append(f"  {c(llm_label, BOLD)}  {row}{ev}")

        lines.append("")
        label_llms = "all LLMs" if not llm_filter else ", ".join(llm_filter)
        lines.append(f"  {c(f'Per-model progress ({label_llms} combined):', DIM)}")
        for model in MAIN_MODEL_ORDER:
            total_done = sum(all_cells[(llm, model)][0] for llm in active_llms)
            total_max  = total * len(active_llms)
            pbar = bar(total_done, total_max)
            pct  = total_done / total_max * 100 if total_max else 0
            short = MAIN_MODEL_SHORT[model]
            done_llms = sum(1 for llm in active_llms
                            if all_cells[(llm, model)][0] >= total > 0)
            lines.append(f"    {c(short, BOLD)} {pbar} {total_done:5d}/{total_max}"
                         f"  {pct:5.1f}%  ({done_llms}/{len(active_llms)} LLMs done)")

    lines.append("")
    lines.append(c("═" * W, CYAN))
    return "\n".join(lines)

# ── render: ablation experiment ───────────────────────────────────────────────

def render_abla(llm_filter: list[str] | None = None) -> str:
    lines: list[str] = []
    W = 88
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    active_llms = llm_filter if llm_filter else ABLA_LLM_ORDER

    lines.append(c("═" * W, YELLOW))
    title = f"  Ablation Experiment Progress   {now}"
    if llm_filter:
        title += f"   [LLM: {', '.join(llm_filter)}]"
    lines.append(c(title, YELLOW, BOLD))
    lines.append(c("═" * W, YELLOW))

    for dataset in DATASET_ORDER:
        total = _dataset_size(dataset)
        label = DATASETS[dataset]["label"]

        all_cells: dict[tuple[str, str], tuple[int, int]] = {}
        for llm in active_llms:
            for model in ABLA_MODEL_ORDER:
                all_cells[(llm, model)] = abla_model_progress(dataset, llm, model)

        finished_runs = sum(1 for (d, t) in all_cells.values() if d >= t > 0)
        total_runs    = len(active_llms) * len(ABLA_MODEL_ORDER)
        evals_done    = sum(1 for llm in active_llms if abla_eval_done(dataset, llm))

        lines.append("")
        lines.append(c(f"  ▌ {label}  ({total} questions)", BOLD, MAGENTA))
        lines.append(c(f"    Inference: {finished_runs}/{total_runs} model-runs done  │  "
                       f"Eval: {evals_done}/{len(active_llms)} LLMs evaluated", DIM))
        lines.append(c("  " + "─" * (W - 2), DIM))

        llm_col = 15
        header_cells = "  ".join(ABLA_MODEL_SHORT[m] for m in ABLA_MODEL_ORDER)
        lines.append(f"  {' ' * llm_col}  {c(header_cells, BOLD)}")

        for llm in active_llms:
            llm_label = ABLA_LLM_SHORT.get(llm, llm[:15])
            cells = []
            for model in ABLA_MODEL_ORDER:
                d, t = all_cells[(llm, model)]
                icon = status_icon(d, t)
                num  = f"{d:3d}/{t}"
                cells.append(f"{icon} {num}")
            row = "  ".join(cells)
            ev = c(" [eval✓]", GREEN) if abla_eval_done(dataset, llm) else c(" [eval·]", DIM)
            lines.append(f"  {c(llm_label, BOLD)}  {row}{ev}")

        lines.append("")
        label_llms = "all LLMs" if not llm_filter else ", ".join(llm_filter)
        lines.append(f"  {c(f'Per-model progress ({label_llms} combined):', DIM)}")
        for model in ABLA_MODEL_ORDER:
            total_done = sum(all_cells[(llm, model)][0] for llm in active_llms)
            total_max  = total * len(active_llms)
            pbar = bar(total_done, total_max)
            pct  = total_done / total_max * 100 if total_max else 0
            short = ABLA_MODEL_SHORT[model]
            disp  = ABLA_MODEL_DISPLAY[model]
            done_llms = sum(1 for llm in active_llms
                            if all_cells[(llm, model)][0] >= total > 0)
            lines.append(f"    {c(short, BOLD)} {pbar} {total_done:5d}/{total_max}"
                         f"  {pct:5.1f}%  ({done_llms}/{len(active_llms)} LLMs done)"
                         f"  {c(disp, DIM)}")

    lines.append("")
    lines.append(c("═" * W, YELLOW))
    return "\n".join(lines)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    mode       = "main"
    watch      = False
    interval   = 30
    llm_filter: list[str] | None = None
    args = sys.argv[1:]

    i = 0
    while i < len(args):
        if args[i] == "--mode":
            if i + 1 < len(args):
                i += 1
                mode = args[i]
                if mode not in ("main", "abla"):
                    print(f"Unknown mode: {mode}  (use 'main' or 'abla')")
                    sys.exit(1)
        elif args[i] == "--watch":
            watch = True
            if i + 1 < len(args):
                try:
                    interval = int(args[i + 1])
                    i += 1
                except ValueError:
                    pass
        elif args[i] == "--llm":
            if i + 1 < len(args):
                i += 1
                llm_filter = [name.strip() for name in args[i].split(",")]
                valid = ABLA_LLM_ORDER if mode == "abla" else MAIN_LLM_ORDER
                unknown = [l for l in llm_filter if l not in valid]
                if unknown:
                    print(f"Unknown LLM(s): {unknown}")
                    print(f"Available: {valid}")
                    sys.exit(1)
        i += 1

    render = render_abla if mode == "abla" else render_main

    if watch:
        while True:
            print("\033[H\033[2J", end="", flush=True)
            print(render(llm_filter), flush=True)
            print(f"\n  (auto-refresh every {interval}s — Ctrl+C to quit)", flush=True)
            time.sleep(interval)
    else:
        print(render(llm_filter))


if __name__ == "__main__":
    main()
