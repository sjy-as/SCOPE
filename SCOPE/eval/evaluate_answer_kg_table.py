'''
Evaluate KG-Table answers across four models.

Layouts (kg_table — same as kg_doc):
  - SCOPE : <dir>/predictions.jsonl                (has sq1/sq2/final)
  - hyprarag  : <dir>/predictions.jsonl                (final only; sq1 skipped)
  - atomr     : <dir>/kgdoc_pred.jsonl                  (one `predicted` string; sq1 skipped)
  - deepserive: <dir>/query_<N>_results.jsonl          (q1 + final_answer; N != gold index)

hyprarag and atomr only participate in sq2 evaluation.
deepserive query_<N> does NOT correspond to gold idx N, so we match by question text.

Run:
python SCOPE_code/eval/evaluate_answer_kg_table.py \
  --gold SCOPE_code/SCOPE/qa_bench/kg-table-160.jsonl \
  --new-model SCOPE_code/eval/result/kg_table/gpt-4o-mini/SCOPE/predictions.jsonl \
  --hyprarag SCOPE_code/eval/result/kg_table/gpt-4o-mini/hydrarag/predictions.jsonl \
  --atomr SCOPE_code/eval/result/kg_table/gpt-4o-mini/atomr/kgdoc_pred.jsonl \
  --deepserice-dir SCOPE_code/eval/result/kg_table/gpt-4o-mini/deepsieve \
  --out-dir SCOPE_code/eval/result/kg_table/gpt-4o-mini/eval_summary \
  --api-key "$JUDGE_LLM_API_KEY" \
  --max-workers 8
'''
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# import matplotlib.pyplot as plt  # plotting disabled
# import numpy as np               # plotting disabled
# from matplotlib import font_manager  # plotting disabled

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

NEW_MODEL_ROOT = Path(os.getenv("SCOPE_NEW_MODEL_DIR", str(Path(__file__).resolve().parents[1] / "SCOPE"))).resolve()
if str(NEW_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEW_MODEL_ROOT))
import pipeline as P
from run import _eval_one, _as_list


# ---------- per-source participation ----------
# 7 simple LLM/RAG baselines from SCOPE_code/baseline/<Method>/.
# They emit predictions.jsonl in SCOPE's schema (sq1/sq2/final), so the
# reader path is the same as SCOPE. Only SelfAsk produces meaningful sq1.
SIMPLE_BASELINES = {"standard_prompt", "cot", "self_ask",
                    "standard_rag", "ircot", "cok", "tog2"}
SIMPLE_BASELINE_FLAG_TO_SLUG = {
    "standard-prompt": "standard_prompt",
    "cot":             "cot",
    "self-ask":        "self_ask",
    "standard-rag":    "standard_rag",
    "ircot":           "ircot",
    "cok":             "cok",
    "tog2":            "tog2",
}

SOURCE_STAGES: Dict[str, Tuple[str, ...]] = {
    "SCOPE": ("sq1", "sq2"),
    "deepserive": ("sq1", "sq2"),
    "atomr": ("sq2",),
    "hyprarag": ("sq2",),
    # Simple baselines: all have sq2/final. Self-Ask also exposes sq1
    # (decomposes follow-ups internally); the other 6 leave sq1 empty.
    "self_ask":        ("sq1", "sq2"),
    "standard_prompt": ("sq2",),
    "cot":             ("sq2",),
    "standard_rag":    ("sq2",),
    "ircot":           ("sq2",),
    "cok":             ("sq2",),
    "tog2":            ("sq2",),
    # Ablations of SCOPE (same predictions.jsonl schema).
    "SCOPE_wo_semantic_directory_few-shot":     ("sq1", "sq2"),
    "SCOPE_wo_semantic_directory_summary": ("sq1", "sq2"),
    "SCOPE_wo_semantic_guide_on_planning":      ("sq1", "sq2"),
    "SCOPE_wo_decomposition":   ("sq2",),
    "SCOPE_wo_reflection":  ("sq1", "sq2"),
    "SCOPE_wo_consolidation": ("sq1", "sq2"),
    "SCOPE_wo_operator_planning":    ("sq1", "sq2"),
}

# Per-source verdict key for the "sq2" stage. Sources that produce a real
# second-hop answer are scored on judge["sq2"]; sources that only emit a
# single final answer (atomr/hyprarag/6 simple baselines) are scored on
# judge["final"] because their sq2_pred is empty.
SQ2_VERDICT_KEY: Dict[str, str] = {
    "SCOPE":                     "sq2",
    "deepserive":                    "sq2",
    "self_ask":                      "sq2",
    "SCOPE_wo_semantic_directory_few-shot":     "sq2",
    "SCOPE_wo_semantic_directory_summary": "sq2",
    "SCOPE_wo_semantic_guide_on_planning":      "sq2",
    "SCOPE_wo_decomposition":   "final",
    "SCOPE_wo_reflection":  "sq2",
    "SCOPE_wo_consolidation": "sq2",
    "SCOPE_wo_operator_planning":    "sq2",
    "atomr":           "final",
    "hyprarag":        "final",
    "standard_prompt": "final",
    "cot":             "final",
    "standard_rag":    "final",
    "ircot":           "final",
    "cok":             "final",
    "tog2":            "final",
}


# ---------- io helpers ----------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _norm_text(x: Any) -> str:
    return " ".join(str(x).strip().lower().split())


def _list_union(*lists: Sequence[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for xs in lists:
        for v in xs or []:
            s = str(v).strip()
            if not s:
                continue
            k = _norm_text(s)
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
    return out


# ---------- gold ----------
def _build_gold_maps(gold_path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    by_q: Dict[str, Dict[str, Any]] = {}
    for row in _load_jsonl(gold_path):
        idx = row.get("index")
        if isinstance(idx, int):
            by_idx[idx] = row
        q = (row.get("question") or "").strip()
        if q:
            by_q[_norm_text(q)] = row
    return by_idx, by_q


# ---------- readers: return (sq1_pred, sq2_pred, final_pred) ----------
def _read_SCOPE_row(row: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    return _as_list(row.get("sq1")), _as_list(row.get("sq2")), _as_list(row.get("final"))


def _read_hyprarag_row(row: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    # hyprarag stores its only answer under `final`; treat that as the sq2 prediction.
    final = _as_list(row.get("final"))
    if not final:
        final = _as_list(row.get("answer_entities"))
    return [], final, final


def _read_atomr_row(row: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    # atomr kg_table only emits one `predicted` string per question.
    p = row.get("predicted")
    p2 = _as_list(p)
    return [], p2, p2


def _read_deepserive_file(fp: Path) -> Tuple[List[str], List[str], List[str], str]:
    """Returns (sq1_pred, sq2_pred, final_pred, question)."""
    q = ""
    final_str: Optional[str] = None
    fallback_str: Optional[str] = None
    q1_answers: List[str] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("type")
            if t == "query_info":
                q = rec.get("query") or ""
            elif t == "final_answer":
                if rec.get("final_answer"):
                    final_str = str(rec["final_answer"])
                if rec.get("fallback_answer"):
                    fallback_str = str(rec["fallback_answer"])
            elif t == "execution_result":
                if rec.get("subquery_id") == "q1":
                    ans = rec.get("answer")
                    if ans:
                        q1_answers.append(str(ans))
    sq1 = _list_union(q1_answers)
    f_ans = final_str or fallback_str or ""
    sq2 = [f_ans] if f_ans else []
    return sq1, sq2, list(sq2), q


# ---------- matplotlib font (disabled) ----------
# CANDIDATE_FONTS = [
#     "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
#     "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
#     "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
# ]
# prop = None
# for fp in CANDIDATE_FONTS:
#     if os.path.exists(fp):
#         font_manager.fontManager.addfont(fp)
#         prop = font_manager.FontProperties(fname=fp)
#         plt.rcParams["font.family"] = prop.get_name()
#         break
# plt.rcParams["axes.unicode_minus"] = False


def _plt_kw() -> Dict[str, Any]:
    return {}  # plotting disabled; original: {"fontproperties": prop} if prop is not None else {}


VERDICTS = ["exact", "partial", "miss"]
STAGES = ["sq1", "sq2"]
# VCOLORS = {"exact": "#2a9d8f", "partial": "#e9c46a", "miss": "#e76f51"}  # plotting disabled


# ---------- per-record evaluation ----------
def _required_judge_keys(stages: Sequence[str]) -> List[Tuple[str, str]]:
    """Stages this source must have non-empty verdicts for. Returns list of
    (judge_field, verdict_key) pairs — e.g. ("sq2_judge", "final")."""
    req: List[Tuple[str, str]] = []
    if "sq1" in stages:
        req.append(("sq1_judge", "sq1"))
    if "sq2" in stages:
        req.append(("sq2_judge", "sq2"))
        req.append(("sq2_judge", "final"))
    return req


def _record_is_complete(rec: Optional[Dict[str, Any]], stages: Sequence[str]) -> bool:
    if not rec:
        return False
    for jfield, vkey in _required_judge_keys(stages):
        j = rec.get(jfield) or {}
        v = (j.get(vkey) or {}).get("verdict")
        if v not in {"exact", "partial", "miss"}:
            return False
    return True


def _evaluate_one_record(
    source_name: str,
    payload: Any,
    gold_idx: Dict[int, Dict[str, Any]],
    gold_q: Dict[str, Dict[str, Any]],
    llm_url: str,
    llm_model: str,
    api_key: str,
    existing_by_index: Optional[Dict[int, Dict[str, Any]]] = None,
    existing_by_q: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    stages = SOURCE_STAGES[source_name]

    if source_name == "SCOPE" or source_name in {
        "SCOPE_wo_semantic_directory_few-shot", "SCOPE_wo_semantic_directory_summary",
        "SCOPE_wo_semantic_guide_on_planning", "SCOPE_wo_decomposition", "SCOPE_wo_reflection", "SCOPE_wo_consolidation", "SCOPE_wo_operator_planning",
    } or source_name in SIMPLE_BASELINES:
        row = payload
        idx = row.get("index")
        gold = gold_idx.get(idx) if isinstance(idx, int) else None
        if gold is None:
            return None
        p1, p2, pf = _read_SCOPE_row(row)
        q = row.get("question") or gold.get("question") or ""
    elif source_name == "hyprarag":
        row = payload
        idx = row.get("index")
        gold = gold_idx.get(idx) if isinstance(idx, int) else None
        if gold is None:
            return None
        p1, p2, pf = _read_hyprarag_row(row)
        q = row.get("question") or gold.get("question") or ""
    elif source_name == "atomr":
        row = payload
        idx = row.get("index")
        gold = gold_idx.get(idx) if isinstance(idx, int) else None
        if gold is None:
            return None
        p1, p2, pf = _read_atomr_row(row)
        q = row.get("question") or gold.get("question") or ""
    elif source_name == "deepserive":
        fp: Path = payload
        p1, p2, pf, q = _read_deepserive_file(fp)
        gold = gold_q.get(_norm_text(q)) if q else None
        if gold is None:
            return None
        idx = gold.get("index")
    else:
        raise ValueError(f"Unknown source: {source_name}")

    # --- resume: reuse already-judged record if it has every required verdict ---
    cached: Optional[Dict[str, Any]] = None
    if existing_by_index is not None and isinstance(idx, int):
        cached = existing_by_index.get(idx)
    if cached is None and existing_by_q is not None and q:
        cached = existing_by_q.get(_norm_text(q))
    if _record_is_complete(cached, stages):
        return cached

    pred = {
        "sq1": p1 if "sq1" in stages else [],
        "sq2": p2 if "sq2" in stages else [],
        "final": pf if "sq2" in stages else [],
    }
    _, judge = _eval_one(q, pred, gold, llm_url, llm_model, api_key)

    sq1_judge = judge if "sq1" in stages else {}
    sq2_judge = judge if "sq2" in stages else {}
    return {
        "source": source_name,
        "index": idx,
        "question": q,
        "sq1_pred": p1,
        "sq2_pred": p2,
        "final_pred": pf,
        "sq1_judge": sq1_judge,
        "sq2_judge": sq2_judge,
        "stages": list(stages),
    }


def _list_source_inputs(source_name: str, source_path: Path) -> List[Any]:
    if source_name in {"SCOPE", "hyprarag", "atomr",
                       "SCOPE_wo_semantic_directory_few-shot", "SCOPE_wo_semantic_directory_summary",
                       "SCOPE_wo_semantic_guide_on_planning", "SCOPE_wo_decomposition", "SCOPE_wo_reflection", "SCOPE_wo_consolidation", "SCOPE_wo_operator_planning"} \
            or source_name in SIMPLE_BASELINES:
        return _load_jsonl(source_path)
    if source_name == "deepserive":
        return sorted(source_path.glob("query_*_results.jsonl"))
    raise ValueError(source_name)


def _build_existing_lookups(
    existing: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    by_q: Dict[str, Dict[str, Any]] = {}
    for r in (existing or []):
        if isinstance(r, dict):
            i = r.get("index")
            if isinstance(i, int):
                by_idx[i] = r
            q = r.get("question") or ""
            if q:
                by_q[_norm_text(q)] = r
    return by_idx, by_q


def _evaluate_source(
    source_name: str,
    source_path: Path,
    gold_idx: Dict[int, Dict[str, Any]],
    gold_q: Dict[str, Dict[str, Any]],
    llm_url: str,
    llm_model: str,
    api_key: str,
    max_workers: int = 8,
    existing_records: Optional[List[Dict[str, Any]]] = None,
    save_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """If save_path is given, the partial record list is flushed to disk after
    every completion, so Ctrl-C mid-source never loses already-judged samples."""
    inputs = _list_source_inputs(source_name, source_path)
    existing_by_idx, existing_by_q = _build_existing_lookups(existing_records)
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_evaluate_one_record, source_name, item, gold_idx, gold_q,
                      llm_url, llm_model, api_key, existing_by_idx, existing_by_q)
            for item in inputs
        ]
        try:
            for fut in tqdm(as_completed(futures), total=len(futures), desc=source_name, unit="qa"):
                rec = fut.result()
                if rec is not None:
                    records.append(rec)
                    if save_path is not None:
                        _write_json(save_path, records)
        except KeyboardInterrupt:
            print(f"\n[eval] Ctrl-C received; flushing {len(records)} records and aborting...", flush=True)
            if save_path is not None:
                _write_json(save_path, records)
            for f in futures:
                f.cancel()
            raise
    if save_path is not None:
        _write_json(save_path, records)
    return records


# ---------- aggregates ----------
def _verdict_counts(records: List[Dict[str, Any]], stage: str) -> Counter:
    """Only counts records that actually participated in this stage AND have a
    real LLM verdict. Records with empty judge (LLM failed even after retries)
    are skipped — they are reported separately as 'unjudged' so they don't
    silently inflate the miss bucket."""
    c = Counter({v: 0 for v in VERDICTS})
    for r in records:
        if stage not in r.get("stages", []):
            continue
        judge = r.get(f"{stage}_judge") or {}
        if not judge:
            continue  # unjudged — exclude from accuracy denominator
        vkey = SQ2_VERDICT_KEY.get(r.get("source", ""), stage) if stage == "sq2" else stage
        verdict = (judge.get(vkey) or {}).get("verdict")
        if verdict not in c:
            continue
        c[verdict] += 1
    return c


def _stage_acc(c: Counter) -> Tuple[float, float]:
    total = sum(c.values()) or 1
    return c["exact"] / total, (c["exact"] + c["partial"]) / total


# ---------- plotting ----------
def _plot_stage_summary(all_results: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    return  # plotting disabled — remove this line to re-enable
    out_dir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        models = [m for m in all_results if stage in SOURCE_STAGES.get(m, ())]
        if not models:
            continue
        x = np.arange(len(models))
        width = 0.35
        strict, loose = [], []
        for model in models:
            c = _verdict_counts(all_results[model], stage)
            s, l = _stage_acc(c)
            strict.append(s); loose.append(l)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width / 2, strict, width, label="Strict (exact)", color="#2a9d8f")
        ax.bar(x + width / 2, loose, width, label="Loose (exact+partial)", color="#e9c46a")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Accuracy", **_plt_kw())
        ax.set_title(f"KG-Table {stage.upper()} comparison")
        for i, v in enumerate(strict):
            ax.text(i - width / 2, v + 0.02, f"{v:.1%}", ha="center")
        for i, v in enumerate(loose):
            ax.text(i + width / 2, v + 0.02, f"{v:.1%}", ha="center")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"compare_{stage}.png", dpi=200)
        plt.close(fig)


def _plot_answer_distribution(all_results: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    return  # plotting disabled — remove this line to re-enable
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(all_results.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        model_stages = [s for s in STAGES if s in SOURCE_STAGES.get(model, ())]
        bottoms = np.zeros(len(model_stages))
        for verdict in VERDICTS:
            vals = []
            for stage in model_stages:
                c = _verdict_counts(all_results[model], stage)
                total = sum(c.values()) or 1
                vals.append(c[verdict] / total)
            ax.bar(model_stages, vals, bottom=bottoms, label=verdict, color=VCOLORS[verdict])
            bottoms += np.array(vals)
        for i, stage in enumerate(model_stages):
            c = _verdict_counts(all_results[model], stage)
            total = sum(c.values()) or 1
            y = 0.0
            for verdict in VERDICTS:
                frac = c[verdict] / total
                if frac > 0.06:
                    ax.text(i, y + frac / 2, f"{frac:.1%}", ha="center", va="center",
                            color="white", fontsize=11, fontweight="bold")
                y += frac
        ax.set_title(f"{model} verdict proportion")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Proportion", **_plt_kw())
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "answer_distribution.png", dpi=200)
    plt.close(fig)


# ---------- main ----------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate KG-Table answers for NewModel / AtomR / DeepSerive / HyprAG")
    parser.add_argument("--gold", required=True, help="Gold jsonl file (kg-table-160.jsonl)")
    parser.add_argument("--new-model", default=None, help="Optional: path to SCOPE/predictions.jsonl")
    parser.add_argument("--hyprarag",  default=None, help="Optional: path to hyprarag/predictions.jsonl")
    parser.add_argument("--atomr",     default=None, help="Optional: path to atomr/kgdoc_pred.jsonl")
    parser.add_argument("--deepserice-dir", default=None, help="Optional: dir containing deepserive query_*_results.jsonl files")
    parser.add_argument("--wo-semantic-directory-few-shot", default=None,
                        help="Optional: path to SCOPE_wo_semantic_directory_few-shot/predictions.jsonl (ablation).")
    parser.add_argument("--wo-semantic-directory-summary", default=None,
                        help="Optional: path to SCOPE_wo_semantic_directory_summary/predictions.jsonl (ablation).")
    parser.add_argument("--wo-decomp", default=None,
                        help="Optional: path to SCOPE_wo_decomposition/predictions.jsonl (ablation).")
    parser.add_argument("--wo-reflection", default=None,
                        help="Optional: path to SCOPE_wo_reflection/predictions.jsonl (ablation).")
    parser.add_argument("--wo-consolidation", default=None,
                        help="Optional: path to SCOPE_wo_consolidation/predictions.jsonl (ablation).")
    parser.add_argument("--wo-semantic-guide-on-planning", default=None,
                        help="Optional: path to SCOPE_wo_semantic_guide_on_planning/predictions.jsonl (ablation).")
    parser.add_argument("--wo-opplan", default=None,
                        help="Optional: path to SCOPE_wo_operator_planning/predictions.jsonl (ablation).")
    # 7 simple LLM/RAG baselines (all optional; predictions.jsonl in SCOPE schema).
    parser.add_argument("--standard-prompt", default=None,
                        help="Optional: path to StandardPrompt/predictions.jsonl")
    parser.add_argument("--cot", default=None,
                        help="Optional: path to CoT/predictions.jsonl")
    parser.add_argument("--self-ask", default=None,
                        help="Optional: path to SelfAsk/predictions.jsonl")
    parser.add_argument("--standard-rag", default=None,
                        help="Optional: path to StandardRAG/predictions.jsonl")
    parser.add_argument("--ircot", default=None,
                        help="Optional: path to IRCoT/predictions.jsonl")
    parser.add_argument("--cok", default=None,
                        help="Optional: path to CoK/predictions.jsonl")
    parser.add_argument("--tog2", default=None,
                        help="Optional: path to ToG2/predictions.jsonl")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "result" / "manual" / "kg_table"),
                        help="Where to write summary and plots")
    parser.add_argument("--llm-url", default=os.getenv("JUDGE_LLM_BASE_URL", os.getenv("LLM_BASE_URL", "")))
    parser.add_argument("--llm-model", default=os.getenv("JUDGE_LLM_MODEL", os.getenv("LLM_MODEL", "deepseek-chat")))
    parser.add_argument("--api-key", default=os.getenv("JUDGE_LLM_API_KEY", os.getenv("LLM_API_KEY", "")))
    parser.add_argument("--max-workers", type=int, default=8, help="Concurrent LLM requests")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing records_<name>.json under --out-dir and re-judge every sample from scratch.")
    parser.add_argument("--max-judge-passes", type=int, default=10,
                        help="Outer retry loop: re-run the LLM judge across all sources until "
                             "every source has 0 unjudged records, capped at this many passes. "
                             "Final summary/plots run only after the loop exits.")
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("[eval] --api-key (or LLM_API_KEY env) is required")

    P.LLM_BASE_URL = args.llm_url
    P.LLM_MODEL = args.llm_model
    P.LLM_API_KEY = args.api_key

    gold_path = Path(args.gold).resolve()
    gold_idx, gold_q = _build_gold_maps(gold_path)

    sources: Dict[str, Path] = {}
    if args.SCOPE:
        sources["SCOPE"] = Path(args.SCOPE).resolve()
    if args.deepserice_dir:
        sources["deepserive"] = Path(args.deepserice_dir).resolve()
    if args.atomr:
        sources["atomr"] = Path(args.atomr).resolve()
    if args.hyprarag:
        sources["hyprarag"] = Path(args.hyprarag).resolve()
    if args.wo_semantic_directory_few_shot:
        sources["SCOPE_wo_semantic_directory_few-shot"] = Path(args.wo_semantic_directory_few_shot).resolve()
    if args.wo_semantic_directory_summary:
        sources["SCOPE_wo_semantic_directory_summary"] = Path(args.wo_semantic_directory_summary).resolve()
    if args.wo_decomp:
        sources["SCOPE_wo_decomposition"] = Path(args.wo_decomp).resolve()
    if args.wo_semantic_guide_on_planning:
        sources["SCOPE_wo_semantic_guide_on_planning"] = Path(args.wo_semantic_guide_on_planning).resolve()
    if args.wo_reflection:
        sources["SCOPE_wo_reflection"] = Path(args.wo_reflection).resolve()
    if args.wo_consolidation:
        sources["SCOPE_wo_consolidation"] = Path(args.wo_consolidation).resolve()
    if args.wo_opplan:
        sources["SCOPE_wo_operator_planning"] = Path(args.wo_opplan).resolve()
    # Optional simple baselines — register only those for which a path was given.
    for _flag, _slug in SIMPLE_BASELINE_FLAG_TO_SLUG.items():
        _val = getattr(args, _flag.replace("-", "_"))
        if _val:
            sources[_slug] = Path(_val).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    max_passes = max(1, args.max_judge_passes)

    for pass_num in range(1, max_passes + 1):
        if pass_num > 1:
            print(f"\n[eval] ====== retry pass {pass_num}/{max_passes} ======", flush=True)
        pass_unjudged: Dict[str, int] = {}
        for name, path in sources.items():
            record_file = out_dir / f"records_{name}.json"
            existing_records: Optional[List[Dict[str, Any]]] = None
            if not args.no_resume and record_file.exists():
                try:
                    with record_file.open("r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, list):
                        existing_records = loaded
                        reusable = sum(1 for r in loaded if _record_is_complete(r, SOURCE_STAGES[name]))
                        print(f"[eval] resume {name}: {reusable}/{len(loaded)} already fully judged in {record_file}",
                              flush=True)
                except Exception as e:
                    print(f"[eval] failed to load {record_file}: {e}; will re-judge from scratch", flush=True)

            print(f"[eval] evaluating {name} (pass {pass_num}) ...", flush=True)
            records = _evaluate_source(name, path, gold_idx, gold_q,
                                       args.llm_url, args.llm_model, args.api_key,
                                       max_workers=args.max_workers,
                                       existing_records=existing_records,
                                       save_path=record_file)
            all_results[name] = records

            src_unj = 0
            for stage in STAGES:
                if stage not in SOURCE_STAGES[name]:
                    continue
                u = sum(1 for r in records
                        if stage in r.get("stages", []) and not (r.get(f"{stage}_judge") or {}))
                src_unj = max(src_unj, u)
            pass_unjudged[name] = src_unj

        total_unj = sum(pass_unjudged.values())
        if total_unj == 0:
            if pass_num > 1:
                print(f"[eval] all sources fully judged after {pass_num} passes", flush=True)
            break
        leftover = ", ".join(f"{n}={u}" for n, u in pass_unjudged.items() if u)
        print(f"[eval] end of pass {pass_num}: total unjudged={total_unj} ({leftover})", flush=True)
        if pass_num == max_passes:
            print(f"[eval] WARNING: hit --max-judge-passes={max_passes}; "
                  f"{total_unj} records remain unjudged. Re-run to continue.", flush=True)

    summaries: Dict[str, Any] = {}
    for name, records in all_results.items():
        unjudged_total = 0
        entry: Dict[str, Any] = {"n": len(records), "stages": list(SOURCE_STAGES[name])}
        for stage in STAGES:
            if stage in SOURCE_STAGES[name]:
                c = _verdict_counts(records, stage)
                strict, loose = _stage_acc(c)
                u = sum(1 for r in records
                        if stage in r.get("stages", []) and not (r.get(f"{stage}_judge") or {}))
                unjudged_total = max(unjudged_total, u)
                entry[stage] = {
                    "exact": c["exact"], "partial": c["partial"], "miss": c["miss"],
                    "strict": strict, "loose": loose, "unjudged": u,
                }
            else:
                entry[stage] = None
        entry["unjudged"] = unjudged_total
        summaries[name] = entry
        if unjudged_total:
            print(f"[eval] WARNING {name}: {unjudged_total} records remain unjudged.", flush=True)

    # plot_dir = out_dir / "plots"        # plotting disabled
    # _plot_stage_summary(all_results, plot_dir)
    # _plot_answer_distribution(all_results, plot_dir)

    report = {
        "gold": str(gold_path),
        "summaries": summaries,
        "llm": {"url": args.llm_url, "model": args.llm_model},
    }
    _write_json(out_dir / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    # print(f"plots   -> {out_dir / 'plots'}")  # plotting disabled
    print(f"summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
