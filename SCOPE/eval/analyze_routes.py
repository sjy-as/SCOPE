"""
Compare routing methods (graph / atomr / deepsieve) against the gold
source labels in one of three benchmarks: kg-table / kg-doc / table-doc.

Gold-label conventions per benchmark:
    kg-table:  type_k2t -> q1=KG,    q2=Table
               type_t2k -> q1=Table, q2=KG
    kg-doc:    type_k2d -> q1=KG,    q2=Doc
               type_d2k -> q1=Doc,   q2=KG
    table-doc: type_t2d -> q1=Table, q2=Doc
               type_d2t -> q1=Doc,   q2=Table

Outputs (under <result-root>/<mode>):
    metrics.json     -- per-method accuracy table
    per_query.csv    -- index | type | gold(q1,q2) | per-mode route predictions and correctness
    fig_<a2b>_acc.png / fig_<b2a>_acc.png -- per-type bar charts
    fig_overall_acc.png -- overall bar chart across both types

Usage:
    cd SCOPE_code
    python eval/analyze_routes.py --mode kg-table
    python eval/analyze_routes.py --mode kg-doc
    python eval/analyze_routes.py --mode table-doc
    python eval/analyze_routes.py --mode all --base-dir eval/result/route
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "SCOPE"

# Per-mode configuration. `deepsieve_dir` accommodates the historical
# `deepsieve` typo under table_doc/.
MODE_CONFIG = {
    "kg-table": {
        "gold_path": ROOT / "qa_bench/kg-table-1147.jsonl",
        "result_subdir": "kg_table",
        "deepsieve_dir": "deepsieve",
        "types": ("type_k2t", "type_t2k"),
        "type_titles": {"type_k2t": "k2t", "type_t2k": "t2k"},
    },
    "kg-doc": {
        "gold_path": ROOT / "qa_bench/kg-doc-1154.jsonl",
        "result_subdir": "kg_doc",
        "deepsieve_dir": "deepsieve",
        "types": ("type_k2d", "type_d2k"),
        "type_titles": {"type_k2d": "k2d", "type_d2k": "d2k"},
    },
    "table-doc": {
        "gold_path": ROOT / "qa_bench/table-doc-1120.jsonl",
        "result_subdir": "table_doc",
        "deepsieve_dir": "deepsieve",
        "types": ("type_t2d", "type_d2t"),
        "type_titles": {"type_t2d": "t2d", "type_d2t": "d2t"},
    },
}


def load_jsonl(p: Path):
    rows = []
    with p.open() as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def norm(s):
    v = (s or "").strip().lower()
    if v == "document":
        v = "doc"
    return v


def run(mode: str, base_dir: Path | None = None, out_dir: Path | None = None):
    cfg = MODE_CONFIG[mode]
    gold_path = cfg["gold_path"]
    if base_dir is None:
        base_dir = ROOT / "eval/result/route" / cfg["result_subdir"]
    if out_dir is None:
        out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # deepsieve dir: prefer canonical "deepsieve"; fall back to the
    # historical "deepsieve" typo (only present in some older table_doc runs).
    ds_canonical = base_dir / "deepsieve"
    ds_legacy = base_dir / cfg["deepsieve_dir"]
    ds_dir = ds_canonical if (ds_canonical / "routes.jsonl").exists() else ds_legacy

    result_dirs = {
        "graph":       base_dir / "graph",
        "atomr":       base_dir / "atomr",
        "deepsieve": ds_dir,
    }
    modes_present = [m for m, p in result_dirs.items() if (p / "routes.jsonl").exists()]

    print(f"[analyze_routes] mode={mode}  comparing methods: {modes_present}")
    if not modes_present:
        print(f"[analyze_routes] no routes.jsonl found under {base_dir}; nothing to do.")
        return {}

    # -------------------------------------------------------------------
    # Load gold + predictions
    # -------------------------------------------------------------------
    gold_rows = load_jsonl(gold_path)
    gold = {
        r["index"]: {
            "type": norm(r.get("type")),
            "q1": norm(r["q1"]["source"]["type"]),
            "q2": norm(r["q2"]["source"]["type"]),
        }
        for r in gold_rows
    }

    preds = {}
    for m in modes_present:
        rows = load_jsonl(result_dirs[m] / "routes.jsonl")
        preds[m] = {r["index"]: r for r in rows}

    # -------------------------------------------------------------------
    # Compute per-method metrics
    # -------------------------------------------------------------------
    per_query = []
    for idx, g in sorted(gold.items()):
        row = {"index": idx, "gold_type": g["type"], "gold_q1": g["q1"], "gold_q2": g["q2"]}
        for m in modes_present:
            r = preds[m].get(idx)
            if not r:
                row[f"{m}_r1_primary"] = ""
                row[f"{m}_r2_primary"] = ""
                row[f"{m}_r1_correct"] = 0
                row[f"{m}_r2_correct"] = 0
                row[f"{m}_joint"] = 0
                continue
            r1 = r.get("route1") or {}
            r2 = r.get("route2") or {}
            p1 = norm(r1.get("primary_source"))
            p2 = norm(r2.get("primary_source"))
            row[f"{m}_r1_primary"] = p1
            row[f"{m}_r2_primary"] = p2
            row[f"{m}_r1_correct"] = int(p1 == g["q1"])
            row[f"{m}_r2_correct"] = int(p2 == g["q2"])
            row[f"{m}_joint"] = int(p1 == g["q1"] and p2 == g["q2"])
        per_query.append(row)

    metrics = {}
    for m in modes_present:
        n = len(per_query)
        r1_acc = sum(r[f"{m}_r1_correct"] for r in per_query) / n if n else 0.0
        r2_acc = sum(r[f"{m}_r2_correct"] for r in per_query) / n if n else 0.0
        metrics[m] = {
            "n": n,
            "r1_primary_acc": r1_acc,
            "r2_primary_acc": r2_acc,
            "joint_primary_acc": sum(r[f"{m}_joint"] for r in per_query) / n if n else 0.0,
            "avg_primary_acc": (r1_acc + r2_acc) / 2 if n else 0.0,
        }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    csv_cols = ["index", "gold_type", "gold_q1", "gold_q2"]
    for m in modes_present:
        csv_cols += [
            f"{m}_r1_primary", f"{m}_r1_correct",
            f"{m}_r2_primary", f"{m}_r2_correct",
            f"{m}_joint",
        ]
    with (out_dir / "per_query.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in per_query:
            w.writerow({k: r.get(k, "") for k in csv_cols})

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------
    plt.rcParams.update({"font.size": 11})

    COLORS = {"graph": "#4C72B0", "atomr": "#DD8452", "deepsieve": "#55A868"}
    LABEL  = {"graph": "Graph",   "atomr": "AtomR",   "deepsieve": "DeepSieve"}

    def annotate(ax, rects, fmt="{:.1%}"):
        for r in rects:
            h = r.get_height()
            ax.annotate(fmt.format(h),
                        xy=(r.get_x() + r.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

    def plot_by_type(type_name: str, out_name: str):
        subset = [r for r in per_query if r["gold_type"] == type_name]
        if not subset:
            return

        fig, ax = plt.subplots(figsize=(max(9, 2.2 + 1.6 * len(modes_present)), 5))
        groups = ["SQ1", "SQ2", "Mean", "Joint"]
        x = np.arange(len(groups))
        width = 0.8 / max(1, len(modes_present))
        offsets = (np.arange(len(modes_present)) - (len(modes_present) - 1) / 2.0) * width

        for i, m in enumerate(modes_present):
            q1_acc = sum(r[f"{m}_r1_correct"] for r in subset) / len(subset)
            q2_acc = sum(r[f"{m}_r2_correct"] for r in subset) / len(subset)
            joint_acc = sum(r[f"{m}_joint"] for r in subset) / len(subset)
            avg_acc = (q1_acc + q2_acc) / 2
            rects = ax.bar(x + offsets[i], [q1_acc, q2_acc, avg_acc, joint_acc], width,
                           color=COLORS[m], label=LABEL[m])
            annotate(ax, rects)

        title_name = cfg["type_titles"].get(type_name, type_name)
        ax.set_xticks(x)
        ax.set_xticklabels(["SQ1", "SQ2", "Mean (SQ1,SQ2)", "Joint (SQ1 & SQ2)"])
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.10)
        ax.set_title(f"Routing accuracy on {title_name} (n={len(subset)})\n"
                     "Primary source vs. gold source label")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / out_name, dpi=160)
        plt.close(fig)

    for t in cfg["types"]:
        plot_by_type(t, f"fig_{cfg['type_titles'][t]}_acc.png")

    # Overall plot across both types
    fig, ax = plt.subplots(figsize=(max(9, 2.2 + 1.6 * len(modes_present)), 5))
    groups = ["SQ1", "SQ2", "Mean", "Joint"]
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(modes_present))
    offsets = (np.arange(len(modes_present)) - (len(modes_present) - 1) / 2.0) * width
    for i, m in enumerate(modes_present):
        q1_acc = sum(r[f"{m}_r1_correct"] for r in per_query) / len(per_query) if per_query else 0.0
        q2_acc = sum(r[f"{m}_r2_correct"] for r in per_query) / len(per_query) if per_query else 0.0
        joint_acc = sum(r[f"{m}_joint"] for r in per_query) / len(per_query) if per_query else 0.0
        avg_acc = (q1_acc + q2_acc) / 2
        rects = ax.bar(x + offsets[i], [q1_acc, q2_acc, avg_acc, joint_acc], width,
                       color=COLORS[m], label=LABEL[m])
        annotate(ax, rects)
    ax.set_xticks(x)
    ax.set_xticklabels(["SQ1", "SQ2", "Mean (SQ1,SQ2)", "Joint (SQ1 & SQ2)"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.10)
    ax.set_title(f"Overall routing accuracy on {mode} (n={len(per_query)})\n"
                 "Primary source vs. gold source label")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_overall_acc.png", dpi=160)
    plt.close(fig)

    print("=" * 70)
    print(f"Outputs written to {out_dir}")
    print("=" * 70)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode_choices = list(MODE_CONFIG.keys()) + ["all"]
    parser.add_argument("--mode", choices=mode_choices, required=True,
                        help="Which benchmark to analyze: kg-table | kg-doc | table-doc | all")
    parser.add_argument("--base-dir", default=None,
                        help="Override result base dir (where <mode>/routes.jsonl live). "
                             "Default: ROOT/eval/result/route/<dataset_subdir>.")
    parser.add_argument("--out-dir", default=None,
                        help="Override output dir (metrics/figures). Default: same as --base-dir.")
    args = parser.parse_args()
    if args.mode == "all":
        base_root = Path(args.base_dir).resolve() if args.base_dir else ROOT / "eval/result/route"
        out_root = Path(args.out_dir).resolve() if args.out_dir else base_root
        all_metrics = {}
        for mode, cfg in MODE_CONFIG.items():
            subdir = cfg["result_subdir"]
            all_metrics[mode] = run(
                mode,
                base_dir=base_root / subdir,
                out_dir=out_root / subdir,
            )
        out_root.mkdir(parents=True, exist_ok=True)
        with (out_root / "metrics_all.json").open("w") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print("=" * 70)
        print(f'Combined metrics written to {out_root / "metrics_all.json"}')
        print("=" * 70)
        return

    base = Path(args.base_dir).resolve() if args.base_dir else None
    out = Path(args.out_dir).resolve() if args.out_dir else None
    run(args.mode, base_dir=base, out_dir=out)


if __name__ == "__main__":
    main()
