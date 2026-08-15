#!/usr/bin/env python3
"""Run DDD_order source-pair order-ablation experiments.

Execution pattern:
  - rounds are serial: round_1 -> round_2 -> round_3
  - within each round, datasets run in parallel: kg-table, kg-doc, table-doc
  - each dataset run invokes new_model/run.py with --workers N
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "new_model"
RUN_PY = MODEL_DIR / "run.py"
SAMPLES_DIR = ROOT / "samples"

DATASET_KB: Dict[str, str] = {
    "kg-table": "kg,table",
    "kg-doc": "kg,doc",
    "table-doc": "table,doc",
}


def _parse_csv(raw: str, allowed: List[str], label: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [x for x in vals if x not in allowed]
    if bad:
        raise SystemExit(f"Unknown {label}: {bad}; allowed={allowed}")
    return vals


def _api_key_for_dataset(args: argparse.Namespace, dataset: str) -> str:
    if args.api_key_map:
        return args.api_key_map[dataset]
    return args.api_key


def _build_command(args: argparse.Namespace, dataset: str, round_id: int, output_dir: Path) -> List[str]:
    sample = SAMPLES_DIR / dataset / f"round_{round_id}.jsonl"
    if not sample.exists():
        raise FileNotFoundError(f"missing sample: {sample}")

    cmd = [
        sys.executable,
        str(RUN_PY),
        "--input", str(sample),
        "--gold", str(sample),
        "--output-dir", str(output_dir),
        "--kb", DATASET_KB[dataset],
        "--stage2-order", args.stage2_order,
        "--workers", str(args.workers),
        "--continue-on-error",
    ]
    if args.resume:
        cmd.append("--resume")
    if args.max is not None:
        cmd.extend(["--max", str(args.max)])
    if args.llm_url:
        cmd.extend(["--llm-url", args.llm_url])
    if args.llm_model:
        cmd.extend(["--llm-model", args.llm_model])
    if args.routing_mode:
        cmd.extend(["--routing-mode", args.routing_mode])
    if args.prompt_version:
        cmd.extend(["--prompt-version", args.prompt_version])
    return cmd


def _run_round(args: argparse.Namespace, round_id: int, datasets: List[str]) -> bool:
    print("\n" + "=" * 88, flush=True)
    print(f"ROUND {round_id}: launching {len(datasets)} dataset runs in parallel", flush=True)
    print("=" * 88, flush=True)

    procs = []
    for dataset in datasets:
        output_dir = args.output_root / dataset / f"round_{round_id}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        cmd = _build_command(args, dataset, round_id, output_dir)
        print(f"[{dataset} round_{round_id}] output -> {output_dir}", flush=True)
        print(f"[{dataset} round_{round_id}] log    -> {log_path}", flush=True)
        print(f"[{dataset} round_{round_id}] cmd    -> LLM_API_KEY=*** {shlex.join(cmd)}", flush=True)
        log_f = log_path.open("a" if args.resume else "w", encoding="utf-8")
        log_f.write("\n" + "=" * 88 + "\n")
        log_f.write(f"START {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.write("LLM_API_KEY=*** " + shlex.join(cmd) + "\n")
        log_f.flush()
        env = os.environ.copy()
        env["LLM_API_KEY"] = _api_key_for_dataset(args, dataset)
        proc = subprocess.Popen(
            cmd,
            cwd=str(MODEL_DIR),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        procs.append((dataset, output_dir, log_path, log_f, proc))

    ok = True
    for dataset, output_dir, log_path, log_f, proc in procs:
        rc = proc.wait()
        log_f.write(f"\nEND {time.strftime('%Y-%m-%d %H:%M:%S')} rc={rc}\n")
        log_f.close()
        if rc == 0:
            print(f"[OK]   {dataset} round_{round_id} -> {output_dir}", flush=True)
        else:
            ok = False
            print(f"[FAIL] {dataset} round_{round_id} rc={rc}; see {log_path}", flush=True)

    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serial rounds, parallel datasets runner for DDD_order experiments."
    )
    parser.add_argument(
        "--datasets",
        default="kg-table,kg-doc,table-doc",
        help="Comma-separated datasets to run. Default: kg-table,kg-doc,table-doc",
    )
    parser.add_argument(
        "--rounds",
        default="1,2,3",
        help="Comma-separated round ids to run serially. Default: 1,2,3",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Workers passed to each dataset run.py invocation. Default: 20",
    )
    parser.add_argument(
        "--stage2-order",
        choices=["route-semantic", "semantic-route"],
        default="semantic-route",
        help="Stage-2 order ablation. Default: semantic-route",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "result" / "swapped_promptfix_parallel",
        help="Root output directory. Default: DDD_order/result/swapped_promptfix_parallel",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", ""),
        help="Single LLM API key. Defaults to LLM_API_KEY env var. Not written to logs.",
    )
    parser.add_argument(
        "--api-keys",
        default=os.getenv("LLM_API_KEYS", ""),
        help="Optional comma-separated per-dataset keys in selected dataset order. Not written to logs.",
    )
    parser.add_argument("--llm-url", default=os.getenv("LLM_BASE_URL", ""))
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", ""))
    parser.add_argument("--routing-mode", default="graph", choices=["graph", "atomr", "deepsieve"])
    parser.add_argument("--prompt-version", default="v2", choices=["v1", "v2"])
    parser.add_argument("--resume", action="store_true", help="Pass --resume to run.py and append logs.")
    parser.add_argument("--max", type=int, default=None, help="Optional smoke-test limit passed to run.py.")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to later rounds even if one dataset fails in a round.",
    )
    args = parser.parse_args()

    if not RUN_PY.exists():
        raise SystemExit(f"Missing run.py: {RUN_PY}")

    datasets = _parse_csv(args.datasets, list(DATASET_KB), "dataset")
    rounds = [int(x) for x in _parse_csv(args.rounds, ["1", "2", "3"], "round")]

    args.api_key_map = None
    if args.api_keys.strip():
        keys = [x.strip() for x in args.api_keys.split(",") if x.strip()]
        if len(keys) != len(datasets):
            raise SystemExit(
                f"--api-keys got {len(keys)} keys, but {len(datasets)} datasets are selected: {datasets}"
            )
        args.api_key_map = dict(zip(datasets, keys))
    elif not args.api_key:
        raise SystemExit("No API key. Export LLM_API_KEY, LLM_API_KEYS, or pass --api-key/--api-keys.")
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"model dir   : {MODEL_DIR}", flush=True)
    print(f"datasets    : {datasets}", flush=True)
    print(f"rounds      : {rounds}", flush=True)
    print(f"workers/run : {args.workers}", flush=True)
    print(f"output root : {args.output_root}", flush=True)
    print(f"resume      : {args.resume}", flush=True)
    print(f"api keys    : {'per-dataset' if args.api_key_map else 'single'}", flush=True)

    all_ok = True
    for round_id in rounds:
        ok = _run_round(args, round_id, datasets)
        all_ok = all_ok and ok
        if not ok and not args.keep_going:
            raise SystemExit(f"Stopping after round_{round_id} failure. Re-run with --keep-going to continue.")

    if all_ok:
        print("\nAll requested runs finished successfully.", flush=True)
    else:
        raise SystemExit("One or more runs failed; inspect run.log files above.")


if __name__ == "__main__":
    main()
