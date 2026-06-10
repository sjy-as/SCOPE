"""Driver for the ToG2 baseline.

Example:
  cd /root/autodl-tmp/baseline/ToG2
  python3 run.py --input <jsonl> --output-dir <dir> --kb kg,table --workers 8 --judge --api-key <KEY>
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# Add baseline/ so `import _common.*` works.
sys.path.insert(0, str(_HERE.parent))

from _common.driver import make_parser, run_baseline  # noqa: E402

# Load the local pipeline.py by absolute path so it can't be shadowed by
# any other ``pipeline`` module on sys.path (HydraRAG's, new_model's, etc.).
_spec = importlib.util.spec_from_file_location(
    "baseline_tog2_pipeline", str(_HERE / "pipeline.py"))
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def main() -> None:
    args = make_parser("ToG2", needs_retrieval=True).parse_args()
    run_baseline(args, name="ToG2", pipeline_fn=pipeline.run, needs_retrieval=True)


if __name__ == "__main__":
    main()
