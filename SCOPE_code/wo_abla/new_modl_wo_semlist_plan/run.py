"""
Ablation: new_model w/o semantic-list metadata in operator planning.

Keeps decomposition, routing, semantic parsing, and fallback execution the
same as full new_model, but disables semantic-list matched-concept metadata
injection only in the operator-tree planner prompt.

Copyable runs:

python /root/autodl-tmp/baseline/new_modl_wo_semlist_plan/run.py \
  --input /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --gold  /root/autodl-tmp/new_model/qa_bench/kg-table-160.jsonl \
  --kb kg,table \
  --workers 8 \
  --api-key "sk-..."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NEW_MODEL_DIR = Path("/root/autodl-tmp/new_model")
BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASELINE_DIR / "result"

if str(NEW_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(NEW_MODEL_DIR))

if "--no-inject-plan-semlist-metadata" not in sys.argv and "--inject-plan-semlist-metadata" not in sys.argv:
    sys.argv.append("--no-inject-plan-semlist-metadata")

if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])

os.chdir(NEW_MODEL_DIR)

from run import main  # noqa: E402

if __name__ == "__main__":
    main()
