"""
Ablation: SCOPE w/o semantic-list metadata in operator planning.

Keeps decomposition, routing, semantic parsing, and fallback execution the
same as full SCOPE, but disables semantic-list matched-concept metadata
injection only in the operator-tree planner prompt.

Copyable runs:

python SCOPE_code/wo_abla/SCOPE_wo_semlist_plan/run.py \
  --input ../../CMQA/qa_bench/kg-table-1147.jsonl \
  --gold  ../../CMQA/qa_bench/kg-table-1147.jsonl \
  --kb kg,table \
  --workers 8 \
  --api-key "$LLM_API_KEY"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

NEW_MODEL_DIR = Path(os.getenv("SCOPE_NEW_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "SCOPE"))).resolve()
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
