"""
Ablation: SCOPE w/o semantic list, with AtomR source routing.

This is a thin wrapper around SCOPE/run.py that forces
--routing-mode=atomr. Under that mode the router returns no `matched_concept`,
so `step2_decompose.semantic.parse_semantic` and
`step2_decompose.operator_plan.build_plan_for_subquery` both receive
`matched_info=None`, which means the semantic-list content is NOT injected
into the query-parsing prompt or the operator-tree-generation prompt. Only
source routing runs against the catalog (and AtomR ignores it entirely).

Default output dir is co-located with this wrapper. Override with --output-dir.

Copyable runs:

python SCOPE_code/wo_abla/SCOPE_wo_semlist_atomr/run.py \
  --input ../../CMQA/qa_bench/kg-table-1147.jsonl \
  --gold  ../../CMQA/qa_bench/kg-table-1147.jsonl \
  --kb kg,table \
  --workers 8 \
  --api-key "$LLM_API_KEY"

python SCOPE_code/wo_abla/SCOPE_wo_semlist_atomr/run.py \
  --input ../../CMQA/qa_bench/kg-doc-1154.jsonl \
  --gold  ../../CMQA/qa_bench/kg-doc-1154.jsonl \
  --kb kg,doc \
  --workers 8 \
  --api-key "$LLM_API_KEY"

python SCOPE_code/wo_abla/SCOPE_wo_semlist_atomr/run.py \
  --input ../../CMQA/qa_bench/table-doc-1120.jsonl \
  --gold  ../../CMQA/qa_bench/table-doc-1120.jsonl \
  --kb table,doc \
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

# Force routing-mode=atomr; reject user override since this script's whole
# point is the AtomR ablation.
if "--routing-mode" in sys.argv:
    i = sys.argv.index("--routing-mode")
    val = sys.argv[i + 1] if i + 1 < len(sys.argv) else ""
    if val != "atomr":
        raise SystemExit(
            f"[SCOPE_wo_semlist_atomr] --routing-mode is fixed to 'atomr', "
            f"got '{val}'. Use SCOPE/run.py directly if you want a different mode."
        )
else:
    sys.argv.extend(["--routing-mode", "atomr"])

# Default --output-dir to <baseline_dir>/result if not provided.
if "--output-dir" not in sys.argv:
    sys.argv.extend(["--output-dir", str(DEFAULT_OUTPUT_DIR)])

# SCOPE/run.py resolves relative input/gold/output paths against its own
# directory, so chdir there before delegating.
os.chdir(NEW_MODEL_DIR)

from run import main  # noqa: E402

if __name__ == "__main__":
    main()
