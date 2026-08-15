#!/usr/bin/env bash
set -euo pipefail

# Rebuild result tables from existing eval_summary/summary.json files.
# Set SCOPE_RESULT_ROOT / SCOPE_ABLA_RESULT_ROOT if your artifacts live elsewhere.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$CODE_ROOT"
python3 run_ex_main.py --tables-only
python3 run_ex_abla.py --tables-only
