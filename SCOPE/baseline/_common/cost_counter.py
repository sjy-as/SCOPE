"""Shim re-exports the global cost counter from SCOPE_code/_common.

When a baseline's run.py imports the local baseline/_common package, it can
shadow the repository-level _common package. This shim loads the repository
implementation by a path relative to this file and re-exports its symbols.
"""
from __future__ import annotations

import importlib.util as _ilu
import os as _os
import sys as _sys
from pathlib import Path as _Path

_REAL = str(_Path(__file__).resolve().parents[2] / "_common" / "cost_counter.py")


def _load_real():
    if not _os.path.exists(_REAL):
        return None
    spec = _ilu.spec_from_file_location("_common_real_cost_counter", _REAL)
    if spec is None or spec.loader is None:
        return None
    mod = _ilu.module_from_spec(spec)
    # Cache so `import _common.cost_counter` returns this same object on reload.
    _sys.modules.setdefault("_common_real_cost_counter", mod)
    spec.loader.exec_module(mod)
    return mod


_real = _load_real()

if _real is not None:
    VERIFY_STAGES = _real.VERIFY_STAGES
    question_scope = _real.question_scope
    bump_llm = _real.bump_llm
    bump_retrieval = _real.bump_retrieval
    reset_aggregator = _real.reset_aggregator
    snapshot = _real.snapshot
    aggregate_totals = _real.aggregate_totals
    dump_summary = _real.dump_summary
    seed_from_existing = _real.seed_from_existing
    format_summary_line = _real.format_summary_line
else:
    # Repository-level _common/cost_counter.py missing; install no-op
    # fallbacks so the driver continues to work.
    from contextlib import contextmanager as _contextmanager

    VERIFY_STAGES = set()

    @_contextmanager
    def question_scope(qid):  # type: ignore
        yield

    def bump_llm(stage=None, n=1): pass
    def bump_retrieval(source=None, n=1): pass
    def reset_aggregator(): pass
    def snapshot(): return {}

    def aggregate_totals():
        return {"n_questions": 0, "llm_calls": 0, "retrieval": {}, "total_retrieval": 0,
                "per_question": {}}

    def dump_summary(out_path):
        return aggregate_totals()

    def seed_from_existing(in_path):
        return 0

    def format_summary_line(summary):
        return ""
