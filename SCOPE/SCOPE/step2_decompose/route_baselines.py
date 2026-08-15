"""
Baseline routers for routing ablation experiments.

Both routers return a dict with the SAME shape as
`step2_decompose_new.sq.source_routing`, so the rest of the
pipeline (stage2b semantic + plan, stage3 reasoner) is untouched.

Returned dict keys:
    matched_kind     : str   ("none" for baselines)
    matched_key      : Optional[str]
    routing_layer    : str   ("atomr" or "deepsieve")
    primary_source   : "kg" | "table" | "doc"
    fallback_source  : "kg" | "table" | "doc"
    confidence       : float
    reasoning        : str

The few-shot AtomR prompt is built dynamically from the live KB
(`enabled_sources`). Labels, descriptions, selection rules, atomic
examples, and bridge examples for any source NOT in the live KB are
stripped from the prompt entirely. This prevents the KG-bias collapse
we saw on kb=table,doc, where the model picked "KG" and the mapper
silently routed everything to table.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from sq import call_llm  # reuse the same LLM caller as graph routing


# =========================================================================
# AtomR-style few-shot routing
#
# The prompt is composed from per-source building blocks. For a 1-source
# KB the model is told it has exactly one option (and is asked to confirm
# it). For 2- or 3-source KBs the prompt enumerates only the live labels,
# only the live descriptions / rules, the balanced atomic examples for
# each live source, and the bridge examples whose every component is live.
# Atomic examples are drawn from real benchmark sub-queries (kg-doc,
# kg-table, table-doc) to match what the router sees in production.
# =========================================================================

# Executor id -> AtomR-style label shown to the LLM.
_SRC_TO_LABEL = {"kg": "KG", "table": "Table", "doc": "Text"}
# Reverse lookup: lowercased label -> executor id.
_LABEL_TO_SRC = {v.lower(): k for k, v in _SRC_TO_LABEL.items()}

# Per-source description for the "Possible knowledge sources" block.
_SRC_DESC = {
    "kg":    "KG: a sports-oriented knowledge graph with structured entities and relations (player-team, draft, coach-of, received-award, plays-position, home-venue, division links).",
    "table": "Table: basketball tables such as season game logs, schedules, rosters, draft pick boards, standings, all-star event score sheets, and per-game / per-event statistics.",
    "doc":   "Text: encyclopedic document passages with biographies, season narratives, award/venue descriptions, injuries, career background, and definition-style explanations.",
}

# Per-source selection rule of thumb.
_SRC_RULE = {
    "kg":    "- Prefer KG for explicit relational or attribute lookups: who drafted whom, which team coached, who played for which team, who received an award, which venue a team uses, what position a player plays.",
    "table": "- Prefer Table for numeric, statistical, ranking, or per-game / per-event questions: highest/lowest score, most/least times, streaks, opponent on a given date, leading scorer, first-round time, attendance.",
    "doc":   "- Prefer Text for definition, biography, descriptive explanation, or background context: what an award recognizes, how a winner is selected, what a player is known for, what injury a player suffered, what term someone coined.",
}

# Atomic (single-source) examples, 4 per source, drawn from sub-queries of
# the kg-doc / kg-table / table-doc benchmarks. Counts are kept balanced
# (4 / 4 / 4) so the few-shot does not bias toward any one source.
_ATOMIC_EXAMPLES: Dict[str, List[Tuple[str, str]]] = {
    "kg": [
        ("Which player received the NBA Most Improved Player Award in 2009?", "KG"),
        ("Who was drafted by the Philadelphia 76ers with the third overall pick in 1999?", "KG"),
        ("Which team did Erik Spoelstra coach?", "KG"),
        ("Which venue did the Detroit Pistons use as their home arena between 1988 and 2017?", "KG"),
    ],
    "table": [
        ("What was Danny Granger's highest score in the February games of the 2007-08 Indiana Pacers season?", "Table"),
        ("How many times did Michael Finley lead his team in scoring during the December games of the 2007-08 San Antonio Spurs season?", "Table"),
        ("Who was the Cleveland Cavaliers' opponent on April 2 in the 2008-09 season?", "Table"),
        ("In the 1947 BAA Draft, which team selected a player from Purdue?", "Table"),
    ],
    "doc": [
        ("How is the winner selected for the NBA Most Valuable Player Award?", "Text"),
        ("What kind of contributions does the Presidential Medal of Freedom recognize?", "Text"),
        ("What injury did Brandon Roy suffer in December 2010?", "Text"),
        ("What is John Havlicek's post-playing career role?", "Text"),
    ],
}

# Bridge (multi-source) examples, keyed by frozenset of executor ids. A
# bridge example is included only when ALL its sources are live. One
# example per combination keeps the prompt compact while still showing
# the comma-separated output format.
_BRIDGE_EXAMPLES: Dict[frozenset, List[Tuple[str, str]]] = {
    frozenset({"kg", "table"}): [
        ("For the player who received the NBA Most Improved Player Award in 2009, what was his highest score in the February games of the 2007-08 Indiana Pacers season?", "KG, Table"),
    ],
    frozenset({"kg", "doc"}): [
        ("For the player drafted by the Philadelphia 76ers in 1999, what genetic disorder forced him to retire prematurely?", "KG, Text"),
    ],
    frozenset({"doc", "table"}): [
        ("For the contestant who posted a first-round time of 35.7 in the 2010 NBA Skills Challenge, what injury did he suffer in December 2010?", "Table, Text"),
    ],
    frozenset({"kg", "table", "doc"}): [
        ("For the player who received the NBA Sportsmanship Award in 2009, how did the team he played for perform during the March games of the 2007-08 season, and what is the award known for recognizing?", "KG, Table, Text"),
    ],
}


def _build_atomr_prompt(question: str, enabled_sources: List[str]) -> str:
    """Build an AtomR few-shot prompt restricted to `enabled_sources`.

    Labels, descriptions, selection rules, atomic examples, and bridge
    examples are all filtered to the live KB so the LLM never sees an
    inactive source.
    """
    live_canonical = [s for s in ("kg", "table", "doc") if s in enabled_sources]
    if not live_canonical:
        raise ValueError(
            "_build_atomr_prompt: enabled_sources must include at least one of {kg, table, doc}"
        )
    live_set = set(live_canonical)
    labels = [_SRC_TO_LABEL[s] for s in live_canonical]

    descs_block = "\n".join(
        f"{i}. {_SRC_DESC[s]}" for i, s in enumerate(live_canonical, 1)
    )

    rules_lines = [_SRC_RULE[s] for s in live_canonical]
    if len(live_canonical) >= 2:
        rules_lines.append(
            "- For bridge questions whose answer depends on more than one source, list every needed source separated by commas (e.g. "
            + ", ".join(labels)
            + ")."
        )
        rules_lines.append(
            "- If the question is ambiguous, list every available source that could plausibly help."
        )
    rules_block = "\n".join(rules_lines)

    example_lines: List[str] = []
    # Atomic examples per live source (balanced).
    for s in live_canonical:
        for q, a in _ATOMIC_EXAMPLES[s]:
            example_lines.append(f"Q: {q}\nA: {a}")
    # Bridge examples whose component sources are all live (and bridge >= 2).
    for combo, examples in _BRIDGE_EXAMPLES.items():
        if len(combo) < 2:
            continue
        if combo.issubset(live_set):
            for q, a in examples:
                example_lines.append(f"Q: {q}\nA: {a}")
    examples_block = "\n".join(example_lines)

    n = len(labels)
    src_word = "source" if n == 1 else "sources"
    prompt = (
        f'Following the examples below, select which knowledge source(s) are best to answer the question "{question}". '
        "You may select multiple sources, but only from the sources listed below.\n"
        f"You have {n} available {src_word}: {labels}\n"
        "Possible knowledge sources:\n"
        f"{descs_block}\n\n"
        "How to select sources:\n"
        f"{rules_block}\n\n"
        "Strictly follow the answer format of the examples below.\n"
        "Examples.\n"
        f"{examples_block}\n\n"
        "Your question.\n"
        f"Q: {question}\n"
        "A: "
    )
    return prompt


def _parse_atomr_response(text: str, allowed_labels: List[str]) -> List[str]:
    """Parse the model's free-form "A: KG, Table" line into a clean list.
    Only labels in `allowed_labels` are kept (in their canonical case).
    Tolerant of multi-line output and extra commentary."""
    if not text:
        return []
    line = text.strip().splitlines()[0].strip()
    # strip a leading "A:" if the model echoed it
    if line.lower().startswith("a:"):
        line = line[2:].strip()
    canon = {s.lower(): s for s in allowed_labels}
    seen: List[str] = []
    for raw in line.replace(";", ",").split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        if tok in canon and canon[tok] not in seen:
            seen.append(canon[tok])
    return seen


def route_atomr_few_shot(
    query: str,
    model: str,
    base_url: str,
    api_key: str,
    available_sources: Optional[List[str]] = None,
    enabled_sources: Optional[List[str]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Few-shot AtomR routing, restricted to the live KB.

    `enabled_sources` is the live KB (executor labels, lowercase). It is
    the SINGLE source of truth for which labels appear in the prompt and
    which can be returned. `available_sources` is kept only for backward
    compatibility; if both are passed, `enabled_sources` wins.
    """
    if enabled_sources:
        live = [s.lower() for s in enabled_sources if s.lower() in _SRC_TO_LABEL]
    elif available_sources:
        # Legacy: AtomR labels like ["KG", "Table"] -> executor ids.
        live = [_LABEL_TO_SRC[s.lower()] for s in available_sources if s.lower() in _LABEL_TO_SRC]
    else:
        live = ["kg", "table"]  # legacy default
    if not live:
        raise ValueError("route_atomr_few_shot: enabled_sources must be non-empty and a subset of {kg, table, doc}")
    # Re-order live in canonical order so prompts are deterministic.
    live = [s for s in ("kg", "table", "doc") if s in live]
    labels = [_SRC_TO_LABEL[s] for s in live]

    prompt = _build_atomr_prompt(query, live)

    # Use a high output budget because reasoning models can spend hidden
    # reasoning tokens before producing visible routing content.
    raw = ""
    chosen_labels: List[str] = []
    retried = False
    for attempt_tokens in (10000,):
        retried = attempt_tokens != 10000
        try:
            raw = call_llm(prompt, model=model, base_url=base_url, api_key=api_key, max_tokens=attempt_tokens)
        except Exception as e:
            raw = ""
            if verbose:
                print(f"[atomr] LLM call failed (max_tokens={attempt_tokens}): {e}")
        chosen_labels = _parse_atomr_response(raw, labels)
        if chosen_labels:
            break
    if verbose:
        print(f"[atomr] live={live} raw='{(raw or '').strip()[:80]}' chosen={chosen_labels}")

    # AtomR label -> executor id. Because the prompt only ever shows live
    # labels, chosen_labels can only contain executor-live sources, so the
    # mapping is direct (no degradation).
    chosen_mapped: List[str] = []
    for lbl in chosen_labels:
        sid = _LABEL_TO_SRC.get(lbl.lower())
        if sid and sid in live and sid not in chosen_mapped:
            chosen_mapped.append(sid)

    def _other_source(primary: str) -> str:
        for s in live:
            if s != primary:
                return s
        return primary  # 1-source KB: nothing else to fall back to.

    parse_failed = False
    if not chosen_mapped:
        # Final fallback: uniform across all routers — always use the first
        # source in canonical kg/table/doc order (live[0]) as primary, the
        # next live source as fallback. live is already canonical-ordered.
        primary = live[0]
        fallback = _other_source(primary)
        reasoning = (
            f"AtomR parse failed after retry (raw='{(raw or '').strip()[:60]}'); "
            f"default to canonical primary {primary}."
        )
        confidence = 0.50
        parse_failed = True
    elif len(chosen_mapped) == 1:
        primary = chosen_mapped[0]
        fallback = _other_source(primary)
        suffix = " (after token-bump retry)" if retried else ""
        reasoning = f"AtomR few-shot selected {chosen_labels}{suffix}."
        confidence = 0.85
    else:
        primary = chosen_mapped[0]
        fallback = chosen_mapped[1] if chosen_mapped[1] != primary else _other_source(primary)
        suffix = " (after token-bump retry)" if retried else ""
        reasoning = f"AtomR few-shot selected {chosen_labels} (first as primary){suffix}."
        confidence = 0.90

    return {
        "matched_kind": "none",
        "matched_key": None,
        "routing_layer": "atomr",
        "primary_source": primary,
        "fallback_source": fallback,
        "confidence": confidence,
        "parse_failed": parse_failed,
        "reasoning": reasoning,
        "atomr_selected_sources": chosen_labels,  # debug: raw labels chosen by LLM
        "atomr_live_sources": live,               # debug: live KB used for the prompt
        "atomr_raw": (raw or "").strip()[:120],   # debug: model output (for parse-fail forensics)
    }


# =========================================================================
# DeepSieve-style profile-based routing (N-source, generalised)
#
# Original DeepSieve was a pairwise SOURCE_A vs SOURCE_B template. We extend
# the same template to any subset of {kg, table, doc} by enumerating each
# live source with its profile and asking the LLM to output one source name.
# For a 1-source KB the answer is forced (no decision to make).
# =========================================================================

# Cache loaded profile files so repeat calls within a run don't re-read JSON.
_DS_PROFILES_CACHE: Dict[str, Dict[str, str]] = {}


def _load_all_source_profiles(profiles_path: str) -> Dict[str, str]:
    """Load {kg, table, doc} profiles from a DeepSieve profiles JSON file.

    Missing sources are returned as ''. Aliases ('graph' -> kg, 'document' ->
    doc) are accepted to match data_sources/source_profiles.json.
    """
    cached = _DS_PROFILES_CACHE.get(profiles_path)
    if cached is not None:
        return cached
    if not os.path.exists(profiles_path):
        raise FileNotFoundError(f"Source profiles file not found: {profiles_path}")
    with open(profiles_path, "r", encoding="utf-8") as f:
        prof = json.load(f)
    out = {
        "kg":    prof.get("kg")    or prof.get("graph")    or prof.get("kg_profile")    or "",
        "table": prof.get("table") or prof.get("table_profile") or "",
        "doc":   prof.get("doc")   or prof.get("document") or prof.get("doc_profile")   or "",
    }
    _DS_PROFILES_CACHE[profiles_path] = out
    return out


def route_deepsieve(
    query: str,
    profiles_path: str,
    enabled_sources: List[str],
    model: str,
    base_url: str,
    api_key: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """DeepSieve N-source profile routing.

    Generalised from the original pairwise DeepSieve template to any subset
    of {kg, table, doc}. `enabled_sources` is the live KB (canonical kg/
    table/doc order). The function:
      - returns the only live source forced when |live| == 1
      - builds an N-source prompt for 2+ live sources
      - retries once with larger max_tokens on parse failure
      - falls back to canonical primary (live[0]) when retry still fails
    """
    profiles = _load_all_source_profiles(profiles_path)
    live = [s for s in enabled_sources if profiles.get(s)]
    if not live:
        return {
            "matched_kind": "none",
            "matched_key": None,
            "routing_layer": "deepsieve",
            "primary_source": "kg",
            "fallback_source": "kg",
            "confidence": 0.0,
            "reasoning": "DeepSieve: no profiles available for the live KB.",
        }
    if len(live) == 1:
        only = live[0]
        return {
            "matched_kind": "none",
            "matched_key": None,
            "routing_layer": "deepsieve",
            "primary_source": only,
            "fallback_source": only,
            "confidence": 1.0,
            "reasoning": f"DeepSieve: only one live source ({only}); no routing decision.",
        }

    blocks = [f"SOURCE NAME: {s}\nSOURCE PROFILE:\n{profiles[s]}" for s in live]
    sources_block = "\n\n".join(blocks)
    choices = " or ".join(f'"{s}"' for s in live)
    prompt = (
        "You are a routing assistant. Your task is to decide which ONE source "
        "should be used to answer the query, given each source's profile.\n\n"
        f"{sources_block}\n\n"
        f"QUERY:\n{query}\n\n"
        f"Please output only one word: {choices}, based on which profile is "
        "most relevant to the query. Do not add any explanation or extra words."
    )

    # Use a high output budget because reasoning models can spend hidden
    # reasoning tokens before producing visible routing content.
    import re as _re
    raw = ""
    err = ""
    chosen_raw = ""
    chosen = ""
    retried = False
    for attempt_tokens in (10000,):
        retried = attempt_tokens != 10000
        try:
            raw = call_llm(prompt, model=model, base_url=base_url, api_key=api_key, max_tokens=attempt_tokens)
            err = ""
        except Exception as e:
            raw = ""
            err = str(e)
            if verbose:
                print(f"[deepsieve] LLM call failed (max_tokens={attempt_tokens}): {e}")
        # Tolerant parse: strip common wrappers, then if not an exact match
        # try whole-word substring against the live source names. Survives
        # 'table.', '"Table"', 'I would pick table.' style outputs.
        chosen_raw = (raw or "").strip().strip('"').strip("'").strip(".").lower()
        chosen = chosen_raw if chosen_raw in live else ""
        if not chosen:
            for _src in live:
                if _re.search(rf"\b{_re.escape(_src)}\b", chosen_raw):
                    chosen = _src
                    break
        if chosen in live:
            break

    parse_failed = False
    if chosen not in live:
        # Final fallback: uniform across all routers — always use the first
        # source in canonical kg/table/doc order (live[0]) as primary, the
        # next live source as fallback.
        primary = live[0]
        confidence = 0.50
        reasoning = (
            f"DeepSieve routing parse failed after retry (got '{chosen_raw or err or raw}'); "
            f"default to canonical primary {primary}."
        )
        parse_failed = True
    else:
        primary = chosen
        confidence = 0.90
        suffix = " (after token-bump retry)" if retried else ""
        reasoning = f"DeepSieve selected {primary} from {live}{suffix}."

    fallback_candidates = [s for s in live if s != primary]
    fallback = fallback_candidates[0] if fallback_candidates else primary
    return {
        "matched_kind": "none",
        "matched_key": None,
        "routing_layer": "deepsieve",
        "primary_source": primary,
        "fallback_source": fallback,
        "confidence": confidence,
        "parse_failed": parse_failed,
        "reasoning": reasoning,
        "deepsieve_live_sources": live,
        "deepsieve_raw": raw,
    }
