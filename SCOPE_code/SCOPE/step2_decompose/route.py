import json
from typing import Any, Dict, List, Optional

from .sq import call_and_parse_with_retry, load_json

# Canonical ordering of every data source the framework knows about. Each
# source (kg / table / doc) is opt-in via the run's --kb flag.
ALL_SOURCES = ["kg", "table", "doc"]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def _norm_sources(enabled_sources: Optional[List[str]]) -> List[str]:
    """Normalize the live-source list: lowercase, dedup, kept in canonical
    (kg, table, doc) order. None / empty means the full catalog."""
    if not enabled_sources:
        return list(ALL_SOURCES)
    want = {str(s).strip().lower() for s in enabled_sources if str(s).strip()}
    norm = [s for s in ALL_SOURCES if s in want]
    return norm or list(ALL_SOURCES)


def _build_semantic_catalog(bundle: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(bundle, dict):
        return {}

    # Preferred format: semantic_list.json (concept -> {concept, overall_description, sources})
    if bundle and all(isinstance(v, dict) and "sources" in v for v in bundle.values()):
        return bundle

    semantic_list = bundle.get("semantic_list") or bundle.get("raw", {}).get("semantic_list")
    if isinstance(semantic_list, dict):
        return semantic_list

    raw_data = bundle.get("raw") or {}
    merged_semantics = raw_data.get("merged_semantics") or {}
    concepts = merged_semantics.get("merged_concepts") or {}
    relations = merged_semantics.get("merged_relations") or {}
    return {**concepts, **relations}


def _source_names(info: Dict[str, Any]) -> Dict[str, Any]:
    sources = info.get("sources") or {}
    return {name.lower(): value for name, value in sources.items() if isinstance(value, dict)}


def _format_semantic_catalog_for_prompt(catalog: Dict[str, Any], limit: int = 60) -> str:
    items = []
    for idx, (key, info) in enumerate(catalog.items()):
        if idx >= limit:
            break
        sources = _source_names(info)
        item = {
            "key": key,
            "concept": info.get("concept", key),
            "overall_description": info.get("overall_description", ""),
            "sources": {
                name: {
                    "description": src.get("description", ""),
                    "elements": src.get("elements") or [],
                    "data_examples": src.get("data_examples") or [],
                }
                for name, src in sources.items()
            },
        }
        items.append(item)
    return json.dumps(items, ensure_ascii=False, indent=2)


def _first_source_name(sources: Dict[str, Any]) -> str:
    if not sources:
        return ""
    return next(iter(sources.keys()))


def _coerce_routing_result(parsed: Any, enabled_sources: List[str]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    valid = list(enabled_sources) or ["kg"]
    primary = _normalize_text(parsed.get("primary_source") or parsed.get("source"))
    fallback = _normalize_text(parsed.get("fallback_source"))
    # Clamp routing to the live knowledge base so the executor never receives
    # a source it has no retriever wired for.
    if primary not in valid:
        primary = "kg" if "kg" in valid else valid[0]
    # Keep the LLM's fallback when it is a valid, distinct source; otherwise
    # pick any other live source. When the KB has a single source, fallback ==
    # primary, which simply disables the cross-source fallback retry.
    if fallback not in valid or fallback == primary:
        others = [s for s in valid if s != primary]
        fallback = others[0] if others else primary
    return {
        "primary_source": primary,
        "fallback_source": fallback,
        "confidence": float(parsed.get("confidence", 0.7)) if parsed.get("confidence") is not None else 0.7,
        "reasoning": parsed.get("reasoning", "LLM routed the query."),
    }


def source_routing(
    matched_kind: str,
    matched_item: Optional[Dict[str, Any]],
    bundle: Dict[str, Any],
    query: str,
    model: str,
    base_url: str,
    api_key: str,
    verbose: bool = False,
    ref_type: str = "entity",
    profiles_path: Optional[str] = None,
    prompt_version: str = "v2",
    enabled_sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # `enabled_sources` is the live knowledge base: the subset of
    # {kg, table, doc} the run was configured with. Routing (primary +
    # fallback) is confined to it, and the semantic catalog passed in is
    # expected to already be filtered to these sources.
    sources = _norm_sources(enabled_sources)
    source_list = ", ".join(sources)
    catalog = _build_semantic_catalog(bundle)

    def _default_route(reason: str) -> Dict[str, Any]:
        primary = "kg" if "kg" in sources else sources[0]
        others = [s for s in sources if s != primary]
        return {
            "matched_concept": None,
            "primary_source": primary,
            "fallback_source": others[0] if others else primary,
            "confidence": 0.5,
            "parse_failed": True,
            "reasoning": reason,
        }

    if not catalog:
        return _default_route("No semantic catalog available; fallback to default source.")

    prompt_catalog = _format_semantic_catalog_for_prompt(catalog)
    matched_hint = matched_item.get("key") if isinstance(matched_item, dict) else None
    matched_desc = matched_item.get("description", "") if isinstance(matched_item, dict) else ""

    # prompt = f"""You are a semantic router for question decomposition.

    #     Your task is to route a sub-query using the semantic catalog.

    #     Inputs:
    #     - Query: {query}
    #     - Upstream matched hint: {matched_hint or "none"}
    #     - Upstream matched description: {matched_desc or "none"}
    #     - Live data sources (knowledge base): {source_list}

    #     Concept selection (Phase 1 — do this BEFORE choosing any source):
    #     a) Parse the query into:
    #        - KNOWN entities: entities explicitly given (including any [k] reference
    #          tokens) and their entity types.
    #        - ASKED target: what the query asks to return and its type.
    #        - CONSTRAINTS: any time / season / event / context scoping conditions
    #          (e.g. "in 2005", "after winning X award", "during the playoffs").

    #     b) Concepts in the catalog fall into three tiers — judge a candidate's tier
    #        on the fly from its description + elements, do not rely on the name:
    #        - SUPER concept (multi-entity relation / event): description says it links
    #          two or more entities; elements are mostly entity references plus a
    #          constraint dimension (e.g. player+award+time, player+team+time,
    #          team+coach, player+team+year).
    #        - BASIC concept (single-anchor record with time/context-scoped attribute
    #          bundle): one primary entity anchor plus season / period / event fields
    #          and a bundle of attributes (e.g. per-season stats, monthly schedule,
    #          per-game record, per-year broadcast assignment).
    #        - META concept (standalone single-entity description): describes one entity
    #          type's intrinsic / static attributes, no temporal scoping in elements
    #          (e.g. a player's profile, a venue's description, an award's definition).

    #     c) Choose the starting tier by query pattern, then verify and fall back DOWN
    #        one tier at a time if verification fails. Never escalate up.

    #        Pattern A — multi-entity with event/relational constraint
    #          (e.g. "in <event-constraint>, the <target> of <known> is …",
    #                "the player who won X award in 2005's team",
    #                "the coach of team T in season S")
    #          → Start at SUPER. A super concept QUALIFIES only if its endpoint set
    #            covers BOTH the KNOWN-entity type AND the ASKED-target type, AND its
    #            elements carry the constraint dimension (time / event / etc.).
    #          → If no super concept qualifies (endpoints or constraint don't line up),
    #            fall back to BASIC: pick the entity-record concept anchored on the
    #            KNOWN entity whose elements plausibly carry the ASKED target.
    #          → If basic also fails, fall back to META on the KNOWN entity.

    #        Pattern B — single-anchor with time / context-scoped attribute
    #          (e.g. "<known>'s <target> is …",
    #                "in <time-constraint>, <known>'s <target> is …",
    #                "Team T's home win rate in 2010", "Player P's points in March 2015")
    #          → Start at BASIC. A basic concept QUALIFIES only if its anchor entity
    #            type matches the KNOWN entity AND its elements plausibly carry the
    #            ASKED attribute (head/anchor matches AND tail/attribute matches).
    #          → If no basic concept qualifies (anchor type or attribute name don't
    #            line up), fall back to META on the KNOWN entity.

    #        Pattern C — plain attribute of a single entity, no time/event scoping
    #          (e.g. "<known>'s height", "what kind of venue is V",
    #                "X award's selection criteria")
    #          → Start at META directly. Match the standalone single-entity concept
    #            whose own record / passages would naturally contain the answer as an
    #            attribute. Do NOT match the concept that merely names the ASKED
    #            target type.

    #     d) Tie-breakers:
    #        - If several KNOWN entities exist, anchor on the one whose own record
    #          would naturally carry the answer.
    #        - Prefer the highest-tier candidate that passes verification; only step
    #          down when verification fails.

    #     Source selection (Phase 2 — after the concept is fixed):
    #     1) Only the live data sources are allowed: {source_list}. If the
    #        matched concept exposes a single live source, route to it and
    #        skip the rest.
    #     2) Answerability check — note the asymmetry between sources:
    #           - KG has a CLOSED schema. Its `elements` list is the full set
    #             of attributes/relations available. If nothing in that list
    #             can plausibly carry the ASKED target, KG is RULED OUT for
    #             this query (do not stretch field names to fit).
    #           - Doc / Table have OPEN content. Their `elements` and
    #             `data_examples` are only samples / summaries — the actual
    #             passages or rows may carry information that is not listed.
    #             You only need a reasonable inference (from description,
    #             elements, or examples) that the source is LIKELY to contain
    #             the answer. Descriptive / definitional / criteria-style
    #             targets are almost always covered by doc text. Tabular /
    #             statistical / per-row targets are typically covered by
    #             table rows.
    #           Label each live source LIKELY / RULED-OUT under these rules.
    #     3) Pick primary_source:
    #           - If KG is the only LIKELY source, primary = kg.
    #           - If KG is RULED OUT, pick primary from the remaining LIKELY
    #             sources (doc for descriptive, table for tabular). Never pick
    #             a RULED-OUT source as primary just because of a heuristic.
    #           - If both KG and another source are LIKELY, prefer the one
    #             whose listed elements most directly name the ASKED target;
    #             KG wins for structured atomic facts, doc wins for
    #             descriptive / criteria / narrative targets, table wins for
    #             statistical / row-shaped targets.
    #     4) Pick fallback_source from the remaining live sources; primary
    #        and fallback must differ whenever more than one source is live.
    #        A RULED-OUT source may still be used as fallback only when no
    #        other live source exists.

    #     General:
    #     - The answerability check above IS the decision procedure; do not
    #       invent extra rule-based scoring.
    #     - The reasoning field MUST contain: (a) the KNOWN entity and the
    #       ASKED target, (b) which pattern (A/B/C) was picked and which
    #       starting tier, (c) the verification outcome — if the starting
    #       tier failed, name each concept you considered and why it was
    #       rejected, and the tier you fell back to, and (d) for each live
    #       source, whether you judged it LIKELY or RULED-OUT and the short
    #       justification (cite an element name for KG, or describe why
    #       doc/table content is expected to cover or not cover it).

    #     Semantic catalog:
    #     {prompt_catalog}

    #     Return valid JSON only with this schema:
    #     {{
    #     "matched_concept": "the matched concept name or null",
    #     "primary_source": "one of: {source_list}",
    #     "fallback_source": "one of: {source_list}",
    #     "confidence": 0.0,
    #     "reasoning": "KNOWN/ASKED, pattern + starting tier, verification & any tier fallback, per-source LIKELY/RULED-OUT"
    #     }}
    # """

    prompt = f"""You are a semantic router for question decomposition.

        Your task is to route a sub-query using the semantic catalog.

        Inputs:
        - Query: {query}
        - Upstream matched hint: {matched_hint or "none"}
        - Upstream matched description: {matched_desc or "none"}
        - Live data sources (knowledge base): {source_list}

        Concept selection (Phase 1 — do this BEFORE choosing any source):
        a) Read the query and identify two things:
           - KNOWN entities: entities explicitly given in the query, including any [k] reference tokens.
           - ASKED target: what the query asks to return (the type of the answer).
        b) Relation-first. A relation/event concept is one whose description says it
           links two or more entities and whose elements are entity references
           (e.g. award_received = player + award + time; contract = player + team;
           coaching_assignment = team + coach). Match such a concept ONLY IF its two
           endpoints are exactly the pair (KNOWN entity type, ASKED target type). A
           relation that connects a different pair of entities does NOT qualify, even
           if it mentions one of them.
        c) Known-entity fallback. If no relation concept qualifies, the query is just
           asking for an attribute or neighbor of a KNOWN entity. Match the concept of
           that KNOWN entity — the anchor you start the search from — and do NOT match
           the concept that merely names the ASKED target type.
           Example: "who is <player>'s coach at <university>" — no relation concept
           links player to coach (coaching_assignment links team to coach, not player
           to coach), so match the KNOWN entity concept `player`, NOT `coach`.
        d) If several known entities exist, pick the one whose own record or passages
           would naturally contain the answer as an attribute or neighbor.

        Source selection (Phase 2 — after the concept is fixed):
        1) Only the live data sources are allowed: {source_list}. If the
           matched concept exposes a single live source, route to it and
           skip the rest.
        2) Answerability check — note the asymmetry between sources:
              - KG has a CLOSED schema. Its `elements` list is the full set
                of attributes/relations available. If nothing in that list
                can plausibly carry the ASKED target, KG is RULED OUT for
                this query (do not stretch field names to fit).
              - Doc / Table have OPEN content. Their `elements` and
                `data_examples` are only samples / summaries — the actual
                passages or rows may carry information that is not listed.
                You only need a reasonable inference (from description,
                elements, or examples) that the source is LIKELY to contain
                the answer. Descriptive / definitional / criteria-style
                targets are almost always covered by doc text. Tabular /
                statistical / per-row targets are typically covered by
                table rows.
              Label each live source LIKELY / RULED-OUT under these rules.
        3) Pick primary_source:
              - If KG is the only LIKELY source, primary = kg.
              - If KG is RULED OUT, pick primary from the remaining LIKELY
                sources (doc for descriptive, table for tabular). Never pick
                a RULED-OUT source as primary just because of a heuristic.
              - If both KG and another source are LIKELY, prefer the one
                whose listed elements most directly name the ASKED target;
                KG wins for structured atomic facts, doc wins for
                descriptive / criteria / narrative targets, table wins for
                statistical / row-shaped targets.
        4) Pick fallback_source from the remaining live sources; primary
           and fallback must differ whenever more than one source is live.
           A RULED-OUT source may still be used as fallback only when no
           other live source exists.

        General:
        - The answerability check above IS the decision procedure; do not
          invent extra rule-based scoring.
        - The reasoning field MUST contain: (a) the KNOWN entity, (b) the
          ASKED target, (c) which Phase-1 rule fired, and (d) for each
          live source, whether you judged it LIKELY or RULED-OUT and the
          short justification (cite an element name for KG, or describe
          why doc/table content is expected to cover or not cover it).

        Semantic catalog:
        {prompt_catalog}

        Return valid JSON only with this schema:
        {{
        "matched_concept": "the matched concept name or null",
        "primary_source": "one of: {source_list}",
        "fallback_source": "one of: {source_list}",
        "confidence": 0.0,
        "reasoning": "name the KNOWN entity, the ASKED target, and which rule fired"
        }}
    """

    parsed, _ = call_and_parse_with_retry(prompt, model, base_url, api_key, required_keys=["matched_concept", "primary_source"])
    routed = _coerce_routing_result(parsed, enabled_sources=sources)
    if not routed:
        return _default_route("LLM routing failed; defaulted to live knowledge base source.")

    matched_concept = None
    if isinstance(parsed, dict):
        matched_concept = parsed.get("matched_concept")
    if not matched_concept and catalog:
        matched_concept = matched_hint if matched_hint in catalog else next(iter(catalog.keys()))

    return {
        "matched_concept": matched_concept,
        "primary_source": routed["primary_source"],
        "fallback_source": routed["fallback_source"],
        "parse_failed": False,
        "reasoning": routed["reasoning"],
    }
