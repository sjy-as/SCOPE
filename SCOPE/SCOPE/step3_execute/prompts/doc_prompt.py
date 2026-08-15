"""
doc_prompt.py - Prompt templates for document (passage) retrieval and verification.

Mirrors prompts/table_prompt.py: every operator prompt asks the LLM to return
strict JSON so the DocSource layer can parse Answer + Evidence deterministically.
The document evidence schema is {"title": ..., "doc_id": ..., "snippet": ...}.
"""

from typing import Dict


class Prompt:
    """Collection of prompt formatters for document-based retrieval tasks."""

    @staticmethod
    def _format_doc_supporting_knowledge(supporting_knowledge: Dict, numbered: bool = True) -> str:
        """Format passage evidence into a readable block for the prompt."""
        doc_items = supporting_knowledge.get("Doc")
        if doc_items is None:
            doc_items = supporting_knowledge.get("Text")
        if not doc_items:
            return ""

        if isinstance(doc_items, list):
            blocks = []
            for i, item in enumerate(doc_items, 1):
                prefix = f"[Passage {i}]\n" if numbered else ""
                blocks.append(f"{prefix}{item}")
            return "\n\n".join(blocks) + "\n"

        return f"{doc_items}\n"

    #########################################################################################
    # Doc Search
    #########################################################################################
    @staticmethod
    def Doc_search(question, target_entity, supporting_knowledge):
        """Doc-only search prompt: select the passages most likely to hold the answer."""
        prompt = f"""You are an intelligent document retrieval judge for multi-hop QA.

Task:
Given a question and candidate passages retrieved from a document collection, select
the passages that are most likely to describe the target entity or contain the answer.
Judge each candidate using its title and passage content.

Question: {question}
Target Entity to find: {target_entity}

Decision rules (highest to lowest priority):
1. Entity Match: Prefer passages whose title or content explicitly refers to the core
   entity (player, team, award, event, season) named in the question.
2. Time/Season Match: Reject passages whose year or season clearly conflicts with the
   question. The Target Entity may carry extra constraints (years, roles); ignore those
   extra constraints here and keep the candidate if it matches the core entity.
3. Answer Coverage: Prefer passages that actually contain facts able to answer the
   question, not merely superficial keyword overlap.
4. Relative Best Match: If nothing is a perfect match, keep the relatively best passages
   that share the core entity. Do NOT reject everything if a plausible candidate exists.

CRITICAL INSTRUCTIONS:
- You MUST use the exact 'doc_id' from the Candidate Passages provided below.
  Do NOT invent, guess, or shorten IDs.

Return JSON ONLY (no markdown, no extra text):
{{
  "selected": [
    {{
      "doc_id": "<exact_id_from_context>",
      "title": "...",
      "decision": "keep",
      "reasons": ["entity_match", "time_match"]
    }}
  ],
  "reasoning": "brief summary of why these were selected"
}}

Your Turn.
Candidate Passages:

"""
        prompt += Prompt._format_doc_supporting_knowledge(supporting_knowledge, numbered=False)
        return prompt

    #########################################################################################
    # Doc Relate
    #########################################################################################
    @staticmethod
    def Doc_relate(question, supporting_knowledge, relation: str = "", target_entity: str = ""):
        """Extract the answer values for a relation from retrieved passages."""
        relation_hint = f"Relation to extract: {relation}" if relation else ""
        target_hint = f"Target entity type: {target_entity}" if target_entity else ""

        prompt = f"""You are an AI assistant extracting answer values from retrieved document passages.

Question: {question}
{relation_hint}
{target_hint}

Your task:
Read the candidate passages below and return BOTH the answer values and the exact
supporting passages (evidence).

How to reason with passages:
1. Use the passage title and body together. The answer must be grounded in the text.
2. If a target entity type is provided, return values of that type.
3. Preserve the original surface form from the passage whenever possible.
4. A single question may have multiple answers spread across passages — collect them all.
5. The question may name several entities at once (e.g. "For A, B, what is his ...").
   Treat each named entity independently and extract an answer for every entity whose
   passage supports one. If only SOME of the named entities have the information,
   still return the answer(s) you DID find — do NOT discard a found answer just
   because another named entity's passage lacks the fact.
6. FOCUS ON THE RELATION ONLY. Your job at this step is to extract every value the
   passage links to the head entity via the requested relation. Extra question
   constraints (specific season, specific team, role qualifiers, etc.) will be
   verified by a later filtering step — do NOT enforce them here.
7. Accept indirect or parenthetical mentions as valid evidence. Phrases like
   "X (who had coached Y when he played for the Nets)" explicitly assert the
   relation between X and Y and must be extracted, even if the relation appears
   inside an aside, a relative clause, or a brief mention.
8. If the passage indicates the head entity served in a coaching/captaining/
   leadership role for a team, any teammates or rostered players mentioned in the
   same passage as being on that team during that tenure also count as valid
   answers under "coached / led / managed" style relations.

IMPORTANT RULES:
1. Extract values STRICTLY from the provided passages — do NOT hallucinate.
2. Do NOT write full sentences as answers — just the values.
3. Output "Answer": ["None"] ONLY when the passages contain ABSOLUTELY NO statement
   linking the head entity to any value via the requested relation. As long as the
   relation is supported for at least one value — even with a mismatched year, team,
   or other qualifier — return that value and let the downstream Filter handle the
   remaining constraints. Do NOT fall back to ["None"] merely because the matched
   value's surrounding context does not satisfy every condition in the question.
4. When answering with numbers, use Arabic numerals (e.g., 1, 2, 3).
5. For Yes/No questions, output "Answer": ["Yes"] or "Answer": ["No"].
6. The 'snippet' in each evidence row must be the exact sentence(s) that support the answer.

Return exactly this JSON schema (no markdown, no extra text):
{{
  "Answer": ["xxx"],
  "Evidence": [
    {{
      "title": "...",
      "doc_id": "...",
      "snippet": "the exact supporting sentence(s) from the passage"
    }}
  ]
}}

Your Turn.
Retrieved Passages:
"""
        prompt += Prompt._format_doc_supporting_knowledge(supporting_knowledge, numbered=True)
        prompt += f"\nQuestion: {question}\n"
        return prompt

    #########################################################################################
    # Doc Filter
    #########################################################################################
    @staticmethod
    def Doc_filter(question, condition, candidate_answers, supporting_knowledge):
        """Filter candidate answers using passage-level evidence.

        ``candidate_answers`` are the answer values an earlier step already
        extracted; the supporting passages are evidence ONLY. The filter must
        keep or drop those candidate answers — it must NOT echo a passage title
        or entity name as the answer.
        """
        if isinstance(candidate_answers, (list, tuple)):
            answers_block = "\n".join(f"  - {a}" for a in candidate_answers if str(a).strip())
        else:
            answers_block = str(candidate_answers or "").strip()
        if not answers_block:
            answers_block = "  (none)"

        prompt = f"""You are an AI assistant responsible for filtering candidate answers using
document passage evidence.

Question: {question}
Condition to satisfy: {condition}

Candidate Answers (extracted by an earlier step — THESE are what you must filter):
{answers_block}

You will be given the candidate answers above along with supporting passage evidence
(title, doc_id, snippet). Keep only the candidates whose evidence explicitly satisfies
the condition. This is only a filtering step — do NOT answer the full question here.

IMPORTANT RULES:
1. Your output answers MUST be drawn from the Candidate Answers list above. NEVER
   invent a new answer — in particular, do NOT output a passage title, a doc_id, or
   an entity name that is not among the Candidate Answers. The passages are evidence
   only, not answer choices.
2. Evaluate ONLY the condition itself, not the final intent of the question. Do NOT
   reject candidates just because the overall question later needs aggregation.
3. If a candidate's passage evidence clearly satisfies the condition, keep it.
4. If a candidate's passage evidence clearly violates the condition, discard it.
5. If a candidate's passage evidence lacks sufficient information to verify the
   condition (the condition may belong to a later reasoning step, or the passage
   simply does not address it), DO NOT automatically discard the candidate. Keep it
   as a tentative candidate marked "kept due to insufficient information".
6-1. If the surviving candidates include BOTH (a) those whose passages explicitly
     satisfy the condition AND (b) those kept due to insufficient information,
     discard group (b) and keep only group (a).
6-2. If the surviving candidates include ONLY group (b) (i.e., NO candidate has
     explicit supporting evidence, but none is contradicted either), KEEP them all
     rather than returning an empty list. Downstream operators will handle the
     remaining condition.
7. Output "Answer": [] ONLY when every candidate's evidence explicitly violates the
   condition.
8. Do NOT deduplicate candidates by label. If the same label is supported by multiple
   passages, keep all of those evidence rows.

Return valid JSON ONLY using this schema:
{{
  "Answer": ["exact candidate label 1", "exact candidate label 2"],
  "Evidence": [
    {{
      "title": "...",
      "doc_id": "...",
      "snippet": "the exact sentence(s) that justify the decision"
    }}
  ]
}}

Your Turn.
Supporting Passages:
"""
        prompt += Prompt._format_doc_supporting_knowledge(supporting_knowledge, numbered=False)
        prompt += f"\nQuestion: {question}\nCondition: {condition}\n"
        return prompt

    #########################################################################################
    # Doc Math
    #########################################################################################
    @staticmethod
    def Doc_math(question, operation, candidate_answers, supporting_knowledge):
        """Compute a math/stat result from passage evidence.

        ``candidate_answers`` are the values an earlier step extracted — they
        are the data the operation runs over. The passages are supporting
        evidence ONLY and must not be mined for a passage title as a result.
        """
        if isinstance(candidate_answers, (list, tuple)):
            answers_block = "\n".join(f"  - {a}" for a in candidate_answers if str(a).strip())
        else:
            answers_block = str(candidate_answers or "").strip()
        if not answers_block:
            answers_block = "  (none)"

        prompt = f"""You are an AI assistant for DOC-MATH.
This step computes a final math/statistical result from provided passage evidence only.

Question: {question}
Operation: {operation}

Candidate Answers (extracted by an earlier step — the data to operate on):
{answers_block}

TASK:
Compute the result strictly from the candidate answers above and the provided passage
evidence.

RULES (STRICT):
1. Use ONLY the provided candidate answers and passages. No external knowledge.
2. Repeated values represent separate evidence rows and must NOT be merged or deduplicated.
3. If the computation uses N passages, output all N of them in the Evidence section.
4. Keep reasoning implicit — only return the final result.
5. If data is insufficient or ambiguous, output "Answer": [].
6. If only one result exists, "Answer" must still be a JSON array with one string.

Return exactly this JSON schema (no markdown, no extra text):
{{
  "Answer": ["<result_1>", "<result_2>"],
  "Evidence": [
    {{
      "title": "...",
      "doc_id": "...",
      "snippet": "the exact sentence(s) used in the computation"
    }}
  ]
}}

Your Turn.
Supporting Passages:
"""
        prompt += Prompt._format_doc_supporting_knowledge(supporting_knowledge, numbered=False)
        prompt += f"\nQuestion: {question}\nOperation: {operation}\n"
        return prompt

    #########################################################################################
    # Doc Final Verification
    #########################################################################################
    @staticmethod
    def Doc_final_verification(question: str, candidate_answers, candidate_evidence: str) -> str:
        """Strict final verification / synthesis over passage-grounded candidates.

        ``candidate_answers`` are the answer values an earlier step already
        extracted; ``candidate_evidence`` is the supporting passage text. The
        verifier must judge the candidate answers — it must NOT re-extract a new
        answer (such as a passage title or award name) from the passages.
        """
        if isinstance(candidate_answers, (list, tuple)):
            answers_block = "\n".join(f"  - {a}" for a in candidate_answers if str(a).strip())
        else:
            answers_block = str(candidate_answers or "").strip()
        if not answers_block:
            answers_block = "  (none)"

        prompt = f"""You are an AI assistant performing final verification and synthesis on
candidate answers that were already extracted from document passages.

Original Question: {question}

Candidate Answers (extracted by an earlier step — THESE are what you must verify):
{answers_block}

Supporting Passages (evidence backing the candidate answers above; each has title,
doc_id, and snippet):
{candidate_evidence}

TASK:
Decide which of the Candidate Answers above correctly answer the Original Question,
using the Supporting Passages as evidence. You may FILTER (drop wrong candidates) or
SYNTHESIZE (merge/clean the candidates into a combined answer).

IMPORTANT RULES:
1. Your output answers MUST be drawn from the Candidate Answers list above (you may
   merge or lightly clean them). NEVER invent a new answer — in particular, do NOT
   output a passage title, an award name, or an entity name that is not among the
   Candidate Answers. The passages are evidence only, not answer choices.
2. Do not assume conjunctions like "both" unless they actually appear in the question.
3. If a candidate's passage evidence explicitly satisfies or violates a constraint,
   filter accordingly: keep those that satisfy, discard those that are contradicted.
4. If a candidate's passage evidence lacks sufficient information to verify a
   constraint (the constraint may belong to a later step, or the passages simply do
   not address it), DO NOT automatically discard the candidate. Keep it as a
   tentative candidate marked "kept due to insufficient information".
5-1. If the surviving candidates include BOTH (a) those whose passages explicitly
     satisfy every constraint AND (b) those kept due to insufficient information,
     discard group (b) and keep only group (a).
5-2. If the surviving candidates include ONLY group (b) (i.e., NO candidate is
     explicitly supported by all constraints, but none is contradicted either),
     KEEP them all rather than returning None.
6. Output "Answer": ["None"] ONLY when EVERY candidate is explicitly contradicted
   by the evidence. If at least one candidate is supported or merely unverified, you
   must return it.
7. BE EXTREMELY CONCISE in reasoning (1-3 sentences maximum).

Return exactly this JSON schema (no markdown, no extra text):
{{
  "Answer": ["exact candidate 1", "exact candidate 2"],
  "Evidence": [
    {{
      "title": "...",
      "doc_id": "...",
      "snippet": "the exact supporting sentence(s)"
    }}
  ]
}}

Your Turn.
"""
        return prompt
