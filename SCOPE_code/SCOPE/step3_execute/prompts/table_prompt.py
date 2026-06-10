"""
prompts.py - Dataset-specific prompt
"""

from typing import Dict


class Prompt:
    """Collection of prompt formatters for different datasets."""

    @staticmethod
    def _format_table_supporting_knowledge(supporting_knowledge: Dict, numbered: bool = True) -> str:
        """Format table evidence while preserving existing prompt interfaces."""
        table_items = supporting_knowledge.get("Table")
        if not table_items:
            return ""

        if isinstance(table_items, list):
            blocks = []
            for i, item in enumerate(table_items, 1):
                prefix = f"[Table {i}]\n" if numbered else ""
                blocks.append(f"{prefix}{item}")
            return "\n\n".join(blocks) + "\n"

        return f"{table_items}\n"

    #########################################################################################
    # Table Search
    #########################################################################################   
    @staticmethod
    def Table_search(question, target_entity, supporting_knowledge):
        """Table-only search prompt: intelligent relevance ranking."""
        prompt = f"""You are an intelligent table retrieval judge for multi-hop QA.

            Task:
            Given a question and candidate tables, select the tables that are most likely to contain the answer.
            Judge each candidate using page_title, section_title, and header/schema clues.

            Question: {question}
            Target Entity to find: {target_entity}

            Decision rules (in priority order from highest to lowest):
            1. Time/Season Match: Reject tables whose year or season explicitly and clearly conflicts with the question. This is the highest priority.
            2. Entity/Scope Relevance: Prefer tables whose page_title or section_title explicitly matches the core entity, team, player, award, or event in the question.
            3. Event Match: Pay attention to event constraints (e.g., playoffs vs finals, regular season vs draft).
            4. Relative Best Match: If no table is a 100% perfect match, keep the relatively best ones that share the core entity and time. Do NOT reject all tables if a plausible candidate exists.

            CRITICAL INSTRUCTIONS:
            - You MUST use the exact 'table_id' from the Candidate Tables provided below. Do NOT invent, guess, or shorten IDs (e.g., do not output "t1" or "t2").
            - If a candidate shows "table_id: 2-18409087-10", you must output exactly "2-18409087-10".

            Return JSON ONLY (no markdown, no extra text):
            {{
            "selected": [
                {{
                "table_id": "<exact_id_from_context>",
                "page_title": "...",
                "section_title": "...",
                "decision": "keep",
                "reasons": ["time_match", "entity_match", "event_match"]
                }}
            ],
            "reasoning": "brief summary of why these were selected"
            }}

            Your Turn.
            Candidate Tables:

        """
        prompt += Prompt._format_table_supporting_knowledge(supporting_knowledge, numbered=False)

        return prompt

    #########################################################################################
    # Table Relation
    #########################################################################################
    @staticmethod
    def Table_relate(question, supporting_knowledge, relation: str = "", target_entity: str = ""):
        """
        从表格/KG/文本中抽取与 relation 对应的值列表。
        relation: 自然语言描述的关系，如 'teams played for'
        target_entity: 目标实体类型，如 'team'（可选）
        """
        relation_hint = f"Relation to extract: {relation}" if relation else ""
        target_hint = f"Target entity type: {target_entity}" if target_entity else ""

        prompt = f"""You are an AI assistant extracting answer values from retrieved table evidence.

        Question: {question}
        {relation_hint}
        {target_hint}

        Your task:
        Read the candidate tables below and return BOTH the answer and the exact supporting evidence rows.

        How to reason with tables:
        1. Use page title, section title, header names, and row values together.
        2. If target entity type is provided, prefer values taken directly from the most relevant answer-bearing column.
        3. If the answer is expressed as a result, outcome, ranking, score, or performance instead of a single obvious column, summarize the relevant cell values concisely from the matching row(s).
        4. Preserve the original surface form from the table whenever possible.
        5. The question may name several entities at once (e.g. "For A, B, what is his ...").
           Treat each named entity independently and extract an answer for every entity whose
           tables support one. If only SOME named entities have the information, still return
           the answer(s) you DID find — do NOT discard a found answer just because another
           named entity's tables lack the fact.
        6. FOCUS ON THE RELATION ONLY. Your job at this step is to extract every value the
           table links to the head entity via the requested relation. Extra question
           constraints (specific season, specific month, specific opponent, role qualifiers,
           etc.) will be verified by a later filtering step — do NOT enforce them here.
        7. If the page_title or section_title already constrains the rows to the right
           team/season scope (e.g. "2006–07 Golden State Warriors season → November"),
           then EVERY player surfaced in those rows under the relevant column counts as a
           valid candidate under relations like "play alongside / teammate of / led /
           backed up / coached". The shared scope IS the evidence — do not require an
           explicit cell that names the relation verb.

        IMPORTANT RULES:
        1. Output ONLY the extracted values as a comma-separated list after 'Answer:'.
        2. Do NOT write full sentences or paragraphs — just the values.
        3. Output Answer: None ONLY when the tables contain ABSOLUTELY NO row that links
           the head entity to any value via the requested relation. As long as the relation
           is supported for at least one value — even with a mismatched year, month, or
           other qualifier — return that value and let the downstream Filter handle the
           remaining constraints. Do NOT fall back to None merely because the matched
           value's surrounding context does not satisfy every condition in the question.
        4. When answering with numbers, use Arabic numerals (e.g., 1, 2, 3).
        5. For Yes/No questions, output Answer: Yes or Answer: No.
        6. Extract values STRICTLY from the provided knowledge — do NOT hallucinate.

        Return exactly this JSON schema:
        {{
          "Answer": ["xxx"],
          "Evidence": [
            {{
              "page_title": "...",
              "section_title": "...",
              "header": ["A", "B", "C"],
              "row": {{"A": "1", "B": "2", "C": "3"}}
            }}
          ]
        }}

        Your Turn.
        Retrieved Tables:
        """

        prompt += Prompt._format_table_supporting_knowledge(supporting_knowledge, numbered=True)
        return prompt
    
    @staticmethod
    def Table_relate_summarize(
            question: str,
            relation: str,
            table_text: str,
        ) -> str:
        """
        当 Relate 无法找到特定列时，让 LLM 直接基于整张表综合生成答案。
        适用于 relation 是抽象概念（如 performance、result、outcome）的情况。
        """
        prompt = f"""You are an AI assistant answering a question from a single table.

            Question: {question}
            What to extract: {relation}

            Table:
            {table_text}

            TASK:
            Based on the full table above, directly answer what the '{relation}' is.
            Summarize only the relevant row or cell content needed to answer.
            Be concise but complete — include key facts like wins, losses, scores, or outcomes.

            IMPORTANT RULES:
            1. Base your answer STRICTLY on the table data provided.
            2. Output a single concise answer after 'Answer:'.
            3. Do NOT hallucinate information not in the table.

            Strictly follow this output format:
            Answer: [your concise answer based on the table]
            Reasoning: [which rows/columns you used]
        """
        return prompt

    #########################################################################################
    # Table Filter
    #########################################################################################

    @staticmethod
    def Table_filter(question, condition, supporting_knowledge):
        prompt = f"""You are an AI assistant responsible for filtering candidate answers derived from tables using row-level evidence.

        Question: {question}
        Condition to satisfy: {condition}

        You will be given candidate answers along with related table context, such as page title, section title, header, and matching row values.
        Your task is to keep only those results whose table evidence explicitly satisfies the condition.
        This is only a filtering step. Do NOT answer the full question here.

        IMPORTANT RULES:
        1. Evaluate ONLY the condition itself, not the final intent of the question. Do NOT reject candidates just because the overall question would later need aggregation across rows. Aggregation is not your job in this step. Those are handled by subsequent operators.
        2. Use a combination of row/cell values, page titles, and section titles as evidence.
        3. If a candidate's evidence contains explicit information that satisfies or violates the condition, filter accordingly: keep those that satisfy, discard those that violate.
        4. If a candidate's evidence lacks sufficient information to verify the condition (the condition may be slated for a later step, or the evidence simply does not address it), DO NOT automatically discard the candidate. Instead, keep it as a candidate and mark it as "kept due to insufficient information".
        5-1. If the surviving candidates include BOTH (a) those whose evidence explicitly satisfies the condition AND (b) those kept due to insufficient information, discard group (b) and keep only group (a).
        5-2. If the surviving candidates include ONLY group (b) (i.e., NO candidate has explicit supporting evidence, but none is contradicted either), KEEP them all rather than returning an empty list. Downstream operators will handle the remaining condition.
        6. Output Answer: [] ONLY when every candidate's evidence explicitly violates the condition.
        7. IMPORTANT: Do NOT deduplicate candidates by label. If the same label appears in multiple different rows/evidence items, keep all of them in the output list (repeated labels allowed).

        Output valid JSON ONLY using this schema:
        {{
          "Answer": ["exact candidate label 1", "exact candidate label 2"],
          "Evidence": [
            {{
              "page_title": "...",
              "section_title": "...",
              "header": ["A", "B", "C"],
              "row": {{"A": "1", "B": "2", "C": "3"}}
            }}
          ]
        }}

        Examples:

        Example 1 (Clear evidence — filter works):
        Table evidence:
        [Candidate: Dell Curry]
        Page: Charlotte Hornets expansion draft
        Section: Draft selections
        Header: Pick | Player | Former Team
        Row: 8 | Dell Curry | Cleveland Cavaliers
        Question: Which player was chosen with pick number 8?
        Condition: pick number is 8
        Answer: ["Dell Curry"]

        Example 2 (Condition satisfied by shared page metadata):
        Table evidence:
        [Candidate: Won 1]
        Page: 1995-96 Miami Heat season
        Section: Schedule
        Header: Date | Opponent | Streak
        Row: Nov 3 | @ IND | Won 1
        Question: What is the best win streak for the team Miami Heat in the 1995-96 schedule?
        Condition: games from 1995-96 season
        Answer: ["Won 1"]

        Your Turn.
        [Formatted table knowledge inserted here]
        """
        prompt += Prompt._format_table_supporting_knowledge(supporting_knowledge, numbered=False)
        return prompt


    #########################################################################################
    # Table Math
    #########################################################################################
    @staticmethod
    def Table_math(question, operation, supporting_knowledge):
        prompt = f"""You are an AI assistant for TABLE-MATH.
        This step computes a final math/stat result from provided table evidence only.

        Question: {question}
        Operation: {operation}

        TASK:
        Compute the result strictly from the provided table evidence.

        RULES (STRICT):
        1) Use ONLY provided evidence. No external knowledge.
        2) Keep Reasoning VERY SHORT (max 5 sentences, max 250 words total).
        3) Output MUST be valid JSON and MUST follow this exact schema:
        Reasoning: <short text>
        Answer: ["<result_1>", "<result_2>"]
        4) If only one result exists, Answer must still be a JSON array with one string.
        5) If data is insufficient/ambiguous, output:
        Reasoning: insufficient evidence
        Answer: []

        Output exactly this schema:
        {{
          "Answer": ["<result_1>", "<result_2>"],
          "Evidence": [
            {{
              "page_title": "...",
              "section_title": "...",
              "header": ["A", "B", "C"],
              "row": {{"A": "1", "B": "2", "C": "3"}}
            }}
          ]
        }}

        Retrieved Tables:
        """
        prompt += Prompt._format_table_supporting_knowledge(supporting_knowledge, numbered=False)
        prompt += f"\nQuestion: {question}\nOperation: {operation}\n"
        return prompt


    #########################################################################################
    # Table Final Verification
    #########################################################################################
    @staticmethod
    def Table_final_verification(question: str, candidate_evidence: str) -> str:
        prompt = f"""You are an AI assistant performing final verification and synthesis on candidate answers extracted from tables.

            Original Question: {question}

            TASK:
            Based on the Original Question, determine if you need to FILTER specific candidates OR SYNTHESIZE a combined answer from the evidence.
            Below is a list of candidate answers along with their detailed row evidence (Header: Value).

            IMPORTANT RULES:
            1. FACTUAL FILTERING (For specific entities): Verify each candidate independently against every constraint stated in the Original Question.
            2. SYNTHESIS / SUMMARIZATION: If the question asks for an overall outcome or count, aggregate the data.
            3. If a candidate's evidence contains explicit information that satisfies or violates a constraint, filter accordingly: keep those that satisfy, discard those that violate.
            4. If a candidate's evidence lacks sufficient information to verify a constraint (the constraint may have been intended for a later reasoning step, or the evidence simply does not address it), DO NOT automatically discard the candidate. Keep it as a tentative candidate marked "kept due to insufficient information".
            5-1. If the surviving candidates include BOTH (a) those whose evidence explicitly satisfies all constraints AND (b) those kept due to insufficient information, discard group (b) and keep only group (a).
            5-2. If the surviving candidates include ONLY group (b) (i.e., NO candidate is explicitly supported by all constraints, but none is contradicted either), KEEP them all rather than returning None.
            6. Output Answer: ["None"] ONLY when every candidate's evidence explicitly contradicts a constraint in the question.
            7. BE EXTREMELY CONCISE: Your reasoning MUST be extremely brief (1 to 3 sentences maximum). Do not over-explain or analyze row-by-row out loud. Get straight to the point.
            8. Output your final result as a valid JSON array of strings containing the distinct answers.
            9. Clean the names: remove extra characters like "&", "and", or years if they are mixed into the host names. Split combined names (e.g., "A and B" -> "A", "B").

            Return exactly this JSON schema:
            {{
              "Answer": ["exact candidate 1", "exact candidate 2"],
              "Evidence": [
                {{
                  "page_title": "...",
                  "section_title": "...",
                  "header": ["A", "B", "C"],
                  "row": {{"A": "1", "B": "2", "C": "3"}}
                }}
              ]
            }}

            Your Turn.
            Candidate Evidence:
            {candidate_evidence}
        """
        return prompt


    #########################################################################################
    # Table Final Answer
    #########################################################################################

    @staticmethod
    def Table_final_answer(
        question: str,
        sq_summaries: str,
        ) -> str:
        """
        基于多跳推理的中间结果，对原始问题生成最终自然语言回答。
        sq_summaries: 每个 subquery 的 id、question、answers 拼成的字符串
        """
        return f"""You are an AI assistant that answers multi-hop questions.

            You have gathered the following intermediate results from multiple reasoning steps:

            {sq_summaries}

            Original question: {question}

            TASK:
            Based on ALL the intermediate results above, provide a concise and direct final answer to the original question.
            Do NOT repeat the intermediate steps. Just give the final answer.

            IMPORTANT RULES:
            1. Use only the information from the intermediate results.
            2. Be concise — one sentence or a short phrase is ideal.
            3. If the evidence contains numbers or scores, summarize them clearly.

            Strictly follow this format:
            Reasoning: [brief explanation of how you derived the answer from the intermediate results]
            Answer: [your final concise answer]
        """
