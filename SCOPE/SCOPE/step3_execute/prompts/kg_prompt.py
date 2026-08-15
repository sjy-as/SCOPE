"""
prompts.py - Dataset-specific prompt
"""

from typing import Dict


class Prompt:
    """Collection of prompt formatters for different datasets."""
    #########################################################################################
    # KG Search
    #########################################################################################
    @staticmethod
    def KG_search(question, target_entity, supporting_knowledge):
        prompt = f"""You are an AI assistant helping to verify if retrieved candidate entities match the CORE target entity requested in the question.

        Question: {question}
        Target Entity to find: {target_entity}

        You will be provided with retrieved knowledge from Knowledge Graphs (KG). 
        Evaluate each candidate's label and metadata to determine if it represents the core entity.

        IMPORTANT INSTRUCTIONS:
        1. The 'Target Entity to find' often contains extra constraints like dates (e.g., '1998'), roles, or event descriptors (e.g., 'induction'). You should IGNORE these extra constraints and ACCEPT the candidate if it accurately matches the CORE entity (e.g., the base organization, award, team, or person). The extra temporal or contextual constraints will be handled in a later filtering step.
        2. Beware of fuzzy matches that share similar tokens but are actually completely different entities (e.g., 'Peter Verhoeven' vs 'Peter Holt').

        Strictly follow this output format:
        Answer: [comma-separated list of exact candidate labels that are correct matches, or 'None']
        Reasoning: [Brief explanation of why they matched (mentioning ignored constraints if any) or why all were rejected]

        Examples:
        Retrieved Knowledge:
        KG: [{{'label': 'North Carolina Sports Hall of Fame', 'meta': {{'matched_by': 'bm25'}}}}, {{'label': 'Ontario Sports Hall of Fame', 'meta': {{'matched_by': 'bm25'}}}}]
        Question: Who is the player who received the North Carolina Sports Hall of Fame induction in 1998?
        Target Entity to find: North Carolina Sports Hall of Fame induction 1998
        Answer: North Carolina Sports Hall of Fame
        Reasoning: The candidate matches the core entity "North Carolina Sports Hall of Fame". The extra descriptors "induction" and "1998" can be safely ignored at this entity-matching stage.

        Retrieved Knowledge:
        KG: [{{'label': 'Peter Verhoeven', 'meta': {{'matched_by': 'fuzzy', 'score': 0.499}}}}, {{'label': 'Peter Thibeaux', 'meta': {{'matched_by': 'fuzzy', 'score': 0.479}}}}]
        Question: Which team was founded by Peter Holt?
        Target Entity to find: Peter Holt
        Answer: None
        Reasoning: The retrieved entities "Peter Verhoeven" and "Peter Thibeaux" are fuzzy matches and represent different people than the core entity "Peter Holt".

        Your Turn.
        Retrieved Knowledge:
        """
        prompt += f"KG: {supporting_knowledge['KG']}\n"
        
        prompt += f"\nQuestion: {question}\nTarget Entity to find: {target_entity}\n"
        
        return prompt
    
    #########################################################################################
    # KG Relate
    #########################################################################################
    @staticmethod
    def KG_relate(question, supporting_knowledge):
        prompt = f"""Please answer the question "{question}" using the retrieved knowledge from knowledge graphs, tables, or texts.  
        Your response should include: a Paraphrase Answer that restates the answer to the question.

        If the provided information is not sufficient to answer the question, end your answer with: (1) Paraphrase Answer: Unknown; (2) Evidence Source List: [].  
        When answering with numbers, always use Arabic numerals, e.g., 1, 2, 3.  
        When answering "Yes" or "No" questions, simply format your answer list as ["Yes"] or ["No"].

        Strictly follow the format of the examples below, and end your answer with: "So the next step in filtering or calculation can be carried out from these sources: [evidence_source_list]"

        Examples.
        Retrieved Knowledge: 
        KG: [1]......|.......
        [2].....|.......
        Question: What is .......of ......?
        Answer: ..... is ....of ... "So, the next step can be carried out from these sources: {"KG": ......, "KG": ......}"

        Your Question.
        Retrieved Knowledge:
        """
        
        prompt += f"\nKG Results: {supporting_knowledge['KG']}"
        
        prompt += f"""\nQuestion: {question}\nAnswer: """
        
        return prompt
    
    @staticmethod
    def KG_relation_mapping(question: str, relation_text: str, available_relations: list):
        """Map natural language relation to actual graph relation names using LLM."""
        prompt = f"""You are a knowledge graph expert. Your task is to map a natural language relation description to the most appropriate relation name in the knowledge graph.

        Question: {question}
        Natural Language Relation: {relation_text}
        Available Relations in KG: {available_relations}

        Based on semantic similarity and the context of the question, output ONLY the exact relation name from the available list that best matches the natural language relation.
        If no relation is a good match, output 'None'.

        Examples:
        Question: Which team did John Havlicek play for?
        Natural Language Relation: team played for
        Available Relations: ['playsFor', 'draftedBy', 'coachedBy', 'hasHomeVenue']
        Answer: playsFor

        Question: Who was the coach of the team?
        Natural Language Relation: has_role
        Available Relations: ['playsFor', 'draftedBy', 'coachedBy', 'hasHomeVenue', 'isSameAs']
        Answer: isSameAs

        Your Turn.
        Question: {question}
        Natural Language Relation: {relation_text}
        Available Relations: {available_relations}
        Answer: """
        return prompt

    @staticmethod
    def KG_property_extraction(question: str, source_entity: str, target_type: str, relation: str, entity_props: list):
        """从源实体的属性中提取目标信息。"""
        prompt = f"""You are a knowledge graph expert. Your task is to identify which property/attribute of the source entity should contain information about the target entity type.

        Question: {question}
        Source Entity: {source_entity}
        Target Entity Type: {target_type}
        Relation: {relation}
        Available Properties: {entity_props}

        Based on the question and the relation semantics, determine which property/attribute of the source entity would contain the target information.

        For example:
        - If looking for "season drafted in", the property might be "work_period_start", "draft_year", "draft_season", "drafted_in_season", but not "date_of_birth"
        - If looking for "team played for", the property might be "team", "plays_for", or "team_name"
        - If looking for "coach of", the property might be "coach", "head_coach", or "coach_name"

        Question: In which season was Hakim Warrick drafted into the NBA?
        Source Entity: Hakim Warrick
        Target Entity Type: season
        Relation: drafted_in_season
        Available Properties: ['enwiki_title', 'country_of_citizenship', 'place_of_birth', 'native_language', 'sex_or_gender', 'date_of_birth', 'height', 'mass', 'work_period_start', 'schools_attended']...
        Property Name: work_period_start

        Question: What award did Hakim Warrick receive?
        Source Entity: Hakim Warrick
        Target Entity Type: award
        Relation: received_award
        Available Properties: ['enwiki_title', 'country_of_citizenship', 'place_of_birth', 'native_language', 'sex_or_gender', 'date_of_birth', 'height', 'mass', 'work_period_start', 'schools_attended']...
        Property Name: None

        Output ONLY the most likely property name that would contain this information.
        If uncertain, output None.

        Question: {question}
        Source Entity: {source_entity}
        Target Type: {target_type}
        Relation: {relation}
        Available Properties: {entity_props}
        Property Name: """
        return prompt

    @staticmethod
    def KG_schema_mapping(question, relation: str, target_entity: str, columns: list, missing_entity: str = "", entity_type: str = ""):
        prompt = f"""You are a database schema expert. Your task is to find the most appropriate column in a database table that can help answer the query.

        QUERY CONTEXT:
        Question: {question}
        Missing Entity: {missing_entity}
        Missing Entity Type: {entity_type}
        Target Table: {target_entity}
        Relation to map: {relation}
        Available Columns: {columns}

        TASK:
        The query is trying to find a {target_entity} that has a specific relationship ('{relation}') with the entity '{missing_entity}' (type: {entity_type}).
        
        You need to identify which column in the {target_entity} table would contain information about this relationship.
        
        Output ONLY the exact column name from the available list that best represents this relationship.
        If no column is a good match, output 'None'.

        Examples:
        Question: Which team was founded by Peter Holt?
        Missing Entity: Peter Holt
        Missing Entity Type: person
        Target Table: team
        Available Columns: ['wikidata_id', 'name', 'enwiki_title', 'inception', 'owners']
        Relation to map: founded team
        Answer: owners

        Question: Which player was drafted by the Boston Celtics?
        Missing Entity: Boston Celtics
        Missing Entity Type: team
        Target Table: player
        Available Columns: ['wikidata_id', 'name', 'draft_team', 'draft_year', 'position']
        Relation to map: drafted by
        Answer: draft_team

        Your Turn.
        Question: {question}
        Missing Entity: {missing_entity}
        Missing Entity Type: {entity_type}
        Target Table: {target_entity}
        Available Columns: {columns}
        Relation to map: {relation}
        Answer: """
        return prompt

    @staticmethod
    def KG_relation_type_check(question: str, relation: str, target_type: str, retrieved_entities: list):
        """验证通过关系查询得到的实体是否真的是目标类型。"""
        prompt = f"""You are a knowledge graph expert. Your task is to verify whether the retrieved entities are of the expected ENTITY TYPE only.

        Question: {question}
        Relation Used: {relation}
        Expected Target Type: {target_type}
        Retrieved Entities: {retrieved_entities}

        Important:
        - Check ONLY the coarse entity category, such as player, team, coach, season, award, city.
        - IGNORE extra constraints from the question such as person name, year, ranking, nationality, or other filtering conditions.
        - If the retrieved entities are players but the question is asking for a specific player name, that should still be PASS for type checking.
        - Return FAIL only when the retrieved entities are mainly the wrong kind of thing, such as teams instead of players, or awards instead of seasons.

        Output format:
        Reasoning: [Brief explanation]
        Verification: [PASS or FAIL]

        Examples:
        Question: Which season was Hakim Warrick drafted in?
        Relation Used: drafted_in_season
        Expected Target Type: season
        Retrieved Entities: ['Memphis Grizzlies']
        Reasoning: The relation 'drafted_in_season' returned 'Memphis Grizzlies' which is a team, not a season. The relation mapping is incorrect - we need to find the season from the source entity's properties instead.
        Verification: FAIL

        Question: Which team did John Havlicek play for?
        Relation Used: playsFor
        Expected Target Type: team
        Retrieved Entities: ['Boston Celtics']
        Reasoning: 'Boston Celtics' is indeed a team, matching the expected type. Any later filtering on other constraints should be handled separately.
        Verification: PASS

        Your Turn.
        Question: {question}
        Relation Used: {relation}
        Expected Target Type: {target_type}
        Retrieved Entities: {retrieved_entities}
        Reasoning:
        Verification: """
        return prompt

    #########################################################################################
    # KG Filter
    #########################################################################################
    @staticmethod
    def KG_filter(question, condition, supporting_knowledge):
        prompt = f"""You are an AI assistant helping to filter candidate answers based on a specific condition.

        Question: {question}
        Condition to satisfy: {condition}

        You will be provided with retrieved knowledge from Knowledge Graphs (KG). Each candidate answer comes with its metadata/evidence.
        Your task is to decide which candidate rows should be kept.

        Important Rules:
        1. If a candidate's metadata contains explicit evidence that satisfies or violates the condition, filter accordingly: keep those that satisfy the condition, discard those that violate it.
        2. If a candidate's metadata lacks sufficient information to verify the condition, DO NOT automatically discard it. Instead, keep it as a candidate and state that it cannot be accurately verified due to insufficient information.
        3-1. If the remaining candidates include both (a) those that passed the condition filter and (b) those kept due to lack of information, then discard the ones kept due to lack of information, keeping only those that passed the condition filter.
        3-2. If the remaining candidates include ONLY those kept due to lack of information, then keep them.
        4. If the same label appears multiple times, treat each row as a separate evidence record (potentially with different metadata, such as year/time) to facilitate later calculations or operations. Do not collapse them into one row.

        Strict output format (PLAIN TEXT ONLY — do NOT wrap section labels in markdown bold `**`, headers, or bullets. Write the labels exactly as `Answer:`, `Reasoning:`, `Evidence:` on their own lines):
        Answer: [comma-separated list of candidate labels to keep]
        Reasoning: [brief explanation]
        Evidence:
        [each kept evidence row on its own line, using exact dict-like Python syntax]

        Evidence row format rules:
        - Copy the original candidate row as a Python dict literal.
        - Include all original fields that are needed to identify the row.
        - Preserve repeated labels as repeated rows if they are separate evidence records.
        - Do not deduplicate.

        Example:
        Retrieved Knowledge:
        KG: [{{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'start time': '1962', 'end time': '1978'}}}}, {{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'start time': '1965', 'end time': '1967'}}}}, {{'label': 'Los Angeles Lakers', 'qid': 'Q456', 'meta': {{'start time': '1979'}}}}]
        Question: Which team did he play for in 1965?
        Condition: affiliation during 1965
        Answer: Boston Celtics, Boston Celtics
        Reasoning: Both Boston Celtics evidence rows include 1965 in their active period.
        Evidence:
        {{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'start time': '1962', 'end time': '1978'}}}}
        {{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'start time': '1965', 'end time': '1967'}}}}

        Your Turn.
        """

        prompt += f"KG: {supporting_knowledge['KG']}\n"
        prompt += f"\nQuestion: {question}\nCondition: {condition}\n"
        prompt += "\nReturn the Answer, Reasoning, and Evidence sections exactly as specified.\n"

        return prompt

    #########################################################################################
    # KG Math
    #########################################################################################
    @staticmethod
    def KG_math(question, operation, supporting_knowledge):
        prompt = f"""You are an AI assistant helping to solve a math-style question over retrieved KG evidence.

        Question: {question}
        Operation: {operation}

        You will be given a list of KG evidence rows. Your task is to compute the answer and also return the exact evidence rows that support the computation.

        Important Rules:
        1. Repeated KG labels represent separate evidence rows and must NOT be merged.
        2. Do not deduplicate evidence rows.
        3. If the computation uses 3 evidence rows, output all 3 rows explicitly in the Evidence section.
        4. The Evidence section must contain the full rows you used for the computation, not just labels.
        5. Keep the original row granularity exactly as provided.

        Strict output format (PLAIN TEXT ONLY — do NOT wrap section labels in markdown bold `**`, headers, or bullets. Write the labels exactly as `Answer:`, `Reasoning:`, `Evidence:` on their own lines):
        Answer: [computed result as a comma-separated list or a single value]
        Reasoning: [brief explanation of how the result was computed]
        Evidence:
        [each supporting evidence row on its own line, using exact dict-like Python syntax]

        Example:
        Retrieved Knowledge:
        KG: [{{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'season': '1964'}}}}, {{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'season': '1965'}}}}, {{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'season': '1966'}}}}]
        Question: How many seasons did Boston Celtics appear in the evidence?
        Operation: count
        Answer: 3
        Reasoning: There are three separate evidence rows, so the count is 3.
        Evidence:
        {{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'season': '1964'}}}}
        {{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'season': '1965'}}}}
        {{'label': 'Boston Celtics', 'qid': 'Q123', 'meta': {{'season': '1966'}}}}

        Your Turn.
        """

        prompt += f"\nKG Results: {supporting_knowledge['KG']}"
        prompt += f"\nQuestion: {question}\nOperation: {operation}\n"
        prompt += "\nReturn Answer, Reasoning, and Evidence exactly as specified.\n"

        return prompt

    #########################################################################################
    # KG Final Verification
    #########################################################################################
    @staticmethod
    def KG_final_verification(question: str, candidate_evidence: str) -> str:
        prompt = f"""You are an AI assistant responsible for performing strict final verification on candidate answers extracted from a Knowledge Graph.

        Original Question: {question}

        Task:
        Read the Original Question carefully and identify all explicit constraints.
        Verify each candidate independently unless the question explicitly asks for a combined/joint condition.

        Important Rules:
        1. Do not assume conjunctions like "both" unless they actually appear in the question.
        2. If the question asks about one entity, do not merge evidence from different entities into a single combined interpretation.
        3. If a candidate appears in multiple rows, use all rows for that same candidate only.
        4. If a candidate's metadata contains explicit evidence that satisfies or violates the condition, filter accordingly: keep those that satisfy the condition, discard those that violate it.
        5. If a candidate's metadata lacks sufficient information to verify the condition, DO NOT automatically discard it. Instead, keep it as a candidate and state that it cannot be accurately verified due to insufficient information.
        6-1. If the remaining candidates include both (a) those that passed the condition filter and (b) those kept due to lack of information, then discard the ones kept due to lack of information, keeping only those that passed the condition filter.
        6-2. If the remaining candidates include ONLY those kept due to lack of information, then keep them.
        7. Output only the exact candidate labels that passed verification.
        8. If no candidates pass, output Answer: None.

        Strictly follow this output format (PLAIN TEXT ONLY — do NOT wrap section labels in markdown bold `**`, headers, or bullets. Write the labels exactly as `Reasoning:`, `Answer:`, `Evidence:` on their own lines):
        Reasoning: [Brief explanation of why candidates were kept or rejected]
        Answer: [comma-separated list of exact candidate labels that passed, or 'None']
        Evidence:
        [each kept evidence row on its own line, using exact dict-like Python syntax]

        Your Turn.
        Candidate Evidence:
        {candidate_evidence}
        Reasoning:
        Answer:
        Evidence:"""
        return prompt