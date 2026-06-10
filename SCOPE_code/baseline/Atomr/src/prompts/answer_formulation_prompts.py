##############################################
### Prompts for Knowledge Source Selection ###
##############################################

# Label shown to the LLM -> canonical executor id.
_LABEL_TO_SRC = {"kg": "kg", "kb": "kg", "table": "table", "text": "doc", "doc": "doc"}
# Canonical executor id -> AtomR-style label shown to the LLM.
_SRC_TO_LABEL = {"kg": "KG", "table": "Table", "doc": "Text"}

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

# Atomic (single-source) examples, balanced 4-per-source, drawn from
# kg-doc / kg-table / table-doc benchmark sub-queries.
_ATOMIC_EXAMPLES = {
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

# Bridge (multi-source) examples, keyed by frozenset of executor ids.
# Included only when ALL the example's sources are live.
_BRIDGE_EXAMPLES = {
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


def format_knowledge_source_selection_prompt(question, available_knowledge_sources):
    """Build an AtomR few-shot prompt restricted to the live knowledge sources.

    Labels, descriptions, selection rules, atomic examples, and bridge
    examples are all filtered to the live KB so the LLM never sees an
    inactive source. This mirrors the prompt-construction logic used by
    `_build_atomr_prompt` in new_model/step2_decompose/route_baselines.py.
    """
    live_canonical = []
    for s in available_knowledge_sources:
        sid = _LABEL_TO_SRC.get(str(s).lower())
        if sid and sid not in live_canonical:
            live_canonical.append(sid)
    # Re-order so prompts are deterministic (canonical kg / table / doc order).
    live_canonical = [s for s in ("kg", "table", "doc") if s in live_canonical]
    if not live_canonical:
        raise ValueError(
            "format_knowledge_source_selection_prompt: available_knowledge_sources "
            "must include at least one of {KG, Table, Text}"
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

    example_lines = []
    for s in live_canonical:
        for q, a in _ATOMIC_EXAMPLES[s]:
            example_lines.append(f"Q: {q}\nA: {a}")
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


################################
### Prompt for direct answer ###
################################

def format_direct_answer_prompt_blendqa(question):
    prompt = f"""Answer the following question "{question}". Formulate your final answer with (1) a Paraphrase Answer that repeats the question, and (2) a clean python Answer List. When answering with numbers, always use arabic numbers i.e. 1,2,3. When answering to "Yes" or "No" questions, simply formulate your Answer list as ["Yes"] or ["No"]. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Examples.
Question: When did the person after whom Nehru Zoological Park is named first visit the US?
Answer: Nehru Zoological Park is named after Jawaharlal Nehru, the first Prime Minister of India. Nehru first visited the United States in October 1949. So the answer is: (1) Paraphrase Answer: The person after whom Nehru Zoological Park is named first visit the US in October 1949; (2) Answer List: ["October 1949"]

Question: Where do I go to get the financial product associated with the industry of RPM Mortgage, Inc.?
Answer: RPM Mortgage, Inc. is associated with the mortgage lending and home financing industry. You can typically get a mortgage loan at a financial institution, such as a bank, credit union or building society. So the answer is: (1) Paraphrase Answer: You can get the financial product associated with the industry of RPM Mortgage, Inc., mortgage loans, at a financial institution, such as a bank, credit union or building society; (2) Answer List: ["at a financial institution, such as a bank, credit union or building society"]

Question: How many people were killed in the mass shooting in the region of the Czech where Daniel Pudil was born on December 21, 2023?
Answer: On December 21, 2023, a mass shooting occurred at Charles University in Prague, Czech Republic. This tragic event resulted in the deaths of 14 people. So the answer is: (1) Paraphrase Answer: 14 people were killed in the mass shooting in the region of the Czech where Daniel Pudil was born on December 21, 2023; (2) Answer List: ["14"]

Your Question.
Question: {question}
Answer: """
    
    return prompt


def format_direct_answer_prompt_mmqa(question):
    prompt = f"""Answer the following question "{question}". Formulate your final answer with (1) a Paraphrase Answer that repeats the question, and (2) a clean python Answer List. Many questions involve sports rosters, schedules, awards, or table-derived facts. When answering with numbers, always use arabic numbers i.e. 1,2,3. When answering to "Yes" or "No" questions, simply formulate your Answer list as ["Yes"] or ["No"]. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Examples.
Question: How did the team John Havlicek played for perform in the 1962-63 NBA Finals?
Answer: John Havlicek played for the Boston Celtics. The Boston Celtics won the 1962-63 NBA Finals. So the answer is: (1) Paraphrase Answer: The team John Havlicek played for, the Boston Celtics, won the 1962-63 NBA Finals; (2) Answer List: ["won the 1962-63 NBA Finals"]

Question: Which team did Erik Spoelstra coach?
Answer: Erik Spoelstra coached the Miami Heat. So the answer is: (1) Paraphrase Answer: Erik Spoelstra coached the Miami Heat; (2) Answer List: ["Miami Heat"]

Question: Which team drafted Mike Conley?
Answer: Mike Conley was drafted by the Memphis Grizzlies. So the answer is: (1) Paraphrase Answer: Mike Conley was drafted by the Memphis Grizzlies; (2) Answer List: ["Memphis Grizzlies"]

Your Question.
Question: {question}
Answer: """
    
    return prompt


#############################
### Prompt for direct RAG ###
#############################

def format_direct_rag_prompt_blendqa(question, supporting_knowledge):
    prompt = f"""Please answer the question "{question}" using the retrieved knowledge from Google, Wikipedia, or Wikidata. Answer the question with (1) a Paraphrase Answer that repeats the question, and (2) a clean python Answer List. If the provided information is not enough to answer the question, answer based on your own knowledge. If neither the provided knowledge nor your own knowledge can answer the question, end your answer with (1) Paraphrase Answer: Unknown; (2) Answer List: []. When answering with numbers, always use arabic numbers i.e. 1,2,3. When answering to "Yes" or "No" questions, simply formulate your Answer list as ["Yes"] or ["No"]. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Examples.
Retrieved Knowledge: 
Wikipedia Passages: [1] Kuhle Wampe | Kuhle Wampe is a 1932 German feature film about unemployment, homelessness and left wing politics in the Weimar Republic produced by Prometheus Film.
Google Results: [1] To Whom Does the World Belong? | Anni Bönike has a badly paid job in a factory ... [2] Kuhle Wampe | Kuhle Wampe is a 1932 German feature film about unemployment, homelessness and left wing politics in the Weimar Republic produced by Prometheus Film. 
Question: What is Kuhle Wampe?
Answer: Kuhle Wampe is a 1932 German feature film. So the answer is: (1) Paraphrase Answer: Kuhle Wampe is a 1932 German feature film; (2) Answer List: ["Kuhle Wampe (German film)"]

Retrieved Knowledge: 
Google Results: [1] Carlos Alcaraz | Carlos Alcaraz (born May 5, 2003, El Palmar, Murcia, Spain) is a Spanish professional tennis player ... [2] Carlos Alcaraz Biography: Childhood, Career and ... | Carlos Alcaraz is a Spanish professional tennis player born on May 5, 2003.
Question: Who is the tennis player born on May 5, 2003?
Answer: The tennis player born on May 5, 2003, is Carlos Alcaraz. So the answer is: (1) Paraphrase Answer: The tennis player born on May 5, 2003, is Carlos Alcaraz; (2) Answer List: ["Carlos Alcaraz"]

Retrieved Knowledge: 
Google Results: [1] an enzyme produced by many organisms and is essential to the complete digestion of whole milk | Lactase (EC 3.2. 1.108) is an enzyme produced by many organisms and is essential to the complete digestion of whole milk. It breaks down the sugar lactose into its component parts, galactose and glucose. [2] LACTASE - Uses, Side Effects, and More | Lactase is an enzyme that breaks down lactose, the sugar in milk. 
Question: What is the enzyme that breaks down lactose into glucose and galactose?
Answer: The enzyme that breaks down lactose into glucose and galactose is lactase. So the answer is: (1) Paraphrase Answer: The enzyme that breaks down lactose into glucose and galactose is lactase; (2) Answer List: ["Lactase"]

Retrieved Knowledge: 
Google Results: [1] US overdoses have fallen sharply in recent months, a ... | A steep drop in deaths from fentanyl is a key factor driving the overall decline. Overdose deaths involving fentanyl and other synthetic ...
Question: What key factor is driving the overall decline in overdose deaths in Puebla?
Answer: The retrieved knowledge indicates that a "key factor driving the overall decline" is "a steep drop in deaths from fentanyl". So the answer is: (1) Paraphrase Answer: a key factor driving the overall decline is a steep drop in deaths from fentanyl; (2) Answer List: ["a steep drop in deaths from fentanyl"]

Retrieved Knowledge: 
Google Results: [1] Cape Verde 2-2 Egypt (Jan 22, 2024) Final Score | Cape Verde's Bryan Teixeira scored a 99th minute equaliser as they held record seven-time champions Egypt to a 2-2 Group B draw at the Africa Cup of Nations. [2] Cape Verde vs. Egypt - Final Score - January 22, 2024 | View the Cape Verde vs. Egypt game played on January 22, 2024. Box score, stats, odds, highlights, play-by-play, social & more.
Question: What was the final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and Cape Verde?
Answer: The final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and Cape Verde, is 2-2. So the answer is: (1) Paraphrase Answer: The final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and Cape Verde, is 2-2; (2) Answer List: ["2-2"]

Retrieved Knowledge: 
Wikipedia Passages: [1] Kuhle Wampe | Kuhle Wampe (full title: Kuhle Wampe, oder: Wem gehört die Welt?, translated in English as Kuhle Wampe or Who Owns the World?, and released in the USA as Whither Germany? by Kinematrade Inc.) is a 1932 German feature film about unemployment, homelessness and left wing politics in the Weimar Republic produced by Prometheus Film. ... [2] Kuhle Wampe | Synopsis: On their return home by train, Anni, Fritz and other workers argue with middle-class, wealthy passengers about the worldwide financial crisis. 
Wikidata Query Results: Great Depression
Question: What is the economic crisis in Kuhle Wampe (German film)?
Answer: The economic crisis depicted in the German film Kuhle Wampe is the Great Depression. So the answer is: (1) Paraphrase Answer: The economic crisis in Kuhle Wampe (German film) is the Great Depression; (2) Answer List: ["Great Depression"]

Retrieved Knowledge: 
Google Results: [1] American hiker found dead on South Africa's Table Mountain | The woman has been identified as a 20-year-old student from North Carolina named Brook Cheuvront. An American woman who went missing while on a hike on Table Mountain in Cape Town, South Africa, has died and her body has been recovered, authorities said on Monday.
Question: Was the body of the missing American hiker found in September 2024 on the Table mountain?
Answer: Yes, based on the retrieved knowledge, an American hiker was found dead on the Table mountain. So the answer is: (1) Paraphrase Answer: Yes, the body of the missing American hiker was found in September 2024 on the Table mountain; (2) Answer List: ["Yes"]

Your Question.
Supporting knowledge: """
    
    if "Text" in supporting_knowledge:
        prompt += f"\nWikipedia Passages: {supporting_knowledge['Text']}"
    if "Web" in supporting_knowledge:
        prompt += f"\nGoogle Results: {supporting_knowledge['Web']}"
    if "KB" in supporting_knowledge:
        prompt += f"\nWikidata Query Results: {supporting_knowledge['KB']}"
    
    prompt += f"""\nQuestion: {question}\nAnswer: """
    
    return prompt


def format_direct_rag_prompt_mmqa(question, supporting_knowledge):
    prompt = f"""Please answer the question "{question}" using the retrieved knowledge from KG, Table, or Text. Questions may involve sports rosters, schedules, awards, or table-derived comparisons. Answer the question with (1) a Paraphrase Answer that repeats the question, and (2) a clean python Answer List. If the provided information is not enough to answer the question, answer based on your own knowledge. If neither the provided knowledge nor your own knowledge can answer the question, end your answer with (1) Paraphrase Answer: Unknown; (2) Answer List: []. When answering with numbers, always use arabic numbers i.e. 1,2,3. When answering to "Yes" or "No" questions, simply formulate your Answer list as ["Yes"] or ["No"]. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Examples.
Retrieved Knowledge: 
Text Passages: [1] John Havlicek | John Havlicek was an American professional basketball player who spent his entire career with the Boston Celtics.
Table Rows: [1] 1963 NBA Finals | The Boston Celtics defeated the Los Angeles Lakers to win the 1963 NBA Finals.
Question: How did the team John Havlicek played for perform in the 1962-63 NBA Finals?
Answer: John Havlicek played for the Boston Celtics, and the Boston Celtics won the 1962-63 NBA Finals. So the answer is: (1) Paraphrase Answer: The team John Havlicek played for, the Boston Celtics, won the 1962-63 NBA Finals; (2) Answer List: ["won the 1962-63 NBA Finals"]

Retrieved Knowledge: 
Text Passages: [1] Erik Spoelstra | Erik Spoelstra is the head coach of the Miami Heat of the National Basketball Association.
Question: Which team did Erik Spoelstra coach?
Answer: Erik Spoelstra coached the Miami Heat. So the answer is: (1) Paraphrase Answer: Erik Spoelstra coached the Miami Heat; (2) Answer List: ["Miami Heat"]

Retrieved Knowledge: 
Text Passages: [1] Mike Conley Jr. | Mike Conley Jr. is an American professional basketball player who was selected by the Memphis Grizzlies in the 2007 NBA draft.
Table Rows: [1] NBA Sportsmanship Award winners | Mike Conley has won the NBA Sportsmanship Award multiple times, more than any other player.
Question: Which team drafted the player who received the NBA Sportsmanship Award the most times?
Answer: The player who received the NBA Sportsmanship Award the most times is Mike Conley, and he was drafted by the Memphis Grizzlies. So the answer is: (1) Paraphrase Answer: The team that drafted the player who received the NBA Sportsmanship Award the most times is the Memphis Grizzlies; (2) Answer List: ["Memphis Grizzlies"]

Your Question.
Supporting knowledge: """

    if "Text" in supporting_knowledge:
        prompt += f"\nText Passages: {supporting_knowledge['Text']}"
    if "Table" in supporting_knowledge:
        prompt += f"\nTable Rows: {supporting_knowledge['Table']}"
    elif "Web" in supporting_knowledge:
        prompt += f"\nTable Rows: {supporting_knowledge['Web']}"
    if "KG" in supporting_knowledge:
        prompt += f"\nKG Triples: {supporting_knowledge['KG']}"
    elif "KB" in supporting_knowledge:
        prompt += f"\nKG Triples: {supporting_knowledge['KB']}"

    prompt += f"""\nQuestion: {question}\nAnswer: """

    return prompt


####################################
### Prompts for Atomic Functions ###
####################################

##### 1. Search

def format_search_prompt_blendqa(question, supporting_knowledge):
    prompt = f"""Using the retrieved knowledge from Google, Wikipedia, or Wikidata, please answer the following entity disambiguation question "{question}". Answer the question with (1) a Paraphrase Answer that repeats the question, and (2) an Answer List that is a clean entity list. You may not answer with an empty entity list: if the provided information is not enough to answer the question, answer based on your own knowledge. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [entity_list]"
Examples.
Retrieved Knowledge: 
Wikipedia Passages: [1] Kuhle Wampe | Kuhle Wampe is a 1932 German feature film about unemployment, homelessness and left wing politics in the Weimar Republic produced by Prometheus Film.
Google Results: [1] To Whom Does the World Belong? | Anni Bönike has a badly paid job in a factory ... [2] Kuhle Wampe | Kuhle Wampe is a 1932 German feature film about unemployment, homelessness and left wing politics in the Weimar Republic produced by Prometheus Film. 
Question: What is Kuhle Wampe?
Answer: Kuhle Wampe is a 1932 German feature film. So the answer is: (1) Paraphrase Answer: Kuhle Wampe is a 1932 German feature film; (2) Answer List: ["Kuhle Wampe (German film)"]

Retrieved Knowledge: 
Google Results: [1] Carlos Alcaraz | Carlos Alcaraz (born May 5, 2003, El Palmar, Murcia, Spain) is a Spanish professional tennis player ... [2] Carlos Alcaraz Biography: Childhood, Career and ... | Carlos Alcaraz is a Spanish professional tennis player born on May 5, 2003.
Question: Who is the tennis player born on May 5, 2003?
Answer: The tennis player born on May 5, 2003, is Carlos Alcaraz. So the answer is: (1) Paraphrase Answer: The tennis player born on May 5, 2003, is Carlos Alcaraz; (2) Answer List: ["Carlos Alcaraz"]

Retrieved Knowledge: 
Wikipedia Passages: [1] Al Jazirah (newspaper) | Al Jazirah (in Arabic الجزيرة meaning The Island) is a daily Arabic newspaper published in Saudi Arabia. Its sister newspaper is Al Masaiya ... [2] Al Jazirah, Sharjah | Al Jazirah (الجزيرة) is a settlement in Sharjah. ... [3] Al Jazirah Al Hamra | Al Jazirah Al Hamra (الجزيرة الحمراء, The Red Island) is a town to the south of the city of Ras Al Khaimah in the United Arab Emirates. 
Question: What is Al Jazirah?
Answer: Al Jazirah could refer to multiple entities, including newspaper Al Jazirah, Sharjah settlement Al Jazirah, Sharjah, or United Arab Emirates town Al Jazirah Al Hamra. So the answer is: (1) Paraphrase Answer: Al Jazirah could refer to Al Jazirah (newspaper), Al Jazirah, Sharjah (settlement), or Al Jazirah Al Hamra (town); (2) Answer List: ["Al Jazirah (newspaper)", "Al Jazirah, Sharjah (settlement)", "Al Jazirah Al Hamra (town)"]

Retrieved Knowledge: 
Google Results: [1] an enzyme produced by many organisms and is essential to the complete digestion of whole milk | Lactase (EC 3.2. 1.108) is an enzyme produced by many organisms and is essential to the complete digestion of whole milk. It breaks down the sugar lactose into its component parts, galactose and glucose. [2] LACTASE - Uses, Side Effects, and More | Lactase is an enzyme that breaks down lactose, the sugar in milk. 
Question: What is the enzyme that breaks down lactose into glucose and galactose?
Answer: The enzyme that breaks down lactose into glucose and galactose is lactase. So the answer is: (1) Paraphrase Answer: The enzyme that breaks down lactose into glucose and galactose is lactase; (2) Answer List: ["Lactase"]

Retrieved Knowledge: 
Google Results: [1] US overdoses have fallen sharply in recent months, a ... | A steep drop in deaths from fentanyl is a key factor driving the overall decline. Overdose deaths involving fentanyl and other synthetic ...
Question: What key factor is driving the overall decline in overdose deaths in Puebla?
Answer: The retrieved knowledge indicates that a "key factor driving the overall decline" is "a steep drop in deaths from fentanyl". So the answer is: (1) Paraphrase Answer: a key factor driving the overall decline is a steep drop in deaths from fentanyl; (2) Answer List: ["a steep drop in deaths from fentanyl"]

Your Question.
Retrieved Knowledge: """
    
    if "Text" in supporting_knowledge:
        prompt += f"\nWikipedia Passages: {supporting_knowledge['Text']}"
    if "Web" in supporting_knowledge:
        prompt += f"\nGoogle Results: {supporting_knowledge['Web']}"
    if "KB" in supporting_knowledge:
        prompt += f"\nWikidata Query Results: {supporting_knowledge['KB']}"
    
    prompt += f"""\nQuestion: {question}\nAnswer: """
    
    return prompt


def format_search_prompt_mmqa(question, supporting_knowledge):
    prompt = f"""Using the retrieved knowledge from KG, Table, or Text, please answer the following entity disambiguation question "{question}". Answer the question with (1) a Paraphrase Answer that repeats the question, and (2) an Answer List that is a clean entity list. You may not answer with an empty entity list: if the provided information is not enough to answer the question, answer based on your own knowledge. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [entity_list]"
Examples.
Retrieved Knowledge: 
Text Passages: [1] John Havlicek | John Havlicek was an American professional basketball player for the Boston Celtics.
Question: Who is John Havlicek?
Answer: John Havlicek is an American professional basketball player. So the answer is: (1) Paraphrase Answer: John Havlicek is an American professional basketball player; (2) Answer List: ["John Havlicek"]

Retrieved Knowledge: 
Text Passages: [1] Erik Spoelstra | Erik Spoelstra is an American professional basketball coach who serves as the head coach of the Miami Heat.
Question: Who is Erik Spoelstra?
Answer: Erik Spoelstra is an American professional basketball coach. So the answer is: (1) Paraphrase Answer: Erik Spoelstra is an American professional basketball coach; (2) Answer List: ["Erik Spoelstra"]

Retrieved Knowledge: 
Text Passages: [1] NBA Sportsmanship Award | The NBA Sportsmanship Award is an annual National Basketball Association award given to the player who best represents the ideals of sportsmanship on the court.
Question: What is the NBA Sportsmanship Award?
Answer: The NBA Sportsmanship Award is an annual National Basketball Association award. So the answer is: (1) Paraphrase Answer: The NBA Sportsmanship Award is an annual National Basketball Association award; (2) Answer List: ["NBA Sportsmanship Award"]

Your Question.
Retrieved Knowledge: """

    if "Text" in supporting_knowledge:
        prompt += f"\nText Passages: {supporting_knowledge['Text']}"
    if "Table" in supporting_knowledge:
        prompt += f"\nTable Rows: {supporting_knowledge['Table']}"
    elif "Web" in supporting_knowledge:
        prompt += f"\nTable Rows: {supporting_knowledge['Web']}"
    if "KG" in supporting_knowledge:
        prompt += f"\nKG Triples: {supporting_knowledge['KG']}"
    elif "KB" in supporting_knowledge:
        prompt += f"\nKG Triples: {supporting_knowledge['KB']}"

    prompt += f"""\nQuestion: {question}\nAnswer: """

    return prompt




##### 2. Relate


def format_relate_prompt_mmqa(question, supporting_knowledge):
    prompt = f"""Please answer the question "{question}" using the retrieved knowledge from KG, Table, or Text. Questions may involve teams, players, awards, schedules, or table-derived facts. Answer the question with (1) a Paraphrase Answer that repeats the question, and (2) a clean python Answer List. If the provided information is not enough to answer the question, answer based on your own knowledge. If neither the provided knowledge nor your own knowledge can answer the question, end your answer with (1) Paraphrase Answer: Unknown; (2) Answer List: []. When answering with numbers, always use arabic numbers i.e. 1,2,3. When answering to "Yes" or "No" questions, simply formulate your Answer list as ["Yes"] or ["No"]. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Note, the answer should not output a series of thethethe......

Examples.
Retrieved Knowledge: 
Text Passages: [1] John Havlicek | John Havlicek spent his entire NBA career with the Boston Celtics.
Question: Which team did John Havlicek play for?
Answer: John Havlicek played for the Boston Celtics. So the answer is: (1) Paraphrase Answer: John Havlicek played for the Boston Celtics; (2) Answer List: ["Boston Celtics"]

Retrieved Knowledge: 
Text Passages: [1] Erik Spoelstra | Erik Spoelstra is the head coach of the Miami Heat.
Question: Which team did Erik Spoelstra coach?
Answer: Erik Spoelstra coached the Miami Heat. So the answer is: (1) Paraphrase Answer: Erik Spoelstra coached the Miami Heat; (2) Answer List: ["Miami Heat"]

Retrieved Knowledge: 
Text Passages: [1] Mike Conley Jr. | Mike Conley Jr. was selected by the Memphis Grizzlies with the fourth overall pick in the 2007 NBA draft.
Question: Which team drafted Mike Conley?
Answer: Mike Conley was drafted by the Memphis Grizzlies. So the answer is: (1) Paraphrase Answer: Mike Conley was drafted by the Memphis Grizzlies; (2) Answer List: ["Memphis Grizzlies"]

Your Question.
Retrieved Knowledge: """

    if "Text" in supporting_knowledge:
        prompt += f"\nText Passages: {supporting_knowledge['Text']}"
    if "Table" in supporting_knowledge:
        prompt += f"\nTable Rows: {supporting_knowledge['Table']}"
    elif "Web" in supporting_knowledge:
        prompt += f"\nTable Rows: {supporting_knowledge['Web']}"
    if "KG" in supporting_knowledge:
        prompt += f"\nKG Triples: {supporting_knowledge['KG']}"
    elif "KB" in supporting_knowledge:
        prompt += f"\nKG Triples: {supporting_knowledge['KB']}"

    prompt += f"""\nQuestion: {question}\nAnswer: """

    return prompt


def format_relate_prompt_blendqa(question, supporting_knowledge):
    prompt = f"""Please answer the question "{question}" using the retrieved knowledge from Wikipedia, Google, or Wikidata. Answer the question with (1) a Paraphrase Answer that repeats the question, and (2) a clean python Answer List. When answering long lists such as album titles, directly write the list in your answer formulation. If the provided information is not enough to answer the question, answer based on your own knowledge. If neither the provided knowledge nor your own knowledge can answer the question, end your answer with (1) Paraphrase Answer: Unknown; (2) Answer List: []. When answering with numbers, always use arabic numbers i.e. 1,2,3. When answering to "Yes" or "No" questions, simply formulate your Answer list as ["Yes"] or ["No"]. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Examples.
Retrieved Knowledge: 
Wikipedia Passages: [1] Daniel Pudil | Daniel Pudil (born 27 September 1985) is a Czech professional footballer who plays for Viktoria Žižkov and the Czech Republic national team as a left back or left winger.
Google Results: [1]  ... 'personal_information': {{'place_of_birth': 'Prague, Czechoslovakia', 'height': '1.83 m (6 ft 0 in)', 'position_s': 'Left back, left winger'}} [2] Daniel Pudil - Wikidata | Daniel Pudil (2014) (Czech). 1 reference. imported from Wikimedia project ... place of birth · Prague. 1 reference. stated in ...
Question: In what region of the Czech was Daniel Pudil born?
Answer: Daniel Pudil was born in Prague. So the answer is: (1) Paraphrase Answer: Daniel Pudil was born in Prague; (2) Answer List: ["Prague"]

Retrieved Knowledge: 
Google Results: [1] Cape Verde 2-2 Egypt (Jan 22, 2024) Final Score | Cape Verde's Bryan Teixeira scored a 99th minute equaliser as they held record seven-time champions Egypt to a 2-2 Group B draw at the Africa Cup of Nations. [2] Cape Verde vs. Egypt - Final Score - January 22, 2024 | View the Cape Verde vs. Egypt game played on January 22, 2024. Box score, stats, odds, highlights, play-by-play, social & more.
Question: What was the final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and Cape Verde?
Answer: The final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and Cape Verde, is 2-2. So the answer is: (1) Paraphrase Answer: The final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and Cape Verde, is 2-2; (2) Answer List: ["2-2"]

Retrieved Knowledge: 
Wikipedia Passages: [1] Kuhle Wampe | Kuhle Wampe (full title: Kuhle Wampe, oder: Wem gehört die Welt?, translated in English as Kuhle Wampe or Who Owns the World?, and released in the USA as Whither Germany? by Kinematrade Inc.) is a 1932 German feature film about unemployment, homelessness and left wing politics in the Weimar Republic produced by Prometheus Film. ... [2] Kuhle Wampe | Synopsis: On their return home by train, Anni, Fritz and other workers argue with middle-class, wealthy passengers about the worldwide financial crisis. 
Wikidata Query Results: Great Depression
Question: What is the economic crisis in Kuhle Wampe (German film)?
Answer: The economic crisis depicted in the German film Kuhle Wampe is the Great Depression. So the answer is: (1) Paraphrase Answer: The economic crisis in Kuhle Wampe (German film) is the Great Depression; (2) Answer List: ["Great Depression"]

Retrieved Knowledge: 
Google Results: [1] US overdoses have fallen sharply in recent months, a ... | A steep drop in deaths from fentanyl is a key factor driving the overall decline. Overdose deaths involving fentanyl and other synthetic ...
Question: What key factor is driving the overall decline in overdose deaths in Puebla?
Answer: The retrieved knowledge indicates that a "key factor driving the overall decline" is "a steep drop in deaths from fentanyl". So the answer is: (1) Paraphrase Answer: a key factor driving the overall decline is a steep drop in deaths from fentanyl; (2) Answer List: ["a steep drop in deaths from fentanyl"]

Retrieved Knowledge: 
Google Results: [1] American hiker found dead on South Africa's Table Mountain | The woman has been identified as a 20-year-old student from North Carolina named Brook Cheuvront. An American woman who went missing while on a hike on Table Mountain in Cape Town, South Africa, has died and her body has been recovered, authorities said on Monday.
Question: Was the body of the missing American hiker found in September 2024 on the Table mountain?
Answer: Yes, based on the retrieved knowledge, an American hiker was found dead on the Table mountain. So the answer is: (1) Paraphrase Answer: Yes, the body of the missing American hiker was found in September 2024 on the Table mountain; (2) Answer List: ["Yes"]

Retrieved Knowledge: 
Wikipedia Passages: [1] Elbow | The elbow is the visible joint between the upper and lower parts of the arm. The elbow joint is the synovial hinge joint between the humerus in the upper arm and the radius and ulna in the forearm which allows the forearm and hand to be moved towards and away from the body. 
Question: What is the anatomical name for the body part associated with the Elbow?
Answer: Wikipedia Passage [1] suggests that the elbow is the "synovial hinge joint" between the humerus in the upper arm and the radius and ulna in the forearm. So the answer is: (1) Paraphrase Answer: The anatomical name for the our body part the Elbow is synovial hinge joint; (2) Answer List: ["synovial hinge joint"]

Retrieved Knowledge: 
Google Results: [1] Paris | Etymology:  \'City of Light\' (La Ville Lumière), both because of its leading role during the Age of Enlightenment and more literally because Paris was one of the first large European cities to use gas street lighting on a grand scale on its boulevards and monuments.
Question: Is Paris known as the city of Love?
Answer: No, Paris is known as the City of Light, not City of Love. So the answer is: (1) Paraphrase Answer: No, Paris is known as the City of Light, not City of Love; (2) Answer List: ["No"]

Retrieved Knowledge: 
Wikipedia Results: [1] The Rachel Maddow Show | Production:  The Rachel Maddow Show is broadcast from Studio 3-A at the NBC Studios, 30 Rockefeller Plaza in New York...
Question: Where is The Rachel Maddow Show broadcast from?
Answer: Wikipedia passage [1] suggests that The Rachel Maddow Show is broadcast from "Studio 3-A at the NBC Studios, 30 Rockefeller Plaza in New York". So the answer is: (1) Paraphrase Answer: The Rachel Maddow Show is broadcast from Studio 3-A at the NBC Studios, 30 Rockefeller Plaza in New York; (2) Answer List: ["Studio 3-A at the NBC Studios, 30 Rockefeller Plaza in New York"]

Your Question.
Retrieved Knowledge: """
    
    if "Text" in supporting_knowledge:
        prompt += f"\nWikipedia Passages: {supporting_knowledge['Text']}"
    if "Web" in supporting_knowledge:
        prompt += f"\nGoogle Results: {supporting_knowledge['Web']}"
    if "KB" in supporting_knowledge:
        prompt += f"\nWikidata Query Results: {supporting_knowledge['KB']}"
    
    prompt += f"""\nQuestion: {question}\nAnswer: """
        
    return prompt




##### 3. Filter

def format_filter_prompt_blendqa(question, condition, supporting_knowledge):
    prompt = f"""Using the retrieved knowledge from Wikipedia, Google, or Wikidata, please answer the following filter question "{question}". Formulate a list of entities as your final answer. If the provided passages does not provide helpful information, answer based on your own knowledge. Answer the question with (1) a Paraphrase Answer that repeats the question's filter condition "{condition}", and (2) a clean python Answer List. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Examples.
Retrieved Knowledge: 
Wikipedia Passages: [1] Pulitzer Prize for Music | History:  guidelines and jury membership will serve that end.” Subsequently, in 2006, a posthumous "Special Citation" was given to jazz composer Thelonious Monk, and in 2007 the prize went to Ornette Coleman, a free jazz composer, who won the prize for his disc Sound Grammar, a recording of a 2005 concert, making it the first time a recording won the music Pulitzer, and a first for purely improvised music. In 2018, rapper Kendrick Lamar won the award for his 2017 hip hop album Damn. The recording was the first musical work not in the jazz or classical genres to win the prize. ... [2] Collected Poems of Robert Frost | Reception:  Frost received a Pulitzer prize in 1931 for the collection. One of the books in the collection, New Hampshire, had received the Pulitzer Prize in 1924...
Question: Which of Damn, Damn. Collector's Edition won the Pulitzer Prize?
Answer: Wikipedia passage [1] suggests that "in 2018, rapper Kendrick Lamar won the award for his 2017 hip hop album Damn", so the album Damn won the Pulitzer Prize. So the answer is: (1) Paraphrase Answer: Damn won the Pulitzer Prize; (2) Answer List: ["Damn"]

Retrieved Knowledge: 
Wikipedia Passages: [1] Mechanical Bull Tour | Setlist: 1) "Charmer" ; 2) "Rock City" ; 3) "My Party" ; 4) "Temple" ; 5) "On Call" ; 6) "Family Tree" ; 7) "Closer" ; 8) "The Immortals" ; 9) "Back Down South" ; 10) "Wait for Me" ; 11) "Supersoaker" ; 12) "Milk" ; 13) "Pyro" ; 14) "Tonight" ; 15) "Radioactive" ; 16) "The Bucket" ; 17) "Don't Matter" ; 18) "Molly's Chambers" ; 19) "Four Kicks" ; 20) "Be Somebody" ; 21) "Notion" ; 22) "Cold Desert" ; 23) "Use Somebody" ; Encore ; 1) "Crawl" ; 2) "Black Thumbnail" ; 3) "Sex on Fire" ... [2] Mechanical Bull (album) | Promotion:  love and fighting," and a "distant cousin of U2's With or Without You". "Beautiful War" and "Don't Matter" were released as singles exclusively in the United Kingdom on December 9, 2013 and June 16, 2014 respectively. "Family Tree" was sent to US modern rock radio as the album's sixth overall single on June 17, 2014...
Question: Among Mechanical Bull, Come Around Sundown, Because of the Times, which album includes the songs "Wait for Me" and "Family Tree"?
Answer: The album Mechanical Bull includes the songs "Wait for Me" and "Family Tree". So the answer is: (1) Paraphrase Answer: Mechanical Bull includes the songs "Wait for Me" and "Family Tree"; (2) Answer List: ["Mechanical Bull"]

Your Question.
Retrieved Knowledge: """
    
    if "Text" in supporting_knowledge:
        prompt += f"\nWikipedia Passages: {supporting_knowledge['Text']}"
    if "Web" in supporting_knowledge:
        prompt += f"\nGoogle Results: {supporting_knowledge['Web']}"
    if "KB" in supporting_knowledge:
        prompt += f"\nWikidata Query Results: {supporting_knowledge['KB']}"
    
    prompt += f"""\nQuestion: {question}\nAnswer: """
    
    return prompt



def format_filter_prompt_mmqa(question, condition, supporting_knowledge):
    prompt = f"""Using the retrieved knowledge from KG, Table, or Text, please answer the following filter question "{question}". Formulate a list of entities as your final answer. If the provided passages does not provide helpful information, answer based on your own knowledge. Answer the question with (1) a Paraphrase Answer that repeats the question's filter condition "{condition}", and (2) a clean python Answer List. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Examples.
Retrieved Knowledge: 
Table Rows: [1] 1963 NBA Finals | The Boston Celtics defeated the Los Angeles Lakers in the 1963 NBA Finals.
Question: Among the Boston Celtics and Los Angeles Lakers, which team won the 1962-63 NBA Finals?
Answer: The Boston Celtics won the 1962-63 NBA Finals. So the answer is: (1) Paraphrase Answer: The Boston Celtics won the 1962-63 NBA Finals; (2) Answer List: ["Boston Celtics"]

Retrieved Knowledge: 
Table Rows: [1] Magic Johnson | height: 6 ft 9 in. [2] Michael Jordan | height: 6 ft 6 in. [3] Kevin Garnett | height: 6 ft 11 in.
Question: Among Magic Johnson, Michael Jordan, and Kevin Garnett, who is the tallest player?
Answer: Kevin Garnett is the tallest player among Magic Johnson, Michael Jordan, and Kevin Garnett. So the answer is: (1) Paraphrase Answer: Kevin Garnett is the tallest player; (2) Answer List: ["Kevin Garnett"]

Your Question.
Retrieved Knowledge: """

    if "Text" in supporting_knowledge:
        prompt += f"\nText Passages: {supporting_knowledge['Text']}"
    if "Table" in supporting_knowledge:
        prompt += f"\nTable Rows: {supporting_knowledge['Table']}"
    elif "Web" in supporting_knowledge:
        prompt += f"\nTable Rows: {supporting_knowledge['Web']}"
    if "KG" in supporting_knowledge:
        prompt += f"\nKG Triples: {supporting_knowledge['KG']}"
    elif "KB" in supporting_knowledge:
        prompt += f"\nKG Triples: {supporting_knowledge['KB']}"

    prompt += f"""\nQuestion: {question}\nAnswer: """

    return prompt


##################################################################
### Prompt for obtaining recursive answer from child questions ###
##################################################################

# Child questions and answers: 
# Q: 1. Who starred in "My Dog Skip"? A: ['Frankie Muniz', 'Diane Lane', 'Luke Wilson', 'Kevin Bacon']
# Q: 2. Who starred in "Malcolm in the Middle"? A: ['Christopher Kennedy Masterson', 'Jane Frances Kaczmarek', 'Justin Tyler Berfield']
# Q: 3. Who is among both [1] and [2]? A: No actor is among both "My Dog Skip" and "Malcolm in the Middle". []
# Question: Who starred in My Dog Skip and Malcolm in the Middle?
# Answer: The child questions and answers indicate that no actor is among both "My Dog Skip" and "Malcolm in the Middle", so the answer is unknown. So the answer is: (1) Paraphrase Answer: Unknown; (2) Answer List: []

def format_answer_from_child_qa_pairs_prompt_blendqa(question, child_qa_pairs):  
    str_child_qa_pairs = ""
    for qa_pair in child_qa_pairs:
        (q, a), = qa_pair.items()
        str_child_qa_pairs += f"Q: {q} A: {a}\n"    
    
    prompt = f"""The complex question "{question}" has been divided into child questions. Based on the child questions and answers, answer the complex question. Answer the question with (1) a Paraphrase Answer that repeats the question, and (2) a clean python Answer List. If the provided information is not enough to answer the question, answer based on your own knowledge. If neither the provided knowledge nor your own knowledge can answer the question, end your answer with (1) Paraphrase Answer: Unknown; (2) Answer List: []. When answering with numbers, always use arabic numbers i.e. 1,2,3. When answering to "Yes" or "No" questions, simply formulate your Answer list as ["Yes"] or ["No"]. Strictly follow the format of the examples below, ending your answer with "So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]"
Answer in one sentence + one list only.

Examples.
Child questions and answers: 
Q: 1. What is the television series that is a notable work of Christian Lee Navarro? A: ["13 Reasons Why"]
Q: 2. Who has 2 tapes in [1]? A: Justin Foley has 2 tapes in 13 Reasons Why. ["Justin Foley"]
Question: Who has 2 tapes in the television series that is a notable work of Christian Lee Navarro?
Answer: Justin Foley has 2 tapes in the television series that is a notable work of Christian Lee Navarro, 13 Reasons Why. So the answer is: (1) Paraphrase Answer: Justin Foley has 2 tapes in the television series that is a notable work of Christian Lee Navarro, 13 Reasons Why; (2) Answer List: ["Justin Foley"]

Child questions and answers: 
Q: 3. What is Praia? A: ["Praia, Cape Verde (city)", "Praia, Cape Verde (municipality)"]
Q: 4. What is the island nation where [3] is located? A: The island nation where Praia, Cape Verde (city) is located is Cape Verde; The island nation where Praia, Cape Verde (municipality) is located is Cape Verde. ["Cape Verde", "Cape Verde"]
Question: What is the island nation where Praia is located?
Answer: Praia is located in Cape Verde. So the answer is: (1) Paraphrase Answer: Praia is located in Cape Verde; (2) Answer List: ["Cape Verde"]

Child questions and answers: 
Q: 1. In what region of the Czech was Daniel Pudil born? A: ["Prague"]
Q: 2. How many people were killed in the mass shooting in [1] on December 21, 2023? A: On December 21, 2023, at least 14 people were killed in the mass shooting in Prague. ["14"]
Question: How many people were killed in the mass shooting in the region of the Czech where Daniel Pudil was born on December 21, 2023?
Answer: On December 21, 2023, 14 people were killed in the mass shooting in the region of the Czech where Daniel Pudil was born. So the answer is: (1) Paraphrase Answer: 14 people were killed in the mass shooting in the region of the Czech where Daniel Pudil was born on December 21, 2023; (2) Answer List: ["14"]

Child questions and answers: 
Q: 1. What mountain is part of the new7wonders of nature? A: ["Table Mountain"]
Q: 2. Was the body of the missing American hiker found in September 2024 on [1]? A: Yes, the body of the missing American hiker found in September 2024 on the Table Mountain. ["Yes"]
Question: Was the body of the missing American hiker found in September 2024 on the mountain that is part of the new7wonders of nature?
Answer: Yes, the body of the missing American hiker was found in September 2024 on the mountain that is part of the new7wonders of nature, Table Mountain. So the answer is: (1) Paraphrase Answer: Yes, the body of the missing American hiker was found in September 2024 on the mountain that is part of the new7wonders of nature, Table Mountain; (2) Answer List: ["Yes"]

Your question.
Child questions and answers:
{str_child_qa_pairs}
Question: {question}
Answer: """
    
    return prompt


def format_answer_from_child_qa_pairs_prompt_mmqa(question, child_qa_pairs):
    str_child_qa_pairs = ""
    for qa_pair in child_qa_pairs:
        (q, a), = qa_pair.items()
        str_child_qa_pairs += f"Q: {q} A: {a}\n"

    prompt = f"""The complex question "{question}" has been divided into child questions. Based on the child questions and answers, answer the complex question.

CRITICAL OUTPUT RULES (must follow):
1) Output exactly ONE final line only.
2) Do NOT include any reasoning, explanation, analysis, notes, or thinking process.
3) Do NOT repeat instructions or examples.
4) Keep the output concise. The entire output must be under 60 words.
5) The output MUST strictly match this format:
So the answer is: (1) Paraphrase Answer: {{paraphrase_answer}}; (2) Answer List: [answer_list]

Answer policy:
- Use child QA pairs first.
- If child QA is insufficient, use your own knowledge.
- If still unanswerable, use: So the answer is: (1) Paraphrase Answer: Unknown; (2) Answer List: []
- Use arabic numbers (1,2,3).
- For yes/no, Answer List must be ["Yes"] or ["No"].

Examples.
Child questions and answers:
Q: 1. Which team did John Havlicek play for? A: ["Boston Celtics"]
Q: 2. How did [1] perform in the 1962-63 NBA Finals? A: The Boston Celtics won the 1962-63 NBA Finals. ["won the 1962-63 NBA Finals"]
Question: How did the team John Havlicek played for perform in the 1962-63 NBA Finals?
Answer: So the answer is: (1) Paraphrase Answer: The team John Havlicek played for, the Boston Celtics, won the 1962-63 NBA Finals; (2) Answer List: ["won the 1962-63 NBA Finals"]

Child questions and answers:
Q: 1. Who received the NBA Sportsmanship Award the most times? A: ["Mike Conley"]
Q: 2. Which team drafted [1]? A: Mike Conley was drafted by the Memphis Grizzlies. ["Memphis Grizzlies"]
Question: Which team drafted the player who received the NBA Sportsmanship Award the most times?
Answer: So the answer is: (1) Paraphrase Answer: The team that drafted the player who received the NBA Sportsmanship Award the most times is the Memphis Grizzlies; (2) Answer List: ["Memphis Grizzlies"]

Your question.
Child questions and answers:
{str_child_qa_pairs}
Question: {question}
Answer: """
    
    return prompt
