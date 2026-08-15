"""
BlendQA
"""

### KG-Web

# type 1
blendqa_ex1_q = "When did the person after whom Nehru Zoological Park is named first visit the US?"
blendqa_ex1 = """{"When did the person after whom Nehru Zoological Park is named first visit the US?": ["1. Who is the person after whom Nehru Zoological Park is named?", "2. When did [1] first visit the US?"], "1. Who is the person after whom Nehru Zoological Park is named?": ["3. What is the Nehru Zoological Park?", "After whom was [3] named?"], "3. What is the Nehru Zoological Park?": "Search(\"Nehru Zoological Park\")", "After whom was [3] named?": "Relate([3], \"named after person\")", "2. When did [1] first visit the US?": "Relate([1], \"first US visit date\")"}"""

blendqa_ex2_q = "What significant military action was taken by the republic that contains the administrative territorial entity of Al Jazirah on September 26, 2024?"
blendqa_ex2 = """{"What significant military action was taken by the republic that contains the administrative territorial entity of Al Jazirah on September 26, 2024?": ["1. What is the republic that contains the administrative territorial entity of Al Jazirah?", "2. What significant military action was taken by [1] on September 26, 2024?"], "1. What is the republic that contains the administrative territorial entity of Al Jazirah?": ["3. What is Al Jazirah?", "4. What is the republic that contains the administrative territorial entity of [3]?"], "3. What is Al Jazirah?": "Search(\"Al Jazirah\")", "4. What is the republic that contains the administrative territorial entity of [3]?": "Relate([3], \"republic that contains the territorial entity\")", "2. What significant military action was taken by [1] on September 26, 2024?": "Relate([1], \"significant military action on September 26, 2024\")"}"""

blendqa_ex3_q = "How many people were killed in the mass shooting in the region of the Czech where Daniel Pudil was born on December 21, 2023?"
blendqa_ex3 = """{"How many people were killed in the mass shooting in the region of the Czech where Daniel Pudil was born on December 21, 2023?": ["1. In what region of the Czech was Daniel Pudil born?", "2. How many people were killed in the mass shooting in [1] on December 21, 2023?"], "1. In what region of the Czech was Daniel Pudil born?": ["3. Who is Daniel Pudil?", "4. In what region of the Czech was [3] born?"], "3. Who is Daniel Pudil?": "Search(\"Daniel Pudil\")", "4. In what region of the Czech was [3] born?": "Relate([3], \"place of birth\")", "2. How many people were killed in the mass shooting in [1] on December 21, 2023?": "Relate(\"mass shooting in [1] on December 21, 2023\", \"number of people killed\")"}"""

# type 2
blendqa_ex4_q = "Was the body of the missing American hiker found in September 2024 on the mountain that is part of the new7wonders of nature?"
blendqa_ex4 = """{"Was the body of the missing American hiker found in September 2024 on the mountain that is part of the new7wonders of nature?": ["1. What mountain is part of the new7wonders of nature?", "2. Was the body of the missing American hiker found in September 2024 on [1]?"], "1. What mountain is part of the new7wonders of nature?": ["3. What is the new7wonders of nature?", "4. What mountain is part of [3]?"], "3. What is the new7wonders of nature?": "Search(\"new7wonders of nature\")", "4. What mountain is part of [3]?": "Relate([3], \"mountain\")", "2. Was the body of the missing American hiker found in September 2024 on [1]?": "Relate(\"missing American hiker\", \"body found on [1] in September 2024\")"}"""

blendqa_ex5_q = "What was the contribution of the tennis player born on May 5, 2003, to Team Europe's victory in the Laver Cup on September 22, 2024?"
blendqa_ex5 = """{"What was the contribution of the tennis player born on May 5, 2003, to Team Europe's victory in the Laver Cup on September 22, 2024?": ["1. Who was the tennis player born on May 5, 2003?", "2. What was [1]'s contribution to Team Europe's victory in the Laver Cup on September 22, 2024?"], "1. Who was the tennis player born on May 5, 2003?": "Search(\"tennis player born on May 5, 2003\")", "2. What was [1]'s contribution to Team Europe's victory in the Laver Cup on September 22, 2024?": "Relate([1], \"contribution to Team Europe's victory in the Laver Cup on September 22, 2024\")"}"""

blendqa_ex6_q = "When did the Republican Governors Association indicate it would stop making further media placements for the campaign of the football player born on July 24, 1981?"
blendqa_ex6 = """{"When did the Republican Governors Association indicate it would stop making further media placements for the campaign of the football player born on July 24, 1981?": ["1. Who is the football player born on July 24, 1981?", "2. When did the Republican Governors Association indicate it would stop making further media placements for the campaign of [1]?"], "1. Who is the football player born on July 24, 1981?": "Search(\"football player born on July 24, 1981\")", "2. When did the Republican Governors Association indicate it would stop making further media placements for the campaign of [1]?": "Relate([1], \"Republican Governors Association media placement stop date\")"}"""

### Text-KG

blendqa_ex7_q = "Who has 2 tapes in the television series that is a notable work of Christian Lee Navarro?"
blendqa_ex7 = """{"Who has 2 tapes in the television series that is a notable work of Christian Lee Navarro?": ["1. What is the television series that is a notable work of Christian Lee Navarro?", "2. Who has 2 tapes in [1]?"], "1. What is the television series that is a notable work of Christian Lee Navarro?": ["3. Who is Christian Lee Navarro?", "4. Which television series is a notable work of [3]?"], "3. Who is Christian Lee Navarro?": "Search(\"Christian Lee Navarro\")", "4. Which television series is a notable work of [3]?": "Relate([3], \"notable television series\")", "2. Who has 2 tapes in [1]?": "Relate([1], \"person who has 2 tapes\")"}"""

blendqa_ex8_q = "When did the economic crisis in Kuhle Wampe reach its peak?"
blendqa_ex8 = """{"When did the economic crisis in Kuhle Wampe reach its peak?": ["1. What is the economic crisis in Kuhle Wampe?", "2. When did [1] reach its peak?"], "1. What is the economic crisis in Kuhle Wampe?": ["3. What is Kuhle Wampe?", "4. What isthe economic crisis in [3]?"], "3. What is Kuhle Wampe?": "Search(\"Kuhle Wampe\")", "4. What is the economic crisis in [3]?": "Relate([3], \"economic crisis\")", "2. When did [1] reach its peak?": "Relate([1], \"reached its peak at time\")"}"""

blendqa_ex9_q = "Where do I go to get the financial product associated with the industry of RPM Mortgage, Inc.?"
blendqa_ex9 = """{"Where do I go to get the financial product associated with the industry of RPM Mortgage, Inc.?": ["1. What is the financial product associated with the industry of RPM Mortgage, Inc.?", "2. Where do I go to get [1]?"], "1. What is the financial product associated with the industry of RPM Mortgage, Inc.?": ["3. What is RPM Mortgage Inc.?", "4. What is the financial product associated with the industry of [3]?"], "3. What is RPM Mortgage Inc.?": "Search(\"RPM Mortgage Inc.\")", "4. What is the financial product associated with the industry of [3]?": "Relate([3], \"financial product\")", "2. Where do I go to get [1]?": "Relate([1], \"typical provider\")"}"""

### Web-Text

blendqa_ex10_q = "Who sang the song by The Eagles about the rock and roll lifestyle?"
blendqa_ex10 = """{"Who sang the song by The Eagles about the rock and roll lifestyle?": ["1. What is the song by The Eagles about the rock and roll lifestyle?", "2. Who sang the song [1]?"], "1. What is the song by The Eagles about the rock and roll lifestyle?": ["3. Who are The Eagles?", "4. What is the song by [3] about the rock and roll lifestyle?"], "3. Who are The Eagles?": "Search(\"The Eagles\", \"band\")", "4. What is the song by [3] about the rock and roll lifestyle?": "Relate([3], \"song about the rock and roll lifestyle\")", "2. Who sang the song [1]?": "Relate([1], \"singer\")"}"""

blendqa_ex11_q = "In the context of the enzyme that breaks down lactose into glucose and galactose, where is it found in the human body?"
blendqa_ex11 = """{"In the context of the enzyme that breaks down lactose into glucose and galactose, where is it found in the human body?": ["1. What is the enzyme that breaks down lactose into glucose and galactose?", "2. In the context of [1], where is it found in the human body?"], "1. What is the enzyme that breaks down lactose into glucose and galactose?": "Search(\"Enzyme that breaks down lactose into glucose and galactose\")", "2. In the context of [1], where is it found in the human body?": "Relate([1], \"location in human body\")"}"""

blendqa_ex12_q = "When were the medieval pilgrimage stories by Geoffrey Chaucer written, and in what language?"
blendqa_ex12 = """{"When were the medieval pilgrimage stories by Geoffrey Chaucer written, and in what language?": ["1. What are the medieval pilgrimage stories by Geoffrey Chaucer?", "2. When were [1] written, and in what language?"], "1. What are the medieval pilgrimage stories by Geoffrey Chaucer?": ["3. Who is Geoffrey Chaucer?", "4. What are the medieval pilgrimage stories by [3]?"], "3. Who is Geoffrey Chaucer?": "Search(\"Geooffrey Chaucer\", \"writer\")", "4. What are the medieval pilgrimage stories by [3]?": "Relate([3], \"medieval pilgrimage stories\")", "2. When were [1] written, and in what language?": "Relate([1], \"written at time and in language\")"}"""

### Mixed

# KG-Web type 2
blendqa_ex13_q = "What does the social networking service that employs Ryan Roslansky use to enhance its artificial intelligence model in September 2024?"
blendqa_ex13 = """{"What does the social networking service that employs Ryan Roslansky use to enhance its artificial intelligence model in September 2024?": ["1. What is the social networking service that employs Ryan Roslansky?", "2. What does [1] use to enhance its artificial intelligence model in September 2024?"], "1. What is the social networking service that employs Ryan Roslansky?": ["3. Who is Ryan Roslansky?", "4. What is the social networking service that employs [3]?"], "3. Who is Ryan Roslansky?": "Search(\"Ryan Roslansky\")", "4. What is the social networking service that employs [3]?": "Relate([3], \"employed by social networking service\")", "2. What does [1] use to enhance its artificial intelligence model in September 2024?": "Search(\"What does [1] use to enhance its artificial intelligence model in September 2024\")"}"""

# Text-KG with [END]
blendqa_ex14_q = "Is the episcopal see where Georges Garreau was born known as the city of love or the city of lights?"
blendqa_ex14 = """{"Is the episcopal see where Georges Garreau was born known as the city of love or the city of lights?": ["1. What is the episcopal see where Georges Garreau was born?", "2. Is [1] known as the city of love?", "3. Is [1] known as the city of lights?", "4. Given answers of [2] and [3], is [1] known as the city of love or city of lights?"], "1. What is the episcopal see where Georges Garreau was born?": "Relate(\"Georges Garreau\", \"place of birth\")", "2. Is [1] known as the city of love?": "Relate([1], \"known as city of love\")", "3. Is [1] known as the city of lights?": "Relate([1], \"known as city of lights\")", "4. Given answers of [2] and [3], is [1] known as the city of love or city of lights?": "[END]"}"""

# Web-Text
blendqa_ex15_q = "How did the bat of the record-holder for most home runs in the home run derby impact Nick Mahrley during the game?"
blendqa_ex15 = """{"How did the bat of the record-holder for most home runs in the home run derby impact Nick Mahrley during the game?": ["1. Who has the record for most home runs in the home run derby?", "2. How did the bat of [1] impact Nick Mahrley during the game?"], "1. Who has the record for most home runs in the home run derby?": "Search(\"Record holder for the most home runs in the home run derby\")", "2. How did the bat of [1] impact Nick Mahrley during the game?": "Search(\"How did the bat of [1] impact Nick Mahrley during the game\")"}"""

# Text-KG
blendqa_ex16_q = "Who was the first person buried at the United States national cemetery where Willard A. Kitts is buried?"
blendqa_ex16 = """{"Who was the first person buried at the United States national cemetery where Willard A. Kitts is buried?": ["1. What is the United States national cemetry where Willard A. Kitts is buried?", "2. Who was the first person buried at [1]?"], "1. What is the United States national cemetry where Willard A. Kitts is buried?": ["3. Who is Willard A. Kitts?", "4. What is the United States national cemetry where [3] is buried?"], "3. Who is Willard A. Kitts?": "Search(\"Willard A. Kitts\")", "4. What is the United States national cemetry where [3] is buried?": "Relate([3], \"buried at United States national cemetry\")", "2. Who was the first person buried at [1]?": "Relate([1], \"first person buried\")"}"""

# Web-Text with Filter
blendqa_ex17_q = "What is the career high points in a game of the electrifying dunker with an NBA-record 22 seasons?"
blendqa_ex17 = """{"What is the career high points in a game of the electrifying dunker with an NBA-record 22 seasons?": ["1. Who is the electrifying dunker with an NBA-record 22 seasons?", "2. What is the career high points in a game of [1]?"], "1. Who is the electrifying dunker with an NBA-record 22 seasons?": ["3. Who holds the NBA record for 22 seasons?", "4. Who among [3] is known as an electrifying dunker?"], "3. Who holds the NBA record for 22 seasons?": "Search(\"NBA player with record 22 seasons\")", "4. Among [3], who is known as an electrifying dunker?": "Filter([3], \"known as electrifying dunker\")", "2. What is the career high points in a game of [1]?": "Relate([1], \"career high points in a game\")"}"""

# Text-KG commonsense long sub-q
blendqa_ex18_q = "What is the anatomical name for the body part associated with the musical group that performs the song 'Asleep in the Back'?"
blendqa_ex18 = """{"What is the anatomical name for the body part associated with the musical group that performs the song 'Asleep in the Back'?": ["1. What is the musical group that performs the song 'Asleep in the Back'?", "2. What is the anatomical name for the body part associated with [1]?"], "1. What is the musical group that performs the song 'Asleep in the Back'?": "Search(\"music group that performs the song Asleep in the Back\")", "2. What is the anatomical name for the body part associated with [1]?": "Relate([1], \"anatomical name for body part\")"}"""

# Web-Text
blendqa_ex19_q = "What key factor is driving the overall decline in overdose deaths in the city famous for its large Cinco de Mayo celebration?"
blendqa_ex19 = """{"What key factor is driving the overall decline in overdose deaths in the city famous for its large Cinco de Mayo celebration?": ["1. What city is famous for its large Cinco de Mayo celebration?", "2. What key factor is driving the overall decilne in overdose deaths in [1]?"], "1. What city is famous for its large Cinco de Mayo celebration?": ["3. What is the Cinco de Mayo celebration?", "4. What city is famous for [3]?"], "3. What is the Cinco de Mayo celebration?": "Search(\"Cinco de Mayo\", \"celebration\")", "4. What city is famous for [3]?": "Relate([3], \"city famous for\")", "2. What key factor is driving the overall decilne in overdose deaths in [1]?": "Search(\"key factor driving the overall decline in overdose deaths in [1]\")"}"""

# Text-KG
blendqa_ex20_q = "Who played Joe in the film directed by Darren Grant featuring cast member Tiffany Evans?"
blendqa_ex20 = """{"Who played Joe in the film directed by Darren Grant featuring cast member Tiffany Evans?": ["1. What is the film directed by Darren Grant featuring cast member Tiffany Evans?", "2. Who played Joe in the film [1]?"], "1. What is the film directed by Darren Grant featuring cast member Tiffany Evans?": ["3. Who is Darren Grant?", "4. What is the film directed by [3] featuring cast member Tiffany Evans?"], "3. Who is Darren Grant?": "Search(\"Darren Grant\", \"film director\")", "4. What is the film directed by [3] featuring cast member Tiffany Evans?": "Relate([3], \"film directed featuring Tiffany Evans\")", "2. Who played Joe in the film [1]?": "Relate([1], \"actor who played Joe\")"}"""

# KG-Web type 1
blendqa_ex21_q = "What was the final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and the island nation where Praia is located?"
blendqa_ex21 = """{"What was the final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and the island nation where Praia is located?": ["1. What is the island nation where Praia is located?", "2. What was the final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and [1]?"], "1. What is the island nation where Praia is located?": ["3. What is Praia?", "4. What is the island nation where [3] is located?"], "3. What is Praia?": "Search(\"Praia\", \"location\")", "4. What is the island nation where [3] is located?": "Relate([3], \"located in island nation\")", "2. What was the final score on January 23, 2024, in the Africa Cup of Nations match between Egypt and [1]?": "Relate([1], \"final score on January 23, 2024 in the Africa Cup of Nations between Egypt\")"}"""


def format_tree_generation_prompt_blendqa(question):
    prompt = f"""You are given 3 atomic functions to help you retrieve and operate knowledge from Google, Wikipedia, or Wikidata:
1. Search(). Input: (name, [optional] descriptor). Output: list[entities]. This function helps you find and disambiguate an entity given its name and optional descriptor. If no descriptor is provided, the most popular entity will be returned. For example, Search(\"Michael Jordan\") returns the famous basketball player ["Michael Jordan"], while Search(\"Michael Jordan\", \"football goalkeeper\") returns the English retired football goalkeeper ["Michael Jordan (footballer)"]. When the question provides explicit entity knowledge, always write a descriptor for the Search() function based on the question's information. If the entity name is unknown, can also query Search() with a descriptive title. For example, Search(\"The first president to be assassinated\") returns ["Abraham Lincoln"], who is the first U.S. president to be assassinated.
2. Relate(). Input: there are 2 input possibilities, (head_entity, relation), or (head_entity, tail_entity). Output: list[tail_entities], or list[relations]. This function helps you find the tail_entities given a head_entity and relation, or relations given a head_entity and tail_entity. For example, Relate(\"Barack Obama\", \"child\") returns ["Malia Obama", "Sasha Obama"], and Relate(\"Barack Obama\", \"Michelle Obama\") returns ["spouse"]. You may also search attribute relations using Relate() by treating attributes as tail entities. For example, Relate(\"Barack Obama\", \"time served as US president\") returns ["1997 to 2004"], and Relate(\"Barack Obama\", \"1961\")" returns ["year of birth"].
3. Filter(). Input: (list[entities], condition). Output: list[entities]. This function helps you filter out entities that satisfy a factual attribute condition. For example, Filter([\"Lionel Messi\", \"Steven Jobs\", \"Bill Gates\"], \"born in 1955\"), returns [\"Bill Gates\", \"Steve Jobs\"], and Filter([\"Lionel Messi\", \"Cristiano Ronaldo\"], \"is Portuguese\") returns ["Cristiano Ronaldo"].
Construct a hierarchical question decomposition tree in json format for the following complex question: \"{question}\". The tree starts with the original complex question as the root node, and each non-root node is a sub-question of its parent. Continue decomposing until a sub-question cannot be further decomposed and could either be: (1) directly answered by calling one of the three atomic functions Search(), Relate(), Filter(), or (2) directly answered by analyzing the answers of at least two previously answered questions, such as comparing, judging, intersecting, counting, etc. In case (1), write this sub-question with its corresponding function call as a leaf node. In case (2), write this sub-question with an [END] mark as a leaf node. For function leaf nodes, do not write nested functions such as Filter(Search(...)): if multiple function calls are required, write each function call with a separate sub-question in a separate leaf node. For [END] leaf questions, format your question as "Given answers of [q_idx_1] and [q_idx_2], ...", where [q_idx_1] and [q_idx_2] are question indices of the previously answered questions required to answer this [END] question. In your question decomposition tree, use double quotes "" to enclose sub-questions and functions, and escape quotes \"\" to enclose work titles and function parameters.
Examples: 
Question: {blendqa_ex1_q}
Decomposition Tree: {blendqa_ex1}
Question: {blendqa_ex2_q}
Decomposition Tree: {blendqa_ex2}
Question: {blendqa_ex3_q}
Decomposition Tree: {blendqa_ex3}
Question: {blendqa_ex4_q}
Decomposition Tree: {blendqa_ex4}
Question: {blendqa_ex5_q}
Decomposition Tree: {blendqa_ex5}
Question: {blendqa_ex6_q}
Decomposition Tree: {blendqa_ex6}
Question: {blendqa_ex7_q}
Decomposition Tree: {blendqa_ex7}
Question: {blendqa_ex8_q}
Decomposition Tree: {blendqa_ex8}
Question: {blendqa_ex9_q}
Decomposition Tree: {blendqa_ex9}
Question: {blendqa_ex10_q}
Decomposition Tree: {blendqa_ex10}
Question: {blendqa_ex11_q}
Decomposition Tree: {blendqa_ex11}
Question: {blendqa_ex12_q}
Decomposition Tree: {blendqa_ex12}
Question: {blendqa_ex13_q}
Decomposition Tree: {blendqa_ex13}
Question: {blendqa_ex14_q}
Decomposition Tree: {blendqa_ex14}
Question: {blendqa_ex15_q}
Decomposition Tree: {blendqa_ex15}
Question: {blendqa_ex16_q}
Decomposition Tree: {blendqa_ex16}
Question: {blendqa_ex17_q}
Decomposition Tree: {blendqa_ex17}
Question: {blendqa_ex18_q}
Decomposition Tree: {blendqa_ex18}
Question: {blendqa_ex19_q}
Decomposition Tree: {blendqa_ex19}
Question: {blendqa_ex20_q}
Decomposition Tree: {blendqa_ex20}
Question: {blendqa_ex21_q}
Decomposition Tree: {blendqa_ex21}

Your Question.
Question: {question}
Decomposition Tree: """
    
    return prompt



"""
MMQA
"""

### KG-Table

mmqa_ex1_q = "How did the team John Havlicek played for perform in the 1962-63 NBA Finals?"
mmqa_ex1 = """{"How did the team John Havlicek played for perform in the 1962-63 NBA Finals?": ["1. Which team did John Havlicek play for?", "2. How did [1] perform in the 1962-63 NBA Finals?"], "1. Which team did John Havlicek play for?": ["3. Who is John Havlicek?", "4. Which team did [3] play for?"], "3. Who is John Havlicek?": "Search(\"John Havlicek\")", "4. Which team did [3] play for?": "Relate([3], \"team played for\")", "2. How did [1] perform in the 1962-63 NBA Finals?": "Relate([1], \"1962-63 NBA Finals performance\")"}"""

mmqa_ex2_q = "In the 2001 NBA All-Star Game, what award did the player from the Vancouver Grizzlies receive?"
mmqa_ex2 = """{"In the 2001 NBA All-Star Game, what award did the player from the Vancouver Grizzlies receive?": ["1. Who was the player from the Vancouver Grizzlies in the 2001 NBA All-Star Game?", "2. What award did [1] receive?"], "1. Who was the player from the Vancouver Grizzlies in the 2001 NBA All-Star Game?": "Search(\"Vancouver Grizzlies 2001 NBA All-Star Game player\")", "2. What award did [1] receive?": "Relate([1], \"award received\")"}"""

mmqa_ex3_q = "How many games did the team founded by Peter Holt play in October 1976-77?"
mmqa_ex3 = """{"How many games did the team founded by Peter Holt play in October 1976-77?": ["1. Which team was founded by Peter Holt?", "2. How many games did [1] play in October 1976-77?"], "1. Which team was founded by Peter Holt?": ["3. Who is Peter Holt?", "4. Which team did [3] found?"], "3. Who is Peter Holt?": "Search(\"Peter Holt\")", "4. Which team did [3] found?": "Relate([3], \"team founded\")", "2. How many games did [1] play in October 1976-77?": "Relate([1], \"games played in October 1976-77\")"}"""

mmqa_ex4_q = "What is the best streak for the team coached by Erik Spoelstra win at most in the 1995-96 Schedule?"
mmqa_ex4 = """{"What is the best streak for the team coached by Erik Spoelstra win at most in the 1995-96 Schedule?": ["1. Which team did Erik Spoelstra coach?", "2. What is the best streak for [1] win at most in the 1995-96 Schedule?"], "1. Which team did Erik Spoelstra coach?": ["3. Who is Erik Spoelstra?", "4. Which team did [3] coach?"], "3. Who is Erik Spoelstra?": "Search(\"Erik Spoelstra\")", "4. Which team did [3] coach?": "Relate([3], \"team coached\")", "2. What is the best streak for [1] win at most in the 1995-96 Schedule?": ["5. What is the streak for [1] win at most in the 1995-96 Schedule?", "6. What is the longest win streak in [5]"], "5. What is the streak for [1] win at most in the 1995-96 Schedule?": "Relate([1], \"win streak in 1995-96 season\")", "6. What is the longest win streak in [5]": "Filter([5], \"longest consecutive wins\")"}"""

mmqa_ex5_q = "What award did the Houston Rockets' No. 4 player with a surname starting with \"B\" receive?"
mmqa_ex5 = """{"What award did the Houston Rockets' No. 4 player with a surname starting with \"B\" receive?": ["1. Who is the Houston Rockets' No. 4 player with a surname starting with \"B\"?", "2. What award did [1] receive?"], "1. Who is the Houston Rockets' No. 4 player with a surname starting with \"B\"?": ["3. What players are on the Houston Rockets roster?", "4. Among [3], who is the No. 4 player with a surname starting with 'B'?"], "3. What players are on the Houston Rockets roster?": "Search(\"Houston Rockets players\")", "4. Among [3], who is the No. 4 player with a surname starting with 'B'?": "Filter([3], \"No. 4 player with surname starting with B\")", "2. What award did [1] receive?": "Relate([1], \"award received\")"}"""

mmqa_ex6_q = "Which team did the player who was the tallest and from the Maine Red Claws in the 2010 NBA All-Star Slam Dunk Contest play for?"
mmqa_ex6 = """{"Which team did the player who was the tallest and from the Maine Red Claws in the 2010 NBA All-Star Slam Dunk Contest play for?": ["1. Who was the tallest player from the Maine Red Claws in the 2010 NBA All-Star Slam Dunk Contest?", "2. Which team did [1] play for?"], "1. Who was the tallest player from the Maine Red Claws in the 2010 NBA All-Star Slam Dunk Contest?": ["3. What players from the Maine Red Claws participated in the 2010 NBA All-Star Slam Dunk Contest?", "4. Among [3], who is the tallest?"], "3. What players from the Maine Red Claws participated in the 2010 NBA All-Star Slam Dunk Contest?": "Search(\"Maine Red Claws players 2010 NBA All-Star Slam Dunk Contest\")", "4. Among [3], who is the tallest?": "Filter([3], \"tallest player\")", "2. Which team did [1] play for?": "Relate([1], \"team played for\")"}"""

mmqa_ex7_q = "Which team drafted the player who received the NBA Sportsmanship Award the most times?"
mmqa_ex7 = """{"Which team drafted the player who received the NBA Sportsmanship Award the most times?": ["1. Who received the NBA Sportsmanship Award the most times?", "2. Which team drafted [1]?"], "1. Who received the NBA Sportsmanship Award the most times?": ["3. What is the NBA Sportsmanship Award?", "4. Which players have received [3]?", "5. Among [4], who received it the most times?"], "3. What is the NBA Sportsmanship Award?": "Search(\"NBA Sportsmanship Award\")", "4. Which players have received [3]?": "Relate([3], \"recipients\")", "5. Among [4], who received it the most times?": "Filter([4], \"player who received it the most times\")", "2. Which team drafted [1]?": "Relate([1], \"drafted by team\")"}"""


def format_tree_generation_prompt_mmqa(question):
    prompt = f"""You are given 3 atomic functions to help you retrieve and operate knowledge from Table, KG, or Text:
1. Search(). Input: (name, [optional] descriptor). Output: list[entities]. This function helps you find and disambiguate an entity given its name and optional descriptor. If no descriptor is provided, the most popular entity will be returned. For example, Search(\"Michael Jordan\") returns the famous basketball player ["Michael Jordan"], while Search(\"Michael Jordan\", \"football goalkeeper\") returns the English retired football goalkeeper ["Michael Jordan (footballer)"]. When the question provides explicit entity knowledge, always write a descriptor for the Search() function based on the question's information. If the entity name is unknown, can also query Search() with a descriptive title. For example, Search(\"The first president to be assassinated\") returns ["Abraham Lincoln"], who is the first U.S. president to be assassinated.
2. Relate(). Input: there are 2 input possibilities, (head_entity, relation), or (head_entity, tail_entity). Output: list[tail_entities], or list[relations]. This function helps you find the tail_entities given a head_entity and relation, or relations given a head_entity and tail_entity. For example, Relate(\"Barack Obama\", \"child\") returns ["Malia Obama", "Sasha Obama"], and Relate(\"Barack Obama\", \"Michelle Obama\") returns ["spouse"]. You may also search attribute relations using Relate() by treating attributes as tail entities. For example, Relate(\"Barack Obama\", \"time served as US president\") returns ["1997 to 2004"], and Relate(\"Barack Obama\", \"1961\")" returns ["year of birth"].
3. Filter(). Input: (list[entities], condition). Output: list[entities]. This function helps you filter out entities that satisfy a factual attribute condition. For example, Filter([\"Lionel Messi\", \"Steven Jobs\", \"Bill Gates\"], \"born in 1955\"), returns [\"Bill Gates\", \"Steve Jobs\"], and Filter([\"Lionel Messi\", \"Cristiano Ronaldo\"], \"is Portuguese\") returns ["Cristiano Ronaldo"].
Construct a hierarchical question decomposition tree in json format for the following complex question: \"{question}\". The tree starts with the original complex question as the root node, and each non-root node is a sub-question of its parent. Continue decomposing until a sub-question cannot be further decomposed and could either be: (1) directly answered by calling one of the three atomic functions Search(), Relate(), Filter(), or (2) directly answered by analyzing the answers of at least two previously answered questions, such as comparing, judging, intersecting, counting, etc. In case (1), write this sub-question with its corresponding function call as a leaf node. In case (2), write this sub-question with an [END] mark as a leaf node. For function leaf nodes, do not write nested functions such as Filter(Search(...)): if multiple function calls are required, write each function call with a separate sub-question in a separate leaf node. For [END] leaf questions, format your question as "Given answers of [q_idx_1] and [q_idx_2], ...", where [q_idx_1] and [q_idx_2] are question indices of the previously answered questions required to answer this [END] question. In your question decomposition tree, use double quotes "" to enclose sub-questions and functions, and escape quotes \"\" to enclose work titles and function parameters.
Examples: 
Question: {mmqa_ex1_q}
Decomposition Tree: {mmqa_ex1}
Question: {mmqa_ex2_q}
Decomposition Tree: {mmqa_ex2}
Question: {mmqa_ex3_q}
Decomposition Tree: {mmqa_ex3}
Question: {mmqa_ex4_q}
Decomposition Tree: {mmqa_ex4}
Question: {mmqa_ex5_q}
Decomposition Tree: {mmqa_ex5}
Question: {mmqa_ex6_q}
Decomposition Tree: {mmqa_ex6}
Question: {mmqa_ex7_q}
Decomposition Tree: {mmqa_ex7}


Your Question.
Question: {question}
Decomposition Tree: """
    
    return prompt
