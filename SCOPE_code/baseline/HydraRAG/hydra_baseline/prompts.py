"""HydraRAG baseline 的全部 prompt。

由原版 Hydra_run/cot_prompt_list.py 适配而来：
  - 去掉 Freebase 的 `m.xxxx` 实体 ID 体系
  - 知识统一表示为  {Head} -[relation]-> {Tail}  的 KG 边 / 多跳路径
  - KG 边与 "Table 行转出的边" 共用同一格式，便于多源融合

每个 prompt 末尾留 `Q:` / `the question is:`，由 pipeline 拼接具体内容。
"""

# ===========================================================================
# Stage A —— 问题理解
# ===========================================================================

SPLIT_QUESTION = """You will receive a multi-hop question about NBA basketball. It chains several facts together.
Decompose it into a Thinking CoT (a relation chain that links the key entities to the final answer)
and exactly TWO split sub-questions that can be solved step by step.

Output the A section only, in exactly this format:
Thinking CoT: <relation chain>
split_question1: <first hop, no dependency>
split_question2: <second hop, may depend on the answer of split_question1>

Example:
Q: In the February games of the 2007-08 Indiana Pacers season, what was his highest score for the player who received the NBA Most Improved Player Award in 2009?
A:
Thinking CoT: "NBA Most Improved Player Award 2009" - received by - answer(player) - playing in - "2007-08 Indiana Pacers season, February games" - with - answer(highest score)
split_question1: Which player received the NBA Most Improved Player Award in 2009?
split_question2: In the February games of the 2007-08 Indiana Pacers season, what was that player's highest score?

The question is:
"""

EXTRACT_TOPIC_ENTITY = """You will receive a multi-hop NBA question. Extract the concrete topic entities
(players, teams, awards, seasons, venues, coaches, etc.) that are explicitly named in the question.
These are the keywords used to anchor retrieval. Extract at most 3.

Give the result in the format:
keywords: {{x}, {x}}
reason: <short reason>

Q:
Question: In the February games of the 2007-08 Indiana Pacers season, what was his highest score for the player who received the NBA Most Improved Player Award in 2009?
A:
keywords: {{2007-08 Indiana Pacers season}, {NBA Most Improved Player Award}}
reason: The question explicitly names the "2007-08 Indiana Pacers season" and the "NBA Most Improved Player Award".

The question is:
"""

# ===========================================================================
# 核心 —— 把检索到的 Table 行转成 KG 边
# ===========================================================================

TABLE_TO_KG = """You will receive a question and one retrieved table.
Your task: read the table and express the facts it contains as knowledge-graph edges,
so that table evidence and KG evidence share ONE unified representation.

Rules:
- Write each edge as:  {Head Entity} -[relation]-> {Tail Entity}
- The relation should be a short natural phrase that also carries the needed context
  (date / month / season / opponent), because a table has no formal relation names.
- Only extract edges relevant to answering the question. Do NOT invent facts not in the table.
- A numeric/statistic answer should appear as the Tail Entity of an edge.
- Output 1 to 8 edges, one per line, inside square brackets.

Output the A section only, in exactly this format:
A:
edges:
[{...} -[...]-> {...}]
[{...} -[...]-> {...}]

Example:
Question: In the February games of the 2007-08 Indiana Pacers season, what was Danny Granger's highest score?
Table:
Table ID: 2-11961051-6
Page: 2007-08 Indiana Pacers season
Section: February
Header: Game | Date | Team | Score | High Points | Record
Row 1: 50 | February 1 | Detroit | L 89-101 | Danny Granger (24) | 24-26
Row 2: 53 | February 8 | Toronto | W 109-95 | Danny Granger (30) | 26-27
Row 3: 55 | February 12 | Boston | L 92-100 | Mike Dunleavy (22) | 26-28
A:
edges:
[{Danny Granger} -[high points, February 1 game, 2007-08 Indiana Pacers]-> {24}]
[{Danny Granger} -[high points, February 8 game, 2007-08 Indiana Pacers]-> {30}]
[{Mike Dunleavy} -[high points, February 12 game, 2007-08 Indiana Pacers]-> {22}]
[{2007-08 Indiana Pacers season, February} -[highest single-game score by Danny Granger]-> {30}]

Now do it for:
"""

# ---------------------------------------------------------------------------
# Stage C —— Doc 头：把检索到的文档段落转成 KG 边（kg-doc 任务用，与 TABLE_TO_KG 对偶）
# ---------------------------------------------------------------------------
DOC_TO_KG = """You will receive a question and one retrieved document passage.
Your task: read the passage and express the facts it contains as knowledge-graph edges,
so that document evidence and KG evidence share ONE unified representation.

Rules:
- Write each edge as:  {Head Entity} -[relation]-> {Tail Entity}
- The relation should be a short natural phrase that also carries the needed context
  (date / year / season / role), because free text has no formal relation names.
- Only extract edges relevant to answering the question. Do NOT invent facts not in the passage.
- A descriptive or numeric answer should appear as the Tail Entity of an edge.
- Output 1 to 8 edges, one per line, inside square brackets.

Output the A section only, in exactly this format:
A:
edges:
[{...} -[...]-> {...}]
[{...} -[...]-> {...}]

Example:
Question: How is the winner selected for the NBA Most Valuable Player Award?
Document:
Doc ID: 631870
Title: NBA Most Valuable Player Award
Passage: The National Basketball Association Most Valuable Player Award (MVP) is an annual
award given since the 1955-56 season to the best performing player of the regular season.
Until the 1979-80 season the MVP was selected by a vote of NBA players. Since then, the
award is decided by a panel of sportswriters and broadcasters throughout the United States
and Canada. The winner receives the Maurice Podoloff Trophy.
A:
edges:
[{NBA Most Valuable Player Award} -[winner selected by, since 1980-81]-> {a panel of sportswriters and broadcasters throughout the United States and Canada}]
[{NBA Most Valuable Player Award} -[winner selected by, until 1979-80]-> {a vote of NBA players}]
[{NBA Most Valuable Player Award} -[winner receives trophy]-> {Maurice Podoloff Trophy}]
[{NBA Most Valuable Player Award} -[first awarded in season]-> {1955-56}]

Now do it for:
"""

# ===========================================================================
# Stage D —— 多源证据融合：LLM 排序
# ===========================================================================

BEAM_SELECT = """You are ranking knowledge-graph edges for a 2-hop NBA question.
You are given the main question, its Thinking CoT, the two split sub-questions (hop 1 and hop 2),
and a numbered list of candidate edges/paths. Edges may come from the KG or be converted from tables.

A useful evidence set must cover BOTH hops:
- some edges resolve hop 1 (the first sub-question / the bridge entity),
- some edges resolve hop 2 (the second sub-question / the final answer).

IMPORTANT: judge each candidate by how well it answers the MAIN question OR EITHER sub-question.
An edge that matches only sub-question 2 (the second hop) is just as important as one matching the
main question — do NOT down-rank an edge merely because it shares few words with the main question.
The bridge entity / final-answer edge often shares almost no words with the original question.

Pick the most useful edges for answering, covering both hops, best first.
Answer in exactly this format, nothing else:
top_list: {Candidate 3, Candidate 7, Candidate 1}

the question is:
"""

# ===========================================================================
# Stage E —— 答案验证 / 迭代判断
# ===========================================================================

ANSWER_VERIFY = """Given a main question, a Thinking CoT, the split sub-questions, and the retrieved
knowledge-graph edges/paths (each edge: {Head} -[relation]-> {Tail}; edges may come from KG or
from tables), decide whether the evidence is SUFFICIENT to answer the main question.

- If sufficient, respond {Yes}, then give the answer (use the exact entity/value from the edges).
- If NOT sufficient, respond {No}, then explain what is still missing.

Answer in exactly this format:
response: {Yes}  or  {No}
answer: {xxx}      (only when Yes)
reason: <one or two sentences>

Example:
Q: Which player received the NBA Most Improved Player Award in 2009, and what is his team?
Thinking CoT: award - received by - player - plays for - team
split_question1: Which player received the award?
split_question2: Which team does he play for?
Edges:
[{NBA Most Improved Player Award} <-[receivesAward]- {Danny Granger}]
[{Danny Granger} -[playsFor]-> {Indiana Pacers}]
A:
response: {Yes}
answer: {Danny Granger, Indiana Pacers}
reason: The edges show Danny Granger received the award and plays for the Indiana Pacers.

the question is:
"""

ANSWER_DIRECT = """Given a main question, a Thinking CoT, the split sub-questions, and the retrieved
knowledge-graph edges/paths, generate the best possible final answer. Even if the evidence is
incomplete, give your most supported answer based strictly on the edges (do not refuse).

Answer in exactly this format:
answer: {xxx}
reason: <short reason citing which edges were used>

the question is:
"""

# ===========================================================================
# Stage E —— 迭代检索
# ===========================================================================

REGEN_QUERY = """Given a main question, the Thinking CoT, the split sub-questions, and the
knowledge-graph edges retrieved so far (which are NOT yet enough), predict what extra evidence
should be retrieved next. Produce ONE focused retrieval query plus a short reasoning hint.

Answer in exactly this format:
query: {xxx}
cot: {xxx}

the question is:
"""

PREDICT_ENTITY = """Given a main question, the Thinking CoT, the split sub-questions, and the
knowledge-graph edges retrieved so far, the reasoning is stuck because a BRIDGE entity is missing
(e.g. the player / team / season that connects the two hops).

Use the edges and your own NBA knowledge to predict up to 3 candidate bridge entities, so retrieval
can be re-anchored on them.

Answer in exactly this format:
Predicted: {xxx, xxx, xxx}
reason: <short reason>

the question is:
"""

# ===========================================================================
# 输出 —— 最终答案综合
# ===========================================================================

FINAL_SYNTHESIS = """You are given a multi-hop NBA question, its split sub-questions with their
solved answers, and the supporting knowledge-graph edges. Produce ONE concise final answer to the
main question. The answer must be the direct value/entity (a name or a number), not a sentence.

Answer in exactly this format:
Answer: <concise direct answer>
Reasoning: <one sentence>

the question is:
"""


# ===========================================================================
# 格式化 helper
# ===========================================================================

def edges_block(paths, with_index: bool = True) -> str:
    """把一组 path 对象渲染成 LLM 可读的编号列表。

    每个 path 是 dict：{"path_text": str, "source": str, "doc_id": str, ...}
    """
    lines = []
    for i, p in enumerate(paths, 1):
        src = p.get("source", "kg")
        tag = f"[{src}]"
        text = p.get("path_text", "")
        if with_index:
            lines.append(f"Candidate {i} {tag}: {text}")
        else:
            lines.append(f"{tag} {text}")
    return "\n".join(lines) if lines else "(no edges retrieved)"


def question_block(question: str, thinking_cot: str = "", split_qs=None) -> str:
    """渲染 question + CoT + split questions 三件套。"""
    parts = [f"Question: {question}"]
    if thinking_cot:
        parts.append(f"Thinking CoT: {thinking_cot}")
    for i, sq in enumerate(split_qs or [], 1):
        parts.append(f"split_question{i}: {sq}")
    return "\n".join(parts)
