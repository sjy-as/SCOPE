"""HydraRAG 多源证据融合：LLM 排序。

原先 step1（词重叠）/ step2（源可信度）是**代码打分**，会在 LLM 看到证据之前就把
第二跳的边筛掉——第二跳的边天然和原问题词重叠低，被系统性埋没。

现改为：
  1. 轻量召回门 _recall_filter —— 仅为控制 LLM 上下文长度把候选压到 cap 条，
     不做打分排序；并保证「表证据 / Table→KG 回探的第二跳边 / 连接多 seed 的
     KG 路径」一定进入候选。
  2. 一次 LLM 调用 _llm_rank —— 对照主问题 + 两个子问题排序选 top-K，
     使第一跳、第二跳的边都得到公平对待。
"""
from __future__ import annotations

import re
from typing import List

try:
    import config
    import prompts
except ImportError:  # 包内导入
    from .. import config, prompts  # type: ignore

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "is", "was", "are",
    "were", "be", "by", "and", "or", "what", "which", "who", "whom", "how",
    "did", "do", "does", "his", "her", "their", "that", "this", "with", "from",
    "during", "season", "game", "games", "player", "team",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(text).lower())
            if t and t not in _STOP and len(t) > 1]


# --------------------------------------------------------------------------
# 召回门：把候选压到 cap 条，仅控制上下文，不打分排序
# --------------------------------------------------------------------------
def _recall_filter(question: str, split_qs: List[str], paths: List[dict], cap: int) -> List[dict]:
    if len(paths) <= cap:
        return list(paths)

    qterms = set(_tokens(question + " " + " ".join(split_qs or [])))

    table_b: List[dict] = []      # 表证据（少且珍贵，全留）
    handoff1_b: List[dict] = []   # Table→KG 回探的 1 跳边（第二跳答案边，全留）
    multi_b: List[dict] = []      # 连接 >=2 个 seed 的 KG 路径（全留）
    handoff2_b: List[dict] = []   # Table→KG 回探的 2 跳边
    rest_b: List[dict] = []       # 其余 KG 路径

    for p in paths:
        src = p.get("source", "kg")
        n_tri = len(p.get("triples", []))
        if src != "kg":
            table_b.append(p)
        elif p.get("handoff") and n_tri <= 1:
            handoff1_b.append(p)
        elif p.get("n_seeds", 0) >= 2:
            multi_b.append(p)
        elif p.get("handoff") and n_tri <= 2:
            handoff2_b.append(p)
        else:
            rest_b.append(p)

    # rest 按与问题/子问题的词重叠度补位（仅用于"补到 cap"，不影响前面优先桶）
    def _key(p: dict):
        ov = len(qterms & set(_tokens(p.get("path_text", ""))))
        return (ov, p.get("n_seeds", 0), -len(p.get("triples", [])))

    rest_b.sort(key=_key, reverse=True)
    ordered = table_b + handoff1_b + multi_b + handoff2_b + rest_b
    return ordered[:cap]


# --------------------------------------------------------------------------
# LLM 排序
# --------------------------------------------------------------------------
def _parse_top_list(resp: str, n_candidates: int) -> List[int]:
    """从 'top_list: {Candidate 3, Candidate 7}' 解析出 1-based 序号。"""
    m = re.search(r"top_list\s*:\s*\{([^}]*)\}", resp, re.IGNORECASE)
    block = m.group(1) if m else resp
    idxs: List[int] = []
    for num in re.findall(r"\d+", block):
        i = int(num)
        if 1 <= i <= n_candidates and i not in idxs:
            idxs.append(i)
    return idxs


def multi_source_beam_search(
    llm, question: str, thinking_cot: str, split_qs: List[str],
    paths: List[dict],
    topn: int = None, recall_cap: int = None,
) -> List[dict]:
    """对 KG 边 + Table 转出的边一起做融合排序，返回 LLM 选出的 top-N 条证据。

    打分完全交给 LLM（对照主问题 + 两个子问题）；代码只负责一个不打分的召回门。
    """
    topn = topn or config.BEAM_TOPN
    recall_cap = recall_cap or config.BEAM_RECALL_CAP
    if not paths:
        return []

    cand = _recall_filter(question, split_qs, paths, recall_cap)
    if llm is None or len(cand) <= topn:
        return cand[:topn]

    block = prompts.edges_block(cand, with_index=True)
    prompt = (
        prompts.BEAM_SELECT
        + prompts.question_block(question, thinking_cot, split_qs)
        + "\n\nCandidate edges:\n" + block
        + f"\n\nPick up to {topn}.\nA:\n"
    )
    try:
        resp, _ = llm.query_gpt4o(prompt, max_tokens=150, stage="beam_select")
        idxs = _parse_top_list(resp, len(cand))
    except Exception:  # noqa: BLE001
        idxs = []

    if not idxs:
        return cand[:topn]

    selected = [cand[i - 1] for i in idxs[:topn]]
    # 不足则按召回门顺序补齐（表证据/第二跳边优先）
    for p in cand:
        if len(selected) >= topn:
            break
        if p not in selected:
            selected.append(p)
    return selected
