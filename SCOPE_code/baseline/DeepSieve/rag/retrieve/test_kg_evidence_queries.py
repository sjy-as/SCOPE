import json
import os
from datetime import datetime

from query_kg import semantic_parsing_api, engine_exec_api


# TEST_QUESTIONS = [
#     "Which NBA team did John Havlicek play for during the 1962-63 season?",
#     "Which team did Walt Frazier play for as a point guard in the 1969 NBA Finals?",
#     "Which team was founded by Peter Holt?",
#     "Who is the player who received the North Carolina Sports Hall of Fame induction in 1998?",
#     "In which season was Hakim Warrick drafted into the NBA?",
#     "Which player received both the Pro Football Hall of Fame and the Canadian Football Hall of Fame?",
#     "Which player named Danny was drafted by the Cleveland Cavaliers?",
# ]


# TEST_QUESTIONS = [
#     "For each player in Glen Selbo, Paul Hoffman, Red Rocha, which NBA/BAA teams did they later play for in their career?",
# ]

# TEST_QUESTIONS = [
#     "Who was the coach of L.A. Lakers in the 1968-69 season?",
# ]

# TEST_QUESTIONS = [
#     "Who is the player who received the North Carolina Sports Hall of Fame induction in 1998?",
# ]

# TEST_QUESTIONS = [
#     "Which team was founded by Peter Holt?",
# ]

# TEST_QUESTIONS = [
#     "Which player received both the Pro Football Hall of Fame and the Canadian Football Hall of Fame?",
# ]

# TEST_QUESTIONS = [
#     "Which player named Danny was drafted by the Cleveland Cavaliers?",
# ]


def _safe_dump(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def run_once(question: str, kg_api_url: str) -> dict:
    # 1. 语义解析
    program = semantic_parsing_api(question, kg_api_url)
    
    # 2. 执行引擎 (新代码中 program 本身就包含了解析出的字段)
    result = engine_exec_api(program, kg_api_url) or {}
    
    # 注意：现在的语义解析结果就在 program 根部，不再有 .get("parsed")
    return {
        "question": question,
        # 兼容旧逻辑，我们把解析后的 program 作为解析信息展示
        "program": result.get("program") or program, 
        "answer": result.get("answer", ""),
        "inner_content": result.get("inner_content", []),
        "evidence_count": len(result.get("evidence") or []),
        "evidence": result.get("evidence") or [],
        "unresolved": result.get("unresolved_mentions", [])
    }

def main() -> None:
    kg_api_url = os.environ.get("KG_API_URL", "http://127.0.0.1:8002/query")
    out_dir = os.environ.get("KG_TEST_OUT_DIR", "./kg_evidence_test_outputs")
    os.makedirs(out_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_results = []

    print(f"[INFO] KG_API_URL = {kg_api_url}")
    print(f"[INFO] Output dir = {os.path.abspath(out_dir)}")
    print("=" * 120)

    for i, q in enumerate(TEST_QUESTIONS, start=1):
        one = run_once(q, kg_api_url)
        merged_results.append(one)

        # 获取解析后的程序信息
        prog = one.get("program", {})

        print(f"\n[{i}] Q: {q}")
        print("    [Parsing Result]:")
        print(f"      - parser: {prog.get('parser', 'N/A')}")
        print(f"      - entities: {prog.get('entities', [])}")
        print(f"      - relation_text: {prog.get('relation_text', '')}")
        # 新增：打印解析出的 Target 意图
        targets = prog.get('relation_targets', [])
        if targets:
            print(f"      - relation_targets: {json.dumps(targets, ensure_ascii=False)}")
        
        if one.get("unresolved"):
            print(f"      - unresolved_mentions: {one['unresolved']}")

        print(f"    [Execution]:")
        print(f"      - answer: {one['answer']}")
        print(f"      - evidence_count: {one['evidence_count']}")

        # 打印证据详情
        for j, ev in enumerate(one["evidence"][:10], start=1): # 最多显示10条防止刷屏
            head = (ev.get("head_entity_snapshot") or {}).get("label", ev.get("head_entity", ""))
            head_qid = ev.get("head_qid", "")
            rel = ev.get("relation", "")
            tail = (ev.get("tail_entity_snapshot") or {}).get("label", ev.get("label", ""))
            tail_qid = ev.get("qid", "")
            sim = ev.get("similarity_score", None)
            
            direction_icon = "-->" if ev.get("direction") == "outgoing" else "<--"
            if ev.get("direction") == "self": direction_icon = "---"
            
            print(f"      - ev#{j}: [{head} ({head_qid})] {direction_icon}[{rel}]{direction_icon} [{tail} ({tail_qid})] (Score: {sim})")

        # 保存单个问题的 JSON
        per_q_file = os.path.join(out_dir, f"q{i}_{ts}.json")
        _safe_dump(per_q_file, one)

    # 保存总结果
    merged_file = os.path.join(out_dir, f"all_questions_{ts}.json")
    _safe_dump(merged_file, merged_results)

    print("\n" + "=" * 120)
    print(f"[DONE] Saved merged output to: {os.path.abspath(merged_file)}")

if __name__ == "__main__":
    main()