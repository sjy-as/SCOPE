"""
Utility functions for formatting and processing knowledge retrieval results.
"""

from typing import List, Dict, Any, Optional
import json


def format_text_passages(passages: List[Dict[str, Any]]) -> str:
    """Format text passages from TextSource into readable string."""
    if not passages:
        return ""
    
    formatted = []
    for i, p in enumerate(passages, 1):
        entity = p.get("entity", "")
        passage = p.get("passage", "")
        
        line = f"[{i}] {entity}"
        if passage:
            line += f" | {passage}"
        formatted.append(line.strip())
    
    return "\n".join([x for x in formatted if x])


def format_table_results(tables: List[Dict[str, Any]]) -> str:
    """Format table results from TableSource into readable string."""
    if not tables:
        return ""
    
    formatted = []
    for i, t in enumerate(tables, 1):
        title = t.get("page_title", "")
        text = t.get("text", "")
        
        if title:
            formatted.append(f"[{i}] {title}")
        if text:
            formatted.append(f"    {text}")
    
    return "\n".join([x for x in formatted if x])


def format_kg_results(results: List[Dict[str, Any]]) -> str:
    """Format KG query results into readable string."""
    if not results:
        return ""
    
    formatted = []
    for i, r in enumerate(results, 1):
        label = r.get("label", "")
        qid = r.get("qid", "")
        
        if label:
            line = f"[{i}] {label}"
            if qid:
                line += f" (QID: {qid})"
            formatted.append(line)
    
    return "\n".join([x for x in formatted if x])


def format_supporting_knowledge(knowledge_dict: Dict[str, Any]) -> str:
    """Format supporting knowledge from all sources for display."""
    output = []
    
    if "Text" in knowledge_dict and knowledge_dict["Text"]:
        if isinstance(knowledge_dict["Text"], list):
            text_str = format_text_passages(knowledge_dict["Text"])
        else:
            text_str = str(knowledge_dict["Text"])
        output.append(f"\n=== Text (Wikipedia) ===\n{text_str}")
    
    if "Table" in knowledge_dict and knowledge_dict["Table"]:
        if isinstance(knowledge_dict["Table"], list):
            table_str = format_table_results(knowledge_dict["Table"])
        else:
            table_str = str(knowledge_dict["Table"])
        output.append(f"\n=== Table ===\n{table_str}")
    
    if "KG" in knowledge_dict and knowledge_dict["KG"]:
        if isinstance(knowledge_dict["KG"], list):
            kg_str = format_kg_results(knowledge_dict["KG"])
        else:
            kg_str = str(knowledge_dict["KG"])
        output.append(f"\n=== KG (Knowledge Graph) ===\n{kg_str}")
    
    return "\n".join(output) if output else ""


def extract_answers_from_evidence(evidence: List[Dict[str, Any]]) -> List[str]:
    """Extract answer strings from evidence list."""
    answers = []
    
    for item in evidence:
        if isinstance(item, dict):
            # Try different possible answer fields
            for key in ["label", "answer", "entity", "text"]:
                if key in item and item[key]:
                    answers.append(str(item[key]))
                    break
        else:
            answers.append(str(item))
    
    return answers


def parse_operator_tree(tree_json: str) -> Dict[str, Any]:
    """Parse operator tree from JSON string."""
    try:
        return json.loads(tree_json)
    except json.JSONDecodeError as e:
        print(f"[Error] Failed to parse operator tree: {e}")
        return {}


def resolve_reference(ref: str, context: Dict[str, Any]) -> Any:
    """Resolve a reference like '${s1}' to actual value in context."""
    if not ref or not ref.startswith("${") or not ref.endswith("}"):
        return ref
    
    ref_name = ref[2:-1]
    return context.get(ref_name)


def substitute_variables(text: str, context: Dict[str, Any]) -> str:
    """Substitute variables in text like '${var}' with values from context."""
    import re
    
    def replace_var(match):
        var_name = match.group(1)
        value = context.get(var_name, match.group(0))
        return str(value)
    
    return re.sub(r'\$\{(\w+)\}', replace_var, text)


def merge_results(results: List[List[str]]) -> List[str]:
    """Merge multiple result lists, removing duplicates while preserving order."""
    seen = set()
    merged = []
    
    for result_list in results:
        for item in result_list:
            item_lower = str(item).lower()
            if item_lower not in seen:
                seen.add(item_lower)
                merged.append(item)
    
    return merged


def collect_results(results: List[List[str]]) -> List[str]:
    """Collect results from multiple operations."""
    collected = []
    for result_list in results:
        collected.extend(result_list)
    return collected


def format_qa_pair(question: str, answer: str, evidence: Optional[str] = None) -> str:
    """Format a Q&A pair for display or logging."""
    output = f"Q: {question}\nA: {answer}"
    if evidence:
        output += f"\nEvidence: {evidence}"
    return output


def log_execution_step(step_num: int, operator: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> str:
    """Log an execution step for debugging."""
    log = f"\n[Step {step_num}] {operator}\n"
    log += f"  Inputs: {json.dumps(inputs, ensure_ascii=False, indent=2)}\n"
    log += f"  Outputs: {json.dumps(outputs, ensure_ascii=False, indent=2)}"
    return log
