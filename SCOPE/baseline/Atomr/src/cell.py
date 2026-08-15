from query_knowledge_source.query_google import execute_web_query
from query_knowledge_source.query_wikikg import semantic_parsing_api, engine_exec_api

# Table smoke test
TABLE_API_URL = "http://127.0.0.1:1216/api/search"
titles, results = execute_web_query("OpenAI", TABLE_API_URL, 3)
print("[Table] titles:", titles[:3])
print("[Table] keys:", results.keys())
print("[Table] top1:", results.get("organic_results", [])[:1])

# KG smoke test
program = semantic_parsing_api("Who is the spouse of Barack Obama?", api_url="")
kg_results = engine_exec_api(program, api_url="")
print("[KG] result:", kg_results)
