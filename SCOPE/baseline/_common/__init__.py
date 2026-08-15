"""Shared infrastructure for the simple LLM / RAG baselines.

Each method under baseline/<Method>/ imports from this module so we
don't duplicate the LLM client, retriever wrappers, judge logic, or
the run-driver across seven directories.
"""
