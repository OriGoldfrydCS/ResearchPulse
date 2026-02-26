"""
ResearchPulse Tools Package.

This package provides all tools for the ReAct agent workflow:

1. fetch_arxiv_papers - Fetch papers from arXiv API
2. check_seen_papers - Identify previously seen papers
3. retrieve_similar_from_pinecone - RAG query for similar documents
4. score_relevance_and_importance - Score paper relevance/novelty
5. decide_delivery_action - Determine delivery actions
6. persist_state - Save paper decisions
7. generate_report - Create final run report
8. terminate_run - End the agent episode

Note: This package uses lazy imports to avoid dependency issues.
Import individual tools directly when needed:
    from tools.fetch_arxiv import fetch_arxiv_papers
"""

# Convenience list of tool names
TOOL_NAMES = [
    "fetch_arxiv_papers",
    "check_seen_papers",
    "retrieve_similar_from_pinecone",
    "score_relevance_and_importance",
    "decide_delivery_action",
    "persist_state",
    "generate_report",
    "terminate_run",
]


def get_all_tools():
    """
    Lazy load and return all tool functions.
    
    Returns:
        dict: Mapping of tool names to tool functions.
    """
    from .fetch_arxiv import fetch_arxiv_papers
    from .check_seen import check_seen_papers
    from .retrieve_similar import retrieve_similar_from_pinecone
    from .score_relevance import score_relevance_and_importance
    from .decide_delivery import decide_delivery_action
    from .persist_state import persist_paper_decision
    from .generate_report import generate_report
    from .terminate_run import terminate_run
    
    return {
        "fetch_arxiv_papers": fetch_arxiv_papers,
        "check_seen_papers": check_seen_papers,
        "retrieve_similar_from_pinecone": retrieve_similar_from_pinecone,
        "score_relevance_and_importance": score_relevance_and_importance,
        "decide_delivery_action": decide_delivery_action,
        "persist_state": persist_paper_decision,
        "generate_report": generate_report,
        "terminate_run": terminate_run,
    }


def get_all_schemas():
    """
    Lazy load and return all tool schemas.
    
    Returns:
        list: List of all tool schemas for LangChain.
    """
    from .fetch_arxiv import FETCH_ARXIV_SCHEMA
    from .check_seen import CHECK_SEEN_PAPERS_SCHEMA
    from .retrieve_similar import RETRIEVE_SIMILAR_SCHEMA
    from .score_relevance import SCORE_RELEVANCE_SCHEMA
    from .decide_delivery import DECIDE_DELIVERY_SCHEMA
    from .persist_state import PERSIST_STATE_SCHEMA
    from .generate_report import GENERATE_REPORT_SCHEMA
    from .terminate_run import TERMINATE_RUN_SCHEMA
    
    return [
        FETCH_ARXIV_SCHEMA,
        CHECK_SEEN_PAPERS_SCHEMA,
        RETRIEVE_SIMILAR_SCHEMA,
        SCORE_RELEVANCE_SCHEMA,
        DECIDE_DELIVERY_SCHEMA,
        PERSIST_STATE_SCHEMA,
        GENERATE_REPORT_SCHEMA,
        TERMINATE_RUN_SCHEMA,
    ]


__all__ = [
    "TOOL_NAMES",
    "get_all_tools",
    "get_all_schemas",
]
