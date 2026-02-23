"""
Tool: fetch_arxiv_papers - Fetch recent papers from arXiv API.

This tool queries the arXiv API with category filters and returns
recent papers matching the researcher's interests.

**API Source:**
Uses the official arXiv API (arxiv.org) via the arxiv Python library.
Returns papers with: title, abstract, authors, categories, published date, links.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Default max papers to fetch from arXiv API (configurable via ARXIV_MAX_RESULTS env var)
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "20"))


# =============================================================================
# Input/Output Models
# =============================================================================

class FetchArxivInput(BaseModel):
    """Input for fetch_arxiv_papers tool."""
    categories_include: List[str] = Field(
        default_factory=list,
        description="arXiv categories to include (e.g., ['cs.CL', 'cs.LG'])"
    )
    categories_exclude: List[str] = Field(
        default_factory=list,
        description="arXiv categories to exclude"
    )
    max_results: int = Field(
        default=ARXIV_MAX_RESULTS,
        ge=1,
        le=100,
        description="Maximum number of papers to fetch (from ARXIV_MAX_RESULTS env var, default: 20)"
    )
    query: Optional[str] = Field(
        None,
        description="Optional keyword query to filter papers"
    )
    days_back: int = Field(
        default=7,
        ge=1,
        le=30,
        description="How many days back to search (default: 7)"
    )


class ArxivPaper(BaseModel):
    """A paper retrieved from arXiv."""
    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2501.00123')")
    title: str = Field(..., description="Paper title")
    abstract: str = Field("", description="Paper abstract")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    categories: List[str] = Field(default_factory=list, description="arXiv categories")
    published: str = Field("", description="Publication date (ISO format)")
    updated: str = Field("", description="Last updated date (ISO format)")
    link: str = Field("", description="URL to paper abstract page")
    pdf_link: str = Field("", description="URL to PDF")


class FetchArxivResult(BaseModel):
    """Result of fetch_arxiv_papers tool."""
    success: bool = Field(..., description="Whether fetch succeeded")
    papers: List[ArxivPaper] = Field(default_factory=list, description="Retrieved papers")
    total_found: int = Field(0, description="Total papers found")
    query_info: Dict[str, Any] = Field(
        default_factory=dict, description="Query parameters used"
    )
    error: Optional[str] = Field(None, description="Error message if failed")


# =============================================================================
# Mock Data for Demo
# =============================================================================

MOCK_PAPERS = [
    {
        "arxiv_id": "2501.01001",
        "title": "Scaling Laws for Large Language Model Training: A Comprehensive Study",
        "abstract": "We present a comprehensive study of scaling laws for large language models, examining how model performance varies with compute, data, and parameter count. Our analysis reveals new insights into optimal allocation of training resources and provides practical guidelines for training future models.",
        "authors": ["Alice Chen", "Bob Smith", "Carol Williams"],
        "categories": ["cs.CL", "cs.LG", "cs.AI"],
        "published": "2026-01-07T00:00:00Z",
        "updated": "2026-01-07T00:00:00Z",
        "link": "https://arxiv.org/abs/2501.01001",
        "pdf_link": "https://arxiv.org/pdf/2501.01001.pdf"
    },
    {
        "arxiv_id": "2501.01002",
        "title": "FlashAttention-4: Efficient Attention Mechanisms for Long Contexts",
        "abstract": "We introduce FlashAttention-4, an I/O-aware exact attention algorithm that extends efficient attention to sequences of 1M+ tokens. Our method achieves 3x speedup over previous approaches while maintaining numerical precision.",
        "authors": ["David Lee", "Eva Martinez", "Frank Johnson"],
        "categories": ["cs.LG", "cs.CL"],
        "published": "2026-01-06T00:00:00Z",
        "updated": "2026-01-07T00:00:00Z",
        "link": "https://arxiv.org/abs/2501.01002",
        "pdf_link": "https://arxiv.org/pdf/2501.01002.pdf"
    },
    {
        "arxiv_id": "2501.01003",
        "title": "Retrieval-Augmented Generation for Scientific Literature: Methods and Applications",
        "abstract": "This paper presents a novel RAG framework specifically designed for scientific literature understanding. We demonstrate significant improvements in factual accuracy and citation generation compared to baseline approaches.",
        "authors": ["Grace Kim", "Henry Wang", "Iris Brown"],
        "categories": ["cs.CL", "cs.IR"],
        "published": "2026-01-05T00:00:00Z",
        "updated": "2026-01-05T00:00:00Z",
        "link": "https://arxiv.org/abs/2501.01003",
        "pdf_link": "https://arxiv.org/pdf/2501.01003.pdf"
    },
    {
        "arxiv_id": "2501.01004",
        "title": "Efficient Fine-tuning of Vision-Language Models with Adapter Networks",
        "abstract": "We propose a parameter-efficient approach to fine-tuning large vision-language models using lightweight adapter networks. Our method achieves comparable performance to full fine-tuning while reducing trainable parameters by 95%.",
        "authors": ["Jack Zhang", "Kelly Liu"],
        "categories": ["cs.CV", "cs.LG"],
        "published": "2026-01-04T00:00:00Z",
        "updated": "2026-01-04T00:00:00Z",
        "link": "https://arxiv.org/abs/2501.01004",
        "pdf_link": "https://arxiv.org/pdf/2501.01004.pdf"
    },
    {
        "arxiv_id": "2501.01005",
        "title": "Neural Machine Translation with Sparse Mixture-of-Experts",
        "abstract": "We explore the application of sparse mixture-of-experts architectures to neural machine translation. Our experiments show improved translation quality across 100+ language pairs while maintaining computational efficiency.",
        "authors": ["Laura Patel", "Mike Thompson", "Nancy Garcia"],
        "categories": ["cs.CL", "cs.AI"],
        "published": "2026-01-03T00:00:00Z",
        "updated": "2026-01-03T00:00:00Z",
        "link": "https://arxiv.org/abs/2501.01005",
        "pdf_link": "https://arxiv.org/pdf/2501.01005.pdf"
    },
    {
        "arxiv_id": "2501.01006",
        "title": "Reinforcement Learning from Human Feedback: Best Practices and Pitfalls",
        "abstract": "This survey examines best practices for implementing RLHF in language model training. We analyze common failure modes and propose solutions based on extensive experiments across multiple model scales.",
        "authors": ["Oscar Rodriguez", "Patricia Wilson"],
        "categories": ["cs.LG", "cs.CL", "cs.AI"],
        "published": "2026-01-02T00:00:00Z",
        "updated": "2026-01-02T00:00:00Z",
        "link": "https://arxiv.org/abs/2501.01006",
        "pdf_link": "https://arxiv.org/pdf/2501.01006.pdf"
    },
    {
        "arxiv_id": "2501.01007",
        "title": "Cryptocurrency Market Prediction Using Transformer Networks",
        "abstract": "We apply transformer-based models to cryptocurrency price prediction, analyzing market patterns and sentiment signals. Our approach demonstrates improved forecasting accuracy over traditional methods.",
        "authors": ["Quinn Adams", "Rachel Lee"],
        "categories": ["q-fin.ST", "cs.LG"],
        "published": "2026-01-01T00:00:00Z",
        "updated": "2026-01-01T00:00:00Z",
        "link": "https://arxiv.org/abs/2501.01007",
        "pdf_link": "https://arxiv.org/pdf/2501.01007.pdf"
    },
    {
        "arxiv_id": "2501.01008",
        "title": "Model Compression via Knowledge Distillation: A Survey",
        "abstract": "A comprehensive survey of knowledge distillation techniques for model compression. We categorize existing approaches, analyze their effectiveness, and identify promising directions for future research.",
        "authors": ["Sam Johnson", "Tina Park", "Uma Singh"],
        "categories": ["cs.LG", "cs.CV"],
        "published": "2026-01-08T00:00:00Z",
        "updated": "2026-01-08T00:00:00Z",
        "link": "https://arxiv.org/abs/2501.01008",
        "pdf_link": "https://arxiv.org/pdf/2501.01008.pdf"
    },
]


# =============================================================================
# Tool Implementation
# =============================================================================

def fetch_arxiv_papers(
    categories_include: List[str] = None,
    categories_exclude: List[str] = None,
    max_results: int = ARXIV_MAX_RESULTS,
    query: Optional[str] = None,
    days_back: int = 7,
    use_mock: bool = True,
    start_index: int = 0,
) -> FetchArxivResult:
    """
    Fetch recent papers from arXiv matching specified criteria.
    
    This tool queries arXiv for papers in specified categories and returns
    structured paper information for further processing.
    
    Args:
        categories_include: List of arXiv categories to include.
            Example: ["cs.CL", "cs.LG", "cs.AI"]
            If empty, searches all categories.
            
        categories_exclude: List of arXiv categories to exclude.
            Papers with these categories will be filtered out.
            
        max_results: Maximum number of papers to return (default: 20).
            Limited to 100 to prevent excessive API usage.
            
        query: Optional keyword query to filter results.
            Searches in title and abstract.
            
        days_back: How many days back to search (default: 7).
            Limits results to papers published within this window.
            
        use_mock: If True, returns mock data for demo (default: True).
            Set to False to use real arXiv API (requires network).
            
    Returns:
        FetchArxivResult with:
        - success: Whether the fetch succeeded
        - papers: List of ArxivPaper objects
        - total_found: Number of papers found
        - query_info: Parameters used for the query
        - error: Error message if failed
        
    Example:
        >>> result = fetch_arxiv_papers(
        ...     categories_include=["cs.CL", "cs.LG"],
        ...     max_results=10,
        ... )
        >>> for paper in result.papers:
        ...     print(f"{paper.arxiv_id}: {paper.title}")
    """
    categories_include = categories_include or []
    categories_exclude = categories_exclude or []
    
    query_info = {
        "categories_include": categories_include,
        "categories_exclude": categories_exclude,
        "max_results": max_results,
        "query": query,
        "days_back": days_back,
        "start_index": start_index,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    # Use mock data for demo
    if use_mock:
        return _fetch_mock_papers(
            categories_include,
            categories_exclude,
            max_results,
            query,
            query_info,
        )
    
    # Real arXiv API fetch with retry on rate-limit (HTTP 429)
    max_retries = 4
    base_delay = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            return _fetch_real_papers(
                categories_include,
                categories_exclude,
                max_results,
                query,
                days_back,
                query_info,
                start_index=start_index,
            )
        except Exception as e:
            err_msg = str(e)
            is_rate_limit = "429" in err_msg or "Too Many Requests" in err_msg

            if is_rate_limit and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))  # 5, 10, 20, 40 …
                logger.warning(
                    "[FETCH_ARXIV] HTTP 429 rate-limited by arXiv. "
                    "Retry %d/%d in %ds…", attempt, max_retries, delay,
                )
                time.sleep(delay)
                continue

            # Non-retryable error or final attempt exhausted
            logger.error("[FETCH_ARXIV] Failed after %d attempt(s): %s", attempt, err_msg)
            return FetchArxivResult(
                success=False,
                papers=[],
                total_found=0,
                query_info=query_info,
                error=f"arXiv API error: {err_msg}",
            )

    # Should never reach here, but just in case
    return FetchArxivResult(
        success=False,
        papers=[],
        total_found=0,
        query_info=query_info,
        error="arXiv API error: retries exhausted",
    )


def _fetch_mock_papers(
    categories_include: List[str],
    categories_exclude: List[str],
    max_results: int,
    query: Optional[str],
    query_info: Dict[str, Any],
) -> FetchArxivResult:
    """Fetch papers from mock data."""
    papers = []
    
    for paper_data in MOCK_PAPERS:
        # Filter by categories
        paper_cats = set(paper_data["categories"])
        
        # Include filter
        if categories_include:
            if not paper_cats.intersection(categories_include):
                continue
        
        # Exclude filter
        if categories_exclude:
            if paper_cats.intersection(categories_exclude):
                continue
        
        # Query filter (simple keyword match)
        if query:
            query_lower = query.lower()
            title_match = query_lower in paper_data["title"].lower()
            abstract_match = query_lower in paper_data["abstract"].lower()
            if not (title_match or abstract_match):
                continue
        
        papers.append(ArxivPaper(**paper_data))
        
        if len(papers) >= max_results:
            break
    
    return FetchArxivResult(
        success=True,
        papers=papers,
        total_found=len(papers),
        query_info=query_info,
        error=None,
    )


def _fetch_real_papers(
    categories_include: List[str],
    categories_exclude: List[str],
    max_results: int,
    query: Optional[str],
    days_back: int,
    query_info: Dict[str, Any],
    start_index: int = 0,
) -> FetchArxivResult:
    """Fetch papers from real arXiv API."""
    try:
        import arxiv
    except ImportError:
        return FetchArxivResult(
            success=False,
            papers=[],
            total_found=0,
            query_info=query_info,
            error="arxiv library not installed. Run: pip install arxiv",
        )
    
    # Build search query
    query_parts = []
    
    if categories_include:
        cat_query = " OR ".join(f"cat:{cat}" for cat in categories_include)
        query_parts.append(f"({cat_query})")
    
    if query:
        # Support multi-topic queries separated by OR
        # Each topic needs its own ti:/abs: prefix for correct arXiv API parsing
        topics = [t.strip() for t in query.split(" OR ") if t.strip()]
        if len(topics) > 1:
            ti_parts = " OR ".join(f'ti:"{t}"' for t in topics)
            abs_parts = " OR ".join(f'abs:"{t}"' for t in topics)
            query_parts.append(f"({ti_parts} OR {abs_parts})")
        else:
            query_parts.append(f"(ti:{query} OR abs:{query})")
    
    search_query = " AND ".join(query_parts) if query_parts else "cat:cs.CL"
    
    # Perform search — request modest overhead for post-filters.
    # start_index allows paginating through results on re-runs.
    fetch_count = min(max_results + 20, max_results * 2, 100)
    search = arxiv.Search(
        query=search_query,
        max_results=start_index + fetch_count,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    
    # Use the built-in Client with a 3-second page delay to respect
    # arXiv's rate-limit policy (max 1 request per 3 seconds).
    client = arxiv.Client(
        page_size=50,
        delay_seconds=3.0,
        num_retries=3,
    )
    
    papers = []
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    skipped = 0
    
    for result in client.results(search):
        # Skip first start_index results for pagination
        if skipped < start_index:
            skipped += 1
            continue

        # Date filter
        if result.published.replace(tzinfo=None) < cutoff_date:
            continue
        
        # Exclude filter
        paper_cats = [cat.split(".")[0] + "." + cat.split(".")[-1] if "." in cat else cat 
                     for cat in result.categories]
        if categories_exclude:
            if set(paper_cats).intersection(categories_exclude):
                continue
        
        # Extract arxiv_id from entry_id
        arxiv_id = result.entry_id.split("/")[-1]
        
        paper = ArxivPaper(
            arxiv_id=arxiv_id,
            title=result.title,
            abstract=result.summary,
            authors=[author.name for author in result.authors],
            categories=result.categories,
            published=result.published.isoformat() + "Z",
            updated=result.updated.isoformat() + "Z",
            link=result.entry_id,
            pdf_link=result.pdf_url or "",
        )
        papers.append(paper)
        
        if len(papers) >= max_results:
            break
    
    return FetchArxivResult(
        success=True,
        papers=papers,
        total_found=len(papers),
        query_info=query_info,
        error=None,
    )


def fetch_single_paper(arxiv_id: str) -> FetchArxivResult:
    """Fetch exactly one paper by its arXiv ID.

    Uses the arXiv API ``id_list`` parameter for a direct lookup.
    Falls back to mock data when the arxiv library is unavailable.

    Args:
        arxiv_id: The arXiv paper ID (e.g. ``"2301.00001"``).

    Returns:
        A ``FetchArxivResult`` containing zero or one papers.
    """
    query_info = {"arxiv_id": arxiv_id, "mode": "single_paper"}

    # Check mock data first
    for paper_data in MOCK_PAPERS:
        if paper_data["arxiv_id"] == arxiv_id:
            return FetchArxivResult(
                success=True,
                papers=[ArxivPaper(**paper_data)],
                total_found=1,
                query_info=query_info,
            )

    try:
        import arxiv as arxiv_lib
    except ImportError:
        return FetchArxivResult(
            success=False, papers=[], total_found=0,
            query_info=query_info,
            error="arxiv library not installed. Run: pip install arxiv",
        )

    try:
        search = arxiv_lib.Search(id_list=[arxiv_id])
        client = arxiv_lib.Client(page_size=1, delay_seconds=3.0, num_retries=3)

        for result in client.results(search):
            entry_id = result.entry_id.split("/")[-1]
            paper = ArxivPaper(
                arxiv_id=entry_id,
                title=result.title,
                abstract=result.summary,
                authors=[a.name for a in result.authors],
                categories=result.categories,
                published=result.published.isoformat() + "Z",
                updated=result.updated.isoformat() + "Z",
                link=result.entry_id,
                pdf_link=result.pdf_url or "",
            )
            return FetchArxivResult(
                success=True, papers=[paper], total_found=1,
                query_info=query_info,
            )

        return FetchArxivResult(
            success=True, papers=[], total_found=0,
            query_info=query_info,
            error=f"Paper {arxiv_id} not found on arXiv",
        )
    except Exception as e:
        logger.error("[FETCH_ARXIV] Single-paper fetch failed: %s", e)
        return FetchArxivResult(
            success=False, papers=[], total_found=0,
            query_info=query_info,
            error=f"arXiv API error: {e}",
        )


def fetch_arxiv_papers_json(
    categories_include: List[str] = None,
    categories_exclude: List[str] = None,
    max_results: int = ARXIV_MAX_RESULTS,
    query: Optional[str] = None,
    days_back: int = 7,
    use_mock: bool = True,
) -> dict:
    """JSON-serializable version of fetch_arxiv_papers."""
    result = fetch_arxiv_papers(
        categories_include=categories_include,
        categories_exclude=categories_exclude,
        max_results=max_results,
        query=query,
        days_back=days_back,
        use_mock=use_mock,
    )
    return {
        "success": result.success,
        "papers": [p.model_dump() for p in result.papers],
        "total_found": result.total_found,
        "query_info": result.query_info,
        "error": result.error,
    }


# =============================================================================
# LangChain Tool Definition (for ReAct agent)
# =============================================================================

FETCH_ARXIV_DESCRIPTION = """
Fetch recent papers from arXiv matching specified category and keyword criteria.

Input:
- categories_include: List of arXiv categories to include (e.g., ["cs.CL", "cs.LG"])
- categories_exclude: List of arXiv categories to exclude (e.g., ["cs.CR"])
- max_results: Maximum number of papers to return (default: 20, max: 100)
- query: Optional keyword query to filter by title/abstract
- days_back: How many days back to search (default: 7)

Output:
- success: Whether the fetch succeeded
- papers: List of papers with arxiv_id, title, abstract, authors, categories, link
- total_found: Number of papers found
- query_info: Query parameters used
- error: Error message if failed

Use this tool as the FIRST step to retrieve papers from arXiv.
Then use check_seen_papers to identify which are new.

Example categories: cs.CL (computation and language), cs.LG (machine learning),
cs.AI (artificial intelligence), cs.CV (computer vision), stat.ML (statistics ML)
"""

FETCH_ARXIV_SCHEMA = {
    "name": "fetch_arxiv_papers",
    "description": FETCH_ARXIV_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "categories_include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "arXiv categories to include (e.g., ['cs.CL', 'cs.LG'])"
            },
            "categories_exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": "arXiv categories to exclude"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of papers to return (default: 20)",
                "default": 20
            },
            "query": {
                "type": "string",
                "description": "Optional keyword query for title/abstract"
            },
            "days_back": {
                "type": "integer",
                "description": "Days back to search (default: 7)",
                "default": 7
            }
        },
        "required": []
    }
}


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for fetch_arxiv_papers tool.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("fetch_arxiv_papers Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Basic fetch with mock data
    print("\n1. Basic Fetch (Mock Data):")
    try:
        result = fetch_arxiv_papers(use_mock=True)
        all_passed &= check("returns FetchArxivResult", isinstance(result, FetchArxivResult))
        all_passed &= check("success is True", result.success)
        all_passed &= check("has papers", len(result.papers) > 0)
        all_passed &= check("total_found > 0", result.total_found > 0)
        all_passed &= check("no error", result.error is None)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 2: Category filtering
    print("\n2. Category Filtering:")
    try:
        result = fetch_arxiv_papers(
            categories_include=["cs.CL"],
            use_mock=True,
        )
        all_passed &= check("returns papers", len(result.papers) > 0)
        # All papers should have cs.CL
        for paper in result.papers:
            has_cat = "cs.CL" in paper.categories
            all_passed &= check(f"paper {paper.arxiv_id} has cs.CL", has_cat)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 3: Category exclusion
    print("\n3. Category Exclusion:")
    try:
        result = fetch_arxiv_papers(
            categories_exclude=["q-fin.ST"],
            use_mock=True,
        )
        all_passed &= check("returns papers", len(result.papers) > 0)
        # No papers should have q-fin.ST
        for paper in result.papers:
            no_excluded = "q-fin.ST" not in paper.categories
            all_passed &= check(f"paper {paper.arxiv_id} excludes q-fin.ST", no_excluded)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 4: Max results limit
    print("\n4. Max Results Limit:")
    try:
        result = fetch_arxiv_papers(max_results=3, use_mock=True)
        all_passed &= check("returns <= 3 papers", len(result.papers) <= 3)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: Paper structure
    print("\n5. Paper Structure:")
    try:
        result = fetch_arxiv_papers(max_results=1, use_mock=True)
        if result.papers:
            paper = result.papers[0]
            all_passed &= check("has arxiv_id", bool(paper.arxiv_id))
            all_passed &= check("has title", bool(paper.title))
            all_passed &= check("has abstract", bool(paper.abstract))
            all_passed &= check("has authors list", isinstance(paper.authors, list))
            all_passed &= check("has categories list", isinstance(paper.categories, list))
            all_passed &= check("has link", bool(paper.link))
        else:
            all_passed &= check("expected at least one paper", False)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: Query info
    print("\n6. Query Info:")
    try:
        result = fetch_arxiv_papers(
            categories_include=["cs.LG"],
            max_results=5,
            use_mock=True,
        )
        all_passed &= check("query_info present", len(result.query_info) > 0)
        all_passed &= check("has categories_include", "categories_include" in result.query_info)
        all_passed &= check("has max_results", "max_results" in result.query_info)
        all_passed &= check("has timestamp", "timestamp" in result.query_info)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 7: JSON output
    print("\n7. JSON Output:")
    try:
        result = fetch_arxiv_papers_json(max_results=2, use_mock=True)
        all_passed &= check("returns dict", isinstance(result, dict))
        all_passed &= check("has success", "success" in result)
        all_passed &= check("has papers", "papers" in result)
        all_passed &= check("papers is list", isinstance(result["papers"], list))
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 8: Tool schema
    print("\n8. Tool Schema:")
    all_passed &= check("schema has name", FETCH_ARXIV_SCHEMA["name"] == "fetch_arxiv_papers")
    all_passed &= check("schema has description", len(FETCH_ARXIV_SCHEMA["description"]) > 50)
    all_passed &= check("schema has parameters", "parameters" in FETCH_ARXIV_SCHEMA)

    # Test 9: Empty results with strict filter
    print("\n9. Empty Results with Strict Filter:")
    try:
        result = fetch_arxiv_papers(
            categories_include=["nonexistent.CAT"],
            use_mock=True,
        )
        all_passed &= check("success even with no results", result.success)
        all_passed &= check("empty papers list", len(result.papers) == 0)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 10: Query keyword filtering
    print("\n10. Query Keyword Filtering:")
    try:
        result = fetch_arxiv_papers(
            query="attention",
            use_mock=True,
        )
        all_passed &= check("returns papers", len(result.papers) > 0)
        # Papers should mention "attention" in title or abstract
        for paper in result.papers:
            has_keyword = (
                "attention" in paper.title.lower() or
                "attention" in paper.abstract.lower()
            )
            all_passed &= check(f"paper {paper.arxiv_id} mentions attention", has_keyword)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("All checks PASSED!")
    else:
        print("Some checks FAILED!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = self_check()
    sys.exit(0 if success else 1)
