"""
Tool: check_seen_papers - Identify which papers have been seen before.

This tool compares a list of papers against the Papers State DB to determine
which papers are new (unseen) and which have already been processed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from db.json_store import get_seen_paper_ids, get_paper_by_id, DEFAULT_DATA_DIR


# =============================================================================
# Input/Output Models
# =============================================================================

class PaperInput(BaseModel):
    """Minimal paper input for seen/unseen check."""
    arxiv_id: str = Field(..., description="arXiv paper ID (e.g., '2501.00123')")
    title: str = Field(..., description="Paper title")
    # Optional additional fields that will be passed through
    abstract: Optional[str] = Field(None, description="Paper abstract")
    authors: Optional[List[str]] = Field(None, description="List of authors")
    categories: Optional[List[str]] = Field(None, description="arXiv categories")
    published: Optional[str] = Field(None, description="Publication date")
    link: Optional[str] = Field(None, description="URL to paper")


class SeenPaperInfo(BaseModel):
    """Information about a previously seen paper."""
    arxiv_id: str = Field(..., description="arXiv paper ID")
    title: str = Field(..., description="Paper title")
    date_seen: str = Field(..., description="When the paper was first seen")
    decision: str = Field(..., description="Previous decision: ignored/saved/shared/logged")
    importance: str = Field(..., description="Previous importance assessment")


class CheckSeenResult(BaseModel):
    """Result of checking papers against seen history."""
    unseen_papers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Papers that have not been seen before"
    )
    seen_papers: List[SeenPaperInfo] = Field(
        default_factory=list,
        description="Papers that have been seen before with their history"
    )
    summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Summary counts"
    )


# =============================================================================
# Tool Implementation
# =============================================================================

def check_seen_papers(
    papers: List[Dict[str, Any]],
    data_dir: Optional[Path] = None
) -> CheckSeenResult:
    """
    Check which papers have been seen before.
    
    Compares the input papers against the Papers State DB to identify
    which are new (unseen) and which have already been processed.
    
    Args:
        papers: List of paper dictionaries. Each must have at minimum:
            - arxiv_id: The arXiv paper ID
            - title: The paper title
            Additional fields will be preserved in the output.
        data_dir: Optional path to data directory. Uses default if not provided.
            
    Returns:
        CheckSeenResult containing:
        - unseen_papers: List of papers not seen before (full input data preserved)
        - seen_papers: List of previously seen papers with their history
        - summary: Counts of total, unseen, and seen papers
        
    Example:
        >>> papers = [
        ...     {"arxiv_id": "2501.00123", "title": "New Paper"},
        ...     {"arxiv_id": "2501.00456", "title": "Old Paper"},  # already in DB
        ... ]
        >>> result = check_seen_papers(papers)
        >>> len(result.unseen_papers)
        1
        >>> result.summary
        {"total": 2, "unseen": 1, "seen": 1}
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    
    # Get the set of all seen paper IDs
    seen_ids = get_seen_paper_ids(data_dir)
    
    unseen_papers: List[Dict[str, Any]] = []
    seen_papers: List[SeenPaperInfo] = []
    
    for paper in papers:
        arxiv_id = paper.get("arxiv_id") or paper.get("paper_id")
        
        if not arxiv_id:
            # Skip papers without an ID, but log a warning
            continue
            
        if arxiv_id in seen_ids:
            # Paper has been seen before - get its history
            paper_record = get_paper_by_id(arxiv_id, data_dir)
            if paper_record:
                seen_papers.append(SeenPaperInfo(
                    arxiv_id=arxiv_id,
                    title=paper.get("title", paper_record.get("title", "Unknown")),
                    date_seen=paper_record.get("date_seen", ""),
                    decision=paper_record.get("decision", "unknown"),
                    importance=paper_record.get("importance", "unknown")
                ))
        else:
            # Paper is new - preserve all input data
            unseen_papers.append(paper)
    
    # Build summary
    summary = {
        "total": len(papers),
        "unseen": len(unseen_papers),
        "seen": len(seen_papers),
    }
    
    return CheckSeenResult(
        unseen_papers=unseen_papers,
        seen_papers=seen_papers,
        summary=summary
    )


def check_seen_papers_json(papers: List[Dict[str, Any]], data_dir: Optional[Path] = None) -> dict:
    """
    JSON-serializable version of check_seen_papers for tool invocation.
    
    Args:
        papers: List of paper dictionaries.
        data_dir: Optional data directory path.
        
    Returns:
        Dictionary with unseen_papers, seen_papers, and summary.
    """
    result = check_seen_papers(papers, data_dir)
    return {
        "unseen_papers": result.unseen_papers,
        "seen_papers": [p.model_dump() for p in result.seen_papers],
        "summary": result.summary
    }


# =============================================================================
# LangChain Tool Definition (for ReAct agent)
# =============================================================================

CHECK_SEEN_PAPERS_DESCRIPTION = """
Check which papers from a list have been seen before.

Input: A list of papers, each with at least 'arxiv_id' and 'title'.
Output: 
- unseen_papers: Papers not previously processed
- seen_papers: Papers already in the database with their history
- summary: Counts of total, unseen, and seen papers

Use this tool after fetching papers from arXiv to identify which ones are new.
"""

# Tool schema for LangChain
CHECK_SEEN_PAPERS_SCHEMA = {
    "name": "check_seen_papers",
    "description": CHECK_SEEN_PAPERS_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "papers": {
                "type": "array",
                "description": "List of papers to check",
                "items": {
                    "type": "object",
                    "properties": {
                        "arxiv_id": {"type": "string", "description": "arXiv paper ID"},
                        "title": {"type": "string", "description": "Paper title"},
                    },
                    "required": ["arxiv_id", "title"]
                }
            }
        },
        "required": ["papers"]
    }
}


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for check_seen_papers.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("check_seen_papers Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Check with known seen paper from demo data
    print("\n1. Check Known Seen Paper:")
    try:
        # Paper ID "2501.00123" should be in the demo papers_state.json
        papers = [
            {"arxiv_id": "2501.00123", "title": "Scaling Laws for Mixture-of-Experts"},
        ]
        result = check_seen_papers(papers)
        all_passed &= check("returns CheckSeenResult", isinstance(result, CheckSeenResult))
        all_passed &= check("identifies seen paper", len(result.seen_papers) == 1)
        all_passed &= check("no unseen papers", len(result.unseen_papers) == 0)
        all_passed &= check("summary total = 1", result.summary["total"] == 1)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 2: Check with unknown paper
    print("\n2. Check Unknown Paper:")
    try:
        papers = [
            {"arxiv_id": "9999.99999", "title": "Completely New Paper"},
        ]
        result = check_seen_papers(papers)
        all_passed &= check("no seen papers", len(result.seen_papers) == 0)
        all_passed &= check("one unseen paper", len(result.unseen_papers) == 1)
        all_passed &= check("unseen preserves data", result.unseen_papers[0]["title"] == "Completely New Paper")
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 3: Mixed seen and unseen
    print("\n3. Mixed Seen and Unseen:")
    try:
        papers = [
            {"arxiv_id": "2501.00123", "title": "Known Paper"},  # seen
            {"arxiv_id": "9999.11111", "title": "New Paper 1"},   # unseen
            {"arxiv_id": "2501.00789", "title": "Another Known"},  # seen (FlashAttention-3)
            {"arxiv_id": "9999.22222", "title": "New Paper 2"},   # unseen
        ]
        result = check_seen_papers(papers)
        all_passed &= check("summary total = 4", result.summary["total"] == 4)
        all_passed &= check("2 seen papers", result.summary["seen"] == 2)
        all_passed &= check("2 unseen papers", result.summary["unseen"] == 2)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 4: Seen paper includes history
    print("\n4. Seen Paper History:")
    try:
        papers = [{"arxiv_id": "2501.00789", "title": "FlashAttention-3"}]
        result = check_seen_papers(papers)
        if result.seen_papers:
            seen = result.seen_papers[0]
            all_passed &= check("has date_seen", bool(seen.date_seen))
            all_passed &= check("has decision", seen.decision in ["ignored", "saved", "shared", "logged"])
            all_passed &= check("has importance", seen.importance in ["high", "medium", "low"])
        else:
            all_passed &= check("paper should be seen", False)
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 5: JSON output format
    print("\n5. JSON Output Format:")
    try:
        papers = [{"arxiv_id": "9999.33333", "title": "Test Paper", "abstract": "Test abstract"}]
        result = check_seen_papers_json(papers)
        all_passed &= check("is dictionary", isinstance(result, dict))
        all_passed &= check("has unseen_papers key", "unseen_papers" in result)
        all_passed &= check("has seen_papers key", "seen_papers" in result)
        all_passed &= check("has summary key", "summary" in result)
        all_passed &= check("preserves extra fields", result["unseen_papers"][0].get("abstract") == "Test abstract")
    except Exception as e:
        all_passed &= check(f"failed: {e}", False)

    # Test 6: Empty input
    print("\n6. Empty Input:")
    try:
        result = check_seen_papers([])
        all_passed &= check("handles empty list", result.summary["total"] == 0)
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
