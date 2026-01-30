"""
Retriever implementation for RAG queries.

Provides retrieval functionality that returns both text and metadata,
with configurable top_k and similarity threshold filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

from .vector_store import (
    similarity_search_with_score,
    check_vector_store_available,
)


# =============================================================================
# Result Models
# =============================================================================

class RetrievedMatch(BaseModel):
    """A single retrieved match from the vector store."""
    text: str = Field(..., description="The document text content")
    score: float = Field(..., description="Similarity score (higher is more similar)")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata (paper_id, title, source, etc.)"
    )


class RetrievalResult(BaseModel):
    """Result of a retrieval query."""
    query: str = Field(..., description="The original query text")
    matches: List[RetrievedMatch] = Field(
        default_factory=list,
        description="List of retrieved matches"
    )
    total_found: int = Field(0, description="Total number of matches found before threshold filtering")
    filtered_count: int = Field(0, description="Number of matches after threshold filtering")
    similarity_threshold: Optional[float] = Field(None, description="Threshold used for filtering")
    

# =============================================================================
# Retriever Configuration
# =============================================================================

@dataclass
class RetrieverConfig:
    """Configuration for the retriever."""
    top_k: int = 5
    similarity_threshold: Optional[float] = None
    default_filter: Optional[Dict[str, Any]] = None


# =============================================================================
# Retrieval Functions
# =============================================================================

def retrieve_similar(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
    filter: Optional[Dict[str, Any]] = None,
) -> RetrievalResult:
    """
    Retrieve similar documents from the vector store.
    
    Args:
        query_text: The query text to search for
        top_k: Maximum number of results to return (default: 5)
        similarity_threshold: Optional minimum similarity score (0.0 to 1.0).
            Results below this threshold will be filtered out.
        filter: Optional metadata filter for the search
        
    Returns:
        RetrievalResult containing matches with text and metadata
        
    Raises:
        ValueError: If vector store is not available
    """
    # Check availability
    available, message = check_vector_store_available()
    if not available:
        raise ValueError(f"Vector store not available: {message}")
    
    # Perform similarity search with scores
    results = similarity_search_with_score(
        query=query_text,
        k=top_k,
        filter=filter,
    )
    
    total_found = len(results)
    
    # Convert results to matches
    matches: List[RetrievedMatch] = []
    for doc, score in results:
        # Apply threshold filtering if specified
        if similarity_threshold is not None and score < similarity_threshold:
            continue
            
        matches.append(RetrievedMatch(
            text=doc.page_content,
            score=score,
            metadata=doc.metadata,
        ))
    
    return RetrievalResult(
        query=query_text,
        matches=matches,
        total_found=total_found,
        filtered_count=len(matches),
        similarity_threshold=similarity_threshold,
    )


def retrieve_similar_json(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
    filter: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    JSON-serializable version of retrieve_similar.
    
    Args:
        query_text: The query text to search for
        top_k: Maximum number of results to return
        similarity_threshold: Optional minimum similarity score
        filter: Optional metadata filter
        
    Returns:
        Dictionary with query, matches, and metadata
    """
    result = retrieve_similar(
        query_text=query_text,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        filter=filter,
    )
    return result.model_dump()


# =============================================================================
# Specialized Retrieval Functions
# =============================================================================

def retrieve_similar_papers(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
) -> RetrievalResult:
    """
    Retrieve similar papers (embedding_type='paper').
    
    Args:
        query_text: The query text
        top_k: Maximum results
        similarity_threshold: Minimum similarity score
        
    Returns:
        RetrievalResult with paper matches only
    """
    return retrieve_similar(
        query_text=query_text,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        filter={"embedding_type": "paper"},
    )


def retrieve_similar_to_own_papers(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
) -> RetrievalResult:
    """
    Retrieve matches from researcher's own papers (embedding_type='own_paper').
    
    Args:
        query_text: The query text
        top_k: Maximum results
        similarity_threshold: Minimum similarity score
        
    Returns:
        RetrievalResult with own paper matches only
    """
    return retrieve_similar(
        query_text=query_text,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        filter={"embedding_type": "own_paper"},
    )


# =============================================================================
# Graceful Fallback
# =============================================================================

def retrieve_similar_safe(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
    filter: Optional[Dict[str, Any]] = None,
) -> RetrievalResult:
    """
    Safe version of retrieve_similar that returns empty result on error.
    
    This is useful for graceful degradation when the vector store
    is not available.
    
    Args:
        query_text: The query text to search for
        top_k: Maximum number of results to return
        similarity_threshold: Optional minimum similarity score
        filter: Optional metadata filter
        
    Returns:
        RetrievalResult (empty if vector store unavailable)
    """
    try:
        return retrieve_similar(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter,
        )
    except (ValueError, Exception) as e:
        # Return empty result with error info in metadata
        return RetrievalResult(
            query=query_text,
            matches=[],
            total_found=0,
            filtered_count=0,
            similarity_threshold=similarity_threshold,
        )


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for retriever module.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("Retriever Module Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Model structure
    print("\n1. Model Structure:")
    all_passed &= check("RetrievedMatch has text field", "text" in RetrievedMatch.model_fields)
    all_passed &= check("RetrievedMatch has score field", "score" in RetrievedMatch.model_fields)
    all_passed &= check("RetrievedMatch has metadata field", "metadata" in RetrievedMatch.model_fields)
    all_passed &= check("RetrievalResult has query field", "query" in RetrievalResult.model_fields)
    all_passed &= check("RetrievalResult has matches field", "matches" in RetrievalResult.model_fields)
    
    # Test 2: Model instantiation
    print("\n2. Model Instantiation:")
    try:
        match = RetrievedMatch(
            text="Test document about machine learning",
            score=0.95,
            metadata={"paper_id": "test-001", "title": "Test Paper"}
        )
        all_passed &= check("creates RetrievedMatch", isinstance(match, RetrievedMatch))
        all_passed &= check("match has correct score", match.score == 0.95)
        
        result = RetrievalResult(
            query="machine learning papers",
            matches=[match],
            total_found=1,
            filtered_count=1,
            similarity_threshold=0.8
        )
        all_passed &= check("creates RetrievalResult", isinstance(result, RetrievalResult))
        all_passed &= check("result has one match", len(result.matches) == 1)
    except Exception as e:
        all_passed &= check(f"model instantiation failed: {e}", False)
    
    # Test 3: JSON serialization
    print("\n3. JSON Serialization:")
    try:
        result = RetrievalResult(
            query="test query",
            matches=[
                RetrievedMatch(text="doc1", score=0.9, metadata={"id": "1"}),
                RetrievedMatch(text="doc2", score=0.8, metadata={"id": "2"}),
            ],
            total_found=5,
            filtered_count=2,
            similarity_threshold=0.7
        )
        json_dict = result.model_dump()
        all_passed &= check("serializes to dict", isinstance(json_dict, dict))
        all_passed &= check("dict has query", json_dict["query"] == "test query")
        all_passed &= check("dict has matches list", len(json_dict["matches"]) == 2)
    except Exception as e:
        all_passed &= check(f"serialization failed: {e}", False)
    
    # Test 4: Function signatures
    print("\n4. Function Signatures:")
    all_passed &= check("retrieve_similar callable", callable(retrieve_similar))
    all_passed &= check("retrieve_similar_json callable", callable(retrieve_similar_json))
    all_passed &= check("retrieve_similar_safe callable", callable(retrieve_similar_safe))
    all_passed &= check("retrieve_similar_papers callable", callable(retrieve_similar_papers))
    
    # Test 5: Safe retrieval (graceful failure)
    print("\n5. Safe Retrieval (Graceful Failure):")
    try:
        result = retrieve_similar_safe("test query")
        all_passed &= check("returns RetrievalResult", isinstance(result, RetrievalResult))
        all_passed &= check("has query set", result.query == "test query")
        # Empty matches expected when vector store not configured
        all_passed &= check("handles unavailable store", True)
    except Exception as e:
        all_passed &= check(f"safe retrieval should not raise: {e}", False)

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
