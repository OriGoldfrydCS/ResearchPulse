"""
Tool: retrieve_similar_from_pinecone - RAG retrieval from Pinecone vector store.

This tool queries the Pinecone vector store using LangChain to find
similar documents (papers, own papers, notes) based on semantic similarity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Add parent to path for sibling imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retriever import (
    retrieve_similar,
    retrieve_similar_safe,
    RetrievalResult,
    RetrievedMatch,
)
from rag.vector_store import check_vector_store_available


# =============================================================================
# Input/Output Models
# =============================================================================

class RetrieveSimilarInput(BaseModel):
    """Input schema for retrieve_similar_from_pinecone tool."""
    query_text: str = Field(
        ...,
        description="The query text to search for similar documents"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of results to return (default: 5)"
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0 to 1.0). Results below this are filtered out."
    )
    embedding_type: Optional[str] = Field(
        default=None,
        description="Filter by embedding type: 'paper', 'own_paper', or 'note'"
    )


class RetrieveSimilarOutput(BaseModel):
    """Output schema for retrieve_similar_from_pinecone tool."""
    success: bool = Field(..., description="Whether the retrieval was successful")
    query: str = Field(..., description="The original query text")
    matches: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of matching documents with text, score, and metadata"
    )
    total_found: int = Field(0, description="Total matches before threshold filtering")
    filtered_count: int = Field(0, description="Matches after threshold filtering")
    error: Optional[str] = Field(None, description="Error message if retrieval failed")


# =============================================================================
# Tool Implementation
# =============================================================================

def retrieve_similar_from_pinecone(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
    embedding_type: Optional[str] = None,
) -> RetrieveSimilarOutput:
    """
    Retrieve similar documents from Pinecone vector store.
    
    This tool uses LangChain and Pinecone to perform semantic similarity
    search against stored embeddings of papers, researcher's own papers,
    and optional notes/summaries.
    
    Args:
        query_text: The text to search for (will be embedded and compared)
        top_k: Maximum number of results to return (default: 5)
        similarity_threshold: Optional minimum similarity score (0.0 to 1.0).
            Results with scores below this threshold will be filtered out.
        embedding_type: Optional filter for embedding type:
            - 'paper': Previously seen/saved papers
            - 'own_paper': Researcher's own papers
            - 'note': Historical agent notes/summaries
            
    Returns:
        RetrieveSimilarOutput containing:
        - success: Whether retrieval succeeded
        - query: The original query
        - matches: List of matches with text, score, and metadata
        - total_found: Count before threshold filtering
        - filtered_count: Count after threshold filtering
        - error: Error message if failed
        
    Example:
        >>> result = retrieve_similar_from_pinecone(
        ...     query_text="transformer attention mechanisms",
        ...     top_k=3,
        ...     similarity_threshold=0.7
        ... )
        >>> for match in result.matches:
        ...     print(f"{match['score']:.2f}: {match['metadata'].get('title')}")
    """
    # Check availability first
    available, message = check_vector_store_available()
    if not available:
        return RetrieveSimilarOutput(
            success=False,
            query=query_text,
            matches=[],
            total_found=0,
            filtered_count=0,
            error=f"Vector store not available: {message}"
        )
    
    # Build filter if embedding_type specified
    filter_dict = None
    if embedding_type:
        filter_dict = {"embedding_type": embedding_type}
    
    try:
        # Perform retrieval
        result = retrieve_similar(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter_dict,
        )
        
        # Convert matches to dict format
        matches_list = [
            {
                "text": m.text,
                "score": m.score,
                "metadata": m.metadata,
            }
            for m in result.matches
        ]
        
        return RetrieveSimilarOutput(
            success=True,
            query=result.query,
            matches=matches_list,
            total_found=result.total_found,
            filtered_count=result.filtered_count,
            error=None,
        )
        
    except Exception as e:
        return RetrieveSimilarOutput(
            success=False,
            query=query_text,
            matches=[],
            total_found=0,
            filtered_count=0,
            error=str(e),
        )


def retrieve_similar_from_pinecone_json(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: Optional[float] = None,
    embedding_type: Optional[str] = None,
) -> dict:
    """
    JSON-serializable version of retrieve_similar_from_pinecone.
    
    Args:
        query_text: The text to search for
        top_k: Maximum number of results
        similarity_threshold: Minimum similarity score
        embedding_type: Optional embedding type filter
        
    Returns:
        Dictionary with retrieval results
    """
    result = retrieve_similar_from_pinecone(
        query_text=query_text,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        embedding_type=embedding_type,
    )
    return result.model_dump()


# =============================================================================
# LangChain Tool Definition (for ReAct agent)
# =============================================================================

RETRIEVE_SIMILAR_DESCRIPTION = """
Retrieve similar documents from Pinecone vector store using semantic search.

Input:
- query_text: The text to search for similar documents
- top_k: Maximum number of results (default: 5)
- similarity_threshold: Minimum similarity score 0.0-1.0 (optional)
- embedding_type: Filter by type - 'paper', 'own_paper', or 'note' (optional)

Output:
- success: Whether retrieval succeeded
- matches: List of similar documents with text, score, and metadata
- total_found/filtered_count: Match counts before/after threshold

Use this tool to:
- Find papers similar to a new paper's abstract (novelty detection)
- Compare new papers against researcher's own work
- Retrieve relevant historical context

Each match includes metadata like paper_id, title, arxiv_categories, publication_date.
"""

RETRIEVE_SIMILAR_SCHEMA = {
    "name": "retrieve_similar_from_pinecone",
    "description": RETRIEVE_SIMILAR_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "query_text": {
                "type": "string",
                "description": "The text to search for similar documents"
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            },
            "similarity_threshold": {
                "type": "number",
                "description": "Minimum similarity score (0.0 to 1.0)",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "embedding_type": {
                "type": "string",
                "enum": ["paper", "own_paper", "note"],
                "description": "Filter by embedding type"
            }
        },
        "required": ["query_text"]
    }
}


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for retrieve_similar_from_pinecone tool.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("retrieve_similar_from_pinecone Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Input model validation
    print("\n1. Input Model Validation:")
    try:
        input_valid = RetrieveSimilarInput(
            query_text="test query about transformers",
            top_k=3,
            similarity_threshold=0.7
        )
        all_passed &= check("creates valid input", isinstance(input_valid, RetrieveSimilarInput))
        all_passed &= check("query_text set", input_valid.query_text == "test query about transformers")
        all_passed &= check("top_k set", input_valid.top_k == 3)
        all_passed &= check("threshold set", input_valid.similarity_threshold == 0.7)
    except Exception as e:
        all_passed &= check(f"input validation failed: {e}", False)

    # Test 2: Input model defaults
    print("\n2. Input Model Defaults:")
    try:
        input_defaults = RetrieveSimilarInput(query_text="minimal query")
        all_passed &= check("default top_k is 5", input_defaults.top_k == 5)
        all_passed &= check("default threshold is None", input_defaults.similarity_threshold is None)
        all_passed &= check("default embedding_type is None", input_defaults.embedding_type is None)
    except Exception as e:
        all_passed &= check(f"defaults test failed: {e}", False)

    # Test 3: Output model structure
    print("\n3. Output Model Structure:")
    try:
        output = RetrieveSimilarOutput(
            success=True,
            query="test query",
            matches=[
                {"text": "doc1", "score": 0.9, "metadata": {"id": "1"}},
                {"text": "doc2", "score": 0.8, "metadata": {"id": "2"}},
            ],
            total_found=5,
            filtered_count=2,
        )
        all_passed &= check("creates output", isinstance(output, RetrieveSimilarOutput))
        all_passed &= check("has success field", output.success is True)
        all_passed &= check("has matches list", len(output.matches) == 2)
    except Exception as e:
        all_passed &= check(f"output model failed: {e}", False)

    # Test 4: Graceful failure when not configured
    print("\n4. Graceful Failure (Not Configured):")
    try:
        result = retrieve_similar_from_pinecone("test query", top_k=3)
        all_passed &= check("returns RetrieveSimilarOutput", isinstance(result, RetrieveSimilarOutput))
        all_passed &= check("has query set", result.query == "test query")
        # When not configured, should fail gracefully
        if not result.success:
            all_passed &= check("has error message", result.error is not None)
            all_passed &= check("empty matches on failure", len(result.matches) == 0)
        else:
            all_passed &= check("succeeded (configured)", True)
    except Exception as e:
        all_passed &= check(f"should not raise exception: {e}", False)

    # Test 5: JSON output format
    print("\n5. JSON Output Format:")
    try:
        result = retrieve_similar_from_pinecone_json("json test query")
        all_passed &= check("returns dict", isinstance(result, dict))
        all_passed &= check("has success key", "success" in result)
        all_passed &= check("has query key", "query" in result)
        all_passed &= check("has matches key", "matches" in result)
    except Exception as e:
        all_passed &= check(f"json output failed: {e}", False)

    # Test 6: Tool schema
    print("\n6. Tool Schema:")
    all_passed &= check("schema has name", RETRIEVE_SIMILAR_SCHEMA.get("name") == "retrieve_similar_from_pinecone")
    all_passed &= check("schema has description", len(RETRIEVE_SIMILAR_SCHEMA.get("description", "")) > 50)
    all_passed &= check("schema has parameters", "parameters" in RETRIEVE_SIMILAR_SCHEMA)
    all_passed &= check("query_text is required", "query_text" in RETRIEVE_SIMILAR_SCHEMA["parameters"]["required"])

    # Test 7: Input validation bounds
    print("\n7. Input Validation Bounds:")
    try:
        # Valid bounds
        input_min = RetrieveSimilarInput(query_text="test", top_k=1, similarity_threshold=0.0)
        all_passed &= check("accepts top_k=1", input_min.top_k == 1)
        all_passed &= check("accepts threshold=0.0", input_min.similarity_threshold == 0.0)
        
        input_max = RetrieveSimilarInput(query_text="test", top_k=100, similarity_threshold=1.0)
        all_passed &= check("accepts top_k=100", input_max.top_k == 100)
        all_passed &= check("accepts threshold=1.0", input_max.similarity_threshold == 1.0)
    except Exception as e:
        all_passed &= check(f"bounds validation failed: {e}", False)

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
