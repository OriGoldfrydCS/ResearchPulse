"""
Vector store wrapper for Pinecone using LangChain.

Provides a unified interface for vector storage operations with
Pinecone backend, configured via environment variables.
"""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from .embeddings import get_embeddings, EmbeddingConfig, check_embeddings_available
from .pinecone_client import (
    PineconeConfig,
    get_pinecone_index,
    get_namespace,
    check_pinecone_available,
    reset_pinecone,
)


# =============================================================================
# Vector Store Factory
# =============================================================================

_vector_store: Optional[PineconeVectorStore] = None


def get_vector_store(
    embedding_config: Optional[EmbeddingConfig] = None,
    pinecone_config: Optional[PineconeConfig] = None,
) -> PineconeVectorStore:
    """
    Get or create the LangChain Pinecone vector store.
    
    Args:
        embedding_config: Optional embedding configuration
        pinecone_config: Optional Pinecone configuration
        
    Returns:
        Configured PineconeVectorStore instance
        
    Raises:
        ValueError: If required env vars are not configured
    """
    global _vector_store
    
    if _vector_store is None:
        embeddings = get_embeddings(embedding_config)
        
        if pinecone_config is None:
            pinecone_config = PineconeConfig.from_env()
        
        _vector_store = PineconeVectorStore(
            index=get_pinecone_index(pinecone_config),
            embedding=embeddings,
            namespace=pinecone_config.namespace,
            text_key="text",
        )
        
    return _vector_store


def reset_vector_store() -> None:
    """Reset the cached vector store instance (useful for testing)."""
    global _vector_store
    _vector_store = None
    reset_pinecone()


# =============================================================================
# Vector Store Operations
# =============================================================================

def similarity_search(
    query: str,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """
    Perform similarity search on the vector store.
    
    Args:
        query: Query text to search for
        k: Number of results to return (default: 5)
        filter: Optional metadata filter
        
    Returns:
        List of matching Document objects
    """
    store = get_vector_store()
    return store.similarity_search(query, k=k, filter=filter)


def similarity_search_with_score(
    query: str,
    k: int = 5,
    filter: Optional[Dict[str, Any]] = None,
) -> List[tuple[Document, float]]:
    """
    Perform similarity search and return results with scores.
    
    Args:
        query: Query text to search for
        k: Number of results to return (default: 5)
        filter: Optional metadata filter
        
    Returns:
        List of (Document, score) tuples, ordered by relevance
    """
    store = get_vector_store()
    return store.similarity_search_with_score(query, k=k, filter=filter)


def add_documents(
    documents: List[Document],
    ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Add documents to the vector store.
    
    Args:
        documents: List of Document objects to add
        ids: Optional list of IDs for the documents
        
    Returns:
        List of document IDs
    """
    store = get_vector_store()
    return store.add_documents(documents, ids=ids)


def add_texts(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    ids: Optional[List[str]] = None,
) -> List[str]:
    """
    Add texts with optional metadata to the vector store.
    
    Args:
        texts: List of text strings to add
        metadatas: Optional list of metadata dicts
        ids: Optional list of IDs
        
    Returns:
        List of document IDs
    """
    store = get_vector_store()
    return store.add_texts(texts, metadatas=metadatas, ids=ids)


# =============================================================================
# Graceful Availability Check
# =============================================================================

def check_vector_store_available() -> tuple[bool, str]:
    """
    Check if the vector store is available and properly configured.
    
    Returns:
        Tuple of (is_available, message)
    """
    # Check embeddings first
    embed_ok, embed_msg = check_embeddings_available()
    if not embed_ok:
        return False, f"Embeddings not configured: {embed_msg}"
    
    # Check Pinecone
    pinecone_ok, pinecone_msg = check_pinecone_available()
    if not pinecone_ok:
        return False, f"Pinecone not configured: {pinecone_msg}"
    
    return True, f"Vector store ready. {embed_msg}. {pinecone_msg}"


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for vector store module.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("Vector Store Module Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Module imports
    print("\n1. Module Imports:")
    all_passed &= check("PineconeVectorStore imported", PineconeVectorStore is not None)
    all_passed &= check("Document imported", Document is not None)
    
    # Test 2: Check availability function
    print("\n2. Availability Check:")
    available, message = check_vector_store_available()
    all_passed &= check("returns tuple", isinstance(available, bool) and isinstance(message, str))
    print(f"     Status: {'Available' if available else 'Not available'}")
    print(f"     Message: {message}")
    
    # Test 3: Function signatures
    print("\n3. Function Signatures:")
    all_passed &= check("get_vector_store callable", callable(get_vector_store))
    all_passed &= check("similarity_search callable", callable(similarity_search))
    all_passed &= check("similarity_search_with_score callable", callable(similarity_search_with_score))
    all_passed &= check("add_documents callable", callable(add_documents))
    all_passed &= check("add_texts callable", callable(add_texts))
    all_passed &= check("reset_vector_store callable", callable(reset_vector_store))
    
    # Test 4: Document creation (mock test)
    print("\n4. Document Creation:")
    try:
        doc = Document(
            page_content="Test paper about neural networks and transformers",
            metadata={
                "paper_id": "test-001",
                "title": "Test Paper",
                "source": "arxiv",
                "embedding_type": "paper",
            }
        )
        all_passed &= check("creates Document", isinstance(doc, Document))
        all_passed &= check("has page_content", doc.page_content is not None)
        all_passed &= check("has metadata", doc.metadata.get("paper_id") == "test-001")
    except Exception as e:
        all_passed &= check(f"Document creation failed: {e}", False)

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
