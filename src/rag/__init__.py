"""
RAG module - Pinecone vector store integration using LangChain.

This module provides:
- Environment-driven embeddings configuration
- Pinecone client initialization
- Vector store wrapper for LangChain
- Retriever for semantic similarity search
"""

from .embeddings import (
    EmbeddingConfig,
    get_embeddings,
    reset_embeddings,
    embed_texts,
    embed_query,
    check_embeddings_available,
)

from .pinecone_client import (
    PineconeConfig,
    get_pinecone_client,
    get_pinecone_index,
    get_namespace,
    reset_pinecone,
    check_pinecone_available,
)

from .vector_store import (
    get_vector_store,
    reset_vector_store,
    similarity_search,
    similarity_search_with_score,
    add_documents,
    add_texts,
    check_vector_store_available,
)

from .retriever import (
    RetrievedMatch,
    RetrievalResult,
    RetrieverConfig,
    retrieve_similar,
    retrieve_similar_json,
    retrieve_similar_safe,
    retrieve_similar_papers,
    retrieve_similar_to_own_papers,
)

__all__ = [
    # Embeddings
    "EmbeddingConfig",
    "get_embeddings",
    "reset_embeddings",
    "embed_texts",
    "embed_query",
    "check_embeddings_available",
    # Pinecone client
    "PineconeConfig",
    "get_pinecone_client",
    "get_pinecone_index",
    "get_namespace",
    "reset_pinecone",
    "check_pinecone_available",
    # Vector store
    "get_vector_store",
    "reset_vector_store",
    "similarity_search",
    "similarity_search_with_score",
    "add_documents",
    "add_texts",
    "check_vector_store_available",
    # Retriever
    "RetrievedMatch",
    "RetrievalResult",
    "RetrieverConfig",
    "retrieve_similar",
    "retrieve_similar_json",
    "retrieve_similar_safe",
    "retrieve_similar_papers",
    "retrieve_similar_to_own_papers",
]
