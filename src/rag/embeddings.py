"""
LangChain embeddings configuration with environment-driven settings.

Supports OpenAI-compatible embedding APIs via environment variables:
- EMBEDDING_API_BASE: API base URL
- EMBEDDING_API_KEY: API key
- EMBEDDING_API_MODEL: Model name
- EMBEDDING_API_DIMENSION: Embedding dimension
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

from langchain_openai import OpenAIEmbeddings


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding API."""
    api_base: str
    api_key: str
    model: str
    dimension: int
    
    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """
        Load embedding configuration from environment variables.
        
        Required env vars:
            EMBEDDING_API_BASE: API base URL
            EMBEDDING_API_KEY: API key
            EMBEDDING_API_MODEL: Model name
            EMBEDDING_API_DIMENSION: Embedding dimension (integer)
            
        Returns:
            EmbeddingConfig instance
            
        Raises:
            ValueError: If required env vars are missing
        """
        api_base = os.getenv("EMBEDDING_API_BASE", "")
        api_key = os.getenv("EMBEDDING_API_KEY", "")
        model = os.getenv("EMBEDDING_API_MODEL", "")
        dimension_str = os.getenv("EMBEDDING_API_DIMENSION", "")
        
        missing = []
        if not api_base:
            missing.append("EMBEDDING_API_BASE")
        if not api_key:
            missing.append("EMBEDDING_API_KEY")
        if not model:
            missing.append("EMBEDDING_API_MODEL")
        if not dimension_str:
            missing.append("EMBEDDING_API_DIMENSION")
            
        if missing:
            raise ValueError(
                f"Missing required embedding environment variables: {', '.join(missing)}. "
                "Please set them in your .env file or environment."
            )
            
        try:
            dimension = int(dimension_str)
        except ValueError:
            raise ValueError(
                f"EMBEDDING_API_DIMENSION must be an integer, got: '{dimension_str}'"
            )
            
        return cls(
            api_base=api_base,
            api_key=api_key,
            model=model,
            dimension=dimension
        )
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if all required embedding env vars are set."""
        return all([
            os.getenv("EMBEDDING_API_BASE"),
            os.getenv("EMBEDDING_API_KEY"),
            os.getenv("EMBEDDING_API_MODEL"),
            os.getenv("EMBEDDING_API_DIMENSION"),
        ])


# =============================================================================
# Embeddings Factory
# =============================================================================

_embeddings_instance: Optional[OpenAIEmbeddings] = None


def get_embeddings(config: Optional[EmbeddingConfig] = None) -> OpenAIEmbeddings:
    """
    Get or create the LangChain embeddings instance.
    
    Uses OpenAIEmbeddings which is compatible with any OpenAI-compatible API.
    
    Args:
        config: Optional EmbeddingConfig. If not provided, loads from env.
        
    Returns:
        Configured OpenAIEmbeddings instance
        
    Raises:
        ValueError: If env vars are not configured
    """
    global _embeddings_instance
    
    if _embeddings_instance is None:
        if config is None:
            config = EmbeddingConfig.from_env()
            
        _embeddings_instance = OpenAIEmbeddings(
            openai_api_base=config.api_base,
            openai_api_key=config.api_key,
            model=config.model,
            dimensions=config.dimension,
        )
        
    return _embeddings_instance


def reset_embeddings() -> None:
    """Reset the cached embeddings instance (useful for testing)."""
    global _embeddings_instance
    _embeddings_instance = None


def embed_texts(texts: List[str], config: Optional[EmbeddingConfig] = None) -> List[List[float]]:
    """
    Embed a list of texts.
    
    Args:
        texts: List of strings to embed
        config: Optional embedding config
        
    Returns:
        List of embedding vectors
    """
    embeddings = get_embeddings(config)
    return embeddings.embed_documents(texts)


def embed_query(text: str, config: Optional[EmbeddingConfig] = None) -> List[float]:
    """
    Embed a single query text.
    
    Args:
        text: Query string to embed
        config: Optional embedding config
        
    Returns:
        Embedding vector
    """
    embeddings = get_embeddings(config)
    return embeddings.embed_query(text)


# =============================================================================
# Graceful Availability Check
# =============================================================================

def check_embeddings_available() -> tuple[bool, str]:
    """
    Check if embeddings are available and properly configured.
    
    Returns:
        Tuple of (is_available, message)
    """
    if not EmbeddingConfig.is_configured():
        missing = []
        if not os.getenv("EMBEDDING_API_BASE"):
            missing.append("EMBEDDING_API_BASE")
        if not os.getenv("EMBEDDING_API_KEY"):
            missing.append("EMBEDDING_API_KEY")
        if not os.getenv("EMBEDDING_API_MODEL"):
            missing.append("EMBEDDING_API_MODEL")
        if not os.getenv("EMBEDDING_API_DIMENSION"):
            missing.append("EMBEDDING_API_DIMENSION")
        return False, f"Missing environment variables: {', '.join(missing)}"
    
    try:
        config = EmbeddingConfig.from_env()
        return True, f"Embeddings configured: {config.model} (dim={config.dimension})"
    except ValueError as e:
        return False, str(e)


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for embeddings module.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("Embeddings Module Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Config class structure
    print("\n1. EmbeddingConfig Structure:")
    all_passed &= check("has api_base field", hasattr(EmbeddingConfig, "__dataclass_fields__"))
    
    # Test 2: Check availability function
    print("\n2. Availability Check:")
    available, message = check_embeddings_available()
    all_passed &= check("returns tuple", isinstance(available, bool) and isinstance(message, str))
    print(f"     Status: {'Available' if available else 'Not available'}")
    print(f"     Message: {message}")
    
    # Test 3: Test with mock environment
    print("\n3. Config from Environment (mocked):")
    original_env = {
        "EMBEDDING_API_BASE": os.environ.get("EMBEDDING_API_BASE"),
        "EMBEDDING_API_KEY": os.environ.get("EMBEDDING_API_KEY"),
        "EMBEDDING_API_MODEL": os.environ.get("EMBEDDING_API_MODEL"),
        "EMBEDDING_API_DIMENSION": os.environ.get("EMBEDDING_API_DIMENSION"),
    }
    
    try:
        # Set mock values
        os.environ["EMBEDDING_API_BASE"] = "https://test.api.com/v1"
        os.environ["EMBEDDING_API_KEY"] = "test-key-123"
        os.environ["EMBEDDING_API_MODEL"] = "test-embed-model"
        os.environ["EMBEDDING_API_DIMENSION"] = "1536"
        
        config = EmbeddingConfig.from_env()
        all_passed &= check("loads api_base", config.api_base == "https://test.api.com/v1")
        all_passed &= check("loads api_key", config.api_key == "test-key-123")
        all_passed &= check("loads model", config.model == "test-embed-model")
        all_passed &= check("loads dimension as int", config.dimension == 1536)
        all_passed &= check("is_configured returns True", EmbeddingConfig.is_configured())
        
    finally:
        # Restore original env
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_embeddings()
    
    # Test 4: Missing env vars error
    print("\n4. Missing Env Vars Handling:")
    original_key = os.environ.pop("EMBEDDING_API_KEY", None)
    try:
        try:
            EmbeddingConfig.from_env()
            all_passed &= check("raises ValueError on missing", False)
        except ValueError as e:
            all_passed &= check("raises ValueError on missing", True)
            all_passed &= check("error mentions missing var", "EMBEDDING_API_KEY" in str(e))
    finally:
        if original_key:
            os.environ["EMBEDDING_API_KEY"] = original_key
        reset_embeddings()
    
    # Test 5: Invalid dimension handling
    print("\n5. Invalid Dimension Handling:")
    try:
        os.environ["EMBEDDING_API_BASE"] = "https://test.api.com/v1"
        os.environ["EMBEDDING_API_KEY"] = "test-key"
        os.environ["EMBEDDING_API_MODEL"] = "model"
        os.environ["EMBEDDING_API_DIMENSION"] = "not-a-number"
        
        try:
            EmbeddingConfig.from_env()
            all_passed &= check("raises on invalid dimension", False)
        except ValueError as e:
            all_passed &= check("raises on invalid dimension", True)
            all_passed &= check("error mentions dimension", "EMBEDDING_API_DIMENSION" in str(e))
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_embeddings()

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
