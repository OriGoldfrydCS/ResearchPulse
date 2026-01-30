"""
Pinecone client initialization with environment-driven settings.

Supports Pinecone configuration via environment variables:
- PINECONE_API_KEY: Pinecone API key
- PINECONE_INDEX_NAME: Name of the Pinecone index
- PINECONE_ENVIRONMENT: Pinecone environment (e.g., 'us-east-1-aws')
- PINECONE_NAMESPACE: Namespace within the index (default: 'demo')
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from pinecone import Pinecone


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class PineconeConfig:
    """Configuration for Pinecone client."""
    api_key: str
    index_name: str
    environment: str
    namespace: str
    
    @classmethod
    def from_env(cls) -> "PineconeConfig":
        """
        Load Pinecone configuration from environment variables.
        
        Required env vars:
            PINECONE_API_KEY: Pinecone API key
            PINECONE_INDEX_NAME: Name of the index
            PINECONE_ENVIRONMENT: Pinecone environment
            
        Optional:
            PINECONE_NAMESPACE: Namespace (default: 'demo')
            
        Returns:
            PineconeConfig instance
            
        Raises:
            ValueError: If required env vars are missing
        """
        api_key = os.getenv("PINECONE_API_KEY", "")
        index_name = os.getenv("PINECONE_INDEX_NAME", "")
        environment = os.getenv("PINECONE_ENVIRONMENT", "")
        namespace = os.getenv("PINECONE_NAMESPACE", "demo")
        
        missing = []
        if not api_key:
            missing.append("PINECONE_API_KEY")
        if not index_name:
            missing.append("PINECONE_INDEX_NAME")
        if not environment:
            missing.append("PINECONE_ENVIRONMENT")
            
        if missing:
            raise ValueError(
                f"Missing required Pinecone environment variables: {', '.join(missing)}. "
                "Please set them in your .env file or environment."
            )
            
        return cls(
            api_key=api_key,
            index_name=index_name,
            environment=environment,
            namespace=namespace
        )
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if all required Pinecone env vars are set."""
        return all([
            os.getenv("PINECONE_API_KEY"),
            os.getenv("PINECONE_INDEX_NAME"),
            os.getenv("PINECONE_ENVIRONMENT"),
        ])


# =============================================================================
# Client Factory
# =============================================================================

_pinecone_client: Optional[Pinecone] = None
_pinecone_index = None


def get_pinecone_client(config: Optional[PineconeConfig] = None) -> Pinecone:
    """
    Get or create the Pinecone client instance.
    
    Args:
        config: Optional PineconeConfig. If not provided, loads from env.
        
    Returns:
        Configured Pinecone client
        
    Raises:
        ValueError: If env vars are not configured
    """
    global _pinecone_client
    
    if _pinecone_client is None:
        if config is None:
            config = PineconeConfig.from_env()
            
        _pinecone_client = Pinecone(api_key=config.api_key)
        
    return _pinecone_client


def get_pinecone_index(config: Optional[PineconeConfig] = None):
    """
    Get or create the Pinecone index instance.
    
    Args:
        config: Optional PineconeConfig. If not provided, loads from env.
        
    Returns:
        Pinecone Index object
        
    Raises:
        ValueError: If env vars are not configured
    """
    global _pinecone_index
    
    if _pinecone_index is None:
        if config is None:
            config = PineconeConfig.from_env()
            
        client = get_pinecone_client(config)
        _pinecone_index = client.Index(config.index_name)
        
    return _pinecone_index


def reset_pinecone() -> None:
    """Reset the cached Pinecone instances (useful for testing)."""
    global _pinecone_client, _pinecone_index
    _pinecone_client = None
    _pinecone_index = None


def get_namespace(config: Optional[PineconeConfig] = None) -> str:
    """
    Get the configured namespace.
    
    Args:
        config: Optional PineconeConfig. If not provided, loads from env.
        
    Returns:
        Namespace string
    """
    if config is None:
        config = PineconeConfig.from_env()
    return config.namespace


# =============================================================================
# Graceful Availability Check
# =============================================================================

def check_pinecone_available() -> tuple[bool, str]:
    """
    Check if Pinecone is available and properly configured.
    
    Returns:
        Tuple of (is_available, message)
    """
    if not PineconeConfig.is_configured():
        missing = []
        if not os.getenv("PINECONE_API_KEY"):
            missing.append("PINECONE_API_KEY")
        if not os.getenv("PINECONE_INDEX_NAME"):
            missing.append("PINECONE_INDEX_NAME")
        if not os.getenv("PINECONE_ENVIRONMENT"):
            missing.append("PINECONE_ENVIRONMENT")
        return False, f"Missing environment variables: {', '.join(missing)}"
    
    try:
        config = PineconeConfig.from_env()
        return True, f"Pinecone configured: index={config.index_name}, namespace={config.namespace}"
    except ValueError as e:
        return False, str(e)


# =============================================================================
# Self-Check
# =============================================================================

def self_check() -> bool:
    """
    Run self-check tests for Pinecone client module.
    
    Returns:
        True if all checks pass, False otherwise.
    """
    print("=" * 60)
    print("Pinecone Client Module Self-Check")
    print("=" * 60)
    
    all_passed = True
    
    def check(name: str, condition: bool) -> bool:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        return condition

    # Test 1: Config class structure
    print("\n1. PineconeConfig Structure:")
    all_passed &= check("has dataclass fields", hasattr(PineconeConfig, "__dataclass_fields__"))
    
    # Test 2: Check availability function
    print("\n2. Availability Check:")
    available, message = check_pinecone_available()
    all_passed &= check("returns tuple", isinstance(available, bool) and isinstance(message, str))
    print(f"     Status: {'Available' if available else 'Not available'}")
    print(f"     Message: {message}")
    
    # Test 3: Test with mock environment
    print("\n3. Config from Environment (mocked):")
    original_env = {
        "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY"),
        "PINECONE_INDEX_NAME": os.environ.get("PINECONE_INDEX_NAME"),
        "PINECONE_ENVIRONMENT": os.environ.get("PINECONE_ENVIRONMENT"),
        "PINECONE_NAMESPACE": os.environ.get("PINECONE_NAMESPACE"),
    }
    
    try:
        # Set mock values
        os.environ["PINECONE_API_KEY"] = "test-api-key-123"
        os.environ["PINECONE_INDEX_NAME"] = "test-index"
        os.environ["PINECONE_ENVIRONMENT"] = "us-east-1-aws"
        os.environ["PINECONE_NAMESPACE"] = "test-namespace"
        
        config = PineconeConfig.from_env()
        all_passed &= check("loads api_key", config.api_key == "test-api-key-123")
        all_passed &= check("loads index_name", config.index_name == "test-index")
        all_passed &= check("loads environment", config.environment == "us-east-1-aws")
        all_passed &= check("loads namespace", config.namespace == "test-namespace")
        all_passed &= check("is_configured returns True", PineconeConfig.is_configured())
        
    finally:
        # Restore original env
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_pinecone()
    
    # Test 4: Default namespace
    print("\n4. Default Namespace:")
    try:
        os.environ["PINECONE_API_KEY"] = "test-key"
        os.environ["PINECONE_INDEX_NAME"] = "test-index"
        os.environ["PINECONE_ENVIRONMENT"] = "test-env"
        os.environ.pop("PINECONE_NAMESPACE", None)
        
        config = PineconeConfig.from_env()
        all_passed &= check("default namespace is 'demo'", config.namespace == "demo")
        
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_pinecone()
    
    # Test 5: Missing env vars error
    print("\n5. Missing Env Vars Handling:")
    original_key = os.environ.pop("PINECONE_API_KEY", None)
    try:
        try:
            PineconeConfig.from_env()
            all_passed &= check("raises ValueError on missing", False)
        except ValueError as e:
            all_passed &= check("raises ValueError on missing", True)
            all_passed &= check("error mentions missing var", "PINECONE_API_KEY" in str(e))
    finally:
        if original_key:
            os.environ["PINECONE_API_KEY"] = original_key
        reset_pinecone()

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
