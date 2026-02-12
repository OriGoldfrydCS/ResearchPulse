"""
Configuration module for ResearchPulse.
"""

from .feature_flags import (
    is_feature_enabled,
    get_feature_config,
    FeatureFlags,
    LLM_NOVELTY_ENABLED,
    AUDIT_LOG_ENABLED,
    PROFILE_EVOLUTION_ENABLED,
    LIVE_DOCUMENT_ENABLED,
)

__all__ = [
    "is_feature_enabled",
    "get_feature_config",
    "FeatureFlags",
    "LLM_NOVELTY_ENABLED",
    "AUDIT_LOG_ENABLED",
    "PROFILE_EVOLUTION_ENABLED",
    "LIVE_DOCUMENT_ENABLED",
]
