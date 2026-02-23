"""
Feature Flags for ResearchPulse Autonomous Components.

This module provides feature flag configuration for the four autonomous components:
1. LLM Novelty Scoring
2. Run Audit Log
3. Profile Evolution Suggestions
4. Live Document

All flags default to False for safety. Enable via environment variables:
- LLM_NOVELTY_ENABLED=true
- AUDIT_LOG_ENABLED=true
- PROFILE_EVOLUTION_ENABLED=true
- LIVE_DOCUMENT_ENABLED=true

Configuration options:
- LLM_RELEVANCE_ENABLED=true
- LLM_NOVELTY_MODEL: Model to use for novelty scoring (default: gpt-4o-mini)
- LLM_NOVELTY_MIN_RELEVANCE: Minimum relevance score to trigger LLM novelty (default: 0.4)
- LLM_NOVELTY_CACHE_TTL_DAYS: Cache TTL for novelty scores (default: 7)
- LLM_RELEVANCE_MODEL: Model for relevance filtering (default: gpt-4o-mini)
- LLM_RELEVANCE_CACHE_TTL_HOURS: Cache TTL for relevance results (default: 48)
- PROFILE_EVOLUTION_MIN_PAPERS: Min high-relevance papers to trigger analysis (default: 3)
- PROFILE_EVOLUTION_COOLDOWN_HOURS: Cooldown between analyses (default: 24)
- LIVE_DOCUMENT_PATH: Path for the live document file (default: artifacts/live_document.md)
- LIVE_DOCUMENT_MAX_ARCHIVE_ENTRIES: Max runs to keep in archive (default: 50)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def _str_to_bool(value: str) -> bool:
    """Convert string environment variable to boolean."""
    return value.lower() in ("true", "1", "yes", "on", "enabled")


def _get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(name, "")
    if not value:
        return default
    return _str_to_bool(value)


def _get_env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.getenv(name, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# =============================================================================
# Feature Flag Constants
# =============================================================================

# Main feature toggles - all default to False for safety
LLM_NOVELTY_ENABLED = _get_env_bool("LLM_NOVELTY_ENABLED", False)
LLM_RELEVANCE_ENABLED = _get_env_bool("LLM_RELEVANCE_ENABLED", False)
AUDIT_LOG_ENABLED = _get_env_bool("AUDIT_LOG_ENABLED", False)
PROFILE_EVOLUTION_ENABLED = _get_env_bool("PROFILE_EVOLUTION_ENABLED", False)
LIVE_DOCUMENT_ENABLED = _get_env_bool("LIVE_DOCUMENT_ENABLED", False)


@dataclass
class LLMNoveltyConfig:
    """Configuration for LLM novelty scoring."""
    enabled: bool = False
    model: str = "gpt-4o-mini"
    min_relevance_threshold: float = 0.4  # Only score papers above this relevance
    cache_ttl_days: int = 7
    max_similar_papers: int = 5  # Top-N similar papers to include in prompt
    max_tokens: int = 1000  # Max tokens for LLM response
    temperature: float = 0.3  # Lower for more consistent scoring
    fallback_on_error: bool = True  # Use heuristic if LLM fails
    
    @classmethod
    def from_env(cls) -> "LLMNoveltyConfig":
        """Load config from environment variables."""
        return cls(
            enabled=_get_env_bool("LLM_NOVELTY_ENABLED", False),
            model=os.getenv("LLM_NOVELTY_MODEL", os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")),
            min_relevance_threshold=_get_env_float("LLM_NOVELTY_MIN_RELEVANCE", 0.4),
            cache_ttl_days=_get_env_int("LLM_NOVELTY_CACHE_TTL_DAYS", 7),
            max_similar_papers=_get_env_int("LLM_NOVELTY_MAX_SIMILAR", 5),
            max_tokens=_get_env_int("LLM_NOVELTY_MAX_TOKENS", 1000),
            temperature=_get_env_float("LLM_NOVELTY_TEMPERATURE", 0.3),
            fallback_on_error=_get_env_bool("LLM_NOVELTY_FALLBACK", True),
        )


@dataclass
class LLMRelevanceConfig:
    """Configuration for LLM relevance filtering."""
    enabled: bool = False
    model: str = "gpt-4o-mini"
    temperature: float = 0.1  # Low for consistent filtering decisions
    max_tokens: int = 300  # Short structured JSON response
    cache_ttl_hours: int = 48  # Cache relevance results
    fallback_on_error: bool = True  # Allow paper through if LLM fails

    @classmethod
    def from_env(cls) -> "LLMRelevanceConfig":
        """Load config from environment variables."""
        return cls(
            enabled=_get_env_bool("LLM_RELEVANCE_ENABLED", False),
            model=os.getenv("LLM_RELEVANCE_MODEL", os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")),
            temperature=_get_env_float("LLM_RELEVANCE_TEMPERATURE", 0.1),
            max_tokens=_get_env_int("LLM_RELEVANCE_MAX_TOKENS", 300),
            cache_ttl_hours=_get_env_int("LLM_RELEVANCE_CACHE_TTL_HOURS", 48),
            fallback_on_error=_get_env_bool("LLM_RELEVANCE_FALLBACK", True),
        )


@dataclass
class AuditLogConfig:
    """Configuration for run audit logging."""
    enabled: bool = False
    include_full_abstracts: bool = False  # Include full abstracts in log (increases size)
    include_rag_results: bool = False  # Include RAG results in log
    retention_days: int = 90  # How long to keep audit logs
    
    @classmethod
    def from_env(cls) -> "AuditLogConfig":
        """Load config from environment variables."""
        return cls(
            enabled=_get_env_bool("AUDIT_LOG_ENABLED", False),
            include_full_abstracts=_get_env_bool("AUDIT_LOG_INCLUDE_ABSTRACTS", False),
            include_rag_results=_get_env_bool("AUDIT_LOG_INCLUDE_RAG", False),
            retention_days=_get_env_int("AUDIT_LOG_RETENTION_DAYS", 90),
        )


@dataclass
class ProfileEvolutionConfig:
    """Configuration for profile evolution suggestions."""
    enabled: bool = False
    min_high_relevance_papers: int = 3  # Min papers to trigger analysis
    min_novelty_for_analysis: float = 0.7  # Min novelty to consider for profile evolution
    cooldown_hours: int = 24  # Hours between analyses
    model: str = "gpt-4o-mini"
    max_suggestions_per_run: int = 3
    auto_expire_days: int = 30  # Auto-expire pending suggestions after N days
    
    @classmethod
    def from_env(cls) -> "ProfileEvolutionConfig":
        """Load config from environment variables."""
        return cls(
            enabled=_get_env_bool("PROFILE_EVOLUTION_ENABLED", False),
            min_high_relevance_papers=_get_env_int("PROFILE_EVOLUTION_MIN_PAPERS", 3),
            min_novelty_for_analysis=_get_env_float("PROFILE_EVOLUTION_MIN_NOVELTY", 0.7),
            cooldown_hours=_get_env_int("PROFILE_EVOLUTION_COOLDOWN_HOURS", 24),
            model=os.getenv("PROFILE_EVOLUTION_MODEL", os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")),
            max_suggestions_per_run=_get_env_int("PROFILE_EVOLUTION_MAX_SUGGESTIONS", 3),
            auto_expire_days=_get_env_int("PROFILE_EVOLUTION_EXPIRE_DAYS", 30),
        )


@dataclass  
class LiveDocumentConfig:
    """Configuration for live research document."""
    enabled: bool = False
    file_path: str = "artifacts/live_document.md"
    export_html: bool = False
    html_path: str = "artifacts/live_document.html"
    max_archive_entries: int = 50
    max_topic_clusters: int = 10
    max_top_papers: int = 10
    max_recent_papers: int = 20
    rolling_window_days: int = 30  # For recurring topics calculation
    save_to_db: bool = True
    save_history: bool = True
    max_history_versions: int = 20
    model: str = "gpt-4o-mini"
    
    @classmethod
    def from_env(cls) -> "LiveDocumentConfig":
        """Load config from environment variables."""
        return cls(
            enabled=_get_env_bool("LIVE_DOCUMENT_ENABLED", False),
            file_path=os.getenv("LIVE_DOCUMENT_PATH", "artifacts/live_document.md"),
            export_html=_get_env_bool("LIVE_DOCUMENT_EXPORT_HTML", False),
            html_path=os.getenv("LIVE_DOCUMENT_HTML_PATH", "artifacts/live_document.html"),
            max_archive_entries=_get_env_int("LIVE_DOCUMENT_MAX_ARCHIVE", 50),
            max_topic_clusters=_get_env_int("LIVE_DOCUMENT_MAX_CLUSTERS", 10),
            rolling_window_days=_get_env_int("LIVE_DOCUMENT_ROLLING_DAYS", 30),
            save_to_db=_get_env_bool("LIVE_DOCUMENT_SAVE_DB", True),
            save_history=_get_env_bool("LIVE_DOCUMENT_SAVE_HISTORY", True),
            max_history_versions=_get_env_int("LIVE_DOCUMENT_MAX_HISTORY", 20),
        )


@dataclass
class FeatureFlags:
    """
    Centralized feature flags for all autonomous components.
    
    Usage:
        flags = FeatureFlags.load()
        if flags.audit_log.enabled:
            # do audit logging
    """
    llm_novelty: LLMNoveltyConfig = field(default_factory=LLMNoveltyConfig)
    llm_relevance: LLMRelevanceConfig = field(default_factory=LLMRelevanceConfig)
    audit_log: AuditLogConfig = field(default_factory=AuditLogConfig)
    profile_evolution: ProfileEvolutionConfig = field(default_factory=ProfileEvolutionConfig)
    live_document: LiveDocumentConfig = field(default_factory=LiveDocumentConfig)
    
    @classmethod
    def load(cls) -> "FeatureFlags":
        """Load all feature flags from environment."""
        return cls(
            llm_novelty=LLMNoveltyConfig.from_env(),
            llm_relevance=LLMRelevanceConfig.from_env(),
            audit_log=AuditLogConfig.from_env(),
            profile_evolution=ProfileEvolutionConfig.from_env(),
            live_document=LiveDocumentConfig.from_env(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API/logging."""
        return {
            "llm_novelty": {
                "enabled": self.llm_novelty.enabled,
                "model": self.llm_novelty.model,
                "min_relevance_threshold": self.llm_novelty.min_relevance_threshold,
            },
            "llm_relevance": {
                "enabled": self.llm_relevance.enabled,
                "model": self.llm_relevance.model,
            },
            "audit_log": {
                "enabled": self.audit_log.enabled,
                "retention_days": self.audit_log.retention_days,
            },
            "profile_evolution": {
                "enabled": self.profile_evolution.enabled,
                "min_papers": self.profile_evolution.min_high_relevance_papers,
            },
            "live_document": {
                "enabled": self.live_document.enabled,
                "file_path": self.live_document.file_path,
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_feature_flags: Optional[FeatureFlags] = None


def get_feature_config() -> FeatureFlags:
    """
    Get the global feature flags configuration.
    
    Loads from environment on first call and caches.
    """
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags.load()
    return _feature_flags


def reload_feature_config() -> FeatureFlags:
    """
    Reload feature flags from environment.
    
    Useful for testing or dynamic reconfiguration.
    """
    global _feature_flags
    _feature_flags = FeatureFlags.load()
    return _feature_flags


def is_feature_enabled(feature_name: str, user_id=None) -> bool:
    """
    Check if a specific feature is enabled.
    
    All autonomous features are always enabled. This function is kept
    for API compatibility but always returns True.
    
    Args:
        feature_name: One of "LLM_NOVELTY", "AUDIT_LOG", "PROFILE_EVOLUTION", "LIVE_DOCUMENT"
        user_id: Optional user UUID (ignored, kept for compatibility)
        
    Returns:
        Always True — features are always on.
    """
    return True


# =============================================================================
# Database-backed Feature Flags
# =============================================================================

def get_feature_flags_from_db(user_id) -> Optional[Dict[str, Any]]:
    """
    Load feature flags from the database for a specific user.
    
    Args:
        user_id: UUID of the user
        
    Returns:
        Dictionary of feature flags or None if not found.
    """
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import UserSettings
        
        if not is_database_configured():
            return None
        
        with get_db_session() as db:
            settings = db.query(UserSettings).filter_by(user_id=user_id).first()
            if not settings:
                return None
            
            return {
                "llm_novelty": {
                    "enabled": settings.feature_llm_novelty_enabled or False,
                    "model": settings.feature_llm_novelty_model or "gpt-4o-mini",
                },
                "audit_log": {
                    "enabled": settings.feature_audit_log_enabled or False,
                },
                "profile_evolution": {
                    "enabled": settings.feature_profile_evolution_enabled or False,
                    "cooldown_hours": settings.feature_profile_evolution_cooldown_hours or 24,
                },
                "live_document": {
                    "enabled": settings.feature_live_document_enabled or False,
                    "max_top_papers": settings.feature_live_document_max_papers or 10,
                },
            }
    except Exception:
        return None


def update_feature_flags_in_db(user_id, flags: Dict[str, Any]) -> bool:
    """
    Update feature flags in the database for a specific user.
    
    Args:
        user_id: UUID of the user
        flags: Dictionary of feature flags to update
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        from db.database import is_database_configured, get_db_session
        from db.orm_models import UserSettings
        from uuid import UUID
        
        if not is_database_configured():
            return False
        
        # Convert string UUID if needed
        if isinstance(user_id, str):
            user_id = UUID(user_id)
        
        with get_db_session() as db:
            settings = db.query(UserSettings).filter_by(user_id=user_id).first()
            
            if not settings:
                # Create UserSettings if it doesn't exist
                settings = UserSettings(user_id=user_id)
                db.add(settings)
            
            # Update LLM Novelty
            if "llm_novelty" in flags:
                llm = flags["llm_novelty"]
                if "enabled" in llm:
                    settings.feature_llm_novelty_enabled = llm["enabled"]
                if "model" in llm:
                    settings.feature_llm_novelty_model = llm["model"]
            
            # Update Audit Log
            if "audit_log" in flags:
                audit = flags["audit_log"]
                if "enabled" in audit:
                    settings.feature_audit_log_enabled = audit["enabled"]
            
            # Update Profile Evolution
            if "profile_evolution" in flags:
                profile = flags["profile_evolution"]
                if "enabled" in profile:
                    settings.feature_profile_evolution_enabled = profile["enabled"]
                if "cooldown_hours" in profile:
                    settings.feature_profile_evolution_cooldown_hours = profile["cooldown_hours"]
            
            # Update Live Document
            if "live_document" in flags:
                live = flags["live_document"]
                if "enabled" in live:
                    settings.feature_live_document_enabled = live["enabled"]
                if "max_top_papers" in live:
                    settings.feature_live_document_max_papers = live["max_top_papers"]
            
            db.commit()
            
            # Reload global config to pick up changes
            reload_feature_config_from_db(user_id)
            
            return True
    except Exception as e:
        import logging
        logging.error(f"Failed to update feature flags: {e}")
        return False


def reload_feature_config_from_db(user_id) -> Optional[FeatureFlags]:
    """
    Reload feature flags from database into the global config.
    
    Args:
        user_id: UUID of the user
        
    Returns:
        Updated FeatureFlags or None if loading failed.
    """
    global _feature_flags
    
    db_flags = get_feature_flags_from_db(user_id)
    if not db_flags:
        return None
    
    # Start with env-based config
    if _feature_flags is None:
        _feature_flags = FeatureFlags.load()
    
    # Override with DB values
    if "llm_novelty" in db_flags:
        _feature_flags.llm_novelty.enabled = db_flags["llm_novelty"].get("enabled", False)
        if db_flags["llm_novelty"].get("model"):
            _feature_flags.llm_novelty.model = db_flags["llm_novelty"]["model"]
    
    if "audit_log" in db_flags:
        _feature_flags.audit_log.enabled = db_flags["audit_log"].get("enabled", False)
    
    if "profile_evolution" in db_flags:
        _feature_flags.profile_evolution.enabled = db_flags["profile_evolution"].get("enabled", False)
        if db_flags["profile_evolution"].get("cooldown_hours"):
            _feature_flags.profile_evolution.cooldown_hours = db_flags["profile_evolution"]["cooldown_hours"]
    
    if "live_document" in db_flags:
        _feature_flags.live_document.enabled = db_flags["live_document"].get("enabled", False)
        if db_flags["live_document"].get("max_top_papers"):
            _feature_flags.live_document.max_archive_entries = db_flags["live_document"]["max_top_papers"]
    
    return _feature_flags
