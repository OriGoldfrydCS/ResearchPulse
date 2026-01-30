"""
Store Interface - Abstract interface for persistence backends.

Provides a unified API for storing and retrieving ResearchPulse data.
Implementations: PostgresStore (production), JSONStore (legacy/dev).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID


class Store(ABC):
    """Abstract base class for persistence stores."""
    
    # =========================================================================
    # User Operations
    # =========================================================================
    
    @abstractmethod
    def get_or_create_default_user(self) -> Dict[str, Any]:
        """Get or create the default user for single-user mode."""
        pass
    
    @abstractmethod
    def get_user(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    def update_user(self, user_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile."""
        pass
    
    # =========================================================================
    # Paper Operations
    # =========================================================================
    
    @abstractmethod
    def upsert_paper(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a paper by source + external_id."""
        pass
    
    @abstractmethod
    def get_paper(self, paper_id: UUID) -> Optional[Dict[str, Any]]:
        """Get paper by ID."""
        pass
    
    @abstractmethod
    def get_paper_by_external_id(self, source: str, external_id: str) -> Optional[Dict[str, Any]]:
        """Get paper by source and external ID."""
        pass
    
    @abstractmethod
    def list_papers(
        self,
        user_id: UUID,
        seen: Optional[bool] = None,
        decision: Optional[str] = None,
        importance: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List papers with filters."""
        pass
    
    @abstractmethod
    def delete_paper_view(self, user_id: UUID, paper_id: UUID) -> bool:
        """Delete a paper view (mark as unseen)."""
        pass
    
    @abstractmethod
    def delete_paper(self, paper_id: UUID) -> bool:
        """Delete a paper and all related data."""
        pass
    
    # =========================================================================
    # Paper View Operations
    # =========================================================================
    
    @abstractmethod
    def upsert_paper_view(self, user_id: UUID, paper_id: UUID, view_data: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a paper view."""
        pass
    
    @abstractmethod
    def get_paper_view(self, user_id: UUID, paper_id: UUID) -> Optional[Dict[str, Any]]:
        """Get a paper view."""
        pass
    
    @abstractmethod
    def is_paper_seen(self, user_id: UUID, source: str, external_id: str) -> bool:
        """Check if a paper has been seen by user."""
        pass
    
    @abstractmethod
    def update_paper_view(self, user_id: UUID, paper_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update paper view (notes, tags, etc.)."""
        pass
    
    # =========================================================================
    # Colleague Operations
    # =========================================================================
    
    @abstractmethod
    def list_colleagues(self, user_id: UUID, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List all colleagues for a user."""
        pass
    
    @abstractmethod
    def get_colleague(self, colleague_id: UUID) -> Optional[Dict[str, Any]]:
        """Get colleague by ID."""
        pass
    
    @abstractmethod
    def create_colleague(self, user_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new colleague."""
        pass
    
    @abstractmethod
    def update_colleague(self, colleague_id: UUID, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update colleague."""
        pass
    
    @abstractmethod
    def delete_colleague(self, colleague_id: UUID) -> bool:
        """Delete colleague."""
        pass
    
    # =========================================================================
    # Run Operations
    # =========================================================================
    
    @abstractmethod
    def create_run(self, user_id: UUID, run_id: str, prompt: str) -> Dict[str, Any]:
        """Create a new run record."""
        pass
    
    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get run by run_id."""
        pass
    
    @abstractmethod
    def update_run(self, run_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update run status, metrics, etc."""
        pass
    
    @abstractmethod
    def list_runs(self, user_id: UUID, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List runs for a user."""
        pass
    
    # =========================================================================
    # Action Operations
    # =========================================================================
    
    @abstractmethod
    def create_action(
        self,
        run_id: UUID,
        user_id: UUID,
        action_type: str,
        paper_id: Optional[UUID] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an action taken during a run."""
        pass
    
    # =========================================================================
    # Email Operations
    # =========================================================================
    
    @abstractmethod
    def create_email(
        self,
        user_id: UUID,
        paper_id: Optional[UUID],
        recipient_email: str,
        subject: str,
        body_text: str,
        body_preview: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an email record."""
        pass
    
    @abstractmethod
    def update_email_status(self, email_id: UUID, status: str, error: Optional[str] = None, provider_id: Optional[str] = None) -> Dict[str, Any]:
        """Update email status after send attempt."""
        pass
    
    @abstractmethod
    def list_emails(self, user_id: UUID, status: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List emails for a user."""
        pass
    
    @abstractmethod
    def email_exists(self, paper_id: UUID, recipient_email: str) -> bool:
        """Check if email already sent for paper to recipient (idempotency)."""
        pass
    
    # =========================================================================
    # Calendar Event Operations
    # =========================================================================
    
    @abstractmethod
    def create_calendar_event(
        self,
        user_id: UUID,
        paper_id: Optional[UUID],
        title: str,
        start_time: datetime,
        duration_minutes: int = 30,
        ics_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a calendar event record."""
        pass
    
    @abstractmethod
    def list_calendar_events(self, user_id: UUID, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List calendar events for a user."""
        pass
    
    @abstractmethod
    def calendar_event_exists(self, paper_id: UUID, start_time: datetime) -> bool:
        """Check if calendar event exists (idempotency)."""
        pass
    
    # =========================================================================
    # Share Operations
    # =========================================================================
    
    @abstractmethod
    def create_share(
        self,
        user_id: UUID,
        paper_id: UUID,
        colleague_id: UUID,
        reason: Optional[str] = None,
        match_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a share record."""
        pass
    
    @abstractmethod
    def update_share_status(self, share_id: UUID, status: str, error: Optional[str] = None) -> Dict[str, Any]:
        """Update share status after send attempt."""
        pass
    
    @abstractmethod
    def list_shares(self, user_id: UUID, colleague_id: Optional[UUID] = None, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List shares for a user."""
        pass
    
    @abstractmethod
    def share_exists(self, paper_id: UUID, colleague_id: UUID) -> bool:
        """Check if share already exists (idempotency)."""
        pass
    
    # =========================================================================
    # Delivery Policy Operations
    # =========================================================================
    
    @abstractmethod
    def get_delivery_policy(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get delivery policy for user."""
        pass
    
    @abstractmethod
    def upsert_delivery_policy(self, user_id: UUID, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update delivery policy."""
        pass
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Check store health. Returns (healthy, message)."""
        pass


def get_store() -> Store:
    """
    Get the appropriate store based on configuration.
    
    Uses STORAGE_BACKEND env var:
    - "db" (default): PostgresStore
    - "json": JSONStore (legacy, dev only)
    
    In production (ENV=production), always uses PostgresStore.
    """
    import os
    
    env = os.getenv("ENV", "development")
    backend = os.getenv("STORAGE_BACKEND", "db")
    
    # Force database in production
    if env == "production":
        backend = "db"
        if not os.getenv("DATABASE_URL"):
            raise RuntimeError("DATABASE_URL is required in production mode")
    
    if backend == "db":
        from .postgres_store import PostgresStore
        return PostgresStore()
    elif backend == "json":
        from .json_store import JSONStore
        return JSONStore()
    else:
        raise ValueError(f"Unknown storage backend: {backend}")


# Global store instance (lazy initialization)
_store_instance: Optional[Store] = None


def get_default_store() -> Store:
    """Get the default store instance (singleton)."""
    global _store_instance
    if _store_instance is None:
        _store_instance = get_store()
    return _store_instance


def reset_store():
    """Reset the store instance (for testing)."""
    global _store_instance
    _store_instance = None
