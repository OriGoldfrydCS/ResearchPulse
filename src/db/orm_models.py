"""
SQLAlchemy ORM Models for ResearchPulse.

These models define the database schema for cloud-safe persistence.
All persistent state is stored here via DATABASE_URL (Supabase Postgres).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float, DateTime,
    ForeignKey, JSON, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .database import Base


# =============================================================================
# User Model
# =============================================================================

class User(Base):
    """User/researcher profile."""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    affiliation = Column(String(500))
    research_topics = Column(JSON, default=list)
    my_papers = Column(JSON, default=list)
    preferred_venues = Column(JSON, default=list)
    avoid_topics = Column(JSON, default=list)
    time_budget_per_week_minutes = Column(Integer, default=120)
    arxiv_categories_include = Column(JSON, default=list)
    arxiv_categories_exclude = Column(JSON, default=list)
    # Research interests (free text descriptions)
    interests_include = Column(Text)  # Free text research interests
    interests_exclude = Column(Text)  # Free text topics to exclude
    # Keywords for filtering
    keywords_include = Column(JSON, default=list)  # Keywords to include
    keywords_exclude = Column(JSON, default=list)  # Keywords to exclude
    stop_policy = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    paper_views = relationship("PaperView", back_populates="user", cascade="all, delete-orphan")
    colleagues = relationship("Colleague", back_populates="user", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="user", cascade="all, delete-orphan")
    emails = relationship("Email", back_populates="user", cascade="all, delete-orphan")
    calendar_events = relationship("CalendarEvent", back_populates="user", cascade="all, delete-orphan")
    shares = relationship("Share", back_populates="user", cascade="all, delete-orphan")
    delivery_policies = relationship("DeliveryPolicy", back_populates="user", cascade="all, delete-orphan")


# =============================================================================
# Paper Model
# =============================================================================

class Paper(Base):
    """Paper record from arXiv or other sources."""
    __tablename__ = "papers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(50), nullable=False, default="arxiv")
    external_id = Column(String(255), nullable=False)
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    authors = Column(JSON, default=list)
    categories = Column(JSON, default=list)
    url = Column(String(500))
    pdf_url = Column(String(500))
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('source', 'external_id', name='uq_paper_source_external_id'),
        Index('ix_paper_source_external_id', 'source', 'external_id'),
    )
    
    # Relationships
    views = relationship("PaperView", back_populates="paper", cascade="all, delete-orphan")
    actions = relationship("Action", back_populates="paper", cascade="all, delete-orphan")
    emails = relationship("Email", back_populates="paper", cascade="all, delete-orphan")
    calendar_events = relationship("CalendarEvent", back_populates="paper", cascade="all, delete-orphan")
    shares = relationship("Share", back_populates="paper", cascade="all, delete-orphan")


# =============================================================================
# PaperView Model
# =============================================================================

class PaperView(Base):
    """User's view/interaction with a paper."""
    __tablename__ = "paper_views"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    seen_count = Column(Integer, default=1)
    
    decision = Column(String(50), default="logged")
    importance = Column(String(20), default="low")
    relevance_score = Column(Float)
    novelty_score = Column(Float)
    heuristic_score = Column(Float)
    
    notes = Column(Text)
    tags = Column(JSON, default=list)
    
    embedded_in_pinecone = Column(Boolean, default=False)
    
    __table_args__ = (
        UniqueConstraint('user_id', 'paper_id', name='uq_paper_view_user_paper'),
        Index('ix_paper_view_user_id', 'user_id'),
        Index('ix_paper_view_decision', 'decision'),
        Index('ix_paper_view_importance', 'importance'),
    )
    
    user = relationship("User", back_populates="paper_views")
    paper = relationship("Paper", back_populates="views")


# =============================================================================
# Colleague Model
# =============================================================================

class Colleague(Base):
    """Colleague researcher for paper sharing."""
    __tablename__ = "colleagues"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    affiliation = Column(String(500))
    
    keywords = Column(JSON, default=list)
    categories = Column(JSON, default=list)
    topics = Column(JSON, default=list)
    
    sharing_preference = Column(String(50), default="weekly")
    enabled = Column(Boolean, default=True)
    
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_colleague_user_id', 'user_id'),
    )
    
    user = relationship("User", back_populates="colleagues")
    shares = relationship("Share", back_populates="colleague", cascade="all, delete-orphan")


# =============================================================================
# Run Model
# =============================================================================

class Run(Base):
    """Agent run/episode record."""
    __tablename__ = "runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(100), unique=True, nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    user_prompt = Column(Text)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    
    status = Column(String(50), default="running")
    stop_reason = Column(String(255))
    error_message = Column(Text)
    
    metrics = Column(JSON, default=dict)
    report = Column(JSON)
    steps = Column(JSON, default=list)
    
    papers_processed = Column(Integer, default=0)
    decisions_made = Column(Integer, default=0)
    artifacts_generated = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_run_user_id', 'user_id'),
        Index('ix_run_status', 'status'),
        Index('ix_run_started_at', 'started_at'),
    )
    
    user = relationship("User", back_populates="runs")
    actions = relationship("Action", back_populates="run", cascade="all, delete-orphan")


# =============================================================================
# Action Model
# =============================================================================

class Action(Base):
    """Action taken during a run."""
    __tablename__ = "actions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True)
    
    action_type = Column(String(50), nullable=False)
    payload = Column(JSON, default=dict)
    
    status = Column(String(50), default="completed")
    error = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_action_run_id', 'run_id'),
        Index('ix_action_action_type', 'action_type'),
    )
    
    run = relationship("Run", back_populates="actions")
    paper = relationship("Paper", back_populates="actions")


# =============================================================================
# Email Model
# =============================================================================

class Email(Base):
    """Email record for paper notifications."""
    __tablename__ = "emails"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True)
    
    recipient_email = Column(String(255), nullable=False)
    subject = Column(String(500), nullable=False)
    body_text = Column(Text)
    body_html = Column(Text)
    body_preview = Column(String(500))
    
    provider_message_id = Column(String(255))
    
    status = Column(String(50), default="queued")
    error = Column(Text)
    sent_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('paper_id', 'recipient_email', name='uq_email_paper_recipient'),
        Index('ix_email_user_id', 'user_id'),
        Index('ix_email_status', 'status'),
    )
    
    user = relationship("User", back_populates="emails")
    paper = relationship("Paper", back_populates="emails")


# =============================================================================
# CalendarEvent Model
# =============================================================================

class CalendarEvent(Base):
    """Calendar event for paper reading reminders."""
    __tablename__ = "calendar_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True)
    
    title = Column(String(500), nullable=False)
    description = Column(Text)
    start_time = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, default=30)
    
    ics_text = Column(Text)
    
    provider_event_id = Column(String(255))
    
    status = Column(String(50), default="created")
    error = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('paper_id', 'start_time', name='uq_calendar_paper_start'),
        Index('ix_calendar_user_id', 'user_id'),
        Index('ix_calendar_start_time', 'start_time'),
    )
    
    user = relationship("User", back_populates="calendar_events")
    paper = relationship("Paper", back_populates="calendar_events")


# =============================================================================
# Share Model
# =============================================================================

class Share(Base):
    """Record of paper shared with colleague."""
    __tablename__ = "shares"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="SET NULL"), nullable=True)
    colleague_id = Column(UUID(as_uuid=True), ForeignKey("colleagues.id", ondelete="SET NULL"), nullable=True)
    
    reason = Column(Text)
    match_score = Column(Float)
    
    status = Column(String(50), default="pending")
    error = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('paper_id', 'colleague_id', name='uq_share_paper_colleague'),
        Index('ix_share_user_id', 'user_id'),
        Index('ix_share_colleague_id', 'colleague_id'),
    )
    
    user = relationship("User", back_populates="shares")
    paper = relationship("Paper", back_populates="shares")
    colleague = relationship("Colleague", back_populates="shares")


# =============================================================================
# DeliveryPolicy Model
# =============================================================================

class DeliveryPolicy(Base):
    """User's delivery policy configuration."""
    __tablename__ = "delivery_policies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    policy_json = Column(JSON, nullable=False)
    
    colleague_share_enabled = Column(Boolean, default=True)
    colleague_share_min_score = Column(Float, default=0.5)
    digest_mode = Column(Boolean, default=False)
    
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_delivery_policy_user_id', 'user_id'),
    )
    
    user = relationship("User", back_populates="delivery_policies")


# =============================================================================
# ArxivCategory Model
# =============================================================================

class ArxivCategoryDB(Base):
    """
    ArXiv category taxonomy - synced from arXiv and cached in DB.
    
    This table stores the complete arXiv category taxonomy, which is
    periodically fetched from arxiv.org and cached for all users.
    """
    __tablename__ = "arxiv_categories"
    
    code = Column(String(50), primary_key=True)  # e.g., "cs.AI"
    name = Column(String(255), nullable=False)  # e.g., "Artificial Intelligence"
    group_name = Column(String(100))  # e.g., "Computer Science"
    description = Column(Text)
    source = Column(String(50), default="arxiv")  # "arxiv" or "fallback"
    
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_arxiv_category_group', 'group_name'),
    )


def arxiv_category_to_dict(cat: ArxivCategoryDB) -> Dict[str, Any]:
    """Convert ArxivCategoryDB model to dictionary."""
    return {
        "code": cat.code,
        "name": cat.name,
        "group": cat.group_name,
        "description": cat.description,
        "source": cat.source,
        "last_updated": cat.last_updated.isoformat() if cat.last_updated else None,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def paper_to_dict(paper: Paper) -> Dict[str, Any]:
    """Convert Paper model to dictionary."""
    return {
        "id": str(paper.id),
        "source": paper.source,
        "external_id": paper.external_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": paper.authors or [],
        "categories": paper.categories or [],
        "url": paper.url,
        "pdf_url": paper.pdf_url,
        "published_at": paper.published_at.isoformat() if paper.published_at else None,
        "created_at": paper.created_at.isoformat() if paper.created_at else None,
    }


def paper_view_to_dict(view: PaperView, include_paper: bool = False) -> Dict[str, Any]:
    """Convert PaperView model to dictionary."""
    result = {
        "id": str(view.id),
        "user_id": str(view.user_id),
        "paper_id": str(view.paper_id),
        "first_seen_at": view.first_seen_at.isoformat() if view.first_seen_at else None,
        "last_seen_at": view.last_seen_at.isoformat() if view.last_seen_at else None,
        "seen_count": view.seen_count,
        "decision": view.decision,
        "importance": view.importance,
        "relevance_score": view.relevance_score,
        "novelty_score": view.novelty_score,
        "heuristic_score": view.heuristic_score,
        "notes": view.notes,
        "tags": view.tags or [],
        "embedded_in_pinecone": view.embedded_in_pinecone,
    }
    if include_paper and view.paper:
        result["paper"] = paper_to_dict(view.paper)
    return result


def colleague_to_dict(colleague: Colleague) -> Dict[str, Any]:
    """Convert Colleague model to dictionary."""
    return {
        "id": str(colleague.id),
        "user_id": str(colleague.user_id),
        "name": colleague.name,
        "email": colleague.email,
        "affiliation": colleague.affiliation,
        "keywords": colleague.keywords or [],
        "categories": colleague.categories or [],
        "topics": colleague.topics or [],
        "sharing_preference": colleague.sharing_preference,
        "enabled": colleague.enabled,
        "notes": colleague.notes,
        "created_at": colleague.created_at.isoformat() if colleague.created_at else None,
        "updated_at": colleague.updated_at.isoformat() if colleague.updated_at else None,
    }


def run_to_dict(run: Run) -> Dict[str, Any]:
    """Convert Run model to dictionary."""
    return {
        "id": str(run.id),
        "run_id": run.run_id,
        "user_id": str(run.user_id),
        "user_prompt": run.user_prompt,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "ended_at": run.ended_at.isoformat() if run.ended_at else None,
        "status": run.status,
        "stop_reason": run.stop_reason,
        "error_message": run.error_message,
        "metrics": run.metrics or {},
        "papers_processed": run.papers_processed,
        "decisions_made": run.decisions_made,
        "artifacts_generated": run.artifacts_generated,
    }


def email_to_dict(email: Email) -> Dict[str, Any]:
    """Convert Email model to dictionary."""
    return {
        "id": str(email.id),
        "user_id": str(email.user_id),
        "paper_id": str(email.paper_id) if email.paper_id else None,
        "recipient_email": email.recipient_email,
        "subject": email.subject,
        "body_preview": email.body_preview,
        "body_text": email.body_text,
        "status": email.status,
        "error": email.error,
        "sent_at": email.sent_at.isoformat() if email.sent_at else None,
        "created_at": email.created_at.isoformat() if email.created_at else None,
    }


def calendar_event_to_dict(event: CalendarEvent) -> Dict[str, Any]:
    """Convert CalendarEvent model to dictionary."""
    return {
        "id": str(event.id),
        "user_id": str(event.user_id),
        "paper_id": str(event.paper_id) if event.paper_id else None,
        "title": event.title,
        "description": event.description,
        "start_time": event.start_time.isoformat() if event.start_time else None,
        "duration_minutes": event.duration_minutes,
        "ics_text": event.ics_text,
        "status": event.status,
        "error": event.error,
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }


def share_to_dict(share: Share, include_details: bool = False) -> Dict[str, Any]:
    """Convert Share model to dictionary."""
    result = {
        "id": str(share.id),
        "user_id": str(share.user_id),
        "paper_id": str(share.paper_id) if share.paper_id else None,
        "colleague_id": str(share.colleague_id) if share.colleague_id else None,
        "reason": share.reason,
        "match_score": share.match_score,
        "status": share.status,
        "error": share.error,
        "created_at": share.created_at.isoformat() if share.created_at else None,
    }
    if include_details:
        if share.colleague:
            result["colleague"] = colleague_to_dict(share.colleague)
        if share.paper:
            result["paper"] = paper_to_dict(share.paper)
    return result


def user_to_dict(user: User) -> Dict[str, Any]:
    """Convert User model to dictionary."""
    return {
        "id": str(user.id),
        "name": user.name,
        "email": user.email,
        "affiliation": user.affiliation,
        "research_topics": user.research_topics or [],
        "my_papers": user.my_papers or [],
        "preferred_venues": user.preferred_venues or [],
        "avoid_topics": user.avoid_topics or [],
        "time_budget_per_week_minutes": user.time_budget_per_week_minutes,
        "arxiv_categories_include": user.arxiv_categories_include or [],
        "arxiv_categories_exclude": user.arxiv_categories_exclude or [],
        "interests_include": user.interests_include or "",
        "interests_exclude": user.interests_exclude or "",
        "keywords_include": user.keywords_include or [],
        "keywords_exclude": user.keywords_exclude or [],
        "stop_policy": user.stop_policy or {},
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "updated_at": user.updated_at.isoformat() if user.updated_at else None,
    }


def policy_to_dict(policy: DeliveryPolicy) -> Dict[str, Any]:
    """Convert DeliveryPolicy model to dictionary."""
    return {
        "id": str(policy.id),
        "user_id": str(policy.user_id),
        "policy_json": policy.policy_json,
        "colleague_share_enabled": policy.colleague_share_enabled,
        "colleague_share_min_score": policy.colleague_share_min_score,
        "digest_mode": policy.digest_mode,
        "updated_at": policy.updated_at.isoformat() if policy.updated_at else None,
    }
