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
    # Keywords for filtering (deprecated - kept for DB compatibility)
    keywords_include = Column(JSON, default=list)  # Keywords to include (deprecated)
    keywords_exclude = Column(JSON, default=list)  # Keywords to exclude (deprecated)
    # Time preferences
    preferred_time_period = Column(String(100), default="last two weeks")  # Preferred search time period
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
    prompt_requests = relationship("PromptRequest", back_populates="user", cascade="all, delete-orphan")
    saved_prompts = relationship("SavedPrompt", back_populates="user", cascade="all, delete-orphan")


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
    
    # Paper status fields
    is_read = Column(Boolean, default=False)  # Has user read this paper?
    is_relevant = Column(Boolean, nullable=True)  # Nullable for tri-state: True/False/None
    is_deleted = Column(Boolean, default=False)  # Soft delete flag
    is_starred = Column(Boolean, default=False)  # Pinned/starred paper
    
    # Extended feedback fields
    relevance_state = Column(String(20), nullable=True)  # 'relevant', 'not_relevant', or null (unknown)
    feedback_reason = Column(Text, nullable=True)  # Optional reason for feedback
    feedback_timestamp = Column(DateTime, nullable=True)  # When last feedback was given
    read_at = Column(DateTime, nullable=True)  # When paper was marked as read
    starred_at = Column(DateTime, nullable=True)  # When paper was starred
    
    # Agent decision tracking
    agent_email_decision = Column(Boolean, nullable=True)  # Did agent recommend email?
    agent_calendar_decision = Column(Boolean, nullable=True)  # Did agent recommend calendar?
    agent_decision_notes = Column(Text)  # Agent's reasoning summary
    
    __table_args__ = (
        UniqueConstraint('user_id', 'paper_id', name='uq_paper_view_user_paper'),
        Index('ix_paper_view_user_id', 'user_id'),
        Index('ix_paper_view_decision', 'decision'),
        Index('ix_paper_view_importance', 'importance'),
    )
    
    user = relationship("User", back_populates="paper_views")
    paper = relationship("Paper", back_populates="views")
    feedback_history = relationship("PaperFeedbackHistory", back_populates="paper_view", cascade="all, delete-orphan")


# =============================================================================
# PaperFeedbackHistory Model
# =============================================================================

class PaperFeedbackHistory(Base):
    """
    Audit trail for paper feedback changes.
    
    Records every change to relevance, star, read status for:
    - Audit/compliance purposes
    - Agent learning from user preferences
    - UI showing feedback history
    """
    __tablename__ = "paper_feedback_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_view_id = Column(UUID(as_uuid=True), ForeignKey("paper_views.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    
    action_type = Column(String(50), nullable=False)  # 'relevance_change', 'star_toggle', 'read_toggle'
    old_value = Column(String(50), nullable=True)
    new_value = Column(String(50), nullable=True)
    note = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_feedback_history_paper_view_id', 'paper_view_id'),
        Index('ix_feedback_history_user_id', 'user_id'),
        Index('ix_feedback_history_paper_id', 'paper_id'),
        Index('ix_feedback_history_action_type', 'action_type'),
        Index('ix_feedback_history_created_at', 'created_at'),
    )
    
    paper_view = relationship("PaperView", back_populates="feedback_history")


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
    
    # Attribution: who triggered this email - 'agent' or 'user'
    triggered_by = Column(String(20), nullable=True, default='agent')
    # For bulk actions: list of paper IDs included in this email
    paper_ids = Column(JSON, nullable=True)
    
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
    reminder_minutes = Column(Integer, default=15)  # Reminder notification before event
    
    ics_text = Column(Text)
    ics_uid = Column(String(255))  # Unique UID for the ICS event for updates
    
    provider_event_id = Column(String(255))
    
    status = Column(String(50), default="created")
    error = Column(Text)
    
    # Attribution: who triggered this reminder - 'agent' or 'user'
    triggered_by = Column(String(20), nullable=True, default='agent')
    # For bulk actions: list of paper IDs included in this reminder
    paper_ids = Column(JSON, nullable=True)
    
    # Rescheduling support
    original_event_id = Column(UUID(as_uuid=True), ForeignKey("calendar_events.id", ondelete="SET NULL"), nullable=True)
    reschedule_note = Column(Text, nullable=True)  # e.g., "Rescheduled after user email reply"
    sequence_number = Column(Integer, default=0)  # ICalendar SEQUENCE for updates
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('paper_id', 'start_time', name='uq_calendar_paper_start'),
        Index('ix_calendar_user_id', 'user_id'),
        Index('ix_calendar_start_time', 'start_time'),
        Index('ix_calendar_ics_uid', 'ics_uid'),
    )
    
    user = relationship("User", back_populates="calendar_events")
    paper = relationship("Paper", back_populates="calendar_events")
    invite_emails = relationship("CalendarInviteEmail", back_populates="calendar_event", cascade="all, delete-orphan")
    original_event = relationship("CalendarEvent", remote_side=[id], backref="rescheduled_events")


# =============================================================================
# CalendarInviteEmail Model
# =============================================================================

class CalendarInviteEmail(Base):
    """
    Tracks calendar invitation emails sent for calendar events.
    
    Links emails to calendar events for tracking replies and rescheduling.
    Stores message_id and thread_id for email reply threading.
    """
    __tablename__ = "calendar_invite_emails"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    calendar_event_id = Column(UUID(as_uuid=True), ForeignKey("calendar_events.id", ondelete="CASCADE"), nullable=False)
    email_id = Column(UUID(as_uuid=True), ForeignKey("emails.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Email threading identifiers for tracking replies
    message_id = Column(String(255), nullable=False, unique=True)  # RFC 2822 Message-ID
    thread_id = Column(String(255))  # Thread identifier (Gmail style or References header)
    in_reply_to = Column(String(255))  # In-Reply-To header for threading
    
    recipient_email = Column(String(255), nullable=False)
    subject = Column(String(500), nullable=False)
    
    # ICS attachment details
    ics_uid = Column(String(255), nullable=False)  # Links to CalendarEvent.ics_uid
    ics_sequence = Column(Integer, default=0)  # Matches CalendarEvent.sequence_number
    ics_method = Column(String(20), default="REQUEST")  # REQUEST, CANCEL, REPLY
    
    # Attribution: who triggered this invite - 'agent' or 'user'
    triggered_by = Column(String(20), nullable=True, default='agent')
    
    # Status tracking
    status = Column(String(50), default="sent")  # sent, delivered, failed, superseded
    is_latest = Column(Boolean, default=True)  # False if superseded by a reschedule
    
    sent_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_calendar_invite_email_calendar_event_id', 'calendar_event_id'),
        Index('ix_calendar_invite_email_message_id', 'message_id'),
        Index('ix_calendar_invite_email_thread_id', 'thread_id'),
        Index('ix_calendar_invite_email_user_id', 'user_id'),
        Index('ix_calendar_invite_email_ics_uid', 'ics_uid'),
    )
    
    calendar_event = relationship("CalendarEvent", back_populates="invite_emails")
    replies = relationship("InboundEmailReply", back_populates="original_invite", cascade="all, delete-orphan")


# =============================================================================
# InboundEmailReply Model
# =============================================================================

class InboundEmailReply(Base):
    """
    Stores inbound email replies to calendar invitations.
    
    Used for agent interpretation of user replies (e.g., reschedule requests).
    Each reply is linked back to the original CalendarInviteEmail.
    """
    __tablename__ = "inbound_email_replies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    original_invite_id = Column(UUID(as_uuid=True), ForeignKey("calendar_invite_emails.id", ondelete="CASCADE"), nullable=False)
    
    # Email headers for threading
    message_id = Column(String(255), nullable=False, unique=True)  # Reply's Message-ID
    in_reply_to = Column(String(255))  # Should match original invite's message_id
    references = Column(Text)  # Full References header chain
    
    from_email = Column(String(255), nullable=False)
    subject = Column(String(500))
    body_text = Column(Text)
    body_html = Column(Text)
    
    # Agent interpretation results
    intent = Column(String(50))  # 'reschedule', 'accept', 'decline', 'unknown'
    extracted_datetime = Column(DateTime, nullable=True)  # Parsed new datetime if rescheduling
    extracted_datetime_text = Column(String(500))  # Raw text that was parsed
    confidence_score = Column(Float)  # Agent's confidence in interpretation
    
    # Processing status
    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime, nullable=True)
    processing_result = Column(JSON, nullable=True)  # Full processing result
    processing_error = Column(Text, nullable=True)
    
    # Action taken based on reply
    action_taken = Column(String(50))  # 'rescheduled', 'no_action', 'error'
    new_event_id = Column(UUID(as_uuid=True), ForeignKey("calendar_events.id", ondelete="SET NULL"), nullable=True)
    
    received_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_inbound_reply_original_invite_id', 'original_invite_id'),
        Index('ix_inbound_reply_message_id', 'message_id'),
        Index('ix_inbound_reply_user_id', 'user_id'),
        Index('ix_inbound_reply_processed', 'processed'),
        Index('ix_inbound_reply_intent', 'intent'),
    )
    
    original_invite = relationship("CalendarInviteEmail", back_populates="replies")


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
# PromptRequest Model - Stores parsed research prompts
# =============================================================================

class PromptRequest(Base):
    """
    Stores parsed research prompts for compliance tracking and audit.
    
    Each user query is parsed, classified by template, and stored with
    all extracted parameters. This ensures:
    - Audit trail of all research requests
    - Template usage analytics
    - Output enforcement compliance tracking
    """
    __tablename__ = "prompt_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    run_id = Column(String(100), nullable=True)  # Links to Run if part of an agent run
    
    # Raw prompt from user
    raw_prompt = Column(Text, nullable=False)
    
    # Parsed template classification
    template = Column(String(50), nullable=False)  # e.g., "TOP_K_PAPERS", "TOPIC_TIME"
    
    # Extracted parameters
    topic = Column(Text)
    venue = Column(String(255))
    time_period = Column(String(100))
    time_days = Column(Integer)
    requested_count = Column(Integer)  # User-requested K (e.g., "top 5")
    output_count = Column(Integer)  # Actual output count (enforced)
    retrieval_count = Column(Integer)  # Internal retrieval limit (e.g., 30)
    method_or_approach = Column(Text)
    application_domain = Column(Text)
    
    # Request type flags
    is_survey_request = Column(Boolean, default=False)
    is_trends_request = Column(Boolean, default=False)
    is_structured_output = Column(Boolean, default=False)
    
    # Compliance tracking
    output_enforced = Column(Boolean, default=False)  # Was output truncated?
    output_insufficient = Column(Boolean, default=False)  # Were there fewer papers than requested?
    compliance_status = Column(String(50), default="pending")  # "compliant", "violation", "pending"
    compliance_message = Column(Text)
    
    # Actual results (for audit)
    papers_retrieved = Column(Integer)  # How many were retrieved internally
    papers_returned = Column(Integer)  # How many were returned to user
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_prompt_request_user_id', 'user_id'),
        Index('ix_prompt_request_run_id', 'run_id'),
        Index('ix_prompt_request_template', 'template'),
        Index('ix_prompt_request_created_at', 'created_at'),
    )
    
    user = relationship("User", back_populates="prompt_requests")


# =============================================================================
# SavedPrompt Model - User-saved prompt templates for quick reuse
# =============================================================================

class SavedPrompt(Base):
    """
    User-saved prompt templates for the Quick Prompt Builder.
    
    Allows users to save frequently used prompt configurations
    for quick access instead of building prompts from scratch.
    """
    __tablename__ = "saved_prompts"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)
    
    # Prompt details
    name = Column(String(255), nullable=False)  # User-given name for the prompt
    prompt_text = Column(Text, nullable=False)  # The generated prompt text
    
    # Builder fields (for recreating the builder state)
    template_type = Column(String(50))  # topic_time, top_k, survey, etc.
    areas = Column(JSON, default=list)  # Selected research areas
    topics = Column(JSON, default=list)  # Focus topics
    time_period = Column(String(50))  # last_week, last_month, etc.
    paper_count = Column(Integer)  # Requested number of papers
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_saved_prompt_user_id', 'user_id'),
        Index('ix_saved_prompt_name', 'name'),
    )
    
    user = relationship("User", back_populates="saved_prompts")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "name": self.name,
            "prompt_text": self.prompt_text,
            "template_type": self.template_type,
            "areas": self.areas or [],
            "topics": self.topics or [],
            "time_period": self.time_period,
            "paper_count": self.paper_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


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


# =============================================================================
# TopicCategoryMapping Model - LLM-inferred topic to category mappings
# =============================================================================

class TopicCategoryMapping(Base):
    """
    Stores topic-to-arXiv-category mappings, either built-in or LLM-inferred.
    
    When a user query doesn't match any known category keywords, the system
    uses an LLM to infer the best categories and stores the mapping here
    for future use. This allows the system to learn and improve over time.
    """
    __tablename__ = "topic_category_mappings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic = Column(String(500), nullable=False)  # Original topic text
    topic_normalized = Column(String(500), nullable=False, unique=True)  # Lowercase, trimmed
    categories = Column(JSON, nullable=False)  # List of category codes, e.g., ["cs.NI", "cs.PF"]
    confidence = Column(Float, nullable=True)  # LLM confidence score (0-1)
    source = Column(String(50), nullable=False)  # 'builtin', 'llm', 'user'
    usage_count = Column(Integer, default=0)  # How many times this mapping was used
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_topic_category_mapping_topic_normalized', 'topic_normalized'),
        Index('ix_topic_category_mapping_source', 'source'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": str(self.id),
            "topic": self.topic,
            "topic_normalized": self.topic_normalized,
            "categories": self.categories or [],
            "confidence": self.confidence,
            "source": self.source,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# =============================================================================
# PromptTemplate Model - Global prompt templates for quick reuse
# =============================================================================

class PromptTemplate(Base):
    """
    Prompt templates for quick prompt builder.
    
    Supports both built-in templates (shipped with the app) and
    custom user-created templates.
    """
    __tablename__ = "prompt_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True)
    text = Column(Text, nullable=False)
    is_builtin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_prompt_template_name', 'name'),
        Index('ix_prompt_template_is_builtin', 'is_builtin'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": str(self.id),
            "name": self.name,
            "text": self.text,
            "is_builtin": self.is_builtin,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


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
        # Paper status fields
        "is_read": view.is_read if hasattr(view, 'is_read') else False,
        "is_relevant": view.is_relevant if hasattr(view, 'is_relevant') else None,
        "is_deleted": view.is_deleted if hasattr(view, 'is_deleted') else False,
        "is_starred": view.is_starred if hasattr(view, 'is_starred') else False,
        # Extended feedback fields
        "relevance_state": view.relevance_state if hasattr(view, 'relevance_state') else None,
        "feedback_reason": view.feedback_reason if hasattr(view, 'feedback_reason') else None,
        "feedback_timestamp": view.feedback_timestamp.isoformat() if hasattr(view, 'feedback_timestamp') and view.feedback_timestamp else None,
        "read_at": view.read_at.isoformat() if hasattr(view, 'read_at') and view.read_at else None,
        "starred_at": view.starred_at.isoformat() if hasattr(view, 'starred_at') and view.starred_at else None,
        # Agent decision tracking
        "agent_email_decision": view.agent_email_decision if hasattr(view, 'agent_email_decision') else None,
        "agent_calendar_decision": view.agent_calendar_decision if hasattr(view, 'agent_calendar_decision') else None,
        "agent_decision_notes": view.agent_decision_notes if hasattr(view, 'agent_decision_notes') else None,
    }
    if include_paper and view.paper:
        result["paper"] = paper_to_dict(view.paper)
    return result


def feedback_history_to_dict(history: PaperFeedbackHistory) -> Dict[str, Any]:
    """Convert PaperFeedbackHistory model to dictionary."""
    return {
        "id": str(history.id),
        "paper_view_id": str(history.paper_view_id),
        "user_id": str(history.user_id),
        "paper_id": str(history.paper_id),
        "action_type": history.action_type,
        "old_value": history.old_value,
        "new_value": history.new_value,
        "note": history.note,
        "created_at": history.created_at.isoformat() if history.created_at else None,
    }


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
        # Attribution field
        "triggered_by": email.triggered_by or 'agent',
        "paper_ids": email.paper_ids or [],
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
        "reminder_minutes": event.reminder_minutes or 15,
        "ics_text": event.ics_text,
        "ics_uid": event.ics_uid,
        "status": event.status,
        "error": event.error,
        "created_at": event.created_at.isoformat() if event.created_at else None,
        "updated_at": event.updated_at.isoformat() if event.updated_at else None,
        # Attribution field
        "triggered_by": event.triggered_by or 'agent',
        "paper_ids": event.paper_ids or [],
        # Rescheduling fields
        "original_event_id": str(event.original_event_id) if event.original_event_id else None,
        "reschedule_note": event.reschedule_note,
        "sequence_number": event.sequence_number or 0,
    }


def calendar_invite_email_to_dict(invite: CalendarInviteEmail) -> Dict[str, Any]:
    """Convert CalendarInviteEmail model to dictionary."""
    return {
        "id": str(invite.id),
        "calendar_event_id": str(invite.calendar_event_id),
        "email_id": str(invite.email_id) if invite.email_id else None,
        "user_id": str(invite.user_id),
        "message_id": invite.message_id,
        "thread_id": invite.thread_id,
        "in_reply_to": invite.in_reply_to,
        "recipient_email": invite.recipient_email,
        "subject": invite.subject,
        "ics_uid": invite.ics_uid,
        "ics_sequence": invite.ics_sequence or 0,
        "ics_method": invite.ics_method or "REQUEST",
        "triggered_by": invite.triggered_by or 'agent',
        "status": invite.status,
        "is_latest": invite.is_latest,
        "sent_at": invite.sent_at.isoformat() if invite.sent_at else None,
        "created_at": invite.created_at.isoformat() if invite.created_at else None,
    }


def inbound_email_reply_to_dict(reply: InboundEmailReply) -> Dict[str, Any]:
    """Convert InboundEmailReply model to dictionary."""
    return {
        "id": str(reply.id),
        "user_id": str(reply.user_id),
        "original_invite_id": str(reply.original_invite_id),
        "message_id": reply.message_id,
        "in_reply_to": reply.in_reply_to,
        "from_email": reply.from_email,
        "subject": reply.subject,
        "body_text": reply.body_text,
        "intent": reply.intent,
        "extracted_datetime": reply.extracted_datetime.isoformat() if reply.extracted_datetime else None,
        "extracted_datetime_text": reply.extracted_datetime_text,
        "confidence_score": reply.confidence_score,
        "processed": reply.processed,
        "processed_at": reply.processed_at.isoformat() if reply.processed_at else None,
        "processing_result": reply.processing_result,
        "processing_error": reply.processing_error,
        "action_taken": reply.action_taken,
        "new_event_id": str(reply.new_event_id) if reply.new_event_id else None,
        "received_at": reply.received_at.isoformat() if reply.received_at else None,
        "created_at": reply.created_at.isoformat() if reply.created_at else None,
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
        "preferred_time_period": user.preferred_time_period or "last two weeks",
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


def prompt_request_to_dict(prompt: PromptRequest) -> Dict[str, Any]:
    """Convert PromptRequest model to dictionary."""
    return {
        "id": str(prompt.id),
        "user_id": str(prompt.user_id),
        "run_id": prompt.run_id,
        "raw_prompt": prompt.raw_prompt,
        "template": prompt.template,
        "topic": prompt.topic,
        "venue": prompt.venue,
        "time_period": prompt.time_period,
        "time_days": prompt.time_days,
        "requested_count": prompt.requested_count,
        "output_count": prompt.output_count,
        "retrieval_count": prompt.retrieval_count,
        "method_or_approach": prompt.method_or_approach,
        "application_domain": prompt.application_domain,
        "is_survey_request": prompt.is_survey_request,
        "is_trends_request": prompt.is_trends_request,
        "is_structured_output": prompt.is_structured_output,
        "output_enforced": prompt.output_enforced,
        "output_insufficient": prompt.output_insufficient,
        "compliance_status": prompt.compliance_status,
        "compliance_message": prompt.compliance_message,
        "papers_retrieved": prompt.papers_retrieved,
        "papers_returned": prompt.papers_returned,
        "created_at": prompt.created_at.isoformat() if prompt.created_at else None,
    }
