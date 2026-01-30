"""Initial schema for ResearchPulse database.

Revision ID: 001
Revises: 
Create Date: 2026-01-30

This migration creates the complete database schema for ResearchPulse:
- users: Researcher profiles
- papers: Paper metadata from arXiv/other sources
- paper_views: User's views/interactions with papers
- colleagues: Colleague researchers for sharing
- runs: Agent run records
- actions: Actions taken during runs
- emails: Email notification records
- calendar_events: Calendar event records
- shares: Paper sharing records
- delivery_policies: User delivery preferences
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("affiliation", sa.String(500)),
        sa.Column("research_topics", postgresql.JSON, server_default="[]"),
        sa.Column("my_papers", postgresql.JSON, server_default="[]"),
        sa.Column("preferred_venues", postgresql.JSON, server_default="[]"),
        sa.Column("avoid_topics", postgresql.JSON, server_default="[]"),
        sa.Column("time_budget_per_week_minutes", sa.Integer, server_default="120"),
        sa.Column("arxiv_categories_include", postgresql.JSON, server_default="[]"),
        sa.Column("arxiv_categories_exclude", postgresql.JSON, server_default="[]"),
        sa.Column("stop_policy", postgresql.JSON, server_default="{}"),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    
    # Create papers table
    op.create_table(
        "papers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("source", sa.String(50), nullable=False, server_default="arxiv"),
        sa.Column("external_id", sa.String(255), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("abstract", sa.Text),
        sa.Column("authors", postgresql.JSON, server_default="[]"),
        sa.Column("categories", postgresql.JSON, server_default="[]"),
        sa.Column("url", sa.String(500)),
        sa.Column("pdf_url", sa.String(500)),
        sa.Column("published_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("source", "external_id", name="uq_paper_source_external_id"),
    )
    op.create_index("ix_paper_source_external_id", "papers", ["source", "external_id"])
    
    # Create paper_views table
    op.create_table(
        "paper_views",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="CASCADE"), nullable=False),
        sa.Column("first_seen_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.Column("last_seen_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.Column("seen_count", sa.Integer, server_default="1"),
        sa.Column("decision", sa.String(50), server_default="logged"),
        sa.Column("importance", sa.String(20), server_default="low"),
        sa.Column("relevance_score", sa.Float),
        sa.Column("novelty_score", sa.Float),
        sa.Column("heuristic_score", sa.Float),
        sa.Column("notes", sa.Text),
        sa.Column("tags", postgresql.JSON, server_default="[]"),
        sa.Column("embedded_in_pinecone", sa.Boolean, server_default="false"),
        sa.UniqueConstraint("user_id", "paper_id", name="uq_paper_view_user_paper"),
    )
    op.create_index("ix_paper_view_user_id", "paper_views", ["user_id"])
    op.create_index("ix_paper_view_decision", "paper_views", ["decision"])
    op.create_index("ix_paper_view_importance", "paper_views", ["importance"])
    
    # Create colleagues table
    op.create_table(
        "colleagues",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("affiliation", sa.String(500)),
        sa.Column("keywords", postgresql.JSON, server_default="[]"),
        sa.Column("categories", postgresql.JSON, server_default="[]"),
        sa.Column("topics", postgresql.JSON, server_default="[]"),
        sa.Column("sharing_preference", sa.String(50), server_default="weekly"),
        sa.Column("enabled", sa.Boolean, server_default="true"),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_colleague_user_id", "colleagues", ["user_id"])
    
    # Create runs table
    op.create_table(
        "runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("run_id", sa.String(100), unique=True, nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_prompt", sa.Text),
        sa.Column("started_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.Column("ended_at", sa.DateTime),
        sa.Column("status", sa.String(50), server_default="running"),
        sa.Column("stop_reason", sa.String(255)),
        sa.Column("error_message", sa.Text),
        sa.Column("metrics", postgresql.JSON, server_default="{}"),
        sa.Column("report", postgresql.JSON),
        sa.Column("steps", postgresql.JSON, server_default="[]"),
        sa.Column("papers_processed", sa.Integer, server_default="0"),
        sa.Column("decisions_made", sa.Integer, server_default="0"),
        sa.Column("artifacts_generated", sa.Integer, server_default="0"),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_run_user_id", "runs", ["user_id"])
    op.create_index("ix_run_status", "runs", ["status"])
    op.create_index("ix_run_started_at", "runs", ["started_at"])
    
    # Create actions table
    op.create_table(
        "actions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="SET NULL")),
        sa.Column("action_type", sa.String(50), nullable=False),
        sa.Column("payload", postgresql.JSON, server_default="{}"),
        sa.Column("status", sa.String(50), server_default="completed"),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_action_run_id", "actions", ["run_id"])
    op.create_index("ix_action_action_type", "actions", ["action_type"])
    
    # Create emails table
    op.create_table(
        "emails",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="SET NULL")),
        sa.Column("recipient_email", sa.String(255), nullable=False),
        sa.Column("subject", sa.String(500), nullable=False),
        sa.Column("body_text", sa.Text),
        sa.Column("body_html", sa.Text),
        sa.Column("body_preview", sa.String(500)),
        sa.Column("provider_message_id", sa.String(255)),
        sa.Column("status", sa.String(50), server_default="queued"),
        sa.Column("error", sa.Text),
        sa.Column("sent_at", sa.DateTime),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("paper_id", "recipient_email", name="uq_email_paper_recipient"),
    )
    op.create_index("ix_email_user_id", "emails", ["user_id"])
    op.create_index("ix_email_status", "emails", ["status"])
    
    # Create calendar_events table
    op.create_table(
        "calendar_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="SET NULL")),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("start_time", sa.DateTime, nullable=False),
        sa.Column("duration_minutes", sa.Integer, server_default="30"),
        sa.Column("ics_text", sa.Text),
        sa.Column("provider_event_id", sa.String(255)),
        sa.Column("status", sa.String(50), server_default="created"),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("paper_id", "start_time", name="uq_calendar_paper_start"),
    )
    op.create_index("ix_calendar_user_id", "calendar_events", ["user_id"])
    op.create_index("ix_calendar_start_time", "calendar_events", ["start_time"])
    
    # Create shares table
    op.create_table(
        "shares",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("paper_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("papers.id", ondelete="SET NULL")),
        sa.Column("colleague_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("colleagues.id", ondelete="SET NULL")),
        sa.Column("reason", sa.Text),
        sa.Column("match_score", sa.Float),
        sa.Column("status", sa.String(50), server_default="pending"),
        sa.Column("error", sa.Text),
        sa.Column("created_at", sa.DateTime, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("paper_id", "colleague_id", name="uq_share_paper_colleague"),
    )
    op.create_index("ix_share_user_id", "shares", ["user_id"])
    op.create_index("ix_share_colleague_id", "shares", ["colleague_id"])
    
    # Create delivery_policies table
    op.create_table(
        "delivery_policies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("policy_json", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("colleague_share_enabled", sa.Boolean, server_default="true"),
        sa.Column("colleague_share_min_score", sa.Float, server_default="0.5"),
        sa.Column("digest_mode", sa.Boolean, server_default="false"),
        sa.Column("updated_at", sa.DateTime, server_default=sa.text("NOW()")),
    )
    op.create_index("ix_delivery_policy_user_id", "delivery_policies", ["user_id"])


def downgrade() -> None:
    # Drop all tables in reverse order
    op.drop_table("delivery_policies")
    op.drop_table("shares")
    op.drop_table("calendar_events")
    op.drop_table("emails")
    op.drop_table("actions")
    op.drop_table("runs")
    op.drop_table("colleagues")
    op.drop_table("paper_views")
    op.drop_table("papers")
    op.drop_table("users")
