"""Add triggered_by and paper_ids fields for bulk actions.

Revision ID: add_bulk_action_fields
Revises: 20260202_paper_feedback_history
Create Date: 2026-02-02

This migration adds:
- triggered_by: String field to track who triggered the action ('agent' or 'user')
- paper_ids: JSON field to store multiple paper IDs for bulk actions
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision = 'add_bulk_action_fields'
down_revision = '20260202_paper_feedback_history'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add triggered_by and paper_ids columns to emails and calendar_events tables."""
    
    # Add triggered_by to emails table
    op.add_column('emails', sa.Column('triggered_by', sa.String(20), nullable=True, default='agent'))
    op.add_column('emails', sa.Column('paper_ids', JSON, nullable=True))
    
    # Add triggered_by to calendar_events table  
    op.add_column('calendar_events', sa.Column('triggered_by', sa.String(20), nullable=True, default='agent'))
    op.add_column('calendar_events', sa.Column('paper_ids', JSON, nullable=True))
    
    # Add description column to calendar_events if it doesn't exist
    # (may already exist, so use batch mode for safety)
    try:
        op.add_column('calendar_events', sa.Column('reminder_minutes', sa.Integer, nullable=True, default=15))
    except Exception:
        pass  # Column may already exist
    
    # Set default values for existing rows
    op.execute("UPDATE emails SET triggered_by = 'agent' WHERE triggered_by IS NULL")
    op.execute("UPDATE calendar_events SET triggered_by = 'agent' WHERE triggered_by IS NULL")


def downgrade() -> None:
    """Remove triggered_by and paper_ids columns."""
    op.drop_column('emails', 'triggered_by')
    op.drop_column('emails', 'paper_ids')
    op.drop_column('calendar_events', 'triggered_by')
    op.drop_column('calendar_events', 'paper_ids')
    
    try:
        op.drop_column('calendar_events', 'reminder_minutes')
    except Exception:
        pass
