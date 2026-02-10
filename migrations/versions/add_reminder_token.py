"""Add reminder_token column to calendar_invite_emails for robust matching.

Revision ID: add_reminder_token
Revises: colleague_email_log
Create Date: 2026-02-10

This migration adds:
1. reminder_token column - unique token embedded in email body for matching replies
2. Index on reminder_token for fast lookups

The token allows reliable reschedule matching even when:
- Gmail modifies Message-ID formatting
- In-Reply-To headers are missing or malformed
- threadId is unavailable
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_reminder_token'
down_revision: Union[str, None] = 'colleague_email_log'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add reminder_token column to calendar_invite_emails."""
    # Add reminder_token column (nullable for backward compatibility with existing rows)
    op.add_column(
        'calendar_invite_emails',
        sa.Column('reminder_token', sa.String(64), nullable=True, unique=True)
    )
    
    # Create index for fast token lookups
    op.create_index(
        'ix_calendar_invite_email_reminder_token',
        'calendar_invite_emails',
        ['reminder_token'],
        unique=True
    )


def downgrade() -> None:
    """Remove reminder_token column from calendar_invite_emails."""
    op.drop_index('ix_calendar_invite_email_reminder_token', table_name='calendar_invite_emails')
    op.drop_column('calendar_invite_emails', 'reminder_token')
