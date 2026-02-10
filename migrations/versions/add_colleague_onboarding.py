"""Add colleague onboarding and derived categories fields.

Revision ID: add_colleague_onboarding
Revises: add_colleague_fields
Create Date: 2026-02-10

This migration adds:
- onboarding_status: Tracks whether colleague has completed onboarding
- onboarding_thread_id: Thread ID for ongoing onboarding conversation
- join_verified: Whether join code has been verified in this thread
- derived_arxiv_categories: Auto-derived arXiv categories from interests
- interests: Free-text research interests input (separate from research_interests for backward compat)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers
revision = 'add_colleague_onboarding'
down_revision = '20260210_inbox_settings'
branch_labels = None
depends_on = None


def upgrade():
    # Add onboarding_status - tracks colleague onboarding state
    # Values: 'complete', 'pending', 'needs_interests', 'needs_name'
    op.add_column(
        'colleagues',
        sa.Column('onboarding_status', sa.String(50), nullable=True, server_default='complete')
    )
    
    # Add onboarding_thread_id - tracks the email thread for ongoing onboarding
    op.add_column(
        'colleagues',
        sa.Column('onboarding_thread_id', sa.String(500), nullable=True)
    )
    
    # Add join_verified - whether join code was verified for this colleague
    op.add_column(
        'colleagues',
        sa.Column('join_verified', sa.Boolean, nullable=True, server_default='false')
    )
    
    # Add derived_arxiv_categories - auto-derived from interests
    # JSON format: {"primary": ["cs.LG", "cs.AI"], "secondary": ["stat.ML"]}
    op.add_column(
        'colleagues',
        sa.Column('derived_arxiv_categories', postgresql.JSON, nullable=True)
    )
    
    # Add interests - separate from research_interests for cleaner data model
    # This stores the raw user-provided interests text
    op.add_column(
        'colleagues',
        sa.Column('interests', sa.Text, nullable=True)
    )
    
    # Add index on onboarding_status for efficient queries
    op.create_index(
        'ix_colleague_onboarding_status',
        'colleagues',
        ['onboarding_status']
    )


def downgrade():
    op.drop_index('ix_colleague_onboarding_status', 'colleagues')
    op.drop_column('colleagues', 'interests')
    op.drop_column('colleagues', 'derived_arxiv_categories')
    op.drop_column('colleagues', 'join_verified')
    op.drop_column('colleagues', 'onboarding_thread_id')
    op.drop_column('colleagues', 'onboarding_status')
