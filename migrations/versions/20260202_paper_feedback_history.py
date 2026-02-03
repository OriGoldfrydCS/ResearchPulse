"""Add paper feedback system with history tracking

Revision ID: 20260202_paper_feedback_history
Revises: add_paper_view_fields
Create Date: 2026-02-02 12:00:00.000000

This migration adds:
- Extended feedback fields to paper_views table
- PaperFeedbackHistory table for audit trail
- Agent-consumable feedback signals
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '20260202_paper_feedback_history'
down_revision: Union[str, None] = 'add_paper_view_fields'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add paper feedback system tables and columns."""
    
    # Add extended feedback fields to paper_views table
    op.add_column('paper_views', sa.Column('relevance_state', sa.String(20), nullable=True))
    op.add_column('paper_views', sa.Column('feedback_reason', sa.Text(), nullable=True))
    op.add_column('paper_views', sa.Column('feedback_timestamp', sa.DateTime(), nullable=True))
    op.add_column('paper_views', sa.Column('read_at', sa.DateTime(), nullable=True))
    op.add_column('paper_views', sa.Column('starred_at', sa.DateTime(), nullable=True))
    
    # Create paper_feedback_history table for audit trail
    op.create_table(
        'paper_feedback_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('paper_view_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('paper_views.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('paper_id', postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('action_type', sa.String(50), nullable=False),
        sa.Column('old_value', sa.String(50), nullable=True),
        sa.Column('new_value', sa.String(50), nullable=True),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    
    # Create indexes for efficient querying
    op.create_index('ix_feedback_history_paper_view_id', 'paper_feedback_history', ['paper_view_id'])
    op.create_index('ix_feedback_history_user_id', 'paper_feedback_history', ['user_id'])
    op.create_index('ix_feedback_history_paper_id', 'paper_feedback_history', ['paper_id'])
    op.create_index('ix_feedback_history_action_type', 'paper_feedback_history', ['action_type'])
    op.create_index('ix_feedback_history_created_at', 'paper_feedback_history', ['created_at'])
    
    # Create index for relevance_state filtering
    op.create_index('ix_paper_views_relevance_state', 'paper_views', ['relevance_state'])


def downgrade() -> None:
    """Remove paper feedback system tables and columns."""
    
    # Drop indexes
    op.drop_index('ix_paper_views_relevance_state', table_name='paper_views')
    op.drop_index('ix_feedback_history_created_at', table_name='paper_feedback_history')
    op.drop_index('ix_feedback_history_action_type', table_name='paper_feedback_history')
    op.drop_index('ix_feedback_history_paper_id', table_name='paper_feedback_history')
    op.drop_index('ix_feedback_history_user_id', table_name='paper_feedback_history')
    op.drop_index('ix_feedback_history_paper_view_id', table_name='paper_feedback_history')
    
    # Drop table
    op.drop_table('paper_feedback_history')
    
    # Drop columns from paper_views
    op.drop_column('paper_views', 'starred_at')
    op.drop_column('paper_views', 'read_at')
    op.drop_column('paper_views', 'feedback_timestamp')
    op.drop_column('paper_views', 'feedback_reason')
    op.drop_column('paper_views', 'relevance_state')
