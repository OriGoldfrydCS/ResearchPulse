"""Add new fields to paper_views for Papers tab improvements.

Revision ID: add_paper_view_fields
Revises: add_prompt_templates
Create Date: 2026-02-02
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_paper_view_fields'
down_revision = 'add_prompt_templates'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add new fields to paper_views table for enhanced paper management."""
    # Add is_read field - tracks if user has read the paper
    op.add_column('paper_views', sa.Column('is_read', sa.Boolean(), nullable=True, default=False))
    
    # Add is_relevant field - nullable for tri-state (True/False/None = unknown)
    op.add_column('paper_views', sa.Column('is_relevant', sa.Boolean(), nullable=True))
    
    # Add is_deleted field - soft delete flag
    op.add_column('paper_views', sa.Column('is_deleted', sa.Boolean(), nullable=True, default=False))
    
    # Add is_starred field - for pinning/starring papers
    op.add_column('paper_views', sa.Column('is_starred', sa.Boolean(), nullable=True, default=False))
    
    # Add agent decision fields
    op.add_column('paper_views', sa.Column('agent_email_decision', sa.Boolean(), nullable=True))
    op.add_column('paper_views', sa.Column('agent_calendar_decision', sa.Boolean(), nullable=True))
    op.add_column('paper_views', sa.Column('agent_decision_notes', sa.Text(), nullable=True))
    
    # Add index for soft delete queries
    op.create_index('ix_paper_view_is_deleted', 'paper_views', ['is_deleted'])
    op.create_index('ix_paper_view_is_read', 'paper_views', ['is_read'])
    op.create_index('ix_paper_view_is_starred', 'paper_views', ['is_starred'])


def downgrade() -> None:
    """Remove added fields from paper_views table."""
    op.drop_index('ix_paper_view_is_starred', 'paper_views')
    op.drop_index('ix_paper_view_is_read', 'paper_views')
    op.drop_index('ix_paper_view_is_deleted', 'paper_views')
    op.drop_column('paper_views', 'agent_decision_notes')
    op.drop_column('paper_views', 'agent_calendar_decision')
    op.drop_column('paper_views', 'agent_email_decision')
    op.drop_column('paper_views', 'is_starred')
    op.drop_column('paper_views', 'is_deleted')
    op.drop_column('paper_views', 'is_relevant')
    op.drop_column('paper_views', 'is_read')
