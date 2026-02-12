"""Add paper summary fields to paper_views table.

Revision ID: add_paper_summary
Revises: add_execution_settings
Create Date: 2026-02-12

This migration adds:
- summary: AI-generated summary of the paper PDF
- summary_generated_at: Timestamp when the summary was generated
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_paper_summary'
down_revision = 'add_execution_settings'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add summary fields to paper_views table."""
    op.add_column('paper_views', sa.Column('summary', sa.Text(), nullable=True))
    op.add_column('paper_views', sa.Column('summary_generated_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Remove summary fields from paper_views table."""
    op.drop_column('paper_views', 'summary_generated_at')
    op.drop_column('paper_views', 'summary')
