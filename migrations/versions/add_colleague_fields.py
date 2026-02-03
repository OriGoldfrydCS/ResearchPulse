"""Add colleague fields for manual vs email addition, research interests, and auto-email preference.

Revision ID: add_colleague_fields
Revises:
Create Date: 2025-02-03
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_colleague_fields'
down_revision = None  # Will be determined by Alembic
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add new columns to colleagues table."""
    # Add research_interests column for storing raw research interests text
    op.add_column('colleagues', sa.Column('research_interests', sa.Text(), nullable=True))
    
    # Add added_by column to track whether colleague was added manually or via email
    # Values: 'manual' (added by owner) or 'email' (self-added via email to ResearchPulse)
    op.add_column('colleagues', sa.Column('added_by', sa.String(50), server_default='manual', nullable=True))
    
    # Add auto_send_emails column to control whether to automatically send research updates
    op.add_column('colleagues', sa.Column('auto_send_emails', sa.Boolean(), server_default='true', nullable=True))


def downgrade() -> None:
    """Remove the added columns."""
    op.drop_column('colleagues', 'auto_send_emails')
    op.drop_column('colleagues', 'added_by')
    op.drop_column('colleagues', 'research_interests')
