"""Add saved_prompts table for Quick Prompt Builder

Revision ID: add_saved_prompts
Revises: add_prompt_requests
Create Date: 2026-02-02 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = 'add_saved_prompts'
down_revision: Union[str, None] = 'add_prompt_requests'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create saved_prompts table for storing user prompt templates."""
    op.create_table(
        'saved_prompts',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('prompt_text', sa.Text(), nullable=False),
        sa.Column('template_type', sa.String(50), nullable=True),
        sa.Column('areas', sa.JSON(), server_default='[]'),
        sa.Column('topics', sa.JSON(), server_default='[]'),
        sa.Column('time_period', sa.String(50), nullable=True),
        sa.Column('paper_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('NOW()')),
    )
    
    # Create indexes for efficient queries
    op.create_index('ix_saved_prompt_user_id', 'saved_prompts', ['user_id'])
    op.create_index('ix_saved_prompt_name', 'saved_prompts', ['name'])


def downgrade() -> None:
    """Drop saved_prompts table."""
    op.drop_index('ix_saved_prompt_name', table_name='saved_prompts')
    op.drop_index('ix_saved_prompt_user_id', table_name='saved_prompts')
    op.drop_table('saved_prompts')
