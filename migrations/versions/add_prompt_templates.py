"""Add prompt_templates table

Revision ID: add_prompt_templates
Revises: add_saved_prompts
Create Date: 2026-02-02
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_prompt_templates'
down_revision = 'add_saved_prompts'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('prompt_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('is_builtin', sa.Boolean(), nullable=True, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_prompt_template_is_builtin', 'prompt_templates', ['is_builtin'], unique=False)
    op.create_index('ix_prompt_template_name', 'prompt_templates', ['name'], unique=False)


def downgrade():
    op.drop_index('ix_prompt_template_name', table_name='prompt_templates')
    op.drop_index('ix_prompt_template_is_builtin', table_name='prompt_templates')
    op.drop_table('prompt_templates')
