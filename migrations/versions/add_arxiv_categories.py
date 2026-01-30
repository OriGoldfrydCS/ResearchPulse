"""Add arxiv_categories table for dynamic category management

Revision ID: add_arxiv_categories
Revises: 
Create Date: 2026-01-30

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_arxiv_categories'
down_revision = None  # Update this to point to your latest migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'arxiv_categories',
        sa.Column('code', sa.String(50), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('group_name', sa.String(100), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source', sa.String(50), server_default='arxiv'),
        sa.Column('last_updated', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
    )
    # Add index on group_name for efficient group queries
    op.create_index('ix_arxiv_categories_group_name', 'arxiv_categories', ['group_name'])


def downgrade() -> None:
    op.drop_index('ix_arxiv_categories_group_name', 'arxiv_categories')
    op.drop_table('arxiv_categories')
