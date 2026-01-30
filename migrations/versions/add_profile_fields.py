"""Add profile fields: interests_include, interests_exclude, keywords_include, keywords_exclude

Adds new fields to users table for storing research profile preferences.

Revision ID: add_profile_fields
Revises: add_arxiv_categories
Create Date: 2026-01-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'add_profile_fields'
down_revision = 'add_arxiv_categories'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new profile fields to users table
    op.add_column('users', sa.Column('interests_include', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('interests_exclude', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('keywords_include', postgresql.JSON(astext_type=sa.Text()), nullable=True, server_default='[]'))
    op.add_column('users', sa.Column('keywords_exclude', postgresql.JSON(astext_type=sa.Text()), nullable=True, server_default='[]'))


def downgrade() -> None:
    # Remove profile fields from users table
    op.drop_column('users', 'keywords_exclude')
    op.drop_column('users', 'keywords_include')
    op.drop_column('users', 'interests_exclude')
    op.drop_column('users', 'interests_include')
