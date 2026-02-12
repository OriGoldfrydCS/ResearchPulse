"""Add feature flag settings columns to user_settings.

This migration adds columns to store feature flag configuration per user:
- LLM Novelty: enabled, model
- Audit Log: enabled
- Profile Evolution: enabled, cooldown_hours
- Live Document: enabled, max_papers

Revision ID: add_feature_flag_settings
Revises: add_autonomous_components
Create Date: 2026-02-12

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_feature_flag_settings'
down_revision = 'add_autonomous_components'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add feature flag columns to user_settings table
    op.add_column('user_settings', sa.Column('feature_llm_novelty_enabled', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('user_settings', sa.Column('feature_llm_novelty_model', sa.String(50), nullable=True, server_default='gpt-4o-mini'))
    op.add_column('user_settings', sa.Column('feature_audit_log_enabled', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('user_settings', sa.Column('feature_profile_evolution_enabled', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('user_settings', sa.Column('feature_profile_evolution_cooldown_hours', sa.Integer(), nullable=True, server_default='24'))
    op.add_column('user_settings', sa.Column('feature_live_document_enabled', sa.Boolean(), nullable=True, server_default='false'))
    op.add_column('user_settings', sa.Column('feature_live_document_max_papers', sa.Integer(), nullable=True, server_default='10'))


def downgrade() -> None:
    # Remove feature flag columns
    op.drop_column('user_settings', 'feature_live_document_max_papers')
    op.drop_column('user_settings', 'feature_live_document_enabled')
    op.drop_column('user_settings', 'feature_profile_evolution_cooldown_hours')
    op.drop_column('user_settings', 'feature_profile_evolution_enabled')
    op.drop_column('user_settings', 'feature_audit_log_enabled')
    op.drop_column('user_settings', 'feature_llm_novelty_model')
    op.drop_column('user_settings', 'feature_llm_novelty_enabled')
