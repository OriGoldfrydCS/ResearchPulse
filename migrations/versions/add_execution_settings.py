"""Add execution settings fields to user_settings table

Revision ID: add_execution_settings
Revises: add_reminder_token
Create Date: 2026-02-10

New columns on user_settings:
- retrieval_max_results: papers per run (default 7)
- execution_mode: 'manual' | 'scheduled' (default 'manual')
- scheduled_frequency: 'daily' | 'every_x_days' | 'weekly' | 'monthly'
- scheduled_every_x_days: integer for every_x_days mode
- last_run_at: timestamp of last execution
- next_run_at: timestamp of next scheduled execution
- colleague_join_code_encrypted: AES-encrypted join code (replaces bcrypt hash)
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_execution_settings'
down_revision = 'add_reminder_token'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Retrieval settings
    op.add_column('user_settings',
        sa.Column('retrieval_max_results', sa.Integer(), nullable=True, server_default='7'))

    # Execution mode settings
    op.add_column('user_settings',
        sa.Column('execution_mode', sa.String(20), nullable=True, server_default='manual'))
    op.add_column('user_settings',
        sa.Column('scheduled_frequency', sa.String(30), nullable=True))
    op.add_column('user_settings',
        sa.Column('scheduled_every_x_days', sa.Integer(), nullable=True))

    # Run tracking for scheduler
    op.add_column('user_settings',
        sa.Column('last_run_at', sa.DateTime(), nullable=True))
    op.add_column('user_settings',
        sa.Column('next_run_at', sa.DateTime(), nullable=True))

    # Encrypted join code (symmetric encryption for display-back)
    op.add_column('user_settings',
        sa.Column('colleague_join_code_encrypted', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('user_settings', 'colleague_join_code_encrypted')
    op.drop_column('user_settings', 'next_run_at')
    op.drop_column('user_settings', 'last_run_at')
    op.drop_column('user_settings', 'scheduled_every_x_days')
    op.drop_column('user_settings', 'scheduled_frequency')
    op.drop_column('user_settings', 'execution_mode')
    op.drop_column('user_settings', 'retrieval_max_results')
