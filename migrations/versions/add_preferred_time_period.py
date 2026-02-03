"""Add preferred_time_period column to users table.

Revision ID: 20260203_preferred_time_period
Revises: 20260202_calendar_invites
Create Date: 2026-02-03

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260203_preferred_time_period'
down_revision = '20260202_calendar_invites'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add preferred_time_period column to users table
    op.add_column('users', sa.Column('preferred_time_period', sa.String(100), nullable=True))
    
    # Set default value for existing rows
    op.execute("UPDATE users SET preferred_time_period = 'last two weeks' WHERE preferred_time_period IS NULL")


def downgrade() -> None:
    op.drop_column('users', 'preferred_time_period')
