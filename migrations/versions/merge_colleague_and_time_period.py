"""merge_colleague_and_time_period

Revision ID: cfb65c235999
Revises: add_colleague_fields, 20260203_preferred_time_period
Create Date: 2026-02-03 16:59:35.048505

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cfb65c235999'
down_revision: Union[str, None] = ('add_colleague_fields', '20260203_preferred_time_period')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
