"""remove icon from badges

Revision ID: 9493b6387770
Revises: 7ddff75f73ab
Create Date: 2026-07-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9493b6387770'
down_revision: Union[str, Sequence[str], None] = '7ddff75f73ab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column('badges', 'icon')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('badges', sa.Column('icon', sa.String(), nullable=True))
