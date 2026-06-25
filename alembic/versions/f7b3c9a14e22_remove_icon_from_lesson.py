"""remove icon from lesson

Revision ID: f7b3c9a14e22
Revises: a898db699e39
Create Date: 2026-06-25 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7b3c9a14e22'
down_revision: Union[str, Sequence[str], None] = 'a898db699e39'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_column('lessons', 'icon')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('lessons', sa.Column('icon', sa.String(), autoincrement=False, nullable=True))
