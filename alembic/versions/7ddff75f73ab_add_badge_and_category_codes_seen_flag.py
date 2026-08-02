"""add badge_code, category_code, and user_badges.is_seen

Revision ID: 7ddff75f73ab
Revises: f7b3c9a14e22
Create Date: 2026-07-21 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7ddff75f73ab'
down_revision: Union[str, Sequence[str], None] = 'f7b3c9a14e22'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Stable machine codes for the categories/badges seeded by scripts/populate_initial_data.py,
# keyed by their current display title. Used to backfill existing rows below.
CATEGORY_CODES_BY_TITLE = {
    "Job Interview": "job_interview",
    "Travel Abroad": "travel_abroad",
    "Emergencies & Safety": "emergencies_safety",
    "Small Talk & Socializing": "small_talk_socializing",
    "Dining & Ordering Food": "dining_ordering_food",
    "Idioms & Everyday Expressions": "idioms_expressions",
    "Tech Support": "tech_support",
    "Public Transport": "public_transport",
}

BADGE_CODES_BY_TITLE = {
    "First Step": "first_step",
    "Perfect Score": "perfect_score",
    "Conversation Starter": "conversation_starter",
    "Lesson Collector": "lesson_collector",
    "7 Day Streak": "streak_7_day",
    "Early Bird": "early_bird",
    "Night Owl": "night_owl",
    "Traveler": "category_complete_travel_abroad",
    "Foodie": "category_complete_dining_ordering_food",
    "Transport Expert": "category_complete_public_transport",
    "Techie": "category_complete_tech_support",
    "Social Butterfly": "category_complete_small_talk_socializing",
    "Badge Collector": "badge_collector",
    "30 Day Streak": "streak_30_day",
    "Mastery": "mastery",
}


def upgrade() -> None:
    """Upgrade schema."""
    # --- categories.category_code ---
    op.add_column('categories', sa.Column('category_code', sa.String(), nullable=True))
    categories = sa.table('categories', sa.column('title', sa.String()), sa.column('category_code', sa.String()))
    for title, code in CATEGORY_CODES_BY_TITLE.items():
        op.execute(
            categories.update().where(categories.c.title == title).values(category_code=code)
        )
    op.alter_column('categories', 'category_code', nullable=False)
    op.create_index(op.f('ix_categories_category_code'), 'categories', ['category_code'], unique=True)

    # --- badges.badge_code ---
    op.add_column('badges', sa.Column('badge_code', sa.String(), nullable=True))
    badges = sa.table('badges', sa.column('title', sa.String()), sa.column('badge_code', sa.String()))
    for title, code in BADGE_CODES_BY_TITLE.items():
        op.execute(
            badges.update().where(badges.c.title == title).values(badge_code=code)
        )
    op.alter_column('badges', 'badge_code', nullable=False)
    op.create_index(op.f('ix_badges_badge_code'), 'badges', ['badge_code'], unique=True)

    # --- user_badges.is_seen ---
    op.add_column(
        'user_badges',
        sa.Column('is_seen', sa.Boolean(), nullable=False, server_default=sa.false())
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('user_badges', 'is_seen')

    op.drop_index(op.f('ix_badges_badge_code'), table_name='badges')
    op.drop_column('badges', 'badge_code')

    op.drop_index(op.f('ix_categories_category_code'), table_name='categories')
    op.drop_column('categories', 'category_code')
