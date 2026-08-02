"""Badge rule predicates.

Each rule is a callable `(db: Session, user_id: str, context: dict) -> bool` that answers
"does this user currently satisfy this badge's criteria?". Rules are pure reads - they never
write to the DB or know about `Badge`/`UserBadge` rows; `badge_service.evaluate_and_award_badges`
owns the awarding side-effect.

`BADGE_RULES` maps a badge's stable `badge_code` (see app/models/domain.py) to its rule, so
badge display copy (title/description) can change freely without touching this file.

All time-based rules (streaks, time-of-day) operate in UTC - see CLAUDE.md for the known
limitation this implies for users in other timezones.
"""
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import extract, func
from sqlalchemy.orm import Session

from app.models.domain import (
    Category,
    Lesson,
    ProgressStatusEnum,
    SentenceAttemptHistory,
    UserBadge,
    UserLessonProgress,
)

BadgeRule = Callable[[Session, str, dict], bool]


# ==========================================
# RULE SHAPE FACTORIES
# ==========================================

def _completed_lesson_count(db: Session, user_id: str) -> int:
    return (
        db.query(UserLessonProgress)
        .filter(
            UserLessonProgress.user_id == user_id,
            UserLessonProgress.status == ProgressStatusEnum.COMPLETED,
        )
        .count()
    )


def _completed_lesson_count_at_least(n: int) -> BadgeRule:
    """e.g. First Step (n=1), Conversation Starter (n=5), Lesson Collector (n=10)."""

    def check(db: Session, user_id: str, context: dict) -> bool:
        return _completed_lesson_count(db, user_id) >= n

    return check


def _perfect_score() -> BadgeRule:
    """Satisfied the moment a just-completed lesson run scored a perfect 100.

    Reads `context["completed_lesson_score"]`, the `final_score` lesson_service.complete_lesson
    already computed for the run being finalized - no extra query needed.
    """

    def check(db: Session, user_id: str, context: dict) -> bool:
        return context.get("completed_lesson_score") == 100

    return check


def _category_complete(category_code: str) -> BadgeRule:
    """e.g. Traveler ("travel_abroad"), Foodie ("dining_ordering_food"), ..."""

    def check(db: Session, user_id: str, context: dict) -> bool:
        category = db.query(Category).filter(Category.category_code == category_code).first()
        if category is None:
            return False

        lesson_ids = [lesson.id for lesson in category.lessons]
        if not lesson_ids:
            return False

        completed = (
            db.query(UserLessonProgress)
            .filter(
                UserLessonProgress.user_id == user_id,
                UserLessonProgress.lesson_id.in_(lesson_ids),
                UserLessonProgress.status == ProgressStatusEnum.COMPLETED,
            )
            .count()
        )
        return completed == len(lesson_ids)

    return check


def _all_lessons_complete() -> BadgeRule:
    """Mastery: every lesson in the entire app has been completed by this user."""

    def check(db: Session, user_id: str, context: dict) -> bool:
        total_lessons = db.query(Lesson).count()
        if total_lessons == 0:
            return False
        return _completed_lesson_count(db, user_id) == total_lessons

    return check


def _distinct_practice_dates(db: Session, user_id: str) -> set:
    """Distinct UTC calendar dates on which the user logged at least one practice attempt."""
    rows = (
        db.query(func.date(SentenceAttemptHistory.created_at))
        .filter(SentenceAttemptHistory.user_id == user_id)
        .distinct()
        .all()
    )
    return {row[0] for row in rows}


def _current_streak_length(practice_dates: set, today) -> int:
    """Length of the unbroken run of consecutive calendar days ending today or yesterday.

    A streak that hasn't been extended today still counts as "current" until the day is
    over, so we allow the most recent practice date to be either today or yesterday before
    walking backwards.
    """
    from datetime import timedelta

    if today in practice_dates:
        cursor = today
    elif (today - timedelta(days=1)) in practice_dates:
        cursor = today - timedelta(days=1)
    else:
        return 0

    streak = 0
    while cursor in practice_dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


def _streak_at_least(days: int) -> BadgeRule:
    """e.g. 7 Day Streak, 30 Day Streak."""

    def check(db: Session, user_id: str, context: dict) -> bool:
        practice_dates = _distinct_practice_dates(db, user_id)
        today = datetime.now(timezone.utc).date()
        return _current_streak_length(practice_dates, today) >= days

    return check


def _time_window(start_hour: int, end_hour: int) -> BadgeRule:
    """e.g. Early Bird (6-8 UTC), Night Owl (0-4 UTC). `end_hour` is exclusive."""

    def check(db: Session, user_id: str, context: dict) -> bool:
        exists = (
            db.query(SentenceAttemptHistory)
            .filter(
                SentenceAttemptHistory.user_id == user_id,
                extract("hour", SentenceAttemptHistory.created_at) >= start_hour,
                extract("hour", SentenceAttemptHistory.created_at) < end_hour,
            )
            .first()
        )
        return exists is not None

    return check


def _badge_count_at_least(n: int) -> BadgeRule:
    """Meta badge, e.g. Badge Collector (earn 10 other badges)."""

    def check(db: Session, user_id: str, context: dict) -> bool:
        earned = db.query(UserBadge).filter(UserBadge.user_id == user_id).count()
        return earned >= n

    return check


# ==========================================
# REGISTRY
# ==========================================
# Meta rules (whose criteria depends on other badges already being awarded) are evaluated in a
# second pass by badge_service.evaluate_and_award_badges - see META_BADGE_CODES below.

BADGE_RULES: dict[str, BadgeRule] = {
    "first_step": _completed_lesson_count_at_least(1),
    "perfect_score": _perfect_score(),
    "conversation_starter": _completed_lesson_count_at_least(5),
    "lesson_collector": _completed_lesson_count_at_least(10),
    "streak_7_day": _streak_at_least(7),
    "early_bird": _time_window(6, 8),
    "night_owl": _time_window(0, 4),
    "category_complete_travel_abroad": _category_complete("travel_abroad"),
    "category_complete_dining_ordering_food": _category_complete("dining_ordering_food"),
    "category_complete_public_transport": _category_complete("public_transport"),
    "category_complete_tech_support": _category_complete("tech_support"),
    "category_complete_small_talk_socializing": _category_complete("small_talk_socializing"),
    "streak_30_day": _streak_at_least(30),
    "mastery": _all_lessons_complete(),
    "badge_collector": _badge_count_at_least(10),
}

# badge_codes whose rule depends on the user's *current* earned-badge count, and therefore must
# be evaluated only after all non-meta badges for this pass have already been awarded.
META_BADGE_CODES: frozenset[str] = frozenset({"badge_collector"})
