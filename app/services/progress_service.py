from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any

from app.models.domain import (
    Category, Badge, UserBadge,
    UserLessonProgress, UserSentenceProgress, ProgressStatusEnum, Sentence
)


# ==========================================
# PROGRESS SERVICES (Business Logic)
# ==========================================

def get_overview_data(db: Session, user_id: str) -> Dict[str, Any]:
    """
    Calculates the high-level overview statistics for the user.

    Args:
        db (Session): The database session.
        user_id (str): The current user's ID.

    Returns:
        Dict: A dictionary matching the OverviewDataResponse schema.
    """
    # 1. Calculate Average Score (across all practiced sentences)
    # Using SQLAlchemy's func.avg to let the database do the math efficiently
    avg_score_result = db.query(func.avg(UserSentenceProgress.highest_score)) \
        .filter(UserSentenceProgress.user_id == user_id).scalar()
    average_score = int(avg_score_result) if avg_score_result else 0

    # 2. Count Total Completed Lessons
    total_completed = db.query(UserLessonProgress) \
        .filter(UserLessonProgress.user_id == user_id,
                UserLessonProgress.status == ProgressStatusEnum.COMPLETED).count()

    # 3. Count Total Earned Badges
    total_badges = db.query(UserBadge).filter(UserBadge.user_id == user_id).count()

    # 4. Fetch Recent Achievements (Top 3 newest badges)
    recent_badges_query = (
        db.query(Badge, UserBadge.achieved_at)
        .join(UserBadge, Badge.id == UserBadge.badge_id)
        .filter(UserBadge.user_id == user_id)
        .order_by(UserBadge.achieved_at.desc())
        .limit(3)
        .all()
    )

    recent_achievements = []
    for badge, achieved_at in recent_badges_query:
        recent_achievements.append({
            "id": badge.id,
            "title": badge.title,
            "icon": badge.icon,
            # Returning ISO format. The Android client will format it to "2 days ago" etc.
            "earned_date": achieved_at.isoformat() if achieved_at else ""
        })

    # Note on Daily Streak: Accurately calculating streaks usually requires a dedicated
    # 'daily_activity_logs' table. For now, we return a placeholder value.
    daily_streak = 0

    return {
        "average_score": average_score,
        "total_completed_lessons": total_completed,
        "total_earned_badges": total_badges,
        "daily_streak": daily_streak,
        "recent_achievements": recent_achievements
    }


def get_category_achievements(db: Session, user_id: str) -> List[Dict[str, Any]]:
    """
    Calculates detailed lesson progress and average score per category.
    """
    categories = db.query(Category).all()
    achievements = []

    for cat in categories:
        # Get all lesson IDs for this category to filter progress efficiently
        lesson_ids = [lesson.id for lesson in cat.lessons]

        if not lesson_ids:
            continue  # Skip empty categories

        total_lessons = len(lesson_ids)

        # Count completed and in-progress lessons for this specific category
        completed_lessons = db.query(UserLessonProgress).filter(
            UserLessonProgress.user_id == user_id,
            UserLessonProgress.lesson_id.in_(lesson_ids),
            UserLessonProgress.status == ProgressStatusEnum.COMPLETED
        ).count()

        in_progress_lessons = db.query(UserLessonProgress).filter(
            UserLessonProgress.user_id == user_id,
            UserLessonProgress.lesson_id.in_(lesson_ids),
            UserLessonProgress.status == ProgressStatusEnum.IN_PROGRESS
        ).count()

        undone_lessons = total_lessons - (completed_lessons + in_progress_lessons)

        # Calculate average score specifically for this category's sentences
        cat_avg_score = db.query(func.avg(UserSentenceProgress.highest_score)) \
            .join(Sentence, UserSentenceProgress.sentence_id == Sentence.id) \
            .filter(Sentence.lesson_id.in_(lesson_ids),
                    UserSentenceProgress.user_id == user_id).scalar()

        average_score = int(cat_avg_score) if cat_avg_score else 0

        achievements.append({
            "category_id": cat.id,
            "category_name": cat.title,
            "icon": cat.icon,
            "average_score": average_score,
            "completed_lessons": completed_lessons,
            "in_progress_lessons": in_progress_lessons,
            "un_done_lessons": undone_lessons,
            "total_lessons": total_lessons
        })

    return achievements


def get_all_badges_status(db: Session, user_id: str) -> List[Dict[str, Any]]:
    """
    Returns all system badges and marks whether the specific user has achieved them.
    """
    all_badges = db.query(Badge).order_by(Badge.order_index).all()

    # Get a set of badge IDs that the user has already earned for O(1) lookup
    earned_badge_ids = {
        ub.badge_id for ub in db.query(UserBadge.badge_id).filter(UserBadge.user_id == user_id).all()
    }

    badges_response = []
    for badge in all_badges:
        badges_response.append({
            "id": badge.id,
            "title": badge.title,
            "description": badge.description,
            "icon": badge.icon,
            "is_achieved": badge.id in earned_badge_ids  # True if user has it, False otherwise
        })

    return badges_response