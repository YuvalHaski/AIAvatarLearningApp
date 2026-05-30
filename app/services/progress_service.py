from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any

from app.models.domain import (
    Category, Badge, UserBadge,
    UserLessonProgress, UserSentenceProgress, ProgressStatusEnum, Sentence
)


# ==========================================
# ATTEMPT PERSISTENCE
# ==========================================

def record_attempt(
        db: Session,
        user_id: str,
        sentence: Sentence,
        final_score: int,
) -> None:
    """Persist one practice attempt to the progress tables.

    - user_sentence_progress: highest_score is monotonic (max of old & new).
    - user_lesson_progress: recomputed from the sentence-level state so the
      lesson row stays consistent with reality (no drift).
    """
    now = datetime.now(timezone.utc)

    # --- 1. Upsert the per-sentence row ---
    sentence_progress = (
        db.query(UserSentenceProgress)
        .filter_by(user_id=user_id, sentence_id=sentence.id)
        .first()
    )

    if sentence_progress is None:
        # User hasn't practiced this sentence before
        sentence_progress = UserSentenceProgress(
            user_id=user_id,
            sentence_id=sentence.id,
            highest_score=final_score,
        )
        db.add(sentence_progress)
    else:
        # Update only if the new score is higher
        if final_score > (sentence_progress.highest_score or 0):
            sentence_progress.highest_score = final_score

    # Flush so the recompute below sees the just-written row.
    db.flush()

    # --- 2. Recompute the lesson-level row from sentence state ---
    lesson_id = sentence.lesson_id
    total_sentences = (
        db.query(Sentence).filter(Sentence.lesson_id == lesson_id).count()
    )

    # We now count ALL practiced sentences, regardless of pass/fail
    completed_sentences = (
        db.query(UserSentenceProgress)
        .join(Sentence, UserSentenceProgress.sentence_id == Sentence.id)
        .filter(
            Sentence.lesson_id == lesson_id,
            UserSentenceProgress.user_id == user_id
        )
        .count()
    )

    progress_pct = (
        (completed_sentences / total_sentences) * 100.0 if total_sentences else 0.0
    )

    if total_sentences and completed_sentences >= total_sentences:
        status = ProgressStatusEnum.COMPLETED
    else:
        # Any attempt at all means the lesson is in progress.
        status = ProgressStatusEnum.IN_PROGRESS

    lesson_progress = (
        db.query(UserLessonProgress)
        .filter_by(user_id=user_id, lesson_id=lesson_id)
        .first()
    )

    if lesson_progress is None:
        lesson_progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
            status=status,
            progress_percentage=progress_pct,
            last_practiced_at=now,
        )
        db.add(lesson_progress)
    else:
        lesson_progress.status = status
        lesson_progress.progress_percentage = progress_pct
        lesson_progress.last_practiced_at = now

    db.commit()


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