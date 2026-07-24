import uuid

from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any

from app.models.domain import (
    Category, Badge, UserBadge,
    UserLessonProgress, SentenceAttemptHistory, ProgressStatusEnum, Sentence
)


# ==========================================
# ATTEMPT PERSISTENCE
# ==========================================

def record_attempt(
        db: Session,
        user_id: str,
        sentence: Sentence,
        final_score: int,
        run_id: str  # NEW: Received from the ASR route
) -> None:
    """
    Persist one practice attempt to the history table and evaluate lesson progress.
    """
    now = datetime.now(timezone.utc)
    lesson_id = sentence.lesson_id

    # --- 1. Insert into Append-Only History Table ---
    attempt_id = str(uuid.uuid4())
    new_attempt = SentenceAttemptHistory(
        id=attempt_id,
        user_id=user_id,
        lesson_id=lesson_id,
        sentence_id=sentence.id,
        run_id=run_id,
        score=final_score,
        created_at=now
    )
    db.add(new_attempt)
    db.flush()  # Push to DB so the recompute below sees it immediately

    # --- 2. Recompute the lesson-level status based on the CURRENT run ---
    total_sentences = db.query(Sentence).filter(Sentence.lesson_id == lesson_id).count()

    # Count distinct sentences completed in THIS specific run
    completed_in_run = (
            db.query(func.count(func.distinct(SentenceAttemptHistory.sentence_id)))
            .filter(
                SentenceAttemptHistory.user_id == user_id,
                SentenceAttemptHistory.lesson_id == lesson_id,
                SentenceAttemptHistory.run_id == run_id
            )
            .scalar() or 0
    )

    lesson_progress = db.query(UserLessonProgress).filter_by(user_id=user_id, lesson_id=lesson_id).first()

    if lesson_progress is None:
        # Fallback mechanism in case the client somehow bypassed the /start route
        lesson_progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
            current_run_id=run_id,
            status=ProgressStatusEnum.COMPLETED if (
                    0 < total_sentences <= completed_in_run) else ProgressStatusEnum.IN_PROGRESS,
            last_practiced_at=now
        )
        db.add(lesson_progress)
    else:
        lesson_progress.last_practiced_at = now

        # If they finished all sentences in this active run, mark the lesson as COMPLETED
        if 0 < total_sentences <= completed_in_run:
            lesson_progress.status = ProgressStatusEnum.COMPLETED
        # If it was NOT_STARTED, move it to IN_PROGRESS.
        # Note: If it's already COMPLETED (from an older run), we leave it as COMPLETED.
        elif lesson_progress.status == ProgressStatusEnum.NOT_STARTED:
            lesson_progress.status = ProgressStatusEnum.IN_PROGRESS

    db.commit()


# ==========================================
# PROGRESS SERVICES (Business Logic)
# ==========================================

def get_overview_data(db: Session, user_id: str) -> Dict[str, Any]:
    """
    Calculates the high-level overview statistics for the user.
    """
    # 1. Calculate Average Score
    # We use a subquery to find the MAX score for each distinct sentence the user practiced,
    # and then calculate the average of those highest scores.
    subquery_max_scores = (
        db.query(func.max(SentenceAttemptHistory.score).label("max_score"))
        .filter(SentenceAttemptHistory.user_id == user_id)
        .group_by(SentenceAttemptHistory.sentence_id)
        .subquery()
    )

    avg_score_result = db.query(func.avg(subquery_max_scores.c.max_score)).scalar()
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
        lesson_ids = [lesson.id for lesson in cat.lessons]

        if not lesson_ids:
            continue  # Skip empty categories

        total_lessons = len(lesson_ids)

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
        # Subquery: get the MAX score per sentence for this specific category
        cat_subquery = (
            db.query(func.max(SentenceAttemptHistory.score).label("max_score"))
            .join(Sentence, SentenceAttemptHistory.sentence_id == Sentence.id)
            .filter(
                SentenceAttemptHistory.user_id == user_id,
                Sentence.lesson_id.in_(lesson_ids)
            )
            .group_by(SentenceAttemptHistory.sentence_id)
            .subquery()
        )

        cat_avg_score = db.query(func.avg(cat_subquery.c.max_score)).scalar()
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
            "is_achieved": badge.id in earned_badge_ids  # True if user has it, False otherwise
        })

    return badges_response


def get_unseen_badges(db: Session, user_id: str) -> List[Dict[str, Any]]:
    """
    Returns badges the user has earned but hasn't yet been shown a celebration for.
    Fallback path for the client to reliably surface "badge earned" UI even if it missed the
    inline `new_badges` field on a /lessons/{id}/complete response (e.g. dropped connection).
    """
    rows = (
        db.query(Badge, UserBadge.achieved_at)
        .join(UserBadge, Badge.id == UserBadge.badge_id)
        .filter(UserBadge.user_id == user_id, UserBadge.is_seen.is_(False))
        .order_by(UserBadge.achieved_at.asc())
        .all()
    )

    return [
        {
            "id": badge.id,
            "title": badge.title,
            "description": badge.description,
            "achieved_at": achieved_at.isoformat() if achieved_at else ""
        }
        for badge, achieved_at in rows
    ]


def mark_badges_seen(db: Session, user_id: str, badge_ids: List[str]) -> None:
    """
    Acknowledges that the client has shown the celebration for the given badges, so they no
    longer appear in get_unseen_badges. Silently ignores badge_ids the user hasn't earned.
    """
    if not badge_ids:
        return

    db.query(UserBadge).filter(
        UserBadge.user_id == user_id,
        UserBadge.badge_id.in_(badge_ids)
    ).update({UserBadge.is_seen: True}, synchronize_session=False)

    db.commit()