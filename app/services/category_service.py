from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from app.models.domain import Category, Lesson, UserLessonProgress, SentenceAttemptHistory, Sentence, ProgressStatusEnum


# ==========================================
# CATEGORY SERVICES (Business Logic)
# ==========================================

def get_all_categories_with_progress(db: Session, user_id: str) -> List[Dict[str, Any]]:
    """
    Fetches all categories and calculates the user's progress for each.

    Args:
        db (Session): The database session.
        user_id (str): The ID of the current logged-in user.

    Returns:
        List[Dict]: A list of dictionaries matching the CategoryResponse schema.
    """
    categories = db.query(Category).all()
    result = []

    for cat in categories:
        total_lessons = db.query(Lesson).filter(Lesson.category_id == cat.id).count()

        completed_lessons = (
            db.query(UserLessonProgress)
            .join(Lesson)
            .filter(
                Lesson.category_id == cat.id,
                UserLessonProgress.user_id == user_id,
                UserLessonProgress.status == ProgressStatusEnum.COMPLETED
            )
            .count()
        )

        progress_percentage = (completed_lessons / total_lessons) if total_lessons > 0 else 0.0

        result.append({
            "id": cat.id,
            "title": cat.title,
            "description": cat.description,
            "icon": cat.icon,
            "total_lessons": total_lessons,
            "completed_lessons": completed_lessons,
            "progress_percentage": progress_percentage
        })

    return result


def get_category_details(db: Session, category_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches a specific category and all its lessons, including the user's progress for each lesson
    based on their currently active run_id.
    """
    category = db.query(Category).filter(Category.id == category_id).first()

    if not category:
        return None

    lessons = db.query(Lesson).filter(Lesson.category_id == category_id).all()
    lesson_responses = []

    for lesson in lessons:
        # Calculate total sentences for this specific lesson
        total_sentences = db.query(Sentence).filter(Sentence.lesson_id == lesson.id).count()

        progress_record = db.query(UserLessonProgress).filter(
            UserLessonProgress.lesson_id == lesson.id,
            UserLessonProgress.user_id == user_id
        ).first()

        completed_in_run = 0
        if progress_record and progress_record.current_run_id:
            # Count UNIQUE sentences practiced in the CURRENT run from the history table
            completed_in_run = (
                    db.query(func.count(func.distinct(SentenceAttemptHistory.sentence_id)))
                    .filter(
                        SentenceAttemptHistory.user_id == user_id,
                        SentenceAttemptHistory.lesson_id == lesson.id,
                        SentenceAttemptHistory.run_id == progress_record.current_run_id
                    )
                    .scalar() or 0
            )

        # Dynamically calculate the percentage for the current run
        prog_pct = (completed_in_run / total_sentences) if total_sentences > 0 else 0.0

        lesson_responses.append({
            "id": lesson.id,
            "title": lesson.title,
            "difficulty": lesson.difficulty,
            "progress_percentage": prog_pct  # Replaces the deleted DB column
        })

    return {
        "id": category.id,
        "title": category.title,
        "description": category.description,
        "icon": category.icon,
        "lessons": lesson_responses
    }