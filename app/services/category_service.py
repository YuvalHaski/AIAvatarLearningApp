from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any

from app.models.domain import Category, Lesson, UserLessonProgress, ProgressStatusEnum


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
    # 1. Fetch all categories from the database
    categories = db.query(Category).all()

    result = []

    # 2. Iterate through each category to calculate specific user progress
    for cat in categories:
        # Count total lessons belonging to this category
        total_lessons = db.query(Lesson).filter(Lesson.category_id == cat.id).count()

        # Count how many lessons the user has COMPLETED in this specific category
        completed_lessons = (
            db.query(UserLessonProgress)
            .join(Lesson)  # Join with Lesson table to filter by category
            .filter(
                Lesson.category_id == cat.id,
                UserLessonProgress.user_id == user_id,
                UserLessonProgress.status == ProgressStatusEnum.COMPLETED
            )
            .count()
        )

        # 3. Calculate progress percentage safely (prevent division by zero)
        progress_percentage = (completed_lessons / total_lessons) if total_lessons > 0 else 0.0

        # 4. Append the formatted data to the result list
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
    Fetches a specific category and all its lessons, including the user's progress for each lesson.

    Args:
        db (Session): The database session.
        category_id (str): The requested category ID.
        user_id (str): The ID of the current logged-in user.

    Returns:
        Dict: A dictionary matching the CategoryDetailsResponse schema, or None if not found.
    """
    # 1. Fetch the specific category
    category = db.query(Category).filter(Category.id == category_id).first()

    # If category doesn't exist, return None. The Router will handle the 404 Error.
    if not category:
        return None

    # 2. Fetch all lessons belonging to this category
    lessons = db.query(Lesson).filter(Lesson.category_id == category_id).all()

    lesson_responses = []

    # 3. Iterate through lessons to fetch user's specific progress for each
    for lesson in lessons:
        progress_record = db.query(UserLessonProgress).filter(
            UserLessonProgress.lesson_id == lesson.id,
            UserLessonProgress.user_id == user_id
        ).first()

        # If the user hasn't started the lesson yet, progress_record will be None
        prog_pct = progress_record.progress_percentage if progress_record else 0.0

        lesson_responses.append({
            "id": lesson.id,
            "title": lesson.title,
            "difficulty": lesson.difficulty,
            "progress_percentage": prog_pct
        })

    # 4. Construct the final detailed response
    return {
        "id": category.id,
        "title": category.title,
        "description": category.description,
        "icon": category.icon,
        "lessons": lesson_responses
    }