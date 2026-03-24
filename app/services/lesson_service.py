from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from app.models.domain import Lesson, Sentence, UserSentenceProgress


# ==========================================
# LESSON SERVICES (Business Logic)
# ==========================================

def get_lesson_details(db: Session, lesson_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches detailed information for a specific lesson, including sentence counts
    and the user's completion status.
    """
    # 1. Fetch the basic lesson details
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()

    if not lesson:
        return None

    # 2. Count the total number of sentences belonging to this lesson
    sentences_count = db.query(Sentence).filter(Sentence.lesson_id == lesson_id).count()

    # 3. Count how many sentences the user has successfully passed in this lesson
    completed_sentences = (
        db.query(UserSentenceProgress)
        .join(Sentence, UserSentenceProgress.sentence_id == Sentence.id)
        .filter(
            Sentence.lesson_id == lesson_id,
            UserSentenceProgress.user_id == user_id,
            UserSentenceProgress.is_passed == True  # Only count successfully passed sentences
        )
        .count()
    )

    # 4. Return the formatted dictionary matching LessonDetailsResponse
    return {
        "id": lesson.id,
        "title": lesson.title,
        # Default to empty string if description is None (to prevent Kotlin crashes)
        "description": lesson.description or "",
        "icon": lesson.icon,
        "sentences_count": sentences_count,
        "completed_sentences": completed_sentences
    }