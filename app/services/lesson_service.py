from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
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
            UserSentenceProgress.user_id == user_id
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

def get_lesson_sentences(db: Session, lesson_id: str) -> List[Dict[str, Any]]:
    """
    Fetches all sentences for a specific lesson, ordered by their sequence,
    and returns them as a list of dictionaries.
    """
    # fetch from db
    sentences_query = (
        db.query(Sentence)
        .filter(Sentence.lesson_id == lesson_id)
        .order_by(Sentence.order_index.asc())
        .all()
    )

    # build the result list
    result = []
    for s in sentences_query:
        result.append({
            "id": s.id,
            "text": s.text,
            "order_index": s.order_index
        })

    return result