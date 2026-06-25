import uuid

from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from app.models.domain import Lesson, Sentence, UserLessonProgress, SentenceAttemptHistory, ProgressStatusEnum


# ==========================================
# LESSON SERVICES (Business Logic)
# ==========================================

def start_lesson(db: Session, lesson_id: str, user_id: str, is_resume: bool = False) -> str:
    """
    Prepares the active session. If is_resume is True and a run exists, it returns the existing run_id.
    Otherwise, it generates a fresh run_id.
    """
    progress = db.query(UserLessonProgress).filter_by(user_id=user_id, lesson_id=lesson_id).first()

    # If resuming, and we already have an active run, just return it!
    if is_resume and progress and progress.current_run_id:
        if progress.status == ProgressStatusEnum.NOT_STARTED:
            progress.status = ProgressStatusEnum.IN_PROGRESS
            db.commit()
        return progress.current_run_id

    new_run_id = str(uuid.uuid4())

    if not progress:
        progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
            current_run_id=new_run_id,
            status=ProgressStatusEnum.IN_PROGRESS
        )
        db.add(progress)
    else:
        progress.current_run_id = new_run_id
        if progress.status == ProgressStatusEnum.NOT_STARTED:
            progress.status = ProgressStatusEnum.IN_PROGRESS

    db.commit()
    return new_run_id


def get_lesson_details(db: Session, lesson_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches detailed information for a specific lesson, including sentence counts
    and the user's completion status based on the CURRENT active run.
    """
    lesson = db.query(Lesson).filter(Lesson.id == lesson_id).first()

    if not lesson:
        return None

    sentences_count = db.query(Sentence).filter(Sentence.lesson_id == lesson_id).count()

    # Get the user's active run for this lesson
    lesson_progress = db.query(UserLessonProgress).filter_by(user_id=user_id, lesson_id=lesson_id).first()
    completed_sentences = 0

    if lesson_progress and lesson_progress.current_run_id:
        # Count UNIQUE sentences successfully practiced in the current run
        # func.count(func.distinct(...)) prevents double-counting if a user practices the same sentence twice in one run.
        completed_sentences = (
            db.query(func.count(func.distinct(SentenceAttemptHistory.sentence_id)))
            .filter(
                SentenceAttemptHistory.user_id == user_id,
                SentenceAttemptHistory.lesson_id == lesson_id,
                SentenceAttemptHistory.run_id == lesson_progress.current_run_id
            )
            .scalar() or 0
        )

    return {
        "id": lesson.id,
        "title": lesson.title,
        "description": lesson.description or "",
        "sentences_count": sentences_count,
        "completed_sentences": completed_sentences
    }


def get_lesson_sentences(db: Session, lesson_id: str) -> List[Dict[str, Any]]:
    """
    Fetches all sentences for a specific lesson, ordered by their sequence.
    """
    sentences_query = (
        db.query(Sentence)
        .filter(Sentence.lesson_id == lesson_id)
        .order_by(Sentence.order_index.asc())
        .all()
    )

    result = []
    for s in sentences_query:
        result.append({
            "id": s.id,
            "text": s.text,
            "order_index": s.order_index
        })

    return result