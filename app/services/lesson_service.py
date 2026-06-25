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


# ==========================================
# LESSON COMPLETION SERVICES
# ==========================================

def complete_lesson(db: Session, lesson_id: str, user_id: str, run_id: str) -> Dict[str, Any]:
    """
    Finalizes the lesson run, calculates the final score using a bulletproof MAX-per-sentence aggregation,
    updates the user's tracking progress status to COMPLETED, and generates contextual verbal feedback.

    Raises ValueError for business rule violations (invalid run or missing progress).
    """
    # 1. Fetch user progress and validate the incoming session run_id
    progress = db.query(UserLessonProgress).filter_by(user_id=user_id, lesson_id=lesson_id).first()

    if not progress:
        raise ValueError("No active progress tracking record found for this lesson.")

    if progress.current_run_id != run_id:
        raise ValueError("Invalid or expired session run ID for this lesson.")

    # 2. Calculate the final score using a bulletproof subquery aggregation:
    # First, we isolate the highest (MAX) score obtained for each unique sentence within this specific run.
    # This safeguards against duplicate double-clicks or multiple attempts on the same sentence.
    score_subquery = (
        db.query(func.max(SentenceAttemptHistory.score).label("max_score"))
        .filter(
            SentenceAttemptHistory.user_id == user_id,
            SentenceAttemptHistory.lesson_id == lesson_id,
            SentenceAttemptHistory.run_id == run_id
        )
        .group_by(SentenceAttemptHistory.sentence_id)
        .subquery()
    )

    # Second, we take the average of those maximum scores.
    average_score_raw = db.query(func.avg(score_subquery.c.max_score)).scalar()

    # Fallback default to 0 if the user somehow completed a lesson without any captured historical attempts
    final_score = int(round(average_score_raw)) if average_score_raw is not None else 0

    # 3. Update the tracking state to COMPLETED
    progress.status = ProgressStatusEnum.COMPLETED

    # 4. Generate the dynamic written context summary feedback text
    feedback = _generate_feedback_text(final_score)

    # 5. Commit all entity modifications atomically within a single database transaction
    db.commit()

    return {
        "lesson_id": lesson_id,
        "status": progress.status,
        "average_score": final_score,
        "feedback_text": feedback
    }


def _generate_feedback_text(score: int) -> str:
    """
    Internal business logic helper to map numerical scores into personalized dynamic text strings.
    Keeps the core database logic cleaner and decoupled from display variants.
    """
    if score >= 90:
        return "Excellent pronunciation! You sound like a native speaker. Keep up the amazing work!"
    elif score >= 75:
        return "Great job! Your fluency and pronunciation are getting much better. Keep it up!"
    elif score >= 50:
        return "Good effort! Practice makes perfect. Try to focus on phrasing and clear accuracy next time."
    else:
        return "Keep going! Continuous practice will help you build your confidence and precise speech."

