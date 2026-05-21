from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from app.models.domain import Lesson, Sentence, UserSentenceProgress, UserLessonProgress, ProgressStatusEnum


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


def update_sentence_progress(db: Session, user_id: str, lesson_id: str, sentence_id: str, score: int) -> Dict[str, Any]:
    """
    Upserts the sentence progress and recalculates the overall lesson progress atomically.
    Returns a dictionary mapping to SentenceProgressResponse.
    """

    # 1. UPSERT SENTENCE PROGRESS
    # Check if the user has already practiced this specific sentence
    sentence_progress = db.query(UserSentenceProgress).filter(
        UserSentenceProgress.user_id == user_id,
        UserSentenceProgress.sentence_id == sentence_id
    ).first()

    if not sentence_progress:
        # First time practicing this sentence, create a new record
        sentence_progress = UserSentenceProgress(
            user_id=user_id,
            sentence_id=sentence_id,
            highest_score=score
        )
        db.add(sentence_progress)
    else:
        # Already practiced, update the highest score only if the new score is better
        if score > sentence_progress.highest_score:
            sentence_progress.highest_score = score

    # BEST PRACTICE: Flush forces the changes to the database without fully committing yet.
    # We need this so the next query (counting completed sentences) includes this new/updated record.
    db.flush()

    # 2. CALCULATE OVERALL LESSON PROGRESS
    # Total sentences in the current lesson
    total_sentences = db.query(Sentence).filter(Sentence.lesson_id == lesson_id).count()

    # Sentences the user has practiced in this lesson (existence of a record means it was practiced)
    completed_sentences = (
        db.query(UserSentenceProgress)
        .join(Sentence, UserSentenceProgress.sentence_id == Sentence.id)
        .filter(
            Sentence.lesson_id == lesson_id,
            UserSentenceProgress.user_id == user_id
        )
        .count()
    )

    # Calculate percentage safely to avoid division by zero
    progress_percentage = 0.0
    if total_sentences > 0:
        progress_percentage = completed_sentences / total_sentences

    # Determine status: if completed all sentences, mark as COMPLETED, otherwise IN_PROGRESS
    new_status = ProgressStatusEnum.COMPLETED if completed_sentences >= total_sentences else ProgressStatusEnum.IN_PROGRESS

    # 3. UPSERT LESSON PROGRESS
    # Update the overall progress for the lesson
    lesson_progress = db.query(UserLessonProgress).filter(
        UserLessonProgress.user_id == user_id,
        UserLessonProgress.lesson_id == lesson_id
    ).first()

    if not lesson_progress:
        lesson_progress = UserLessonProgress(
            user_id=user_id,
            lesson_id=lesson_id,
            status=new_status,
            progress_percentage=progress_percentage
        )
        db.add(lesson_progress)
    else:
        lesson_progress.status = new_status
        lesson_progress.progress_percentage = progress_percentage
        # FORCE UPDATE THE TIMESTAMP (Even if the score/percentage didn't change)
        lesson_progress.last_practiced_at = func.now()

    # 4. COMMIT TRANSACTION
    # Save both sentence progress and lesson progress atomically
    db.commit()

    return {
        "highest_score": sentence_progress.highest_score,
        "lesson_progress_percentage": progress_percentage
    }