from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.auth import get_current_user

from app.schemas.lesson import LessonDetailsResponse, SentenceResponse, SentenceProgressRequest, SentenceProgressResponse
from app.services.lesson_service import get_lesson_details, get_lesson_sentences, update_sentence_progress

# ==========================================
# LESSONS ROUTER
# ==========================================

router = APIRouter(prefix="/lessons", tags=["Lessons"])


@router.get("/{lesson_id}", response_model=LessonDetailsResponse)
def read_lesson_details(
        lesson_id: str,
        db: Session = Depends(get_db),
        current_user_id: str = Depends(get_current_user)
):
    """
    Retrieve details and user progress for a specific lesson.
    Requires a valid Firebase Bearer token.
    """
    # Calling the business logic we built in Service
    lesson_details = get_lesson_details(db=db, lesson_id=lesson_id, user_id=current_user_id)

    # If the class does not exist in the database, we will return a professional 404 error.
    if not lesson_details:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Lesson with ID {lesson_id} not found."
        )

    return lesson_details


@router.get("/{lesson_id}/sentences",
            response_model=List[SentenceResponse],
            dependencies=[Depends(get_current_user)])
def read_lesson_sentences(
        lesson_id: str,
        db: Session = Depends(get_db)
):
    """
    Retrieve all sentences for a specific lesson.
    Used by the Android app when starting the interactive Lesson Progress screen.
    Requires a valid Firebase Bearer token.
    """
    # 1. Fetch sentences from the service layer
    sentences = get_lesson_sentences(db=db, lesson_id=lesson_id)

    # 2. Check if the lesson has sentences (or if it exists)
    if not sentences:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sentences found for Lesson ID {lesson_id}."
        )

    return sentences


@router.post("/progress/{lesson_id}/sentences/{sentence_id}", response_model=SentenceProgressResponse)
def save_sentence_progress(
        lesson_id: str,
        sentence_id: str,
        payload: SentenceProgressRequest,
        db: Session = Depends(get_db),
        current_user_id: str = Depends(get_current_user)
):
    """
    Saves the user's evaluation score for a specific sentence.
    It automatically updates the highest score and recalculates the overall lesson progress.
    Requires a valid Firebase Bearer token.
    """
    # Delegate the complex business logic to the service layer
    result = update_sentence_progress(
        db=db,
        user_id=current_user_id,
        lesson_id=lesson_id,
        sentence_id=sentence_id,
        score=payload.score
    )

    return result