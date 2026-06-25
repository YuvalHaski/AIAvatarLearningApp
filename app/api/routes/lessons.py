from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.auth import get_current_user

from app.schemas.lesson import LessonDetailsResponse, SentenceResponse, LessonStartResponse, LessonCompleteRequest, LessonCompleteResponse
from app.services.lesson_service import get_lesson_details, get_lesson_sentences, start_lesson, complete_lesson

# ==========================================
# LESSONS ROUTER
# ==========================================

router = APIRouter(prefix="/lessons", tags=["Lessons"])


@router.post("/{lesson_id}/start", response_model=LessonStartResponse)
def start_lesson_endpoint(
        lesson_id: str,
        is_resume: bool = False,
        db: Session = Depends(get_db),
        current_user_id: str = Depends(get_current_user)
):
    """
    Initialize a new run for the lesson.
    The Android app should call this exactly once when the user clicks 'Start/Restart Lesson'.
    Returns a unique run_id.
    """
    run_id = start_lesson(db=db, lesson_id=lesson_id, user_id=current_user_id, is_resume=is_resume)
    return LessonStartResponse(run_id=run_id)


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
    lesson_details = get_lesson_details(db=db, lesson_id=lesson_id, user_id=current_user_id)

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
    """
    sentences = get_lesson_sentences(db=db, lesson_id=lesson_id)

    if not sentences:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No sentences found for Lesson ID {lesson_id}."
        )

    return sentences


@router.post("/{lesson_id}/complete", response_model=LessonCompleteResponse)
def complete_lesson_endpoint(
        lesson_id: str,
        request: LessonCompleteRequest,
        db: Session = Depends(get_db),
        current_user_id: str = Depends(get_current_user)
):
    """
    Marks the current lesson run as completed, calculates the final score,
    and returns the summary along with dynamic feedback.
    """
    try:
        # Pass the execution to the business logic layer (Service)
        result = complete_lesson(
            db=db,
            lesson_id=lesson_id,
            user_id=current_user_id,
            run_id=request.run_id
        )
        return result

    except ValueError as e:
        # Catch the business logic error from the service and return a proper HTTP 400 Client Error
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Catch any unexpected database or server errors and return a 500 Server Error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while completing the lesson."
        )