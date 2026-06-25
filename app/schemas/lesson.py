from pydantic import BaseModel, ConfigDict
from app.models.domain import DifficultyEnum

# ==========================================
# LESSON SCHEMAS
# ==========================================

class LessonResponse(BaseModel):
    """
    Maps to the Kotlin `Lesson` data class.
    Used to represent a single lesson's data and user progress.
    """
    id: str
    title: str
    progress_percentage: float
    difficulty: DifficultyEnum

    model_config = ConfigDict(from_attributes=True)


class LessonDetailsResponse(BaseModel):
    """
    Maps to the Kotlin `LessonDetails` data class.
    Provides comprehensive details about a specific lesson and the user's progress in it.
    """
    id: str
    title: str
    description: str
    sentences_count: int
    completed_sentences: int  # This now reflects the CURRENT run count

    model_config = ConfigDict(from_attributes=True)

class SentenceResponse(BaseModel):
    """
    Represents a single sentence within a lesson.
    """
    id: str
    text: str
    order_index: int

    model_config = ConfigDict(from_attributes=True)


class LessonStartResponse(BaseModel):
    """
    Response returned when the client initiates a lesson.
    The client MUST hold onto this run_id and send it with every ASR request.
    """
    run_id: str


class SentenceProgressRequest(BaseModel):
    """
    Payload received from the Android app when a user completes a sentence.
    Contains the score evaluated by the ASR service.
    """
    score: int


class SentenceProgressResponse(BaseModel):
    """
    Response sent back to the app after updating the progress in the database.
    """
    highest_score: int
    lesson_progress_percentage: float