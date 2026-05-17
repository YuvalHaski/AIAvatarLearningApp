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
    progress_percentage: float  # Value between 0.0 and 1.0
    difficulty: DifficultyEnum

    # This config allows Pydantic to read data directly from SQLAlchemy models
    model_config = ConfigDict(from_attributes=True)


class LessonDetailsResponse(BaseModel):
    """
    Maps to the Kotlin `LessonDetails` data class.
    Provides comprehensive details about a specific lesson and the user's progress in it.
    """
    id: str
    title: str
    description: str
    icon: str | None = None
    sentences_count: int
    completed_sentences: int

    model_config = ConfigDict(from_attributes=True)

class SentenceResponse(BaseModel):
    """
    Represents a single sentence within a lesson.
    """
    id: str
    text: str
    order_index: int

    model_config = ConfigDict(from_attributes=True)