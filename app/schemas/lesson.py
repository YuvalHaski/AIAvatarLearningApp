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