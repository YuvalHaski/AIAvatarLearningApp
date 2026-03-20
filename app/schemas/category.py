from pydantic import BaseModel, ConfigDict
from typing import List
from app.schemas.lesson import LessonResponse

# ==========================================
# CATEGORY SCHEMAS
# ==========================================

class CategoryResponse(BaseModel):
    """
    Maps to the Kotlin `Category` data class used in the Home screen.
    Note: 'iconRes' (Int) in Android is replaced by 'icon' (str) in the Backend.
    The Client should map the string (URL/Emoji) to local resources or load it via a library like Coil/Glide.
    """
    id: str
    title: str
    description: str
    icon: str | None = None
    total_lessons: int
    completed_lessons: int
    progress_percentage: float  # Calculated by the backend (completedLessons / totalLessons)

    # This config allows Pydantic to read data directly from SQLAlchemy models
    model_config = ConfigDict(from_attributes=True)


class CategoryDetailsResponse(BaseModel):
    """
    Maps to the Kotlin `CategoryDetails` data class.
    Used when a user clicks on a specific category to see its lessons.
    """
    id: str
    title: str
    description: str
    icon: str | None = None
    lessons: List[LessonResponse]

    # This config allows Pydantic to read data directly from SQLAlchemy models
    model_config = ConfigDict(from_attributes=True)