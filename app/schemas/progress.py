from pydantic import BaseModel, ConfigDict
from typing import List

# ==========================================
# OVERVIEW TAB SCHEMAS
# ==========================================

class RecentBadge(BaseModel):
    """
    Maps to the Kotlin `RecentBadge` data class.
    Represents a simplified badge model for the "Recent Achievements" section.
    """
    id: str
    title: str
    icon: str | None = None
    earned_date: str  # Pre-formatted string or ISO-8601

    model_config = ConfigDict(from_attributes=True)


class OverviewDataResponse(BaseModel):
    """
    Maps to the Kotlin `OverviewData` data class.
    Aggregates the user's overall progress across the entire app.
    """
    average_score: int
    total_completed_lessons: int
    total_earned_badges: int
    daily_streak: int
    recent_achievements: List[RecentBadge]

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# CATEGORY ACHIEVEMENTS TAB SCHEMAS
# ==========================================

class CategoryAchievementResponse(BaseModel):
    """
    Maps to the Kotlin `CategoryAchievement` data class.
    Provides detailed progress breakdown per category.
    """
    category_id: str
    category_name: str
    icon: str | None = None
    average_score: int
    completed_lessons: int
    in_progress_lessons: int
    un_done_lessons: int
    total_lessons: int

    model_config = ConfigDict(from_attributes=True)