from pydantic import BaseModel, ConfigDict

class BadgeResponse(BaseModel):
    """
    Maps to the Kotlin `Badge` data class.
    Used in the "Badges" tab to show all available badges and their unlock status.
    """
    id: str
    title: str
    description: str
    is_achieved: bool  # Computed by the backend (True if user has the badge, False otherwise)

    # This config allows Pydantic to read data directly from SQLAlchemy models
    model_config = ConfigDict(from_attributes=True)


class UnseenBadgeResponse(BaseModel):
    """
    A badge the user has earned but not yet been shown a celebration for.
    Returned by GET /progress/badges/unseen; acknowledged via POST /progress/badges/seen.
    """
    id: str
    title: str
    description: str
    achieved_at: str  # ISO-8601

    model_config = ConfigDict(from_attributes=True)


class MarkBadgesSeenRequest(BaseModel):
    """
    Payload sent by the Android app once it has shown the celebration for the given badges.
    """
    badge_ids: list[str]