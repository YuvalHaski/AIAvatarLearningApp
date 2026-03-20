from pydantic import BaseModel, ConfigDict

class BadgeResponse(BaseModel):
    """
    Maps to the Kotlin `Badge` data class.
    Used in the "Badges" tab to show all available badges and their unlock status.
    """
    id: str
    title: str
    description: str
    icon: str | None = None
    is_achieved: bool  # Computed by the backend (True if user has the badge, False otherwise)

    # This config allows Pydantic to read data directly from SQLAlchemy models
    model_config = ConfigDict(from_attributes=True)