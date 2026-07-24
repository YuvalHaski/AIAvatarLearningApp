from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.schemas.progress import OverviewDataResponse, CategoryAchievementResponse
from app.schemas.badge import BadgeResponse, UnseenBadgeResponse, MarkBadgesSeenRequest
from app.services import progress_service
from app.api.auth import get_current_user

# ==========================================
# PROGRESS ROUTER
# ==========================================

router = APIRouter(prefix="/progress", tags=["Progress"])

@router.get("/overview", response_model=OverviewDataResponse)
def get_overview_data(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """
    Returns the high-level overview data (Average score, streak, recent badges).
    """
    return progress_service.get_overview_data(db, user_id)

@router.get("/categories", response_model=List[CategoryAchievementResponse])
def get_category_achievements(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """
    Returns detailed progress statistics per category.
    """
    return progress_service.get_category_achievements(db, user_id)

@router.get("/badges", response_model=List[BadgeResponse])
def get_badges_status(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """
    Returns all available badges and indicates which ones the user has achieved.
    """
    return progress_service.get_all_badges_status(db, user_id)

@router.get("/badges/unseen", response_model=List[UnseenBadgeResponse])
def get_unseen_badges(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """
    Returns badges the user has earned but hasn't yet been shown a celebration for.
    Fallback path for the client (e.g. called once on app foreground) in case it missed the
    inline `new_badges` field on a /lessons/{id}/complete response.
    """
    return progress_service.get_unseen_badges(db, user_id)

@router.post("/badges/seen", status_code=status.HTTP_204_NO_CONTENT)
def mark_badges_seen(
    request: MarkBadgesSeenRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user)
):
    """
    Acknowledges that the client has shown the celebration for the given badges.
    """
    progress_service.mark_badges_seen(db, user_id, request.badge_ids)