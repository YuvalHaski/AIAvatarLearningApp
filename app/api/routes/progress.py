from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.schemas.progress import OverviewDataResponse, CategoryAchievementResponse
from app.schemas.badge import BadgeResponse
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