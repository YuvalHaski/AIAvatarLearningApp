from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.schemas.progress import OverviewDataResponse, CategoryAchievementResponse
from app.schemas.badge import BadgeResponse
from app.services import progress_service

# ==========================================
# PROGRESS ROUTER
# ==========================================

router = APIRouter(prefix="/progress", tags=["Progress"])

# TODO: Replace with real authenticated user ID
CURRENT_USER_ID = "dummy-user-id"

@router.get("/overview", response_model=OverviewDataResponse)
def get_overview_data(db: Session = Depends(get_db)):
    """
    Returns the high-level overview data (Average score, streak, recent badges).
    Used for the first tab in the Progress Screen.
    """
    return progress_service.get_overview_data(db, CURRENT_USER_ID)

@router.get("/categories", response_model=List[CategoryAchievementResponse])
def get_category_achievements(db: Session = Depends(get_db)):
    """
    Returns detailed progress statistics per category.
    Used for the second tab in the Progress Screen.
    """
    return progress_service.get_category_achievements(db, CURRENT_USER_ID)

@router.get("/badges", response_model=List[BadgeResponse])
def get_badges_status(db: Session = Depends(get_db)):
    """
    Returns all available badges and indicates which ones the user has achieved.
    Used for the third tab in the Progress Screen.
    """
    return progress_service.get_all_badges_status(db, CURRENT_USER_ID)