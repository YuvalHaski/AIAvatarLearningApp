from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.schemas.category import CategoryResponse, CategoryDetailsResponse
from app.services import category_service
from app.api.auth import get_current_user

# ==========================================
# CATEGORIES ROUTER
# ==========================================

router = APIRouter(prefix="/categories", tags=["Categories"])


@router.get("/", response_model=List[CategoryResponse])
def get_categories(
        db: Session = Depends(get_db),
        user_id: str = Depends(get_current_user)
):
    """
    Returns a list of all categories along with the user's specific progress.
    Requires a valid Firebase ID Token in the Authorization header.
    """
    return category_service.get_all_categories_with_progress(db, user_id)


@router.get("/{category_id}", response_model=CategoryDetailsResponse)
def get_category_details(
        category_id: str,
        db: Session = Depends(get_db),
        user_id: str = Depends(get_current_user)
):
    """
    Returns the details of a specific category, including all its lessons
    and the user's progress in each lesson.
    """
    result = category_service.get_category_details(db, category_id, user_id)

    if not result:
        raise HTTPException(status_code=404, detail=f"Category with ID {category_id} not found.")

    return result