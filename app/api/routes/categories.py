from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.schemas.category import CategoryResponse, CategoryDetailsResponse
from app.services import category_service

# ==========================================
# CATEGORIES ROUTER
# ==========================================
# The 'prefix' means all routes here will automatically start with "/categories"
# The 'tags' group them beautifully in the Swagger documentation

router = APIRouter(prefix="/categories", tags=["Categories"])

# TODO: Replace this with actual user ID from JWT authentication token later
CURRENT_USER_ID = "dummy-user-id"


@router.get("/", response_model=List[CategoryResponse])
def get_categories(db: Session = Depends(get_db)):
    """
    Returns a list of all categories along with the user's specific progress.
    Used for the Home Screen.
    """
    return category_service.get_all_categories_with_progress(db, CURRENT_USER_ID)


@router.get("/{category_id}", response_model=CategoryDetailsResponse)
def get_category_details(category_id: str, db: Session = Depends(get_db)):
    """
    Returns the details of a specific category, including all its lessons
    and the user's progress in each lesson.
    """
    result = category_service.get_category_details(db, category_id, CURRENT_USER_ID)

    # If the service returns None, we throw a standard 404 Not Found error
    if not result:
        raise HTTPException(status_code=404, detail=f"Category with ID {category_id} not found.")

    return result