from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from firebase_admin import auth

from app.core.database import get_db
from app.models.domain import User

# ==========================================
# SECURITY SCHEME CONFIGURATION
# ==========================================
# This initializes FastAPI's built-in HTTPBearer security scheme.
# It automatically looks for an 'Authorization' header with the format 'Bearer <token>'.
# If the header is missing or malformed, FastAPI automatically rejects the request (403 Forbidden).
security = HTTPBearer()


def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
) -> str:
    """
    Core authentication dependency for protecting API routes.

    This function intercepts incoming requests, validates the provided Firebase ID Token,
    and synchronizes the authenticated user with our PostgreSQL database.

    Args:
        credentials (HTTPAuthorizationCredentials): The parsed Bearer token from the request header.
        db (Session): The active database session.

    Returns:
        str: The authenticated user's ID (which matches the Firebase UID).

    Raises:
        HTTPException (401): If the token is invalid, expired, or verification fails.
    """
    # Extract the actual JWT string from the credentials object
    token = credentials.credentials

    try:
        # 1. Verify the Token via Firebase Admin SDK
        # This contacts Google's servers to ensure the token is authentic, correctly signed, and not expired. Allows up to 10 seconds of time difference between the device and the server
        # It throws an exception if verification fails.
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=10)

        # 2. Extract User Data from the Decoded Token
        uid = decoded_token.get("uid")
        email = decoded_token.get("email")
        name = decoded_token.get("name", "")  # May be empty depending on the sign-up method

        # 3. Check for User Existence in Our Database
        # We query the DB to see if we already have a record for this Firebase UID.
        user = db.query(User).filter(User.id == uid).first()

        # 4. Synchronize/Create User if Missing
        # If this is the user's first time accessing the API after signing up via Firebase,
        # we silently create their profile in our DB to ensure foreign key constraints (like progress) work.
        if not user:
            user = User(
                id=uid,
                email=email,
                display_name=name
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        # 5. Return the Authenticated User ID
        # The calling router can now safely use this ID to fetch user-specific data.
        return user.id

    except Exception as e:
        # Catch-all for any verification failures (expired token, malicious payload, etc.)
        #raise HTTPException(
        #    status_code=status.HTTP_401_UNAUTHORIZED,
        #    detail="Invalid authentication credentials or token expired.",
        #    headers={"WWW-Authenticate": "Bearer"},
        #)
        # BEST PRACTICE DEBUGGING: Print the exact error to the server console!
        print(f"AUTHENTICATION CRASH: {str(e)}")
        import traceback
        traceback.print_exc()  # This will print the exact line number that failed

        # Catch-all for any verification failures (expired token, malicious payload, etc.)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Auth Failed: {str(e)}",  # Sending the error to the Android App too!
            headers={"WWW-Authenticate": "Bearer"},
        )