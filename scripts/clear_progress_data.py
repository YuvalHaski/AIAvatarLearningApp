import sys
import os

# Add the root directory to sys.path to allow importing from 'app'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models.domain import UserLessonProgress, SentenceAttemptHistory

def clear_progress_data():
    """
    Clears all progress and attempt history data to allow a clean restart.
    """
    db = SessionLocal()

    try:
        print("Starting to clear progress data...")

        # 1. Clear SentenceAttemptHistory first (often has many records)
        attempts_count = db.query(SentenceAttemptHistory).count()
        db.query(SentenceAttemptHistory).delete()
        print(f"Deleted {attempts_count} rows from SentenceAttemptHistory.")

        # 2. Clear UserLessonProgress
        progress_count = db.query(UserLessonProgress).count()
        db.query(UserLessonProgress).delete()
        print(f"Deleted {progress_count} rows from UserLessonProgress.")

        # Commit changes
        db.commit()
        print("Database progress tables cleared successfully! 🎉")

    except Exception as e:
        # Rollback in case of error to keep the database consistent
        db.rollback()
        print(f"An error occurred during the clearing process: {e}")
    finally:
        # Close the session
        db.close()

if __name__ == "__main__":
    clear_progress_data()