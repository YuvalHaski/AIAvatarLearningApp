import sys
import os

# Add the root directory to sys.path to allow importing from 'app'
# This ensures the script can run seamlessly from the scripts directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models.domain import Category

# Map the exact Category titles to their new Firebase Storage URLs.
# Replace the placeholder URLs with the actual Download URLs from your Firebase console.
CATEGORY_ICONS_MAPPING = {
    "Job Interview": "https://firebasestorage.googleapis.com/v0/b/learningapp-5ee92.firebasestorage.app/o/Categories%20Icons%2Fcat_job_interview.png?alt=media&token=37cd0608-920c-49a2-bdf8-cb463149d133",
    "Travel Abroad": "https://firebasestorage.googleapis.com/v0/b/learningapp-5ee92.firebasestorage.app/o/Categories%20Icons%2Fcat_travel_abroad.png?alt=media&token=964fbe8c-f6b5-44ea-b1d6-e40ca4264c0d",
    "Emergencies & Safety": "https://firebasestorage.googleapis.com/v0/b/learningapp-5ee92.firebasestorage.app/o/Categories%20Icons%2Fcat_emergencies_and_safety.png?alt=media&token=ab9a39c1-05d4-437c-bcb4-3b584901a6a9",
    "Small Talk & Socializing": "https://firebasestorage.googleapis.com/v0/b/learningapp-5ee92.firebasestorage.app/o/Categories%20Icons%2Fcat_small_talk_and_socializing.png?alt=media&token=934da7af-d923-4701-a26e-c22fbcc434e7",
    "Dining & Ordering Food": "https://firebasestorage.googleapis.com/v0/b/learningapp-5ee92.firebasestorage.app/o/Categories%20Icons%2Fcat_dining_and_ordering_food.png?alt=media&token=92de6f65-9fb6-4336-81e8-2d451ddf36c2",
    "Idioms & Everyday Expressions": "https://firebasestorage.googleapis.com/v0/b/learningapp-5ee92.firebasestorage.app/o/Categories%20Icons%2Fcat_idioms_and_everyday_expressions.png?alt=media&token=d3f65743-89da-4005-bbe3-24cbffea5739",
    "Tech Support": "https://firebasestorage.googleapis.com/v0/b/learningapp-5ee92.firebasestorage.app/o/Categories%20Icons%2Fcat_tech_support.png?alt=media&token=c9036134-0471-4c38-bf4c-9bd4b9d16c35",
    "Public Transport": "https://firebasestorage.googleapis.com/v0/b/learningapp-5ee92.firebasestorage.app/o/Categories%20Icons%2Fcat_public_transport.png?alt=media&token=c8330669-0476-46af-9e54-a25cfeca9fa0"
}


def update_category_icons():
    """
    Connects to the database and updates the 'icon' column for existing Categories.
    This is a safe approach that preserves user progress and relational data.
    """
    db = SessionLocal()

    try:
        print("Starting category icons update...")
        updated_count = 0

        for title, icon_url in CATEGORY_ICONS_MAPPING.items():
            # Query the database for the specific category by its title
            category = db.query(Category).filter(Category.title == title).first()

            if category:
                # Update the icon URL property
                category.icon = icon_url
                updated_count += 1
                print(f"Updated icon for category: '{title}'")
            else:
                print(f"Warning: Category '{title}' not found in the database. Skipping.")

        # Commit all the changes to the database as a single transaction
        db.commit()
        print(f"Process completed successfully! 🎉 {updated_count} categories updated.")

    except Exception as e:
        # Rollback the transaction if any error occurs to maintain data integrity
        db.rollback()
        print(f"An error occurred during the update: {e}")
    finally:
        # Ensure the database session is closed regardless of success or failure
        db.close()


if __name__ == "__main__":
    update_category_icons()