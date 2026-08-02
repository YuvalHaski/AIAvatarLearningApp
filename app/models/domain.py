from sqlalchemy import Column, String, Integer, Float, Boolean, ForeignKey, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

# Import Base from our new core database file
from app.core.database import Base


# ==========================================
# ENUMS
# ==========================================

class DifficultyEnum(enum.Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class ProgressStatusEnum(enum.Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"


# ==========================================
# CORE TABLES
# ==========================================

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    display_name = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Category(Base):
    __tablename__ = "categories"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String)
    icon = Column(String)
    # Stable machine identifier (e.g. "travel_abroad") used by badge rule logic,
    # decoupled from the display `title` so copy/localization changes never break earning logic.
    category_code = Column(String, unique=True, nullable=False, index=True)

    # Relationship to Lessons
    lessons = relationship("Lesson", back_populates="category", cascade="all, delete-orphan")


class Lesson(Base):
    __tablename__ = "lessons"

    id = Column(String, primary_key=True, index=True)
    category_id = Column(String, ForeignKey("categories.id"), nullable=False)
    title = Column(String, nullable=False)
    description = Column(String)
    difficulty = Column(Enum(DifficultyEnum), default=DifficultyEnum.EASY)

    # Relationships
    category = relationship("Category", back_populates="lessons")
    sentences = relationship("Sentence", back_populates="lesson", cascade="all, delete-orphan")


class Sentence(Base):
    __tablename__ = "sentences"

    id = Column(String, primary_key=True, index=True)
    lesson_id = Column(String, ForeignKey("lessons.id"), nullable=False)
    text = Column(String, nullable=False)
    order_index = Column(Integer, nullable=False)

    # Relationship
    lesson = relationship("Lesson", back_populates="sentences")


class Badge(Base):
    __tablename__ = "badges"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(String)
    order_index = Column(Integer, nullable=False)
    # Stable machine identifier (e.g. "perfect_score") used by the badge rule engine,
    # decoupled from the display `title` so copy/localization changes never break earning logic.
    badge_code = Column(String, unique=True, nullable=False, index=True)


# ==========================================
# PROGRESS TABLES
# ==========================================

class UserLessonProgress(Base):
    __tablename__ = "user_lesson_progress"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    lesson_id = Column(String, ForeignKey("lessons.id"), primary_key=True)
    status = Column(Enum(ProgressStatusEnum), default=ProgressStatusEnum.NOT_STARTED)

    # Tracks the UUID of the currently active session/run for this lesson.
    # Nullable because a lesson might be completed or not yet started.
    current_run_id = Column(String, nullable=True)

    last_practiced_at = Column(DateTime(timezone=True), onupdate=func.now())


class SentenceAttemptHistory(Base):
    """
    Append-only table to track every pronunciation attempt made by the user.
    """
    __tablename__ = "sentence_attempt_history"

    # Unique identifier for each individual attempt
    id = Column(String, primary_key=True, index=True)

    # Foreign keys with indexes for fast querying and aggregation
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    lesson_id = Column(String, ForeignKey("lessons.id"), nullable=False, index=True)
    sentence_id = Column(String, ForeignKey("sentences.id"), nullable=False, index=True)

    # The ID linking this attempt to a specific lesson run (matches current_run_id)
    run_id = Column(String, nullable=False, index=True)

    score = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserBadge(Base):
    __tablename__ = "user_badges"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    badge_id = Column(String, ForeignKey("badges.id"), primary_key=True)
    achieved_at = Column(DateTime(timezone=True), server_default=func.now())
    # False until the client has acknowledged/shown the "badge earned" celebration,
    # via GET /progress/badges/unseen + POST /progress/badges/seen.
    is_seen = Column(Boolean, nullable=False, server_default="false")