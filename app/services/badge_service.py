"""Badge evaluation & awarding.

`evaluate_and_award_badges` is the single entry point that turns badge_rules predicates into
persisted `UserBadge` rows. It is intentionally decoupled from *when* it's called - today the
only caller is lesson_service.complete_lesson (a lesson finishing is the one meaningful
"achievement" event in the app), but any state-changing event could call it safely since it is
idempotent (already-earned badges are always skipped).
"""
import logging
from typing import Optional

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.domain import Badge, UserBadge
from app.services.badge_rules import BADGE_RULES, META_BADGE_CODES

logger = logging.getLogger(__name__)


def _earned_badge_ids(db: Session, user_id: str) -> set:
    rows = db.query(UserBadge.badge_id).filter(UserBadge.user_id == user_id).all()
    return {row[0] for row in rows}


def _award(db: Session, user_id: str, badge: Badge) -> bool:
    """Persist one UserBadge row. Returns True if newly awarded.

    Commits immediately (rather than batching) so that a race between two concurrent requests
    awarding the same badge to the same user - e.g. the same lesson completed from two devices
    at once - only rolls back the single losing insert (via the UserBadge composite primary key),
    never any badge already awarded earlier in this same evaluation pass.
    """
    db.add(UserBadge(user_id=user_id, badge_id=badge.id, is_seen=False))
    try:
        db.commit()
        return True
    except IntegrityError:
        db.rollback()
        return False


def evaluate_and_award_badges(
    db: Session, user_id: str, *, context: Optional[dict] = None
) -> list[Badge]:
    """
    Evaluate the full badge catalog against the user's current progress and award any newly
    satisfied badges. Returns the badges newly awarded during this call (empty list if none).

    Runs in two passes: ordinary badges first, then META_BADGE_CODES (badges whose criteria
    depends on how many *other* badges the user has earned, e.g. "Badge Collector") - so a single
    event that crosses both an ordinary threshold and the meta threshold awards both in one call.
    """
    context = context or {}
    all_badges = db.query(Badge).all()
    earned_ids = _earned_badge_ids(db, user_id)
    newly_awarded: list[Badge] = []

    def run_pass(badges: list) -> None:
        for badge in badges:
            if badge.id in earned_ids:
                continue
            rule = BADGE_RULES.get(badge.badge_code)
            if rule is None:
                logger.warning("No badge rule registered for badge_code=%r", badge.badge_code)
                continue
            if rule(db, user_id, context) and _award(db, user_id, badge):
                earned_ids.add(badge.id)
                newly_awarded.append(badge)

    ordinary = [b for b in all_badges if b.badge_code not in META_BADGE_CODES]
    meta = [b for b in all_badges if b.badge_code in META_BADGE_CODES]

    run_pass(ordinary)
    run_pass(meta)

    return newly_awarded
