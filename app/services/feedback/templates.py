"""Deterministic template renderer for spoken feedback.

This is the offline fallback for Stage 2: if the fine-tuned model service is
unreachable, the avatar still gets sensible feedback. It is also handy as a
sanity baseline when generating the training dataset.
"""
import random

from app.schemas.asr import ErrorReport, FeedbackPoint

_PRAISE_PASS = ["Great job!", "Well done!", "Nicely done!", "Excellent work!"]
_PRAISE_FAIL = ["Good effort!", "Nice try!", "Almost there!", "You're getting close!"]
_CLOSERS = ["Give it another try.", "Let's try once more.", "Try it again."]


def _render_point(point: FeedbackPoint) -> str | None:
    """Turn one prioritized feedback point into a spoken sentence."""
    if point.kind == "praise":
        return None  # handled by the opening line
    if point.kind == "missing_word":
        return f"You missed the word '{point.word}'."
    if point.kind == "extra_word":
        return f"Try not to add the word '{point.word}'."
    if point.kind == "substitution":
        heard = point.detail or "a different word"
        return f"You {heard} instead of '{point.word}'."
    if point.kind == "mispronunciation":
        sentence = f"Work on how you say '{point.word}'."
        if point.detail:
            sentence += f" Focus on {point.detail}."
        return sentence
    if point.kind == "pattern":
        return f"I noticed {point.detail}."
    if point.kind == "fluency":
        return f"Try to {point.detail}."
    return None


def render_feedback(report: ErrorReport) -> str:
    """Render a full 2-4 sentence spoken feedback string from an ErrorReport."""
    parts: list[str] = []

    opener_pool = _PRAISE_PASS if report.is_passed else _PRAISE_FAIL
    parts.append(random.choice(opener_pool))

    for point in report.feedback_points:
        sentence = _render_point(point)
        if sentence:
            parts.append(sentence)

    if report.is_passed and len(parts) == 1:
        parts.append("That sentence sounded clear and accurate.")
    if not report.is_passed:
        parts.append(random.choice(_CLOSERS))

    return " ".join(parts)
