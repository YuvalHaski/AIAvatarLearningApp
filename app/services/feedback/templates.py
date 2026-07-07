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


def render_point(point: FeedbackPoint) -> str | None:
    """Turn one prioritized feedback point into a spoken sentence."""
    if point.kind == "praise":
        return None  # handled by the opening line
    if point.kind == "missing_word":
        return f"You missed the word '{point.word}'."
    if point.kind == "extra_word":
        return f"Try not to add the word '{point.word}'."
    if point.kind == "substitution":
        heard = point.detail or "a different word"
        # Minimal pair: we can't know the learner's intent from audio alone,
        # so phrase it as an acoustic observation ("sounded more like") and
        # attach the sound_hint teaching the ONE sound that distinguishes
        # the pair. General subs (no sound_hint) stay as "you said X instead
        # of Y" — those are real word-level swaps where naming it is enough.
        if point.sound_hint:
            return (
                f"Your '{point.word}' sounded more like '{heard}'. "
                f"Practice {point.sound_hint}."
            )
        return f"You said '{heard}' instead of '{point.word}'."
    if point.kind == "mispronunciation":
        # `detail` is the spoken-friendly anchor hint (correct_hint). Mirror the
        # model's frame so the fallback reads the same: "For 'thought', practice
        # the 'th' sound like in 'thin'." When there's no usable hint, fall back
        # to a generic line rather than fabricate one.
        if point.detail:
            return f"For '{point.word}', practice {point.detail}."
        return f"Work on the way you say '{point.word}'."
    if point.kind in ("silent_letter", "hard_soft_c"):
        # `detail` is a full rule clause, e.g. "the 'k' in 'knife' is silent —
        # say it like 'nife'". Frame with "Remember," so it reads naturally.
        return f"Remember, {point.detail}." if point.detail else None
    if point.kind == "cluster":
        # Same frame as mispronunciation — cluster hints ARE anchor phrases.
        if point.detail:
            return f"For '{point.word}', practice {point.detail}."
        return None
    if point.kind == "pattern":
        return f"I noticed {point.detail}."
    if point.kind == "fluency":
        return f"Try to {point.detail}."
    if point.kind == "polish":
        return f"To make it perfect, {point.detail}."
    return None


def render_feedback(report: ErrorReport) -> str:
    """Render a full 2-4 sentence spoken feedback string from an ErrorReport."""
    parts: list[str] = []

    opener_pool = _PRAISE_PASS if report.is_passed else _PRAISE_FAIL
    parts.append(random.choice(opener_pool))

    for point in report.feedback_points:
        sentence = render_point(point)
        if sentence:
            parts.append(sentence)

    if report.is_passed and len(parts) == 1:
        parts.append("That sentence sounded clear and accurate.")
    if not report.is_passed:
        parts.append(random.choice(_CLOSERS))

    return " ".join(parts)
