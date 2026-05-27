"""Stage 1 analyzer: turn an AsrResult into a structured ErrorReport.

This module makes every *decision*: what went wrong, how bad it was, the final
score, pass/fail, and which 2-3 things the feedback should focus on. Stage 2
only puts this into words.
"""
from collections import Counter

from app.schemas.asr import (
    AsrResult,
    ErrorReport,
    FeedbackPoint,
    MispronouncedWord,
    Substitution,
    WordResult,
)
from app.services.feedback import scoring
from app.services.feedback.alignment import align, tokenize

# A word counts as mispronounced below this accuracy score (0-100).
WORD_MISPRONOUNCED_THRESHOLD = 60
# A phoneme inside a word counts as "weak" below this accuracy score.
PHONEME_WEAK_THRESHOLD = 50
# Azure fluency score below this is worth mentioning.
FLUENCY_LOW_THRESHOLD = 60
# A weak phoneme seen in this many words becomes a "pattern".
PATTERN_MIN_OCCURRENCES = 2
# Most feedback points to surface, so we don't overwhelm the learner.
MAX_FEEDBACK_POINTS = 3


def _weak_phonemes(word: WordResult) -> list[str]:
    return [
        ph.phoneme
        for ph in word.phonemes
        if ph.accuracy_score < PHONEME_WEAK_THRESHOLD and ph.phoneme
    ]


def _find_mispronounced(words: list[WordResult]) -> list[MispronouncedWord]:
    """Words Azure flagged as mispronounced (or that scored low overall)."""
    result: list[MispronouncedWord] = []
    for word in words:
        is_mispronounced = (
            word.error_type == "Mispronunciation"
            or word.accuracy_score < WORD_MISPRONOUNCED_THRESHOLD
        )
        # Omission / Insertion are handled by text alignment, not here.
        if is_mispronounced and word.error_type not in ("Omission", "Insertion"):
            result.append(
                MispronouncedWord(
                    word=word.word,
                    accuracy_score=word.accuracy_score,
                    weak_phonemes=_weak_phonemes(word),
                )
            )
    return result


def _detect_patterns(mispronounced: list[MispronouncedWord]) -> list[str]:
    """A phoneme that fails across multiple words is a recurring pattern."""
    counts: Counter[str] = Counter()
    for word in mispronounced:
        for phoneme in {ph.lower() for ph in word.weak_phonemes}:
            counts[phoneme] += 1
    return [
        f"recurring difficulty with the '{phoneme}' sound"
        for phoneme, count in counts.items()
        if count >= PATTERN_MIN_OCCURRENCES
    ]


def _select_feedback_points(
    *,
    is_passed: bool,
    missing_words: list[str],
    substitutions: list[Substitution],
    mispronounced: list[MispronouncedWord],
    extra_words: list[str],
    patterns: list[str],
    fluency: float,
) -> list[FeedbackPoint]:
    """Pick and prioritize the handful of points worth speaking about."""
    candidates: list[FeedbackPoint] = []

    for word in missing_words:
        candidates.append(FeedbackPoint(kind="missing_word", priority=1, word=word))
    for pattern in patterns:
        candidates.append(FeedbackPoint(kind="pattern", priority=2, detail=pattern))
    for sub in substitutions:
        candidates.append(
            FeedbackPoint(
                kind="substitution", priority=2, word=sub.expected,
                detail=f"said '{sub.heard}'",
            )
        )
    for word in mispronounced:
        detail = (
            "the " + ", ".join(f"'{p}'" for p in word.weak_phonemes) + " sound"
            if word.weak_phonemes
            else None
        )
        candidates.append(
            FeedbackPoint(kind="mispronunciation", priority=3, word=word.word, detail=detail)
        )
    for word in extra_words:
        candidates.append(FeedbackPoint(kind="extra_word", priority=4, word=word))
    if fluency < FLUENCY_LOW_THRESHOLD:
        candidates.append(
            FeedbackPoint(kind="fluency", priority=5, detail="speak smoothly, with fewer pauses")
        )

    candidates.sort(key=lambda p: p.priority)

    if not candidates:
        return [FeedbackPoint(kind="praise", priority=0)]

    points = candidates[:MAX_FEEDBACK_POINTS]
    # When the learner passed, lead with encouragement before the nitpicks.
    if is_passed:
        points = [FeedbackPoint(kind="praise", priority=0)] + points[: MAX_FEEDBACK_POINTS - 1]
    return points


def analyze(target_sentence: str, asr: AsrResult) -> ErrorReport:
    """Build the full deterministic ErrorReport for one attempt."""
    target_tokens = tokenize(target_sentence)
    recognized_tokens = tokenize(asr.recognized_text)
    ops = align(target_tokens, recognized_tokens)

    missing_words = [op.target for op in ops if op.kind == "missing"]
    extra_words = [op.recognized for op in ops if op.kind == "extra"]
    substitutions = [
        Substitution(expected=op.target, heard=op.recognized)
        for op in ops
        if op.kind == "substitution"
    ]

    mispronounced = _find_mispronounced(asr.words)
    patterns = _detect_patterns(mispronounced)

    final_score = scoring.compute_score(
        asr.scores,
        num_substitutions=len(substitutions),
        num_extra_words=len(extra_words),
    )
    is_passed = scoring.is_passing(final_score)

    feedback_points = _select_feedback_points(
        is_passed=is_passed,
        missing_words=missing_words,
        substitutions=substitutions,
        mispronounced=mispronounced,
        extra_words=extra_words,
        patterns=patterns,
        fluency=asr.scores.fluency,
    )

    return ErrorReport(
        target_sentence=target_sentence,
        recognized_text=asr.recognized_text,
        scores=asr.scores,
        missing_words=missing_words,
        extra_words=extra_words,
        substitutions=substitutions,
        mispronounced_words=mispronounced,
        patterns=patterns,
        final_score=final_score,
        is_passed=is_passed,
        feedback_points=feedback_points,
    )
