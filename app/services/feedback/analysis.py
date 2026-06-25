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
    PronunciationScores,
    Substitution,
    WordResult,
)
from app.services.feedback import scoring
from app.services.feedback.alignment import align, tokenize
from app.services.feedback.phoneme_hints import (
    canonical_phoneme,
    correct_hint_for,
    plausible_weak_phonemes,
)

# A word counts as mispronounced below this accuracy score (0-100). Azure's
# scripted assessment is very lenient — audibly accented words routinely score
# in the high 70s / low 80s with ErrorType "None" — so this is set high (85) to
# surface them. The trade-off is more false positives on accented-but-acceptable
# speech; this is a deliberate sensitivity choice, not a neutral default.
WORD_MISPRONOUNCED_THRESHOLD = 85
# A phoneme inside a word counts as "weak" below this accuracy score. Kept in
# step with the lenient word cutoff above so a flagged word actually has weak
# phonemes to point at (e.g. a barely-rolled "r" that Azure scores ~78).
PHONEME_WEAK_THRESHOLD = 80
# Azure fluency score below this is worth mentioning.
FLUENCY_LOW_THRESHOLD = 60
# A weak phoneme seen in this many words becomes a "pattern".
PATTERN_MIN_OCCURRENCES = 2
# Cap for "framing" feedback points (missing / extra / substitution /
# pattern / fluency). Mispronunciations are NOT subject to this cap —
# they are always all surfaced, since dropping them is the bug that this
# whole module was reworked to fix.
MAX_FRAMING_POINTS = 3
# Safety cap on mispronunciations per attempt so the avatar response stays
# a reasonable length even on a totally garbled sentence.
MAX_MISPRONUNCIATION_POINTS = 4
# When an attempt has no word-level errors but `final_score < 100`, the gap
# comes from one of Azure's sub-scores being below this value. We pick the
# weakest such sub-score and surface a single "polish" tip explaining why
# the score wasn't perfect. Sub-scores at or above this are not worth
# pointing out — the learner won't perceive the difference.
SUBSCORE_POLISH_THRESHOLD = 95

# Spoken-friendly polish tips per Azure sub-score dimension. Used both as
# the `detail` of a `polish` FeedbackPoint and (via that point) as the
# `polish_tip` field sent to the Stage 2 model.
_POLISH_TIPS: dict[str, str] = {
    "fluency": "speak with a smoother flow, with fewer pauses",
    "prosody": "vary your tone a little more naturally",
    "accuracy": "enunciate each sound a bit more sharply",
    "completeness": "say every word fully",
}


def _weak_phonemes(word: WordResult) -> list[str]:
    """Canonical codes for the phonemes Azure scored below the weak threshold,
    worst (lowest accuracy) first so the hint leads with the biggest problem
    sound instead of whichever weak sound happens to come first in the word."""
    scored = [
        (canonical_phoneme(ph.phoneme), ph.accuracy_score)
        for ph in word.phonemes
        if ph.accuracy_score < PHONEME_WEAK_THRESHOLD and ph.phoneme
    ]
    scored.sort(key=lambda cs: cs[1])
    ordered: list[str] = []
    seen: set[str] = set()
    for code, _ in scored:
        if code and code not in seen:
            seen.add(code)
            ordered.append(code)
    return ordered


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
            weak = _weak_phonemes(word)
            # Azure sometimes flags a word without any single phoneme dropping
            # below the weak threshold (typical for "th"). Derive the hint from
            # the difficult sounds the spelling contains so the learner still
            # gets a concrete target instead of a generic "work on this word".
            # `weak_phonemes` itself stays Azure-only so pattern detection isn't
            # polluted by orthographic guesses.
            hint_phonemes = weak or plausible_weak_phonemes(word.word)
            result.append(
                MispronouncedWord(
                    word=word.word,
                    accuracy_score=word.accuracy_score,
                    weak_phonemes=weak,
                    correct_hint=correct_hint_for(hint_phonemes),
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


def _polish_point(scores: PronunciationScores) -> FeedbackPoint | None:
    """When an attempt has no flagged errors but `final_score < 100`, pick
    the weakest sub-score below SUBSCORE_POLISH_THRESHOLD and turn it into
    a single 'polish' tip explaining the gap to the learner. Returns None
    if every sub-score is at the threshold or above (truly perfect)."""
    candidates = [
        (name, getattr(scores, name)) for name in _POLISH_TIPS
    ]
    weak = [(n, v) for n, v in candidates if v < SUBSCORE_POLISH_THRESHOLD]
    if not weak:
        return None
    name, _ = min(weak, key=lambda nv: nv[1])
    return FeedbackPoint(kind="polish", priority=6, detail=_POLISH_TIPS[name])


def _select_feedback_points(
    *,
    is_passed: bool,
    final_score: int,
    scores: PronunciationScores,
    missing_words: list[str],
    substitutions: list[Substitution],
    mispronounced: list[MispronouncedWord],
    extra_words: list[str],
    patterns: list[str],
    fluency: float,
) -> list[FeedbackPoint]:
    """Pick and prioritize the points worth speaking about.

    Mispronunciations are surfaced separately from the other (framing)
    issues and are never evicted by them. The framing items are still
    prioritized + capped so the avatar doesn't read a paragraph, but
    every mispronounced word (up to a safety cap) makes it through.
    """
    framing: list[FeedbackPoint] = []

    for word in missing_words:
        framing.append(FeedbackPoint(kind="missing_word", priority=1, word=word))
    for pattern in patterns:
        framing.append(FeedbackPoint(kind="pattern", priority=2, detail=pattern))
    for sub in substitutions:
        framing.append(
            FeedbackPoint(
                kind="substitution", priority=2, word=sub.expected,
                detail=f"said '{sub.heard}'",
            )
        )
    for word in extra_words:
        framing.append(FeedbackPoint(kind="extra_word", priority=4, word=word))
    if fluency < FLUENCY_LOW_THRESHOLD:
        framing.append(
            FeedbackPoint(kind="fluency", priority=5, detail="speak smoothly, with fewer pauses")
        )

    framing.sort(key=lambda p: p.priority)
    framing = framing[:MAX_FRAMING_POINTS]

    mispron_points: list[FeedbackPoint] = []
    for word in mispronounced[:MAX_MISPRONUNCIATION_POINTS]:
        # Prefer the spoken anchor hint; fall back to a raw phoneme list when
        # no anchor is available so the model still has something to say.
        detail = word.correct_hint or (
            "the " + ", ".join(f"'{p}'" for p in word.weak_phonemes) + " sound"
            if word.weak_phonemes
            else None
        )
        mispron_points.append(
            FeedbackPoint(kind="mispronunciation", priority=3, word=word.word, detail=detail)
        )

    points = framing + mispron_points

    # No flagged errors but the score isn't perfect — surface a polish tip
    # so the learner can hear *why* they didn't get 100.
    if not points and final_score < 100:
        polish = _polish_point(scores)
        if polish is not None:
            points.append(polish)

    if not points:
        return [FeedbackPoint(kind="praise", priority=0)]
    if is_passed:
        return [FeedbackPoint(kind="praise", priority=0), *points]
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
        final_score=final_score,
        scores=asr.scores,
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
