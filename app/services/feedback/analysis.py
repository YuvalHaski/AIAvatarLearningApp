"""Stage 1 analyzer: turn an AsrResult into a structured ErrorReport.

This module makes every *decision*: what went wrong, how bad it was, the final
score, pass/fail, and which 2-3 things the feedback should focus on. Stage 2
only puts this into words.
"""
from collections import Counter

from app.schemas.asr import (
    AsrResult,
    ClusterError,
    ErrorReport,
    FeedbackPoint,
    HardSoftCError,
    MispronouncedWord,
    PronunciationScores,
    SilentLetterError,
    Substitution,
    WordResult,
)
from app.services.feedback import scoring
from app.services.feedback.alignment import align, tokenize
from app.services.feedback.phoneme_hints import (
    canonical_phoneme,
    contrast_hint_for,
    has_anchor,
    is_vowel,
)
from app.services.feedback.word_hints import (
    sound_hint_for_substitution,
)

# A word with Azure ErrorType "None" only gets considered below this conservative
# accuracy score. Even then, it is surfaced only when a strongly weak phoneme
# gives us something evidence-based to coach.
WORD_MISPRONOUNCED_THRESHOLD = 75
# A phoneme inside an Azure-confirmed mispronounced word counts as "weak" below
# this accuracy score.
PHONEME_WEAK_THRESHOLD = 80
# For words Azure did not mark as mispronunciations, require much stronger
# phoneme evidence before surfacing feedback.
PHONEME_STRONG_WEAK_THRESHOLD = 65
# Only name the produced sound in a contrast hint when Azure's top
# NBestPhonemes candidate is confident and clearly ahead of the expected sound.
PRODUCED_PHONEME_MIN_SCORE = 70
PRODUCED_PHONEME_MIN_MARGIN = 20
# Product priority: coach these high-value sounds whenever Azure gives direct
# phoneme evidence, even if the whole word score stayed relatively high.
PRIORITY_FEEDBACK_PHONEMES = frozenset({"th", "dh", "r"})
# Priority sounds get a slightly more sensitive detector because they are the
# product focus right now. This uses direct Azure phoneme/NBest evidence only;
# it still does not guess from spelling.
PRIORITY_PHONEME_WEAK_THRESHOLD = 85
PRIORITY_PRODUCED_PHONEME_MIN_SCORE = 70
PRIORITY_PRODUCED_PHONEME_MIN_MARGIN = 10
PRIORITY_CONTRAST_PHONEMES: dict[str, frozenset[str]] = {
    "th": frozenset({"f", "s", "t"}),
    "dh": frozenset({"d", "z", "v"}),
    "r": frozenset({"l", "w"}),
}
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
# Combined safety cap for silent-letter / hard-soft-c / cluster errors. These
# are spelling-driven and rare per attempt; capping them together keeps the
# avatar response short if a sentence happens to be dense with them.
MAX_SPELLING_ERROR_POINTS = 3
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
    "accuracy": "enunciate each sound a bit more sharply",
    "completeness": "say every word fully",
}


def _expected_candidate_score(phoneme, expected: str) -> float:
    """Best NBest score for the expected phoneme, or Azure's expected score."""
    scores = [
        candidate.score
        for candidate in phoneme.candidates
        if canonical_phoneme(candidate.phoneme) == expected
    ]
    return max(scores) if scores else phoneme.accuracy_score


def _confident_produced_phoneme(phoneme, expected: str) -> str | None:
    """Return the produced phoneme only with strong contrast evidence."""
    if not phoneme.candidates:
        return None

    top = phoneme.candidates[0]
    produced = canonical_phoneme(top.phoneme)
    if not produced:
        return None
    if produced == expected:
        return expected
    if is_vowel(expected) != is_vowel(produced):
        return None

    expected_score = _expected_candidate_score(phoneme, expected)
    if (
        top.score >= PRODUCED_PHONEME_MIN_SCORE
        and top.score - expected_score >= PRODUCED_PHONEME_MIN_MARGIN
    ):
        return produced
    return None


def _confident_priority_produced_phoneme(phoneme, expected: str) -> str | None:
    """Like _confident_produced_phoneme, tuned for th/r feedback.

    Priority sounds are the product focus right now, so a smaller margin is
    acceptable. The produced sound still has to be a common, readable
    substitution for that expected sound; otherwise we keep the feedback on the
    target sound only instead of saying something dubious like "not p" for a
    messy "three" attempt.
    """
    if not phoneme.candidates:
        return None

    top = phoneme.candidates[0]
    produced = canonical_phoneme(top.phoneme)
    if not produced:
        return None
    if produced == expected:
        return expected
    if produced not in PRIORITY_CONTRAST_PHONEMES.get(expected, frozenset()):
        return None
    if is_vowel(expected) != is_vowel(produced):
        return None

    expected_score = _expected_candidate_score(phoneme, expected)
    if (
        top.score >= PRIORITY_PRODUCED_PHONEME_MIN_SCORE
        and top.score - expected_score >= PRIORITY_PRODUCED_PHONEME_MIN_MARGIN
    ):
        return produced
    return None


def _weak_sound_pairs(
    word: WordResult,
    *,
    threshold: float = PHONEME_WEAK_THRESHOLD,
    include_threshold: bool = False,
) -> list[tuple[str, str | None]]:
    """(expected, produced|None) for the phonemes worth coaching, worst
    (lowest accuracy) first so the hint leads with the biggest problem sound.

    A phoneme is coachable when Azure scored it below the weak threshold and we
    have an anchor for the expected sound. NBestPhonemes is used conservatively:
    if the top candidate is the expected sound, the low score is treated as
    noise; if a different top candidate is confident enough, we include it for a
    contrast hint; otherwise `produced` is None and the hint stays focused on
    the correct sound only."""
    scored: list[tuple[str, str | None, float]] = []
    for ph in word.phonemes:
        is_weak = (
            ph.accuracy_score <= threshold
            if include_threshold
            else ph.accuracy_score < threshold
        )
        if not ph.phoneme or not is_weak:
            continue
        expected = canonical_phoneme(ph.phoneme)
        # Only coach sounds we can name. Drops schwa ('ax'), glides ('y'), and
        # unanchored consonants ('k', ...) so they never pollute weak_phonemes
        # or fabricate a "recurring 'ax' difficulty" pattern.
        if not expected or not has_anchor(expected):
            continue
        produced = _confident_produced_phoneme(ph, expected)
        # Azure's best guess of what was produced IS the expected sound: the
        # learner hit the target, so the low score is probably noise.
        if produced == expected:
            continue
        scored.append((expected, produced, ph.accuracy_score))
    scored.sort(key=lambda eps: eps[2])
    seen: set[str] = set()
    pairs: list[tuple[str, str | None]] = []
    for expected, produced, _ in scored:
        if expected in seen:
            continue
        seen.add(expected)
        pairs.append((expected, produced))
    return pairs


def _filter_priority_sound_pairs(
    pairs: list[tuple[str, str | None]]
) -> list[tuple[str, str | None]]:
    """Only the product-priority sounds we want to coach first: th and r."""
    return [
        pair for pair in pairs
        if pair[0] in PRIORITY_FEEDBACK_PHONEMES
    ]


def _priority_sound_pairs(word: WordResult) -> list[tuple[str, str | None]]:
    """Direct detector for priority th/r sounds.

    This intentionally runs before the generic word-score detector. It catches
    cases where Azure keeps `ErrorType=None` and the word score is acceptable,
    but the priority phoneme itself or its NBest candidates clearly show trouble.
    """
    scored: list[tuple[str, str | None, float]] = []
    for ph in word.phonemes:
        expected = canonical_phoneme(ph.phoneme)
        if expected not in PRIORITY_FEEDBACK_PHONEMES or not has_anchor(expected):
            continue

        produced = _confident_priority_produced_phoneme(ph, expected)
        if produced == expected:
            continue

        if produced and ph.accuracy_score <= PRIORITY_PHONEME_WEAK_THRESHOLD:
            scored.append((expected, produced, ph.accuracy_score))
            continue

        if ph.accuracy_score <= PRIORITY_PHONEME_WEAK_THRESHOLD:
            scored.append((expected, None, ph.accuracy_score))

    scored.sort(key=lambda eps: eps[2])
    seen: set[str] = set()
    pairs: list[tuple[str, str | None]] = []
    for expected, produced, _ in scored:
        if expected in seen:
            continue
        seen.add(expected)
        pairs.append((expected, produced))
    return pairs


def _has_priority_sound(word: MispronouncedWord) -> bool:
    return any(ph in PRIORITY_FEEDBACK_PHONEMES for ph in word.weak_phonemes)


def _mispronounced_word_from_pairs(
    word: WordResult,
    pairs: list[tuple[str, str | None]],
) -> MispronouncedWord:
    weak = [expected for expected, _ in pairs]
    target_only_pairs = [(expected, None) for expected, _ in pairs]
    return MispronouncedWord(
        word=word.word,
        accuracy_score=word.accuracy_score,
        weak_phonemes=weak,
        correct_hint=contrast_hint_for(target_only_pairs) if pairs else None,
    )


def _find_mispronounced(words: list[WordResult]) -> list[MispronouncedWord]:
    """Words Azure flagged as mispronounced (or that scored low overall)."""
    result: list[MispronouncedWord] = []
    for word in words:
        # Omission / Insertion are handled by text alignment, not here.
        if word.error_type in ("Omission", "Insertion"):
            continue

        priority_pairs = _priority_sound_pairs(word)
        if priority_pairs:
            result.append(_mispronounced_word_from_pairs(word, priority_pairs))
            continue

        if word.error_type == "Mispronunciation":
            pairs = _weak_sound_pairs(word, threshold=PHONEME_WEAK_THRESHOLD)
            result.append(_mispronounced_word_from_pairs(
                word,
                _filter_priority_sound_pairs(pairs) or pairs,
            ))
            continue

        if (
            word.error_type == "None"
            and word.accuracy_score < WORD_MISPRONOUNCED_THRESHOLD
        ):
            pairs = _weak_sound_pairs(
                word,
                threshold=PHONEME_STRONG_WEAK_THRESHOLD,
                include_threshold=True,
            )
            if not pairs:
                continue
            result.append(_mispronounced_word_from_pairs(
                word,
                _filter_priority_sound_pairs(pairs) or pairs,
            ))

    priority_result = [word for word in result if _has_priority_sound(word)]
    return priority_result or result


def _split_spelling_errors(
    mispronounced: list[MispronouncedWord],
) -> tuple[
    list[MispronouncedWord],
    list[SilentLetterError],
    list[HardSoftCError],
    list[ClusterError],
]:
    """Keep spelling-rule buckets empty unless explicit evidence is available.

    The current ASR result tells us that a word was mispronounced, but not that a
    silent letter was pronounced, a hard/soft-c rule was confused, or a cluster
    was simplified. Curated spelling tables are useful hints for future explicit
    detectors, but table membership alone is not evidence. Until the analyzer
    receives that evidence, keep these as generic mispronunciations.
    """
    return mispronounced, [], [], []


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
    silent_letters: list[SilentLetterError],
    hard_soft_c: list[HardSoftCError],
    clusters: list[ClusterError],
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
        # `detail` carries just the heard word now — templates.py picks the
        # framing sentence based on whether sound_hint is populated. Minimal
        # pairs (sound_hint set) render as "Your 'X' sounded more like 'Y'";
        # general subs render as "You said 'Y' instead of 'X'".
        framing.append(
            FeedbackPoint(
                kind="substitution", priority=2, word=sub.expected,
                detail=sub.heard,
                sound_hint=sub.sound_hint,
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

    # Spelling-driven errors share one combined cap. Priority 3 puts them
    # alongside phoneme mispronunciations so the avatar names all specific
    # word issues before generic framing.
    spelling_points: list[FeedbackPoint] = []
    for err in silent_letters:
        spelling_points.append(FeedbackPoint(
            kind="silent_letter", priority=3, word=err.word, detail=err.hint,
        ))
    for err in hard_soft_c:
        spelling_points.append(FeedbackPoint(
            kind="hard_soft_c", priority=3, word=err.word, detail=err.hint,
        ))
    for err in clusters:
        spelling_points.append(FeedbackPoint(
            kind="cluster", priority=3, word=err.word, detail=err.hint,
        ))
    spelling_points = spelling_points[:MAX_SPELLING_ERROR_POINTS]

    points = framing + mispron_points + spelling_points

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
        Substitution(
            expected=op.target,
            heard=op.recognized,
            # If the (expected, heard) pair is a curated minimal pair, attach
            # the sound_hint so the avatar can coach the ONE distinguishing
            # phoneme instead of just naming the swap. None otherwise.
            sound_hint=sound_hint_for_substitution(op.target, op.recognized),
        )
        for op in ops
        if op.kind == "substitution"
    ]

    all_mispronounced = _find_mispronounced(asr.words)
    # Keep spelling-rule buckets empty until Stage 1 has explicit evidence for
    # them; table membership alone is not enough to relabel a mispronunciation.
    mispronounced, silent_letters, hard_soft_c, clusters = _split_spelling_errors(
        all_mispronounced,
    )
    # Patterns only track pure phoneme-level recurring difficulties. Spelling
    # errors don't contribute so we don't invent a fake "recurring 'k' pattern"
    # from silent-letter words.
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
        silent_letters=silent_letters,
        hard_soft_c=hard_soft_c,
        clusters=clusters,
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
        silent_letter_errors=silent_letters,
        hard_soft_c_errors=hard_soft_c,
        cluster_errors=clusters,
        patterns=patterns,
        final_score=final_score,
        is_passed=is_passed,
        feedback_points=feedback_points,
    )
