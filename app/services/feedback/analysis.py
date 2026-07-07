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
    correct_hint_for,
    has_anchor,
    is_vowel,
    plausible_weak_phonemes,
)
from app.services.feedback.word_hints import (
    cluster_hint_for,
    hard_soft_c_for,
    silent_letter_for,
    sound_hint_for_substitution,
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
    "prosody": "vary your tone a little more naturally",
    "accuracy": "enunciate each sound a bit more sharply",
    "completeness": "say every word fully",
}


def _weak_sound_pairs(word: WordResult) -> list[tuple[str, str | None]]:
    """(expected, produced|None) for the phonemes worth coaching, worst
    (lowest accuracy) first so the hint leads with the biggest problem sound.

    A phoneme is coachable when Azure scored it below the weak threshold AND its
    top NBestPhonemes candidate isn't already the expected sound. Using NBest
    this way (a) filters positions the learner actually hit — the low-score
    noise that used to make hints lead with a sound the learner said correctly —
    and (b) records what they produced instead, for a "you said X not Y"
    contrast. `produced` is None when NBest is unavailable (older data /
    unsupported locale), so the hint degrades to a plain anchor."""
    scored: list[tuple[str, str | None, float]] = []
    for ph in word.phonemes:
        if not ph.phoneme or ph.accuracy_score >= PHONEME_WEAK_THRESHOLD:
            continue
        expected = canonical_phoneme(ph.phoneme)
        # Only coach sounds we can name. Drops schwa ('ax'), glides ('y'), and
        # unanchored consonants ('k', ...) so they never pollute weak_phonemes
        # or fabricate a "recurring 'ax' difficulty" pattern.
        if not expected or not has_anchor(expected):
            continue
        produced = canonical_phoneme(ph.candidates[0]) if ph.candidates else ""
        if produced:
            # Azure's best guess of what was produced IS the expected sound: the
            # learner hit the target, the low score is noise — don't coach it.
            if produced == expected:
                continue
            # A vowel slot reported as a consonant (or vice versa) is Azure's
            # segmentation collapsing on a badly-said word, not a real swap
            # (e.g. menu's 'uw' came back as 'n'). Drop it so the hint leads
            # with the genuine sound error instead of this noise.
            if is_vowel(expected) != is_vowel(produced):
                continue
        scored.append((expected, produced or None, ph.accuracy_score))
    scored.sort(key=lambda eps: eps[2])
    seen: set[str] = set()
    pairs: list[tuple[str, str | None]] = []
    for expected, produced, _ in scored:
        if expected in seen:
            continue
        seen.add(expected)
        pairs.append((expected, produced))
    return pairs


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
            pairs = _weak_sound_pairs(word)
            weak = [expected for expected, _ in pairs]
            if pairs:
                # Coach the actually-wrong sounds, naming what was said instead
                # ("the 'e' sound like in 'bed', not the 'a' sound").
                hint = contrast_hint_for(pairs)
            else:
                # Azure flagged the word but no single phoneme is weak-and-wrong
                # (typical for "th", or when every weak phoneme was actually hit).
                # Derive the hint from the difficult sounds the spelling contains
                # so the learner still gets a concrete target. `weak_phonemes`
                # stays Azure-only so pattern detection isn't polluted by these
                # orthographic guesses.
                hint = correct_hint_for(plausible_weak_phonemes(word.word))
            # Suppress words flagged ONLY by the lenient acc<85 threshold that
            # have no coachable sound to point at. Azure scores unstressed
            # function words like "the" in the low 80s on a schwa we can't
            # anchor, which otherwise becomes generic "work on this word"
            # noise. Words Azure itself marks "Mispronunciation" are always
            # kept (it's confident they're wrong) even without a hint.
            soft_only = word.error_type != "Mispronunciation"
            if soft_only and hint is None:
                continue
            result.append(
                MispronouncedWord(
                    word=word.word,
                    accuracy_score=word.accuracy_score,
                    weak_phonemes=weak,
                    correct_hint=hint,
                )
            )
    return result


def _split_spelling_errors(
    mispronounced: list[MispronouncedWord],
) -> tuple[
    list[MispronouncedWord],
    list[SilentLetterError],
    list[HardSoftCError],
    list[ClusterError],
]:
    """Route each flagged word into its most specific coaching category.

    Preference order: silent letter → hard/soft-c → consonant cluster →
    phoneme mispronunciation. Each word is coached in EXACTLY ONE category
    so the avatar doesn't say two things about the same word. Detection
    only fires when Azure already flagged the word — the curated tables
    don't invent errors, they just re-label ones Azure already saw.
    """
    remaining: list[MispronouncedWord] = []
    silent: list[SilentLetterError] = []
    hard_soft: list[HardSoftCError] = []
    clusters: list[ClusterError] = []

    for word in mispronounced:
        sl = silent_letter_for(word.word)
        if sl is not None:
            letter, hint = sl
            silent.append(SilentLetterError(
                word=word.word, silent_letter=letter, hint=hint,
            ))
            continue
        hsc = hard_soft_c_for(word.word)
        if hsc is not None:
            rule, hint = hsc
            hard_soft.append(HardSoftCError(
                word=word.word, rule=rule, hint=hint,
            ))
            continue
        cl = cluster_hint_for(word.word)
        if cl is not None:
            cluster, hint = cl
            clusters.append(ClusterError(
                word=word.word, cluster=cluster, hint=hint,
            ))
            continue
        remaining.append(word)
    return remaining, silent, hard_soft, clusters


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
    # Route flagged words into their most specific category; each word is
    # coached exactly once, in the most specific bucket that matches.
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
