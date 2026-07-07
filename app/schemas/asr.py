"""Schemas for the pronunciation assessment flow.

These models are used both internally (between asr_service -> Stage 1 -> Stage 2)
and at the API boundary, so everything here is JSON-serializable.
"""
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ==========================================================================
# ASR layer  -  produced by asr_service from Azure's response
# ==========================================================================

class PhonemeCandidate(BaseModel):
    """One Azure NBestPhonemes candidate and its confidence score."""
    phoneme: str
    score: float


class PhonemeScore(BaseModel):
    """A single (expected) phoneme and how accurately it was pronounced (0-100)."""
    phoneme: str
    accuracy_score: float
    # Azure NBestPhonemes: the phonemes Azure thinks the learner ACTUALLY
    # produced at this position, highest-confidence first. Empty when NBest is
    # off or the locale doesn't support it. When the top entry differs from
    # `phoneme`, the learner may have substituted a sound. The candidate score
    # lets Stage 1 require strong evidence before saying "not the X sound".
    # String candidates are still accepted for older tests and synthetic data.
    candidates: list[PhonemeCandidate] = Field(default_factory=list)

    @field_validator("candidates", mode="before")
    @classmethod
    def _coerce_candidates(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        converted: list[Any] = []
        for candidate in value:
            if isinstance(candidate, str):
                converted.append({"phoneme": candidate, "score": 100.0})
            else:
                converted.append(candidate)
        return converted


class WordResult(BaseModel):
    """Per-word result from Azure's scripted Pronunciation Assessment."""
    word: str
    accuracy_score: float
    # Azure ErrorType: "None" | "Mispronunciation" | "Omission" | "Insertion"
    error_type: str = "None"
    phonemes: list[PhonemeScore] = Field(default_factory=list)


class PronunciationScores(BaseModel):
    """Sentence-level scores returned by Azure (scripted mode), each 0-100."""
    accuracy: float
    fluency: float
    completeness: float
    prosody: float
    pronunciation: float  # Azure's own overall aggregate


class AsrResult(BaseModel):
    """Full structured output of the ASR layer."""
    recognized_text: str
    scores: PronunciationScores
    words: list[WordResult] = Field(default_factory=list)


# ==========================================================================
# Stage 1 analyzer output  -  the ErrorReport (input to Stage 2)
# ==========================================================================

class Substitution(BaseModel):
    expected: str
    heard: str
    # When (expected, heard) is a known minimal pair (sheep/ship, bad/bed,
    # right/light, wine/vine), this is the phoneme anchor teaching the ONE
    # sound that distinguishes them — e.g. "the long 'ee' sound like in
    # 'tree'". None for general word-level confusions (was/wants), where
    # naming the swap is already the correction.
    sound_hint: str | None = None


class MispronouncedWord(BaseModel):
    word: str
    accuracy_score: float
    # The specific phonemes inside the word that scored poorly.
    weak_phonemes: list[str] = Field(default_factory=list)
    # Spoken-friendly anchor for the correct pronunciation, e.g.
    # "the 'th' in 'thin'". None when no usable anchor is available.
    correct_hint: str | None = None


class SilentLetterError(BaseModel):
    """A word whose silent letter was likely pronounced.

    Emitted only when Stage 1 has explicit evidence for the spelling-rule
    mistake. Table membership alone is not enough. `hint` is the verbatim rule
    the model must speak, e.g. "the 'k' in 'knife' is silent — say it like
    'nife'".
    """
    word: str
    silent_letter: str
    hint: str


class HardSoftCError(BaseModel):
    """A word likely pronounced with the wrong C rule.

    Emitted only when Stage 1 has explicit evidence for the rule mistake.
    `rule` is one of: soft | hard | cc-ks | cc-k. `hint` is the verbatim
    coaching line, e.g. "in 'city' the 'c' sounds like 's' — say it like
    'sity'".
    """
    word: str
    rule: str
    hint: str


class ClusterError(BaseModel):
    """A word whose consonant cluster (str, spr, thr, ...) was likely
    vowel-inserted or simplified. Emitted only when Stage 1 has explicit
    evidence for that cluster error. `cluster` is the raw trigraph; `hint` is
    the coaching line, e.g. "the 'str' cluster — blend it smoothly like in
    'street'"."""
    word: str
    cluster: str
    hint: str


class FeedbackPoint(BaseModel):
    """One prioritized thing the feedback should talk about.

    kind: praise | missing_word | extra_word | substitution |
          mispronunciation | fluency | pattern | silent_letter |
          hard_soft_c | cluster | polish
    """
    kind: str
    priority: int  # lower = more important
    word: str | None = None
    detail: str | None = None
    # Only populated on `substitution` kind when the (expected, heard) pair
    # is a curated minimal pair — the phoneme anchor the learner should
    # practice ("the long 'ee' sound like in 'tree'"). When set, the
    # substitution is phrased more honestly ("Your 'X' sounded more like
    # 'Y'") because we can't know from audio alone whether the learner
    # *meant* the wrong word or merely produced it.
    sound_hint: str | None = None


class ErrorReport(BaseModel):
    """Deterministic analysis of one attempt. Every decision lives here;
    Stage 2 only turns this into spoken text."""
    target_sentence: str
    recognized_text: str
    scores: PronunciationScores

    missing_words: list[str] = Field(default_factory=list)
    extra_words: list[str] = Field(default_factory=list)
    substitutions: list[Substitution] = Field(default_factory=list)
    mispronounced_words: list[MispronouncedWord] = Field(default_factory=list)
    # Spelling/orthography-driven errors that phoneme scoring alone can't
    # coach — see app/services/feedback/word_hints.py.
    silent_letter_errors: list[SilentLetterError] = Field(default_factory=list)
    hard_soft_c_errors: list[HardSoftCError] = Field(default_factory=list)
    cluster_errors: list[ClusterError] = Field(default_factory=list)
    # Recurring issues, e.g. "th-sound", "dropped-final-consonant".
    patterns: list[str] = Field(default_factory=list)

    final_score: int = Field(ge=0, le=100)
    is_passed: bool
    feedback_points: list[FeedbackPoint] = Field(default_factory=list)


# ==========================================================================
# API response
# ==========================================================================

class AssessmentResponse(BaseModel):
    """What the /asr endpoint returns to the UI."""
    sentence_id: str
    run_id: str
    recognized_text: str
    target_sentence: str
    scores: PronunciationScores
    words: list[WordResult]

    final_score: int = Field(ge=0, le=100)
    is_passed: bool

    missing_words: list[str]
    extra_words: list[str]
    substitutions: list[Substitution]
    mispronounced_words: list[MispronouncedWord]
    silent_letter_errors: list[SilentLetterError] = Field(default_factory=list)
    hard_soft_c_errors: list[HardSoftCError] = Field(default_factory=list)
    cluster_errors: list[ClusterError] = Field(default_factory=list)

    feedback_points: list[FeedbackPoint]
    # The paragraph the avatar reads aloud.
    feedback_text: str
