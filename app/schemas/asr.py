"""Schemas for the pronunciation assessment flow.

These models are used both internally (between asr_service -> Stage 1 -> Stage 2)
and at the API boundary, so everything here is JSON-serializable.
"""
from pydantic import BaseModel, Field


# ==========================================================================
# ASR layer  -  produced by asr_service from Azure's response
# ==========================================================================

class PhonemeScore(BaseModel):
    """A single (expected) phoneme and how accurately it was pronounced (0-100)."""
    phoneme: str
    accuracy_score: float
    # Azure NBestPhonemes: the phonemes Azure thinks the learner ACTUALLY
    # produced at this position, highest-confidence first. Empty when NBest is
    # off or the locale doesn't support it. When the top entry differs from
    # `phoneme`, the learner substituted a sound (the basis for "you said X
    # instead of Y"); when it equals `phoneme`, they hit the target sound and
    # the low score is noise we can filter out.
    candidates: list[str] = Field(default_factory=list)


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


class MispronouncedWord(BaseModel):
    word: str
    accuracy_score: float
    # The specific phonemes inside the word that scored poorly.
    weak_phonemes: list[str] = Field(default_factory=list)
    # Spoken-friendly anchor for the correct pronunciation, e.g.
    # "the 'th' in 'thin'". None when no usable anchor is available.
    correct_hint: str | None = None


class FeedbackPoint(BaseModel):
    """One prioritized thing the feedback should talk about.

    kind: praise | missing_word | extra_word | substitution |
          mispronunciation | fluency | pattern
    """
    kind: str
    priority: int  # lower = more important
    word: str | None = None
    detail: str | None = None


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

    feedback_points: list[FeedbackPoint]
    # The paragraph the avatar reads aloud.
    feedback_text: str
