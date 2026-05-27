"""Scoring rubric for a repeat-after-me attempt.

This is our own rubric, not Azure's aggregate. We blend Azure's four
sub-scores with explicit weights, then subtract penalties for word-level
mistakes that Azure's pronunciation score does not fully capture.

All weights and thresholds are named constants so they are easy to show,
justify, and tune.
"""
from app.schemas.asr import PronunciationScores

# Weights for the four Azure sub-scores. Must sum to 1.0.
# Accuracy (phoneme correctness) matters most; completeness (did they say the
# whole sentence) is next; fluency and prosody are smaller polish factors.
WEIGHT_ACCURACY = 0.40
WEIGHT_COMPLETENESS = 0.30
WEIGHT_FLUENCY = 0.20
WEIGHT_PROSODY = 0.10

# Per-mistake penalties applied on top of the weighted base score.
SUBSTITUTION_PENALTY = 4  # said a different word than the target
EXTRA_WORD_PENALTY = 2    # added a word that isn't in the target

# A learner "passes" the sentence at or above this final score.
PASS_THRESHOLD = 80


def compute_score(
    scores: PronunciationScores,
    num_substitutions: int,
    num_extra_words: int,
) -> int:
    """Return the final 0-100 score for an attempt.

    Missing words are intentionally not penalized here: they already pull down
    Azure's `completeness` score, so penalizing again would double-count.
    """
    base = (
        scores.accuracy * WEIGHT_ACCURACY
        + scores.completeness * WEIGHT_COMPLETENESS
        + scores.fluency * WEIGHT_FLUENCY
        + scores.prosody * WEIGHT_PROSODY
    )
    penalty = (
        num_substitutions * SUBSTITUTION_PENALTY
        + num_extra_words * EXTRA_WORD_PENALTY
    )
    return round(max(0.0, min(100.0, base - penalty)))


def is_passing(final_score: int) -> bool:
    return final_score >= PASS_THRESHOLD
