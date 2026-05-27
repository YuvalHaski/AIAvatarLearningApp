from app.schemas.asr import PronunciationScores
from app.services.feedback import scoring


def _scores(accuracy, fluency, completeness, prosody):
    return PronunciationScores(
        accuracy=accuracy,
        fluency=fluency,
        completeness=completeness,
        prosody=prosody,
        pronunciation=accuracy,
    )


def test_perfect_attempt_scores_100_and_passes():
    score = scoring.compute_score(_scores(100, 100, 100, 100), 0, 0)
    assert score == 100
    assert scoring.is_passing(score) is True


def test_weighted_blend_with_no_penalties():
    # 80*.4 + 60*.2 + 90*.3 + 50*.1 = 32 + 12 + 27 + 5 = 76
    score = scoring.compute_score(_scores(80, 60, 90, 50), 0, 0)
    assert score == 76
    assert scoring.is_passing(score) is False


def test_substitution_and_extra_word_penalties_apply():
    base = scoring.compute_score(_scores(90, 90, 90, 90), 0, 0)
    penalized = scoring.compute_score(_scores(90, 90, 90, 90), 2, 1)
    # 2 substitutions * 4 + 1 extra * 2 = 10
    assert base - penalized == 10


def test_score_never_goes_below_zero():
    score = scoring.compute_score(_scores(0, 0, 0, 0), 10, 10)
    assert score == 0


def test_pass_threshold_boundary():
    assert scoring.is_passing(scoring.PASS_THRESHOLD) is True
    assert scoring.is_passing(scoring.PASS_THRESHOLD - 1) is False
