from app.schemas.asr import PhonemeScore, WordResult
from app.services.feedback.analysis import analyze
from tests.factories import make_asr_result


def test_missing_word_appears_in_report():
    report = analyze(
        "please close the door",
        make_asr_result("please close door", completeness=75),
    )
    assert "the" in report.missing_words


def test_perfect_attempt_passes_and_leads_with_praise():
    words = [WordResult(word=w, accuracy_score=98) for w in ("hello", "world")]
    report = analyze("hello world", make_asr_result("hello world", words=words))
    assert report.is_passed is True
    assert report.feedback_points[0].kind == "praise"


def test_mispronounced_word_and_weak_phoneme_flagged():
    words = [
        WordResult(
            word="three",
            accuracy_score=40,
            error_type="Mispronunciation",
            phonemes=[
                PhonemeScore(phoneme="th", accuracy_score=20),
                PhonemeScore(phoneme="r", accuracy_score=92),
                PhonemeScore(phoneme="iy", accuracy_score=88),
            ],
        )
    ]
    report = analyze("three", make_asr_result("three", accuracy=40, words=words))
    assert report.mispronounced_words
    flagged = report.mispronounced_words[0]
    assert flagged.word == "three"
    assert flagged.weak_phonemes == ["th"]


def test_recurring_phoneme_becomes_a_pattern():
    words = [
        WordResult(
            word=w,
            accuracy_score=45,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=20)],
        )
        for w in ("three", "think")
    ]
    report = analyze(
        "three think", make_asr_result("three think", accuracy=45, words=words)
    )
    assert any("th" in pattern for pattern in report.patterns)
