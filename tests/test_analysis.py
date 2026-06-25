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


def test_mispronounced_word_gets_correct_hint():
    words = [
        WordResult(
            word="three",
            accuracy_score=40,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=20)],
        )
    ]
    report = analyze("three", make_asr_result("three", accuracy=40, words=words))
    flagged = report.mispronounced_words[0]
    assert flagged.correct_hint == "the 'th' sound like in 'thin'"


def test_borderline_accuracy_word_is_flagged():
    # 82 is the real-world case (Azure's lenient "software"): below the 85
    # cutoff and reported with ErrorType "None", yet still flagged.
    words = [
        WordResult(word="software", accuracy_score=82, error_type="None")
    ]
    report = analyze("software", make_asr_result("software", accuracy=82, words=words))
    assert report.mispronounced_words
    assert report.mispronounced_words[0].word == "software"


def test_voiced_th_is_flagged_and_shown_as_th():
    # Azure reports voiced "th" as SAPI "dh"; the learner should never see "dh".
    words = [
        WordResult(
            word="this",
            accuracy_score=40,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="dh", accuracy_score=22)],
        )
    ]
    report = analyze("this", make_asr_result("this", accuracy=40, words=words))
    flagged = report.mispronounced_words[0]
    assert flagged.weak_phonemes == ["dh"]
    assert flagged.correct_hint == "the 'th' sound like in 'this'"
    assert "dh" not in flagged.correct_hint


def test_ipa_th_symbol_is_normalized():
    words = [
        WordResult(
            word="thin",
            accuracy_score=40,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="θ", accuracy_score=20)],  # IPA theta
        )
    ]
    report = analyze("thin", make_asr_result("thin", accuracy=40, words=words))
    flagged = report.mispronounced_words[0]
    assert flagged.weak_phonemes == ["th"]
    assert flagged.correct_hint == "the 'th' sound like in 'thin'"


def test_weak_phoneme_threshold_catches_lenient_th():
    # 78 is the kind of "barely off" score Azure gives a weak sound; it sits
    # below the 80 phoneme cutoff so it is still surfaced as weak.
    words = [
        WordResult(
            word="three",
            accuracy_score=90,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=78)],
        )
    ]
    report = analyze("three", make_asr_result("three", accuracy=90, words=words))
    assert report.mispronounced_words[0].weak_phonemes == ["th"]


def test_weak_phonemes_are_ordered_worst_first():
    # 'r' is the worst sound but the last phoneme; it must still lead the hint.
    words = [
        WordResult(
            word="world",
            accuracy_score=55,
            error_type="Mispronunciation",
            phonemes=[
                PhonemeScore(phoneme="w", accuracy_score=78),
                PhonemeScore(phoneme="r", accuracy_score=30),
                PhonemeScore(phoneme="l", accuracy_score=60),
                PhonemeScore(phoneme="d", accuracy_score=95),
            ],
        )
    ]
    report = analyze("world", make_asr_result("world", accuracy=55, words=words))
    flagged = report.mispronounced_words[0]
    assert flagged.weak_phonemes == ["r", "l", "w"]  # ascending by score
    assert flagged.correct_hint.startswith("the 'r' sound like in 'red'")


def test_orthographic_fallback_hint_when_no_weak_phoneme():
    # Word flagged mispronounced, but no single phoneme dipped below threshold.
    words = [
        WordResult(
            word="think",
            accuracy_score=60,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="ih", accuracy_score=95)],
        )
    ]
    report = analyze("think", make_asr_result("think", accuracy=60, words=words))
    flagged = report.mispronounced_words[0]
    assert flagged.weak_phonemes == []  # stays Azure-only
    assert flagged.correct_hint is not None
    assert "th" in flagged.correct_hint


def test_all_mispronunciations_survive_alongside_missing_word():
    words = [
        WordResult(
            word="think", accuracy_score=40, error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=20)],
        ),
        WordResult(
            word="this", accuracy_score=42, error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="dh", accuracy_score=22)],
        ),
        WordResult(
            word="thing", accuracy_score=44, error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=24)],
        ),
    ]
    report = analyze(
        "I think this thing",
        make_asr_result(
            "think this thing", accuracy=45, completeness=80, words=words
        ),
    )
    assert "i" in report.missing_words  # tokenizer lowercases
    mispron_words = {
        p.word for p in report.feedback_points if p.kind == "mispronunciation"
    }
    assert {"think", "this", "thing"} <= mispron_words
