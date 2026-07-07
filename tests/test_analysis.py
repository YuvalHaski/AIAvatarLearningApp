from app.schemas.asr import PhonemeCandidate, PhonemeScore, WordResult
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


def test_strong_candidate_evidence_adds_contrast_hint():
    words = [
        WordResult(
            word="three",
            accuracy_score=40,
            error_type="Mispronunciation",
            phonemes=[
                PhonemeScore(
                    phoneme="th",
                    accuracy_score=20,
                    candidates=[
                        PhonemeCandidate(phoneme="s", score=95),
                        PhonemeCandidate(phoneme="th", score=20),
                    ],
                )
            ],
        )
    ]
    report = analyze("three", make_asr_result("three", accuracy=40, words=words))
    hint = report.mispronounced_words[0].correct_hint
    assert hint == "the 'th' sound like in 'thin', not the 's' sound"


def test_weak_candidate_evidence_keeps_correct_sound_without_contrast():
    words = [
        WordResult(
            word="three",
            accuracy_score=55,
            error_type="Mispronunciation",
            phonemes=[
                PhonemeScore(
                    phoneme="th",
                    accuracy_score=50,
                    candidates=[
                        PhonemeCandidate(phoneme="s", score=58),
                        PhonemeCandidate(phoneme="th", score=52),
                    ],
                )
            ],
        )
    ]
    report = analyze("three", make_asr_result("three", accuracy=55, words=words))
    hint = report.mispronounced_words[0].correct_hint
    assert hint == "the 'th' sound like in 'thin'"
    assert "not the 's' sound" not in hint


def test_top_candidate_matching_expected_sound_does_not_create_false_hint():
    words = [
        WordResult(
            word="three",
            accuracy_score=55,
            error_type="Mispronunciation",
            phonemes=[
                PhonemeScore(
                    phoneme="th",
                    accuracy_score=50,
                    candidates=[
                        PhonemeCandidate(phoneme="th", score=92),
                        PhonemeCandidate(phoneme="s", score=40),
                    ],
                )
            ],
        )
    ]
    report = analyze("three", make_asr_result("three", accuracy=55, words=words))
    flagged = report.mispronounced_words[0]
    assert flagged.weak_phonemes == []
    assert flagged.correct_hint is None


def test_string_candidates_are_still_accepted_for_synthetic_data():
    score = PhonemeScore(phoneme="th", accuracy_score=20, candidates=["s"])
    assert score.candidates[0].phoneme == "s"
    assert score.candidates[0].score == 100


def test_error_type_none_score_around_82_without_weak_phoneme_is_not_flagged():
    words = [
        WordResult(word="software", accuracy_score=82, error_type="None")
    ]
    report = analyze("software", make_asr_result("software", accuracy=82, words=words))
    assert report.mispronounced_words == []
    assert not any(p.kind == "mispronunciation" for p in report.feedback_points)


def test_error_type_none_low_score_requires_strong_weak_phoneme():
    words = [
        WordResult(
            word="three",
            accuracy_score=70,
            error_type="None",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=70)],
        ),
        WordResult(
            word="thin",
            accuracy_score=70,
            error_type="None",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=60)],
        ),
    ]
    report = analyze(
        "three thin",
        make_asr_result("three thin", accuracy=70, words=words),
    )
    assert [word.word for word in report.mispronounced_words] == ["thin"]


def test_error_type_none_boundary_vowel_substitution_is_flagged():
    words = [
        WordResult(
            word="dish",
            accuracy_score=69,
            error_type="None",
            phonemes=[
                PhonemeScore(
                    phoneme="d",
                    accuracy_score=55,
                    candidates=[
                        PhonemeCandidate(phoneme="b", score=92),
                        PhonemeCandidate(phoneme="d", score=91),
                    ],
                ),
                PhonemeScore(
                    phoneme="ih",
                    accuracy_score=65,
                    candidates=[
                        PhonemeCandidate(phoneme="eh", score=97),
                        PhonemeCandidate(phoneme="ae", score=82),
                        PhonemeCandidate(phoneme="ih", score=71),
                    ],
                ),
                PhonemeScore(
                    phoneme="sh",
                    accuracy_score=67,
                    candidates=[
                        PhonemeCandidate(phoneme="s", score=89),
                        PhonemeCandidate(phoneme="sh", score=86),
                    ],
                ),
            ],
        )
    ]
    report = analyze(
        "dish",
        make_asr_result("dish", accuracy=85, fluency=96, prosody=74, words=words),
    )
    flagged = report.mispronounced_words[0]
    assert flagged.word == "dish"
    assert flagged.weak_phonemes == ["ih"]
    assert flagged.correct_hint == "the 'i' sound like in 'sit', not the 'e' sound"


def test_low_prosody_alone_does_not_create_tone_feedback():
    report = analyze(
        "hello",
        make_asr_result(
            "hello",
            accuracy=100,
            fluency=100,
            completeness=100,
            prosody=70,
            words=[WordResult(word="hello", accuracy_score=100)],
        ),
    )
    assert not any(p.kind == "polish" for p in report.feedback_points)


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


def test_mispronunciation_without_clean_weak_phoneme_gets_generic_feedback():
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
    assert flagged.correct_hint is None
    point = next(p for p in report.feedback_points if p.kind == "mispronunciation")
    assert point.word == "think"
    assert point.detail is None


def test_word_containing_r_does_not_get_r_hint_without_weak_r():
    words = [
        WordResult(
            word="software",
            accuracy_score=60,
            error_type="Mispronunciation",
            phonemes=[
                PhonemeScore(phoneme="s", accuracy_score=95),
                PhonemeScore(phoneme="ao", accuracy_score=92),
                PhonemeScore(phoneme="f", accuracy_score=93),
                PhonemeScore(phoneme="t", accuracy_score=94),
                PhonemeScore(phoneme="w", accuracy_score=96),
                PhonemeScore(phoneme="eh", accuracy_score=91),
                PhonemeScore(phoneme="r", accuracy_score=90),
            ],
        )
    ]
    report = analyze("software", make_asr_result("software", accuracy=60, words=words))
    flagged = report.mispronounced_words[0]
    assert flagged.weak_phonemes == []
    assert flagged.correct_hint is None


def test_silent_letter_word_is_not_automatically_routed_to_silent_letter_errors():
    words = [
        WordResult(
            word="knife",
            accuracy_score=60,
            error_type="Mispronunciation",
            phonemes=[
                PhonemeScore(phoneme="n", accuracy_score=95),
                PhonemeScore(phoneme="ay", accuracy_score=92),
                PhonemeScore(phoneme="f", accuracy_score=94),
            ],
        )
    ]
    report = analyze("knife", make_asr_result("knife", accuracy=60, words=words))
    assert report.silent_letter_errors == []
    assert report.mispronounced_words[0].word == "knife"
    assert report.mispronounced_words[0].correct_hint is None


def test_real_weak_vowel_phoneme_produces_vowel_hint_not_spelling_guess():
    words = [
        WordResult(
            word="soup",
            accuracy_score=55,
            error_type="Mispronunciation",
            phonemes=[
                PhonemeScore(phoneme="s", accuracy_score=94),
                PhonemeScore(phoneme="uw", accuracy_score=30, candidates=["uh"]),
                PhonemeScore(phoneme="p", accuracy_score=93),
            ],
        )
    ]
    report = analyze("soup", make_asr_result("soup", accuracy=55, words=words))
    flagged = report.mispronounced_words[0]
    assert flagged.weak_phonemes == ["uw"]
    assert flagged.correct_hint == "the 'oo' sound like in 'food'"
    assert "see" not in flagged.correct_hint


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
