from app.core.config import settings
from app.schemas.asr import PhonemeScore, WordResult
from app.services.feedback import generator
from app.services.feedback.analysis import analyze
from tests.factories import make_asr_result


def test_falls_back_to_template_when_no_model_configured(monkeypatch):
    monkeypatch.setattr(settings, "FEEDBACK_MODEL_URL", None)
    report = analyze("please close the door", make_asr_result("please close door"))
    text = generator.generate_feedback(report)
    assert isinstance(text, str)
    assert text.strip()


def test_falls_back_to_template_when_model_unreachable(monkeypatch):
    # Port 9 (discard) refuses connections fast -> exercises the except path.
    monkeypatch.setattr(settings, "FEEDBACK_MODEL_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(settings, "FEEDBACK_MODEL_TIMEOUT", 0.5)
    report = analyze("hello world", make_asr_result("hello world"))
    text = generator.generate_feedback(report)
    assert isinstance(text, str)
    assert text.strip()


def test_build_model_messages_shape():
    report = analyze("hello world", make_asr_result("hello world"))
    messages = generator.build_model_messages(report)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "hello world" in messages[1]["content"]


def _stub_model_reply(monkeypatch, content: str):
    """Make generator.generate_feedback hit a fake model returning `content`."""
    monkeypatch.setattr(settings, "FEEDBACK_MODEL_URL", "http://fake/v1")
    monkeypatch.setattr(settings, "FEEDBACK_MODEL_NAME", "stub")
    monkeypatch.setattr(settings, "FEEDBACK_MODEL_TIMEOUT", 5)

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(generator.httpx, "post", lambda *a, **k: _FakeResponse())


def test_verifier_appends_mispronounced_word_the_model_omitted(monkeypatch):
    # Model returns only praise and drops the mispronounced word entirely.
    _stub_model_reply(monkeypatch, "Nice try!")
    words = [
        WordResult(
            word="three",
            accuracy_score=40,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=20)],
        )
    ]
    report = analyze("three", make_asr_result("three", accuracy=40, words=words))
    text = generator.generate_feedback(report).lower()
    # The verifier must name the word and include the anchor hint.
    assert "three" in text
    assert "thin" in text


def test_verifier_leaves_complete_feedback_untouched(monkeypatch):
    reply = "Nice try! For 'three', practice the 'th' sound like in 'thin'."
    _stub_model_reply(monkeypatch, reply)
    words = [
        WordResult(
            word="three",
            accuracy_score=40,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=20)],
        )
    ]
    report = analyze("three", make_asr_result("three", accuracy=40, words=words))
    text = generator.generate_feedback(report)
    assert text == reply  # nothing appended when coverage is already complete


def test_model_reply_missing_exact_mispronunciation_hint_uses_template(monkeypatch):
    _stub_model_reply(
        monkeypatch,
        "Great job! For the word 'developer', focus on the initial d and final er.",
    )
    words = [
        WordResult(
            word="developer",
            accuracy_score=48,
            error_type="Mispronunciation",
            phonemes=[PhonemeScore(phoneme="r", accuracy_score=28)],
        )
    ]
    report = analyze(
        "developer",
        make_asr_result("developer", accuracy=48, words=words),
    )
    text = generator.generate_feedback(report).lower()
    assert "initial d" not in text
    assert "final er" not in text
    assert "developer" in text
    assert "the 'r' sound like in 'red'" in text


def test_model_reply_with_not_the_sound_contrast_uses_template(monkeypatch):
    _stub_model_reply(
        monkeypatch,
        "Nice try! For 'three', practice the 'th' sound like in 'thin', not the 'p' sound.",
    )
    words = [
        WordResult(
            word="three",
            accuracy_score=82,
            error_type="None",
            phonemes=[PhonemeScore(phoneme="th", accuracy_score=52)],
        )
    ]
    report = analyze(
        "three",
        make_asr_result("three", accuracy=72, words=words),
    )
    text = generator.generate_feedback(report).lower()
    assert "three" in text
    assert "the 'th' sound like in 'thin'" in text
    assert "not the" not in text
