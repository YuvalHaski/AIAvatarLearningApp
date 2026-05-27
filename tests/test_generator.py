from app.core.config import settings
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
