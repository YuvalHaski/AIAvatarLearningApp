"""Stage 2: turn an ErrorReport into spoken feedback.

Primary path: call our fine-tuned feedback model, served behind an
OpenAI-compatible endpoint (vLLM). Fallback path: the deterministic template
renderer, so the endpoint never fails just because the model service is down.

`build_model_messages` is the single source of truth for the prompt format -
the training scripts import it so the model is trained on exactly what it is
served at request time.
"""
import json

import httpx

from app.core.config import settings
from app.schemas.asr import ErrorReport
from app.services.feedback import templates

SYSTEM_PROMPT = (
    "You are a warm, encouraging language tutor. You are given a JSON analysis "
    "of how a student repeated a target sentence in a 'repeat after me' "
    "exercise. Write 2 to 4 short, spoken-friendly sentences of feedback for an "
    "avatar to read aloud. Start with brief encouragement. Only mention issues "
    "that appear in the analysis - never invent mistakes. If the student "
    "passed, keep it positive and light. Plain text only, no markdown."
)


def _report_payload(report: ErrorReport) -> dict:
    """The compact, model-facing view of an ErrorReport."""
    return {
        "target_sentence": report.target_sentence,
        "passed": report.is_passed,
        "score": report.final_score,
        "missing_words": report.missing_words,
        "extra_words": report.extra_words,
        "substitutions": [s.model_dump() for s in report.substitutions],
        "mispronounced": [
            {"word": w.word, "weak_sounds": w.weak_phonemes}
            for w in report.mispronounced_words
        ],
        "patterns": report.patterns,
        "focus_points": [
            {"kind": p.kind, "word": p.word, "detail": p.detail}
            for p in report.feedback_points
        ],
    }


def build_model_messages(report: ErrorReport) -> list[dict]:
    """Build the chat messages for the feedback model. Shared with training."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(_report_payload(report), ensure_ascii=False),
        },
    ]


def generate_feedback(report: ErrorReport) -> str:
    """Return spoken feedback text for an attempt.

    Calls the fine-tuned model service when configured; falls back to the
    template renderer on any missing config, timeout, or service error.
    """
    if not settings.FEEDBACK_MODEL_URL:
        return templates.render_feedback(report)

    url = settings.FEEDBACK_MODEL_URL.rstrip("/") + "/chat/completions"
    try:
        response = httpx.post(
            url,
            json={
                "model": settings.FEEDBACK_MODEL_NAME,
                "messages": build_model_messages(report),
                "temperature": 0.3,
                "max_tokens": 160,
            },
            timeout=settings.FEEDBACK_MODEL_TIMEOUT,
        )
        response.raise_for_status()
        text = (
            response.json()["choices"][0]["message"]["content"] or ""
        ).strip()
        return text or templates.render_feedback(report)
    except Exception:
        # Model service unreachable / slow / malformed -> deterministic fallback.
        return templates.render_feedback(report)
