"""Stage 2: turn an ErrorReport into spoken feedback.

Primary path: call our fine-tuned feedback model, served behind an
OpenAI-compatible endpoint (vLLM). Fallback path: the deterministic template
renderer, so the endpoint never fails just because the model service is down.

`build_model_messages` is the single source of truth for the prompt format -
the training scripts import it so the model is trained on exactly what it is
served at request time.

A post-generation verifier guarantees that every concrete mistake in the
ErrorReport (missing / extra / substituted / mispronounced word) is named in
the final text, regardless of whether the model remembered to mention it.
"""
import json
import re

import httpx

from app.core.config import settings
from app.schemas.asr import ErrorReport, FeedbackPoint
from app.services.feedback import templates

SYSTEM_PROMPT = (
    "You are a warm, encouraging English tutor for a 'repeat after me' "
    "exercise. You receive a JSON analysis of one student attempt. Write "
    "2 to 5 short, spoken-friendly sentences for an avatar to read aloud.\n"
    "\n"
    "Rules - follow EXACTLY:\n"
    "- Start with ONE brief encouragement (e.g. \"Nice try!\", "
    "\"Great job!\").\n"
    "- For EVERY word in `missing_words`, name it: "
    "\"You skipped '<word>'.\"\n"
    "- For EVERY word in `extra_words`, name it: "
    "\"You added the word '<word>'.\"\n"
    "- For EVERY entry in `substitutions`, name both sides: "
    "\"You said '<heard>' instead of '<expected>'.\"\n"
    "- For EVERY entry in `mispronounced`, name the word and use its "
    "`correct_hint` verbatim in a natural tutor sentence such as: "
    "\"For '<word>', practice <correct_hint>.\" You may vary the verb "
    "(practice / work on / try saying) but the word and the hint must "
    "appear exactly as given. If `correct_hint` is null, say: "
    "\"Work on the way you say '<word>'.\"\n"
    "- When several `mispronounced` words share the SAME `correct_hint`, "
    "group them into ONE sentence naming all of them, e.g. \"For "
    "'software' and 'developer', practice the 'r' sound like in 'red'.\" "
    "Never repeat an identical hint in separate sentences.\n"
    "- Never skip an item that appears in any of the four lists above. "
    "Never invent items that are not in the JSON.\n"
    "- If `polish_tip` is non-null (this only happens when all four "
    "error lists are empty), explain the score gap by adding exactly "
    "one sentence: \"To make it perfect, <polish_tip>.\" Use the "
    "`polish_tip` value verbatim.\n"
    "- If all four lists are empty AND `polish_tip` is null, keep it "
    "short and positive (one sentence).\n"
    "- Plain text only. No markdown, no bullet points, no numbered lists.\n"
    "\n"
    "Example 1 (passed, one mispronunciation):\n"
    "Input: {\"passed\": true, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [], \"mispronounced\": [{\"word\": \"thought\", "
    "\"correct_hint\": \"the 'th' sound like in 'thin'\"}]}\n"
    "Output: Great job! For 'thought', practice the 'th' sound like in "
    "'thin'.\n"
    "\n"
    "Example 2 (failed, substitution plus two mispronunciations):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [{\"expected\": \"coffee\", \"heard\": \"tea\"}], "
    "\"mispronounced\": [{\"word\": \"please\", \"correct_hint\": "
    "\"the 'z' sound like in 'zoo'\"}, {\"word\": \"thank\", "
    "\"correct_hint\": \"the 'th' sound like in 'thin'\"}]}\n"
    "Output: Nice try! You said 'tea' instead of 'coffee'. For 'please', "
    "practice the 'z' sound like in 'zoo'. And for 'thank', work on the "
    "'th' sound like in 'thin'.\n"
    "\n"
    "Example 3 (passed, no errors, polish tip):\n"
    "Input: {\"passed\": true, \"score\": 97, \"missing_words\": [], "
    "\"extra_words\": [], \"substitutions\": [], \"mispronounced\": [], "
    "\"polish_tip\": \"speak with a smoother flow, with fewer pauses\"}\n"
    "Output: Great job! To make it perfect, speak with a smoother flow, "
    "with fewer pauses.\n"
    "\n"
    "Example 4 (two mispronunciations sharing one hint - group them):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [], \"mispronounced\": [{\"word\": \"software\", "
    "\"correct_hint\": \"the 'r' sound like in 'red'\"}, {\"word\": "
    "\"developer\", \"correct_hint\": \"the 'r' sound like in 'red'\"}]}\n"
    "Output: Nice try! For 'software' and 'developer', practice the 'r' "
    "sound like in 'red'."
)


def _polish_tip_for(report: ErrorReport) -> str | None:
    """Pull the polish tip (if any) out of the analyzer's feedback points."""
    for point in report.feedback_points:
        if point.kind == "polish" and point.detail:
            return point.detail
    return None


def _report_payload(report: ErrorReport) -> dict:
    """The compact, model-facing view of an ErrorReport.

    `focus_points` was intentionally removed: it was a truncated, prioritized
    list and the model learned to treat it as the authoritative set, which
    caused mispronunciations to be silently dropped. The per-list rules in
    the system prompt are now the contract.

    `polish_tip` is set by Stage 1 only when the attempt had no flagged
    errors but `score < 100` — it tells the model why the score wasn't
    perfect so the avatar can explain the gap.
    """
    return {
        "target_sentence": report.target_sentence,
        "passed": report.is_passed,
        "score": report.final_score,
        "missing_words": report.missing_words,
        "extra_words": report.extra_words,
        "substitutions": [s.model_dump() for s in report.substitutions],
        "mispronounced": [
            {
                "word": w.word,
                "weak_sounds": w.weak_phonemes,
                "correct_hint": w.correct_hint,
            }
            for w in report.mispronounced_words
        ],
        "patterns": report.patterns,
        "polish_tip": _polish_tip_for(report),
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


def _word_in_text(text: str, word: str) -> bool:
    """Case-insensitive whole-word search for `word` inside `text`."""
    if not word:
        return True
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def _polish_covered(text: str, detail: str) -> bool:
    """Heuristic: the polish tip is considered covered when either the
    standard 'to make it perfect' lead-in appears, or the tip's literal
    phrase appears verbatim."""
    lower = text.lower()
    return "to make it perfect" in lower or detail.lower() in lower


def _ensure_coverage(text: str, report: ErrorReport) -> str:
    """Append template-rendered sentences for any feedback point the model
    failed to name. Coverage is judged by whether the point's `word` (or,
    for polish points, a fragment of `detail`) appears in the text. Soft
    points (praise / fluency / pattern) are not enforced."""
    extras: list[str] = []
    for point in report.feedback_points:
        if point.kind in ("praise", "fluency", "pattern"):
            continue
        if point.kind == "polish":
            if not point.detail or _polish_covered(text, point.detail):
                continue
        else:
            target_word = point.word
            if not target_word or _word_in_text(text, target_word):
                continue
        sentence = templates.render_point(point)
        if sentence:
            extras.append(sentence)

    if not extras:
        return text

    glue = "" if text.rstrip().endswith((".", "!", "?")) else "."
    return (text.rstrip() + glue + " " + " ".join(extras)).strip()


def generate_feedback(report: ErrorReport) -> str:
    """Return spoken feedback text for an attempt.

    Calls the fine-tuned model service when configured; falls back to the
    template renderer on any missing config, timeout, or service error. In
    both cases the result is run through `_ensure_coverage` so every
    concrete mistake in the report is named in the spoken text.
    """
    if not settings.FEEDBACK_MODEL_URL:
        return _ensure_coverage(templates.render_feedback(report), report)

    url = settings.FEEDBACK_MODEL_URL.rstrip("/") + "/chat/completions"
    try:
        response = httpx.post(
            url,
            json={
                "model": settings.FEEDBACK_MODEL_NAME,
                "messages": build_model_messages(report),
                "temperature": 0.1,
                "max_tokens": 200,
            },
            timeout=settings.FEEDBACK_MODEL_TIMEOUT,
        )
        response.raise_for_status()
        text = (
            response.json()["choices"][0]["message"]["content"] or ""
        ).strip()
        if not text:
            return _ensure_coverage(templates.render_feedback(report), report)
        return _ensure_coverage(text, report)
    except Exception:
        # Model service unreachable / slow / malformed -> deterministic fallback.
        return _ensure_coverage(templates.render_feedback(report), report)
