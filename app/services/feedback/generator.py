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
    "- For EVERY entry in `substitutions`, use ONE of two phrasings based "
    "on `sound_hint`:\n"
    "  * If `sound_hint` is non-null (minimal-pair confusion — the two "
    "words differ by just one sound), phrase it acoustically as "
    "\"Your '<expected>' sounded more like '<heard>'.\" Then add ONE "
    "more sentence teaching the sound: \"Practice <sound_hint>.\" "
    "(verb may vary: practice / work on / try saying). The `sound_hint` "
    "text must appear verbatim.\n"
    "  * If `sound_hint` is null (general word confusion), phrase it "
    "as \"You said '<heard>' instead of '<expected>'.\" and do NOT add "
    "a hint sentence.\n"
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
    "- For EVERY entry in `silent_letter_errors`, use its `hint` verbatim "
    "in a natural tutor sentence starting with \"Remember,\": "
    "\"Remember, <hint>.\" The hint is a full clause and must appear "
    "exactly as given.\n"
    "- For EVERY entry in `hard_soft_c_errors`, use its `hint` verbatim: "
    "\"Remember, <hint>.\" Same rule as silent-letter errors.\n"
    "- For EVERY entry in `cluster_errors`, name the word and use its "
    "`hint` verbatim in a natural tutor sentence such as: \"For "
    "'<word>', practice <hint>.\" You may vary the verb.\n"
    "- Never skip an item that appears in any of the seven lists above. "
    "Never invent items that are not in the JSON.\n"
    "- Hint strings may come in two styles — both are valid, always use "
    "them VERBATIM without rewording:\n"
    "  * Anchor style: \"the 'th' sound like in 'thin'\", \"the 'r' "
    "sound like in 'red'\".\n"
    "  * Articulation style: \"tipping your tongue lightly between your "
    "teeth — just let air flow\", \"curling your tongue up and back — "
    "but don't let it touch the roof\". These read naturally after "
    "\"practice\" or \"try\" (\"For 'thought', try tipping your tongue "
    "lightly between your teeth — just let air flow.\").\n"
    "- If `polish_tip` is non-null (this only happens when all seven "
    "error lists are empty), explain the score gap by adding exactly "
    "one sentence: \"To make it perfect, <polish_tip>.\" Use the "
    "`polish_tip` value verbatim.\n"
    "- If all seven lists are empty AND `polish_tip` is null, keep it "
    "short and positive (one sentence).\n"
    "- Plain text only. No markdown, no bullet points, no numbered lists.\n"
    "\n"
    "Example 1 (passed, one mispronunciation):\n"
    "Input: {\"passed\": true, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [], \"mispronounced\": [{\"word\": \"thought\", "
    "\"correct_hint\": \"the 'th' sound like in 'thin'\"}], "
    "\"silent_letter_errors\": [], \"hard_soft_c_errors\": [], "
    "\"cluster_errors\": []}\n"
    "Output: Great job! For 'thought', practice the 'th' sound like in "
    "'thin'.\n"
    "\n"
    "Example 2 (failed, general substitution WITHOUT sound_hint plus two "
    "mispronunciations, using articulation-style hints):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [{\"expected\": \"coffee\", \"heard\": \"copy\", "
    "\"sound_hint\": null}], \"mispronounced\": [{\"word\": \"please\", "
    "\"correct_hint\": \"humming a long 's' — same shape, but voice it "
    "like a bee\"}, {\"word\": \"thank\", \"correct_hint\": \"tipping "
    "your tongue lightly between your teeth — just let air flow\"}], "
    "\"silent_letter_errors\": [], \"hard_soft_c_errors\": [], "
    "\"cluster_errors\": []}\n"
    "Output: Nice try! You said 'copy' instead of 'coffee'. For 'please', "
    "try humming a long 's' — same shape, but voice it like a bee. And "
    "for 'thank', practice tipping your tongue lightly between your teeth "
    "— just let air flow.\n"
    "\n"
    "Example 3 (passed, no errors, polish tip):\n"
    "Input: {\"passed\": true, \"score\": 97, \"missing_words\": [], "
    "\"extra_words\": [], \"substitutions\": [], \"mispronounced\": [], "
    "\"silent_letter_errors\": [], \"hard_soft_c_errors\": [], "
    "\"cluster_errors\": [], \"polish_tip\": \"speak with a smoother "
    "flow, with fewer pauses\"}\n"
    "Output: Great job! To make it perfect, speak with a smoother flow, "
    "with fewer pauses.\n"
    "\n"
    "Example 4 (two mispronunciations sharing one hint - group them):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [], \"mispronounced\": [{\"word\": \"software\", "
    "\"correct_hint\": \"the 'r' sound like in 'red'\"}, {\"word\": "
    "\"developer\", \"correct_hint\": \"the 'r' sound like in 'red'\"}], "
    "\"silent_letter_errors\": [], \"hard_soft_c_errors\": [], "
    "\"cluster_errors\": []}\n"
    "Output: Nice try! For 'software' and 'developer', practice the 'r' "
    "sound like in 'red'.\n"
    "\n"
    "Example 5 (minimal-pair substitution WITH sound_hint — note the "
    "acoustic-honest phrasing):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [{\"expected\": \"sheep\", \"heard\": \"ship\", "
    "\"sound_hint\": \"the long 'ee' sound like in 'tree'\"}], "
    "\"mispronounced\": [], \"silent_letter_errors\": [], "
    "\"hard_soft_c_errors\": [], \"cluster_errors\": []}\n"
    "Output: Nice try! Your 'sheep' sounded more like 'ship'. Practice "
    "the long 'ee' sound like in 'tree'.\n"
    "\n"
    "Example 6 (silent letter):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [], \"mispronounced\": [], "
    "\"silent_letter_errors\": [{\"word\": \"knife\", \"silent_letter\": "
    "\"k\", \"hint\": \"the 'k' in 'knife' is silent — say it like "
    "'nife'\"}], \"hard_soft_c_errors\": [], \"cluster_errors\": []}\n"
    "Output: Nice try! Remember, the 'k' in 'knife' is silent — say it "
    "like 'nife'.\n"
    "\n"
    "Example 7 (hard/soft c):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [], \"mispronounced\": [], "
    "\"silent_letter_errors\": [], \"hard_soft_c_errors\": [{\"word\": "
    "\"city\", \"rule\": \"soft\", \"hint\": \"in 'city' the 'c' sounds "
    "like 's' — say it like 'sity'\"}], \"cluster_errors\": []}\n"
    "Output: Nice try! Remember, in 'city' the 'c' sounds like 's' — say "
    "it like 'sity'.\n"
    "\n"
    "Example 8 (consonant cluster with articulation-style hint):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [], \"mispronounced\": [], "
    "\"silent_letter_errors\": [], \"hard_soft_c_errors\": [], "
    "\"cluster_errors\": [{\"word\": \"street\", \"cluster\": \"str\", "
    "\"hint\": \"blending all the sounds together smoothly — don't "
    "slip a vowel in between\"}]}\n"
    "Output: Nice try! For 'street', try blending all the sounds "
    "together smoothly — don't slip a vowel in between.\n"
    "\n"
    "Example 9 (minimal-pair sub with articulation-style sound_hint):\n"
    "Input: {\"passed\": false, \"missing_words\": [], \"extra_words\": [], "
    "\"substitutions\": [{\"expected\": \"men\", \"heard\": \"man\", "
    "\"sound_hint\": \"opening your mouth halfway — tongue in the "
    "middle, jaw relaxed\"}], \"mispronounced\": [], "
    "\"silent_letter_errors\": [], \"hard_soft_c_errors\": [], "
    "\"cluster_errors\": []}\n"
    "Output: Nice try! Your 'men' sounded more like 'man'. Practice "
    "opening your mouth halfway — tongue in the middle, jaw relaxed."
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
        "silent_letter_errors": [
            e.model_dump() for e in report.silent_letter_errors
        ],
        "hard_soft_c_errors": [
            e.model_dump() for e in report.hard_soft_c_errors
        ],
        "cluster_errors": [
            e.model_dump() for e in report.cluster_errors
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


def _missing_required_coverage(text: str, report: ErrorReport) -> bool:
    """True when model output skipped concrete report content.

    For mispronunciation hints, word coverage alone is not enough: the model
    must include the exact `correct_hint`, otherwise it can mention the word
    while inventing unrelated advice.
    """
    lower = text.lower()
    if re.search(r"\bnot\s+the\s+'[^']+'\s+sound\b", lower):
        return True
    for point in report.feedback_points:
        if point.kind in ("praise", "fluency", "pattern"):
            continue
        if point.kind == "polish":
            if point.detail and not _polish_covered(text, point.detail):
                return True
            continue
        if point.word and not _word_in_text(text, point.word):
            return True
        if point.kind in ("mispronunciation", "silent_letter", "hard_soft_c", "cluster"):
            if point.detail and point.detail.lower() not in lower:
                return True

    for sub in report.substitutions:
        if sub.sound_hint and sub.sound_hint.lower() not in lower:
            return True
    return False


def _ensure_coverage(text: str, report: ErrorReport) -> str:
    """Append template-rendered sentences for any feedback point the model
    failed to name. Coverage is judged by whether the point's `word` (or,
    for polish points, a fragment of `detail`) appears in the text. Soft
    points (praise / fluency / pattern) are not enforced.

    For substitutions we additionally check the sound_hint: even when the
    swap itself was named, a missing sound_hint means the coaching is
    incomplete and we append it.
    """
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

    # Substitutions carry an optional sound_hint that isn't tied to a
    # feedback_point word. Enforce it separately: if the hint text isn't in
    # the reply, append a coaching sentence.
    for sub in report.substitutions:
        if not sub.sound_hint:
            continue
        if sub.sound_hint.lower() in text.lower():
            continue
        extras.append(f"Practice {sub.sound_hint}.")

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
        # TEMP DEBUG: raw model output before the verifier appends anything.
        # If this rambles about syllables/toning, the regression is the model;
        # if it's clean but the final text isn't, it's _ensure_coverage.
        print(f"[FB DEBUG] raw model = {text!r}")
        if not text:
            return _ensure_coverage(templates.render_feedback(report), report)
        if _missing_required_coverage(text, report):
            return _ensure_coverage(templates.render_feedback(report), report)
        return _ensure_coverage(text, report)
    except Exception:
        # Model service unreachable / slow / malformed -> deterministic fallback.
        return _ensure_coverage(templates.render_feedback(report), report)
