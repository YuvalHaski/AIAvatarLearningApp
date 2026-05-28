"""Pronunciation assessment endpoint for the repeat-after-me exercise.

Flow: look up the target sentence in the DB -> normalize audio -> Azure
Pronunciation Assessment -> Stage 1 deterministic analyzer -> Stage 2 feedback
generator -> persist the attempt to the progress tables -> response.
"""
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.api.auth import get_current_user
from app.core.database import get_db
from app.models.domain import Sentence
from app.schemas.asr import AssessmentResponse
from app.services.asr_service import AsrError, assess_pronunciation
from app.services.audio_utils import normalize_to_wav, resolve_locale
from app.services.feedback.analysis import analyze
from app.services.feedback.generator import generate_feedback
from app.services.progress_service import record_attempt

router = APIRouter()

MAX_AUDIO_MB = 20


@router.post("/asr", response_model=AssessmentResponse)
async def asr(
    file: UploadFile = File(...),
    sentence_id: str = Form(...),
    language: str | None = Form(None),
    current_user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not file.content_type or not file.content_type.startswith(("audio/", "video/")):
        raise HTTPException(
            status_code=400, detail=f"Unsupported content-type: {file.content_type}"
        )

    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_AUDIO_MB}MB)")

    # The target sentence is the canonical text from the DB - never trust the
    # client to send the text they're being graded against.
    sentence = db.query(Sentence).filter(Sentence.id == sentence_id).first()
    if sentence is None:
        raise HTTPException(status_code=404, detail=f"Sentence '{sentence_id}' not found")
    target_sentence = sentence.text

    locale = resolve_locale(language)

    try:
        wav_bytes = normalize_to_wav(audio_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ASR + pronunciation assessment (Azure).
    try:
        asr_result = assess_pronunciation(wav_bytes, target_sentence, locale)
    except AsrError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Speech service error: {exc}")

    if not asr_result.recognized_text:
        raise HTTPException(
            status_code=422, detail="I couldn't hear you clearly. Please try again."
        )

    # Stage 1: deterministic analysis + scoring.
    report = analyze(target_sentence, asr_result)

    # Stage 2: spoken feedback (fine-tuned model, template fallback).
    feedback_text = generate_feedback(report)

    # Persist this attempt to user_sentence_progress + user_lesson_progress.
    record_attempt(
        db=db,
        user_id=current_user_id,
        sentence=sentence,
        final_score=report.final_score
    )

    return AssessmentResponse(
        sentence_id=sentence.id,
        recognized_text=report.recognized_text,
        target_sentence=target_sentence,
        scores=report.scores,
        words=asr_result.words,
        final_score=report.final_score,
        is_passed=report.is_passed,
        missing_words=report.missing_words,
        extra_words=report.extra_words,
        substitutions=report.substitutions,
        mispronounced_words=report.mispronounced_words,
        feedback_points=report.feedback_points,
        feedback_text=feedback_text,
    )
