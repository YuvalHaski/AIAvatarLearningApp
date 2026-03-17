import io
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.schemas.asr import ASRCombinedOut
from app.services.ai_service import call_llm, client

# We use APIRouter instead of 'app' to group related endpoints
router = APIRouter()

MAX_AUDIO_MB = 20

@router.post("/asr", response_model=ASRCombinedOut)
async def asr(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    target_sentence: str | None = Form(None),
):
    if not file.content_type or not file.content_type.startswith(("audio/", "video/")):
        raise HTTPException(status_code=400, detail=f"Unsupported content-type: {file.content_type}")

    audio_bytes = await file.read()
    if len(audio_bytes) > MAX_AUDIO_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (>{MAX_AUDIO_MB}MB)")

    if not target_sentence:
        raise HTTPException(status_code=400, detail="target_sentence is required")

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = file.filename or "audio.wav"

    try:
        # ASR (Speech-to-Text) using Whisper
        asr_res = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language or "en",
            prompt=target_sentence or "",
            temperature=0,
            response_format="verbose_json",
        )

        asr_data = asr_res.model_dump() if hasattr(asr_res, "model_dump") else dict(asr_res)
        transcript = (asr_data.get("text") or "").strip()

        if not transcript:
            raise HTTPException(status_code=422, detail="I couldn't hear you clearly. Please try again.")

        segments = asr_data.get("segments") or []
        if segments:
            no_speech = [s.get("no_speech_prob", 0) for s in segments if isinstance(s, dict)]
            avg_logprob = [s.get("avg_logprob", 0) for s in segments if isinstance(s, dict)]

            if no_speech and (sum(no_speech) / len(no_speech)) > 0.6:
                raise HTTPException(status_code=422, detail="I couldn't hear you clearly. Please try again.")
            if avg_logprob and (sum(avg_logprob) / len(avg_logprob)) < -1.2:
                raise HTTPException(status_code=422, detail="Audio was unclear. Please try again more slowly.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ASR error: {str(e)}")

    # Call our Service to get the LLM Evaluation
    llm_out = call_llm(transcript, target_sentence, language)

    return {"transcript": transcript, "llm": llm_out.model_dump()}