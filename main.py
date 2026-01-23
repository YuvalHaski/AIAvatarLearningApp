import os, io, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MAX_AUDIO_MB = 20


class Substitution(BaseModel):
    expected: str
    heard: str


class LLMOut(BaseModel):
    is_correct: bool
    corrected_text: str
    missing_words: list[str]
    extra_words: list[str]
    substitutions: list[Substitution]
    feedback: list[str]
    score_0_to_100: int = Field(ge=0, le=100)
    detected_language: str | None = None


class ASRCombinedOut(BaseModel):
    transcript: str
    llm: LLMOut


def call_llm(transcript: str, target_sentence: str | None, language: str | None) -> LLMOut:
    if not target_sentence or not target_sentence.strip():
        raise HTTPException(status_code=400, detail="target_sentence is required for repeat-after-me mode")

    system = (
        "You are a language teacher for a repeat-after-me speaking exercise. "
        "The student is NOT creating their own sentence. They are trying to repeat the target sentence exactly.\n\n"
        "You will receive:\n"
        "- target_sentence: what the avatar asked the student to say\n"
        "- transcript: what the ASR (speech-to-text) heard\n\n"
        "Your job:\n"
        "1) Decide if the transcript matches the target closely enough.\n"
        "2) If it does not match, identify what was wrong using simple word-level comparison:\n"
        "   - missing_words: words that appear in target but not in transcript\n"
        "   - extra_words: words that appear in transcript but not in target\n"
        "   - substitutions: where a target word was replaced by a different word\n"
        "3) Give feedback that is short, spoken-friendly, and encouraging (the avatar will read it aloud).\n"
        "4) If you mention pronunciation, phrase it as a suggestion (because ASR is imperfect).\n\n"
        "Scoring rules (repeat accuracy):\n"
        "- 95–100: exact or near-exact match (tiny filler words ok)\n"
        "- 80–94: small mistakes (1–2 word issues)\n"
        "- 60–79: several word issues but still recognizable\n"
        "- 0–59: far from target\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        "- is_correct (boolean)\n"
        "- corrected_text (string) = the target sentence (clean punctuation/capitalization)\n"
        "- missing_words (string[])\n"
        "- extra_words (string[])\n"
        "- substitutions (array of {\"expected\": string, \"heard\": string})\n"
        "- feedback (string[]) (2–5 short teacher sentences)\n"
        "- score_0_to_100 (integer 0–100)\n"
        "- detected_language (string|null)\n\n"
        "Do not use markdown. Do not add extra fields."
    )

    payload = {
        "target_sentence": target_sentence,
        "transcript": transcript,
        "expected_language": language,
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

    content = resp.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {content}")

    try:
        return LLMOut(**data)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM JSON did not match schema: {str(e)}")


@app.post("/asr", response_model=ASRCombinedOut)
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
        asr_res = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language or "en",          # help Whisper
            prompt=target_sentence or "",       # repeat-after-me hint
            temperature=0,                      # more deterministic
            response_format="verbose_json",     # gives segments w/ signals
        )

        asr_data = asr_res.model_dump() if hasattr(asr_res, "model_dump") else dict(asr_res)
        transcript = (asr_data.get("text") or "").strip()

        # Basic "bad ASR" gate
        if not transcript:
            raise HTTPException(status_code=422, detail="I couldn't hear you clearly. Please try again.")

        segments = asr_data.get("segments") or []
        if segments:
            no_speech = [s.get("no_speech_prob", 0) for s in segments if isinstance(s, dict)]
            avg_logprob = [s.get("avg_logprob", 0) for s in segments if isinstance(s, dict)]

            # Heuristics (tune if needed)
            if no_speech and (sum(no_speech) / len(no_speech)) > 0.6:
                raise HTTPException(status_code=422, detail="I couldn't hear you clearly. Please try again.")
            if avg_logprob and (sum(avg_logprob) / len(avg_logprob)) < -1.2:
                raise HTTPException(status_code=422, detail="Audio was unclear. Please try again more slowly.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ASR error: {str(e)}")

    # LLM call stays the same
    try:
        llm_out = call_llm(transcript, target_sentence, language)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    return {"transcript": transcript, "llm": llm_out.model_dump()}
