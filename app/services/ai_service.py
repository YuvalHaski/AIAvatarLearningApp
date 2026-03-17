import os
import json
from openai import OpenAI
from fastapi import HTTPException
from dotenv import load_dotenv
from app.schemas.asr import LLMOut

# Load env variables (ensures OPENAI_API_KEY is available)
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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

    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",  # Note: Verify if this is your intended model name (maybe gpt-4o-mini?)
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        return LLMOut(**data)

    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"LLM returned invalid JSON: {content}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")