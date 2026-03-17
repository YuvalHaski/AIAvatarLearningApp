from pydantic import BaseModel, Field

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