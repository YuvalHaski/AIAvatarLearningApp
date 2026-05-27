"""Small builders for test fixtures."""
from app.schemas.asr import AsrResult, PronunciationScores, WordResult


def make_asr_result(
    recognized: str,
    *,
    accuracy: float = 90,
    fluency: float = 90,
    completeness: float = 100,
    prosody: float = 90,
    words: list[WordResult] | None = None,
) -> AsrResult:
    return AsrResult(
        recognized_text=recognized,
        scores=PronunciationScores(
            accuracy=accuracy,
            fluency=fluency,
            completeness=completeness,
            prosody=prosody,
            pronunciation=accuracy,
        ),
        words=words or [],
    )
