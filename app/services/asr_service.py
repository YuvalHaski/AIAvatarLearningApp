"""ASR layer: Azure Speech Pronunciation Assessment (scripted mode).

The student is repeating a known sentence, so we pass that sentence as the
`reference_text`. Azure returns sentence-level accuracy / fluency / completeness /
prosody scores plus per-word and per-phoneme scores with an error type. This
replaces Whisper, which can only transcribe and cannot judge pronunciation.
"""
import io
import json
import wave

import azure.cognitiveservices.speech as speechsdk

from app.core.config import settings
from app.schemas.asr import AsrResult, PhonemeScore, PronunciationScores, WordResult


class AsrError(Exception):
    """Raised when speech recognition / assessment cannot be completed."""


def _read_pcm(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
    """Extract raw PCM frames and format from a normalized WAV blob."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())
    return pcm, channels, sample_width, frame_rate


def _parse_words(json_result: str) -> list[WordResult]:
    """Parse per-word / per-phoneme detail from Azure's JSON payload."""
    try:
        data = json.loads(json_result)
    except (json.JSONDecodeError, TypeError):
        return []

    nbest = data.get("NBest") or []
    if not nbest:
        return []

    words: list[WordResult] = []
    for raw_word in nbest[0].get("Words") or []:
        assessment = raw_word.get("PronunciationAssessment") or {}
        phonemes = [
            PhonemeScore(
                phoneme=ph.get("Phoneme", ""),
                accuracy_score=(ph.get("PronunciationAssessment") or {}).get(
                    "AccuracyScore", 0.0
                ),
            )
            for ph in (raw_word.get("Phonemes") or [])
        ]
        words.append(
            WordResult(
                word=raw_word.get("Word", ""),
                accuracy_score=assessment.get("AccuracyScore", 0.0),
                error_type=assessment.get("ErrorType", "None"),
                phonemes=phonemes,
            )
        )
    return words


def assess_pronunciation(
    wav_bytes: bytes, reference_text: str, locale: str
) -> AsrResult:
    """Run Azure scripted Pronunciation Assessment on a normalized WAV blob.

    `wav_bytes` must already be 16 kHz mono 16-bit PCM WAV (see audio_utils).
    Raises AsrError on no-speech, cancellation, or configuration problems.
    """
    settings.require_azure()

    pcm, channels, sample_width, frame_rate = _read_pcm(wav_bytes)
    if not pcm:
        raise AsrError("Audio contained no samples.")

    stream_format = speechsdk.audio.AudioStreamFormat(
        samples_per_second=frame_rate,
        bits_per_sample=sample_width * 8,
        channels=channels,
    )
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format)
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

    speech_config = speechsdk.SpeechConfig(
        subscription=settings.AZURE_SPEECH_KEY,
        region=settings.AZURE_SPEECH_REGION,
    )
    speech_config.speech_recognition_language = locale

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    pa_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True,
    )
    # Pin the phoneme alphabet to SAPI so the codes Azure returns (e.g. "th",
    # "dh", "sh") match the anchor table in phoneme_hints. The default can vary
    # by locale/model; without this, voiced "th" can come back as IPA and slip
    # past our hint mapping, so "th" mistakes never get coached.
    try:
        pa_config.phoneme_alphabet = "SAPI"
    except Exception:
        pass
    # Prosody scoring is opt-in and only available for some locales.
    try:
        pa_config.enable_prosody_assessment()
    except Exception:
        pass
    pa_config.apply_to(recognizer)

    push_stream.write(pcm)
    push_stream.close()

    result = recognizer.recognize_once()

    if result.reason == speechsdk.ResultReason.NoMatch:
        raise AsrError("I couldn't hear you clearly. Please try again.")
    if result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        raise AsrError(f"Speech service error: {cancellation.reason} {cancellation.error_details}")
    if result.reason != speechsdk.ResultReason.RecognizedSpeech:
        raise AsrError(f"Unexpected recognition result: {result.reason}")

    pa_result = speechsdk.PronunciationAssessmentResult(result)
    accuracy = pa_result.accuracy_score or 0.0
    prosody = pa_result.prosody_score
    scores = PronunciationScores(
        accuracy=accuracy,
        fluency=pa_result.fluency_score or 0.0,
        completeness=pa_result.completeness_score or 0.0,
        # Prosody can be None when the locale doesn't support it; fall back to a
        # neutral value so it doesn't distort the Stage 1 score.
        prosody=prosody if prosody is not None else accuracy,
        pronunciation=pa_result.pronunciation_score or 0.0,
    )

    json_result = result.properties.get(
        speechsdk.PropertyId.SpeechServiceResponse_JsonResult
    )
    words = _parse_words(json_result)

    return AsrResult(
        recognized_text=(result.text or "").strip(),
        scores=scores,
        words=words,
    )
