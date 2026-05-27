"""Audio normalization for Azure Pronunciation Assessment.

Browsers and mobile clients send audio in many containers (webm/ogg/m4a/wav).
Azure's recognizer is fed a 16 kHz, mono, 16-bit PCM WAV stream, so we convert
everything to that shape up front. Relies on `ffmpeg` being installed on the host.
"""
import io

from pydub import AudioSegment

# Azure Pronunciation Assessment expects this exact PCM shape.
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SAMPLE_WIDTH = 2  # bytes -> 16-bit

# Short language code -> Azure BCP-47 locale.
_LOCALE_MAP = {
    "en": "en-US",
    "en-us": "en-US",
    "en-gb": "en-GB",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "it": "it-IT",
    "pt": "pt-BR",
    "he": "he-IL",
}
DEFAULT_LOCALE = "en-US"


def resolve_locale(language: str | None) -> str:
    """Map a loose language code to an Azure locale, defaulting to en-US."""
    if not language:
        return DEFAULT_LOCALE
    return _LOCALE_MAP.get(language.strip().lower(), DEFAULT_LOCALE)


def normalize_to_wav(audio_bytes: bytes) -> bytes:
    """Decode any supported audio and re-encode as 16 kHz mono 16-bit PCM WAV.

    Raises ValueError if the bytes cannot be decoded as audio.
    """
    try:
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
    except Exception as exc:  # pydub raises a variety of errors / CouldntDecodeError
        raise ValueError(f"Could not decode audio: {exc}") from exc

    segment = (
        segment.set_frame_rate(TARGET_SAMPLE_RATE)
        .set_channels(TARGET_CHANNELS)
        .set_sample_width(TARGET_SAMPLE_WIDTH)
    )

    out = io.BytesIO()
    segment.export(out, format="wav")
    return out.getvalue()
