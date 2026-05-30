"""Audio normalization for Azure Pronunciation Assessment.

Uses imageio-ffmpeg to run a direct conversion from M4A to 16kHz Mono 16-bit PCM WAV.
Bypasses pydub entirely to avoid ffprobe dependencies.
"""
import subprocess
import imageio_ffmpeg

# Azure Pronunciation Assessment expects this exact PCM shape.
TARGET_SAMPLE_RATE = "16000"
TARGET_CHANNELS = "1"

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
    """Decode audio and re-encode as 16 kHz mono 16-bit PCM WAV using direct FFmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Build the direct FFmpeg command
    command = [
        ffmpeg_exe,
        "-i", "pipe:0",           # Read input from stdin
        "-f", "wav",              # Force output format to WAV
        "-acodec", "pcm_s16le",   # Audio codec: 16-bit PCM
        "-ac", TARGET_CHANNELS,   # Channels: 1 (Mono)
        "-ar", TARGET_SAMPLE_RATE,# Sample Rate: 16000 Hz
        "pipe:1"                  # Write output to stdout
    ]

    try:
        # Run ffmpeg, feed the m4a bytes to stdin, get wav bytes from stdout
        process = subprocess.run(
            command,
            input=audio_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return process.stdout
    except subprocess.CalledProcessError as exc:
        # If ffmpeg fails, print its error log
        error_msg = exc.stderr.decode(errors="ignore")
        raise ValueError(f"FFmpeg conversion failed: {error_msg}")