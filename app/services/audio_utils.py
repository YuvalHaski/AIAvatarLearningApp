"""Audio normalization for Azure Pronunciation Assessment.

Uses imageio-ffmpeg to run a direct conversion from M4A to 16kHz Mono 16-bit PCM WAV.
Bypasses pydub entirely to avoid ffprobe dependencies.
"""
import os
import subprocess
import tempfile

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

# Silence padded around the speech, in seconds. The Android recorder calls
# MediaRecorder.stop() the instant the user releases the button, which clips
# the tail off the final word. With no trailing silence Azure can't see the
# word's full release, so it segments the last word short and scores it low.
# Padding both ends gives Azure clean boundaries to assess against.
_LEAD_SILENCE_SEC = "0.3"
_TRAIL_SILENCE_SEC = "0.8"

# Sanity check on the decoded result. 16 kHz * 2 bytes per 16-bit sample.
_BYTES_PER_SEC = 16000 * 2
_WAV_HEADER_BYTES = 78
_PADDING_BYTES = int((float(_LEAD_SILENCE_SEC) + float(_TRAIL_SILENCE_SEC)) * _BYTES_PER_SEC)
# Shorter than this and there is no utterance worth assessing (0.25 s).
_MIN_SPEECH_BYTES = int(0.25 * _BYTES_PER_SEC)


def normalize_to_wav(audio_bytes: bytes) -> bytes:
    """Decode audio and re-encode as 16 kHz mono 16-bit PCM WAV using direct FFmpeg.

    A short pad of silence is added before and after the speech so Azure's
    Pronunciation Assessment can segment the first and last words cleanly
    (see the constants above).
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # adelay prepends leading silence; apad appends trailing silence.
    pad_filter = (
        f"adelay={int(float(_LEAD_SILENCE_SEC) * 1000)}:all=1,"
        f"apad=pad_dur={_TRAIL_SILENCE_SEC}"
    )

    # The input MUST be a real file, not stdin. M4A/MP4 keeps its index (the
    # `moov` atom) at the end of the file, so a demuxer reading a non-seekable
    # pipe cannot find it: ffmpeg logs "Error during demuxing" but still exits
    # 0, and emits a WAV containing nothing but the padding above. Azure then
    # sees silence and marks every word as an omission, scoring the attempt 0.
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    command = [
        ffmpeg_exe,
        "-i", tmp_path,           # Seekable input, so MP4 demuxing works
        "-af", pad_filter,        # Pad silence on both ends for clean segmentation
        "-f", "wav",              # Force output format to WAV
        "-acodec", "pcm_s16le",   # Audio codec: 16-bit PCM
        "-ac", TARGET_CHANNELS,   # Channels: 1 (Mono)
        "-ar", TARGET_SAMPLE_RATE,# Sample Rate: 16000 Hz
        "pipe:1"                  # Write output to stdout
    ]

    try:
        process = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as exc:
        error_msg = exc.stderr.decode(errors="ignore")
        raise ValueError(f"FFmpeg conversion failed: {error_msg}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    wav = process.stdout
    # ffmpeg can exit 0 while producing only the padding (see above), so verify
    # that actual speech survived instead of sending silence to Azure.
    speech_bytes = len(wav) - _WAV_HEADER_BYTES - _PADDING_BYTES
    if speech_bytes < _MIN_SPEECH_BYTES:
        detail = process.stderr.decode(errors="ignore").strip().splitlines()
        reason = detail[-1] if detail else "no decoder output"
        raise ValueError(
            f"Audio decoded to little or no speech ({max(speech_bytes, 0)} bytes). "
            f"ffmpeg: {reason}"
        )
    return wav