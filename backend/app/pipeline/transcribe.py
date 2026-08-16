"""
Stage 1: transcribe(audio_bytes) -> Transcript

Integrates with Sarvam AI's Speech-to-Text API (Endpoint: https://api.sarvam.ai/speech-to-text).
Uses model `saarika:v2` by default. Handles transient failures with exponential retries.
"""
import time

import requests

from app.config import settings
from app.pipeline.types import Transcript, timed_stage


class TranscriptionError(Exception):
    """Raised when transcription fails after all retry attempts or due to missing credentials."""
    pass


def _call_sarvam(audio_bytes: bytes, filename: str = "audio.wav", mime_type: str = "audio/wav") -> dict:
    if not settings.sarvam_api_key:
        raise TranscriptionError(
            "SARVAM_API_KEY is missing. Please set SARVAM_API_KEY in backend/.env"
        )

    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": settings.sarvam_api_key,
    }
    files = {
        "file": (filename, audio_bytes, mime_type),
    }
    data = {
        "model": settings.sarvam_model,
        "mode": settings.sarvam_mode,
        "language_code": "unknown",
    }

    resp = requests.post(
        url,
        headers=headers,
        files=files,
        data=data,
        timeout=settings.stt_timeout_seconds,
    )

    if resp.status_code == 401:
        raise TranscriptionError("Sarvam API authentication failed (HTTP 401). Check SARVAM_API_KEY.")

    resp.raise_for_status()
    return resp.json()


@timed_stage("transcribe")
def transcribe(audio_bytes: bytes, filename: str = "audio.wav", mime_type: str = "audio/wav") -> Transcript:
    """
    Transcribes audio bytes using Sarvam STT API.
    Retries up to settings.stt_max_retries times upon transient failure.
    """
    if not audio_bytes:
        raise TranscriptionError("Empty audio payload received.")

    last_err: Exception | None = None
    for attempt in range(settings.stt_max_retries + 1):
        try:
            data = _call_sarvam(audio_bytes, filename=filename, mime_type=mime_type)
            transcript_text = data.get("transcript", "").strip()

            if not transcript_text and "text" in data:
                transcript_text = data.get("text", "").strip()

            return Transcript(
                text=transcript_text,
                language=data.get("language_code"),
                confidence=data.get("confidence"),
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < settings.stt_max_retries:
                time.sleep(0.3 * (attempt + 1))
            continue

    raise TranscriptionError(
        f"Sarvam STT failed after {settings.stt_max_retries + 1} attempts: {last_err}"
    )
