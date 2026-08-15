"""
Stage 1: transcribe(audio) -> Transcript

Uses Sarvam's STT API. Retries on transient failure, times out per config.
Swap the HTTP call inside `_call_sarvam` if you end up using a different
Sarvam SDK/endpoint shape than assumed here — check their docs for the exact
request format before the hackathon deadline, don't trust this blindly.
"""
import time

import requests

from app.config import settings
from app.pipeline.types import Transcript, timed_stage


class TranscriptionError(Exception):
    pass


def _call_sarvam(audio_bytes: bytes) -> dict:
    resp = requests.post(
        "https://api.sarvam.ai/speech-to-text",
        headers={"api-subscription-key": settings.sarvam_api_key},
        files={"file": ("audio.wav", audio_bytes, "audio/wav")},
        data={"model": "saaras:v3"},
        timeout=settings.stt_timeout_seconds,
    )
    resp.raise_for_status()
    return resp.json()


@timed_stage("transcribe")
def transcribe(audio_bytes: bytes) -> Transcript:
    last_err: Exception | None = None
    for attempt in range(settings.stt_max_retries + 1):
        try:
            data = _call_sarvam(audio_bytes)
            return Transcript(
                text=data.get("transcript", "").strip(),
                language=data.get("language_code"),
                confidence=data.get("confidence"),
            )
        except Exception as e:  # noqa: BLE001 — deliberately broad, retry policy below
            last_err = e
            if attempt < settings.stt_max_retries:
                time.sleep(0.3 * (attempt + 1))  # small backoff
            continue

    # Defined failure mode instead of a stack trace to the user.
    raise TranscriptionError(
        f"STT failed after {settings.stt_max_retries + 1} attempts: {last_err}"
    )
