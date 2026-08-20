"""
Stage 1: transcribe(audio_bytes) -> Transcript

Integrates with Sarvam AI's Speech-to-Text API (Endpoint: https://api.sarvam.ai/speech-to-text).
Uses model `saaras:v3` and language `hi-IN` in plain transcribe mode.
Distinguishes non-retryable errors (missing API key, client payload/auth HTTP 400/401/403/422) from retryable network/5xx errors.
Extracts and logs detailed error messages from Sarvam API response bodies.
"""
import io
import logging
import time
import requests

from app.config import settings
from app.pipeline.types import Transcript, timed_stage

logger = logging.getLogger("ragingoa.transcribe")


class TranscriptionError(Exception):
    """Raised when transcription fails after all retry attempts or due to missing credentials."""
    pass


class NonRetryableTranscriptionError(Exception):
    """Internal exception for non-retryable errors (missing keys, auth failures, 400/422 client errors)."""
    pass


def _call_sarvam_api(audio_bytes: bytes, filename: str = "audio.wav", mime_type: str = "audio/wav") -> dict:
    if not settings.sarvam_api_key:
        raise NonRetryableTranscriptionError("SARVAM_API_KEY is missing in configuration.")

    model = getattr(settings, "sarvam_model", "saaras:v3") or "saaras:v3"
    mode = getattr(settings, "sarvam_mode", "transcribe") or "transcribe"
    language_code = "hi-IN"

    logger.info("Calling Sarvam STT via REST HTTP API (model=%s, lang=%s)...", model, language_code)
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": settings.sarvam_api_key,
    }
    files = {
        "file": (filename, audio_bytes, mime_type),
    }
    data = {
        "model": model,
        "mode": mode,
        "language_code": language_code,
    }

    resp = requests.post(
        url,
        headers=headers,
        files=files,
        data=data,
        timeout=getattr(settings, "stt_timeout_seconds", 8.0),
    )

    # 400, 401, 403, 422 are non-retryable client side errors
    if resp.status_code in (400, 401, 403, 422):
        err_msg = resp.text
        try:
            err_json = resp.json()
            err_msg = (
                err_json.get("error", {}).get("message")
                or err_json.get("detail")
                or err_json.get("message")
                or resp.text
            )
        except Exception:
            pass
        raise NonRetryableTranscriptionError(f"Sarvam API Client Error (HTTP {resp.status_code}): {err_msg}")

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        err_msg = resp.text
        try:
            err_json = resp.json()
            err_msg = err_json.get("error", {}).get("message") or err_json.get("detail") or resp.text
        except Exception:
            pass
        raise requests.HTTPError(f"Sarvam API Server Error (HTTP {resp.status_code}): {err_msg}", response=resp) from e

    return resp.json()


@timed_stage("transcribe")
def transcribe(audio_bytes: bytes, filename: str = "audio.wav", mime_type: str = "audio/wav") -> Transcript:
    """
    Transcribes audio bytes using Sarvam STT API with model saaras:v3 and language hi-IN.
    Non-retryable errors (missing API key, HTTP 400/401/403/422) fail fast without delay.
    Retryable network errors use exponential backoff.
    Returns Transcript populated on success, or empty Transcript(text="") on failure.
    """
    if not audio_bytes:
        logger.warning("Sarvam STT received empty audio bytes payload. Returning empty transcript.")
        return Transcript(text="", language=None, confidence=None)

    max_retries = min(getattr(settings, "stt_max_retries", 2), 3)
    last_err: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            data = _call_sarvam_api(audio_bytes, filename=filename, mime_type=mime_type)
            transcript_text = (data.get("transcript") or data.get("text") or "").strip()
            lang = data.get("language_code", "hi-IN")
            conf = data.get("confidence") or data.get("language_probability")

            logger.info("Sarvam STT succeeded: transcript_len=%d, language=%s, confidence=%s", len(transcript_text), lang, conf)
            return Transcript(text=transcript_text, language=lang, confidence=conf)

        except NonRetryableTranscriptionError as e:
            logger.error("Sarvam STT non-retryable error (aborting retries): %s", e)
            return Transcript(text="", language=None, confidence=None)

        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < max_retries:
                delay = 0.3 * (2 ** attempt)
                logger.warning("Sarvam STT attempt %d/%d failed: %s. Retrying in %.2fs...", attempt + 1, max_retries + 1, e, delay)
                time.sleep(delay)
            else:
                logger.error("Sarvam STT failed after %d attempts: %s", max_retries + 1, e)

    return Transcript(text="", language=None, confidence=None)




