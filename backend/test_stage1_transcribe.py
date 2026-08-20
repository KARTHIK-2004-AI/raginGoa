"""
Test script for Stage 1: transcribe.py

Run manually from the backend directory:
    cd backend
    python -m app.pipeline.transcribe_test  # or python scratch_test.py
"""
import sys
import logging
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(r"c:\Users\YS TECH CENTER\Downloads\raginGoa\raginGoa\backend")
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Configure logging to stdout so user sees exact execution logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

from app.config import settings
from app.pipeline.transcribe import transcribe
from app.pipeline.types import StageTiming

def run_tests():
    print("=" * 60)
    print("TEST 1: Empty audio payload (Early Return Fast Fail)")
    print("=" * 60)
    timings_1: list[StageTiming] = []
    t0 = time.perf_counter()
    res_1 = transcribe(b"", timings=timings_1)
    dt_1 = (time.perf_counter() - t0) * 1000
    print(f"-> Result Dataclass: {res_1}")
    print(f"-> Total Time Taken: {dt_1:.2f} ms")
    print(f"-> Recorded Timings: {timings_1}\n")

    print("=" * 60)
    print("TEST 2: Missing API Key (Non-retryable Fast Fail)")
    print("=" * 60)
    original_key = settings.sarvam_api_key
    # Temporarily remove API key
    object.__setattr__(settings, "sarvam_api_key", "")
    timings_2: list[StageTiming] = []
    t0 = time.perf_counter()
    res_2 = transcribe(b"fake_audio_header_bytes", timings=timings_2)
    dt_2 = (time.perf_counter() - t0) * 1000
    print(f"-> Result Dataclass: {res_2}")
    print(f"-> Total Time Taken: {dt_2:.2f} ms (no retry delays!)")
    print(f"-> Recorded Timings: {timings_2}\n")

    print("=" * 60)
    print("TEST 3: Corrupted / Bad API Key (HTTP 401 Auth Failure Fast Fail)")
    print("=" * 60)
    object.__setattr__(settings, "sarvam_api_key", "invalid_key_401_test")
    timings_3: list[StageTiming] = []
    t0 = time.perf_counter()
    res_3 = transcribe(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00", timings=timings_3)
    dt_3 = (time.perf_counter() - t0) * 1000
    print(f"-> Result Dataclass: {res_3}")
    print(f"-> Total Time Taken: {dt_3:.2f} ms")
    print(f"-> Recorded Timings: {timings_3}\n")

    # Restore key
    object.__setattr__(settings, "sarvam_api_key", original_key)

    audio_files = sorted(
        list(backend_dir.glob("*.wav")) +
        list(backend_dir.glob("*.mp3")) +
        list(backend_dir.glob("*.m4a"))
    )

    if audio_files:
        print("=" * 60)
        print(f"TEST 4: Live Audio Transcription for {len(audio_files)} audio file(s)")
        print("=" * 60)
        for audio_file in audio_files:
            print(f"\n--- Transcribing: '{audio_file.name}' ({audio_file.stat().st_size} bytes) ---")
            audio_bytes = audio_file.read_bytes()
            timings: list[StageTiming] = []
            t0 = time.perf_counter()
            res = transcribe(audio_bytes, filename=audio_file.name, mime_type="audio/wav", timings=timings)
            dt = (time.perf_counter() - t0) * 1000
            print(f"-> Transcribed Text: '{res.text}'")
            print(f"-> Language: {res.language}, Confidence: {res.confidence}")
            print(f"-> Total Latency: {dt:.2f} ms")
            print(f"-> Recorded Timings: {timings}")
        print()
    elif settings.sarvam_api_key:
        print("=" * 60)
        print("TEST 4: Live Call against Sarvam AI API (Real Key Present)")
        print("=" * 60)
        timings_4: list[StageTiming] = []
        t0 = time.perf_counter()
        wav_header = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        res_4 = transcribe(wav_header, timings=timings_4)
        dt_4 = (time.perf_counter() - t0) * 1000
        print(f"-> Result Dataclass: {res_4}")
        print(f"-> Total Time Taken: {dt_4:.2f} ms")
        print(f"-> Recorded Timings: {timings_4}\n")

    print("=" * 60)
    print("SUMMARY: ALL STAGE 1 TRANSCRIBE TESTS COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
