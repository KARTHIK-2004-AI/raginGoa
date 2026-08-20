"""
Integration test for FastAPI endpoint POST /ask and GET /health
using fastapi.testclient.TestClient against app.main.app.

Run manually from backend directory:
    python test_integration_ask.py
"""
import sys
import logging
import json
import base64
import requests
from pathlib import Path

from fastapi.testclient import TestClient

backend_dir = Path(r"c:\Users\YS TECH CENTER\Downloads\raginGoa\raginGoa\backend")
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

from app.config import settings
from app.main import app

client = TestClient(app)
logger = logging.getLogger("ragingoa.test_integration")


def run_tests():
    print("=" * 60)
    print("TEST 1: GET /health Endpoint")
    print("=" * 60)
    res_health = client.get("/health")
    print(f"-> Status Code: {res_health.status_code}")
    print(f"-> Response Body: {res_health.json()}")
    assert res_health.status_code == 200, "Health check should return HTTP 200"
    assert res_health.json() == {"status": "ok"}, "Health check body mismatch"

    print("\n" + "=" * 60)
    print("TEST 2: POST /ask with Empty Audio Bytes (Validation Check)")
    print("=" * 60)
    res_empty = client.post("/ask", files={"audio": ("empty.wav", b"", "audio/wav")})
    print(f"-> Status Code: {res_empty.status_code}")
    print(f"-> Response Body: {res_empty.json()}")
    assert res_empty.status_code == 400, "Empty audio upload must return HTTP 400"
    assert "empty audio upload" in res_empty.json().get("detail", "").lower()

    # Discover existing audio files in backend directory
    audio_files = sorted(
        list(backend_dir.glob("*.wav")) + list(backend_dir.glob("*.mp3"))
    )
    unique_audio_files = [f for f in audio_files if f.stat().st_size > 0]

    print("\n" + "=" * 60)
    print(f"TEST 3: POST /ask Real Audio Files Upload Test ({len(unique_audio_files)} files found)")
    print("=" * 60)

    for audio_path in audio_files:
        print(f"\n--- Uploading Audio File: '{audio_path.name}' ({audio_path.stat().st_size} bytes) ---")
        audio_bytes = audio_path.read_bytes()

        res_ask = client.post("/ask", files={"audio": (audio_path.name, audio_bytes, "audio/wav")})
        print(f"-> HTTP Status Code: {res_ask.status_code}")
        body = res_ask.json()
        print(f"-> Response Body JSON:\n{json.dumps(body, indent=2, ensure_ascii=False)}")

        assert res_ask.status_code == 200, "Pipeline HTTP upload should return HTTP 200"

        expected_keys = {"answer", "citations", "is_fully_grounded", "flagged_claims", "stopped_at", "latency_ms"}
        assert expected_keys.issubset(body.keys()), f"Missing expected keys in response JSON: {expected_keys - body.keys()}"

        assert isinstance(body["answer"], str), "answer must be a string"
        assert isinstance(body["citations"], list), "citations must be a list"
        assert isinstance(body["is_fully_grounded"], bool), "is_fully_grounded must be a boolean"
        assert isinstance(body["flagged_claims"], list), "flagged_claims must be a list"
        assert isinstance(body["stopped_at"], str), "stopped_at must be a string"
        assert isinstance(body["latency_ms"], dict), "latency_ms must be a dictionary"

        for fc in body["flagged_claims"]:
            assert isinstance(fc, dict), "Each flagged claim entry must be a dictionary"
            claim_keys = {"claim_text", "citation_index", "chunk_id", "reason"}
            assert claim_keys.issubset(fc.keys()), f"Flagged claim dict missing keys: {claim_keys - fc.keys()}"

        if "manhattan" in audio_path.name.lower():
            ans = body["answer"]
            (backend_dir / "task2_answer.txt").write_text(ans, encoding="utf-8")
            import re
            print("\n--- TASK 2 TERMINAL-INDEPENDENT WHITESPACE ANALYSIS ---")
            print(f"-> Length: {len(ans)} chars")
            print(f"-> Contains double space ('  '): {'  ' in ans}")
            multi_ws = re.findall(r"\s{2,}", ans)
            print(f"-> Multi-whitespace runs: {multi_ws}")
            print(f"-> Dumped exact UTF-8 string to: '{backend_dir / 'task2_answer.txt'}'")
            assert body["stopped_at"] == "verify", "On-topic audio query must reach verify stage"
            assert len(ans) > 0, "On-topic answer text must not be empty"

    if unique_audio_files:
        print("\n" + "=" * 60)
        print("TEST 4: Consecutive Warm-Load Retrieval Latency Benchmarking (3 Runs)")
        print("=" * 60)
        sample_audio = unique_audio_files[0]
        audio_bytes = sample_audio.read_bytes()
        retrieval_latencies = []
        for i in range(1, 4):
            res_repeat = client.post("/ask", files={"audio": (sample_audio.name, audio_bytes, "audio/wav")})
            body_rep = res_repeat.json()
            r_ms = body_rep.get("latency_ms", {}).get("stages", {}).get("retrieve", 0.0)
            retrieval_latencies.append(r_ms)
            print(f"\n--- TEST 4 Run {i} Full Response Body JSON ---")
            print(json.dumps(body_rep, indent=2, ensure_ascii=False))
            print(f"-> Run {i} Retrieve Stage Latency: {r_ms:.2f} ms (Total Pipeline: {body_rep.get('latency_ms', {}).get('total', 0.0):.2f} ms)")

        print(f"\n-> Consecutive Retrieval Latencies across warm runs: {retrieval_latencies}")
        assert retrieval_latencies[1] < 2000, f"Retrieve stage should be warm on run 2, got {retrieval_latencies[1]}ms"

    print("\n" + "=" * 60)
    print("SUMMARY: ALL PHASE 2 CLOSEOUT & INTEGRATION TESTS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
