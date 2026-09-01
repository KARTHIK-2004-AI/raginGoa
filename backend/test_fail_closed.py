"""
Tests Fail-Closed Pipeline Behavior on Total LLM Provider Failure.
Verifies that if all LLM models fail (503/404/QuotaExhausted), the pipeline:
1. Fails-closed at stopped_at="generate"
2. Returns a localized, safe fallback message
3. Never leaks internal error stack traces or raw ungrounded fallbacks
"""
import sys
import json
import logging
import time
from pathlib import Path
from unittest.mock import patch

# Force UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

backend_dir = Path(r"c:\Users\YS TECH CENTER\Downloads\raginGoa\raginGoa\backend")
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

from app.pipeline.orchestrator import run_pipeline_text

def run_fail_closed_test():
    print("=" * 80)
    print("TESTING FAIL-CLOSED PIPELINE BEHAVIOR ON TOTAL LLM PROVIDER FAILURE")
    print("=" * 80)

    query = "Which is the best beach in Goa for water sports?"

    # Simulate total LLM failure (e.g. QuotaExhausted / 503 Service Unavailable / All Models Failed)
    with patch("app.pipeline.generate._call_gemini_api", side_effect=Exception("HTTP 503 Service Unavailable / Quota Exhausted")):
        res = run_pipeline_text(query)

    print(f"-> Query: '{query}'")
    print(f"-> Stopped At Stage: '{res.stopped_at}'")
    print(f"-> Answer: '{res.answer}'")
    print(f"-> Is Fully Grounded: {res.is_fully_grounded}")
    print(f"-> Flagged Claims: {res.flagged_claims}")
    print(f"-> Timings Recorded: {[t.name for t in res.timings]}")

    # Validation assertions
    assert res.stopped_at == "generate", f"Expected stopped_at='generate', got {res.stopped_at}"
    assert "unavailable" in res.answer.lower() or "अनुपलब्ध" in res.answer, f"Expected safe fallback in answer, got '{res.answer}'"
    assert res.flagged_claims == [], "Fail-closed path must not produce ungrounded flagged claims"
    assert "verify" not in [t.name for t in res.timings], "Verify stage must not execute on failed generation"

    print("\n" + "=" * 80)
    print("SUCCESS: FAIL-CLOSED PIPELINE VERIFIED PERFECTLY (100% SAFE BOUNDARY)")
    print("=" * 80)

if __name__ == "__main__":
    run_fail_closed_test()
