"""
Test script for Stage 6: verify.py (Deterministic Factual Grounding Verification)

Run manually from backend directory:
    python test_stage6_verify.py
"""
import sys
import logging
import time
from pathlib import Path

backend_dir = Path(r"c:\Users\YS TECH CENTER\Downloads\raginGoa\raginGoa\backend")
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)

from app.eval.eval_recall import load_ground_truth
from app.pipeline.generate import generate
from app.pipeline.retrieve import retrieve
from app.pipeline.verify import verify
from app.pipeline.types import Answer, Chunk, StageTiming, VerifiedAnswer


def run_tests():
    dummy_chunks = [
        Chunk(id="c1", text="मैनहट्टन परियोजना 1939 में द्वितीय विश्व युद्ध के दौरान शुरू हुई थी।", score=0.9),
        Chunk(id="c2", text="अमेरिका ने 1945 में परमाणु बम का सफल परीक्षण किया।", score=0.85),
    ]

    print("=" * 60)
    print("TEST 1: Empty Answer Text")
    print("=" * 60)
    timings_1: list[StageTiming] = []
    t0 = time.perf_counter()
    v_1 = verify(Answer(text="", citations=[]), dummy_chunks, timings=timings_1)
    dt_1 = (time.perf_counter() - t0) * 1000
    print(f"-> Verified Result: {v_1}")
    print(f"-> Latency: {dt_1:.2f} ms")
    assert v_1.is_fully_grounded is True, "Empty answer should be trivially grounded"
    assert v_1.flagged_claims == [], "Empty answer should have 0 flagged claims"

    print("\n" + "=" * 60)
    print("TEST 2: Answer with Citations but No Numeric Claims")
    print("=" * 60)
    ans_no_num = Answer(text="मैनहट्टन परियोजना एक शोध और विकास कार्यक्रम था [1] ।", citations=["c1"])
    timings_2: list[StageTiming] = []
    t0 = time.perf_counter()
    v_2 = verify(ans_no_num, dummy_chunks, timings=timings_2)
    dt_2 = (time.perf_counter() - t0) * 1000
    print(f"-> Verified Result: {v_2}")
    print(f"-> Latency: {dt_2:.2f} ms")
    assert v_2.is_fully_grounded is True, "Non-numeric claim answer should be trivially grounded"
    assert len(v_2.flagged_claims) == 0, "No claims should be flagged when no numbers exist"

    print("\n" + "=" * 60)
    print("TEST 3: Live End-to-End Pipeline (Retrieve -> Generate -> Verify)")
    print("=" * 60)
    eq = load_ground_truth(language="hi", sample_limit=200)[0]
    print(f"Query: '{eq.query_text}'")

    chunks = retrieve(eq.query_text, k=3, collection="chunks_semantic")
    print(f"Retrieved {len(chunks)} chunks from Qdrant.")
    assert len(chunks) > 0, "Retrieval returned 0 chunks"

    ans_live = generate(eq.query_text, chunks)
    print(f"-> Generated Answer Text:\n'{ans_live.text}'")

    timings_3: list[StageTiming] = []
    t0 = time.perf_counter()
    v_3 = verify(ans_live, chunks, timings=timings_3)
    dt_3 = (time.perf_counter() - t0) * 1000
    print(f"\n-> Verification Grounded: {v_3.is_fully_grounded}")
    print(f"-> Flagged Claims ({len(v_3.flagged_claims)}): {v_3.flagged_claims}")
    print(f"-> Verification Latency: {dt_3:.2f} ms")
    print(f"-> Recorded Timings: {timings_3}")
    assert isinstance(v_3, VerifiedAnswer), "Verification output should be a VerifiedAnswer instance"

    print("\n" + "=" * 60)
    print("TEST 4: Critical Negative Test (Fabricated Wrong Year / Ungrounded Number)")
    print("=" * 60)
    fabricated_ans = Answer(
        text="मैनहट्टन परियोजना वर्ष 1899 में शुरू की गई थी [1] और इसका समापन 1945 में हुआ [2] ।",
        citations=["c1", "c2"]
    )
    timings_4: list[StageTiming] = []
    t0 = time.perf_counter()
    v_4 = verify(fabricated_ans, dummy_chunks, timings=timings_4)
    dt_4 = (time.perf_counter() - t0) * 1000
    print(f"-> Verified Result: {v_4}")
    print(f"-> Flagged Claims: {v_4.flagged_claims}")
    print(f"-> Latency: {dt_4:.2f} ms")
    assert v_4.is_fully_grounded is False, "Fabricated wrong year MUST be flagged as ungrounded"
    assert len(v_4.flagged_claims) > 0, "Expected at least 1 flagged claim for wrong year 1899"
    assert any("1899" in fc.claim_text for fc in v_4.flagged_claims), "Claim text 1899 should be flagged"

    print("\n" + "=" * 60)
    print("SUMMARY: STAGE 6 VERIFY TESTS PASSED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
