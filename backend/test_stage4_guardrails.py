"""
Test script for Stage 4: guardrails.py (Grounding Gate)

Run manually from backend directory:
    python test_stage4_guardrails.py
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

from app.config import settings
from app.pipeline.guardrails import check_grounding
from app.pipeline.retrieve import retrieve
from app.pipeline.types import Chunk, StageTiming

def run_tests():
    print("=" * 60)
    print("TEST 1: Empty Chunk List")
    print("=" * 60)
    timings_1: list[StageTiming] = []
    t0 = time.perf_counter()
    dec_1 = check_grounding([], timings=timings_1)
    dt_1 = (time.perf_counter() - t0) * 1000
    print(f"-> Decision: {dec_1}")
    print(f"-> Latency: {dt_1:.2f} ms")
    assert not dec_1.can_answer, "Expected can_answer=False for empty chunks"
    assert dec_1.top_score == 0.0

    print("\n" + "=" * 60)
    print("TEST 2: Mock Low Similarity Score (Below Threshold 0.83)")
    print("=" * 60)
    low_score_chunks = [Chunk(id="c1", text="Unrelated passage", score=0.8122)]
    timings_2: list[StageTiming] = []
    t0 = time.perf_counter()
    dec_2 = check_grounding(low_score_chunks, timings=timings_2)
    dt_2 = (time.perf_counter() - t0) * 1000
    print(f"-> Chunks Top Score: {low_score_chunks[0].score:.4f}")
    print(f"-> Decision: {dec_2}")
    print(f"-> Latency: {dt_2:.2f} ms")
    assert not dec_2.can_answer, "Expected can_answer=False for 0.8122 < 0.83"

    print("\n" + "=" * 60)
    print("TEST 3: Mock High Similarity Score (Above Threshold 0.83)")
    print("=" * 60)
    high_score_chunks = [Chunk(id="c2", text="Grounded passage", score=0.8942)]
    timings_3: list[StageTiming] = []
    t0 = time.perf_counter()
    dec_3 = check_grounding(high_score_chunks, timings=timings_3)
    dt_3 = (time.perf_counter() - t0) * 1000
    print(f"-> Chunks Top Score: {high_score_chunks[0].score:.4f}")
    print(f"-> Decision: {dec_3}")
    print(f"-> Latency: {dt_3:.2f} ms")
    assert dec_3.can_answer, "Expected can_answer=True for 0.8942 >= 0.83"

    print("\n" + "=" * 60)
    print("TEST 4: Live Integration Test (Retrieve + Grounding Gate)")
    print("=" * 60)
    ungrounded_query = "How to bake a chocolate chip cookie recipe at home?"
    chunks_un = retrieve(ungrounded_query, k=5, collection="chunks_semantic")
    dec_un = check_grounding(chunks_un)
    print(f"\n[Ungrounded Query]: '{ungrounded_query}'")
    print(f"-> Top Score: {chunks_un[0].score if chunks_un else 0.0:.4f}")
    print(f"-> Decision: {dec_un}")
    assert not dec_un.can_answer, f"Expected can_answer=False for ungrounded query, got {dec_un.can_answer}"

    from app.eval.eval_recall import load_ground_truth
    grounded_eq = load_ground_truth(language="hi", sample_limit=200)[0]
    chunks_gr = retrieve(grounded_eq.query_text, k=5, collection="chunks_semantic")
    dec_gr = check_grounding(chunks_gr)
    print(f"\n[Grounded Query]: '{grounded_eq.query_text[:50]}...'")
    print(f"-> Top Score: {chunks_gr[0].score if chunks_gr else 0.0:.4f}")
    print(f"-> Decision: {dec_gr}")
    assert dec_gr.can_answer, f"Expected can_answer=True for grounded query, got {dec_gr.can_answer}"

    print("\n" + "=" * 60)
    print("SUMMARY: ALL STAGE 4 GUARDRAILS TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
