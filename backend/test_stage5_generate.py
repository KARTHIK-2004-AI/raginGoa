"""
Test script for Stage 5: generate.py (Gemini LLM Answer Generation)

Run manually from backend directory:
    python test_stage5_generate.py
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
from app.pipeline.types import Answer, Chunk, StageTiming

def run_tests():
    print("=" * 60)
    print("TEST 1: Empty Chunks or Empty Query Input")
    print("=" * 60)
    timings_1: list[StageTiming] = []
    t0 = time.perf_counter()
    ans_1 = generate("dummy query", [], timings=timings_1)
    dt_1 = (time.perf_counter() - t0) * 1000
    print(f"-> Answer: {ans_1}")
    print(f"-> Latency: {dt_1:.2f} ms")
    assert ans_1.text == "" and ans_1.citations == [], "Expected empty Answer for empty chunks"

    print("\n" + "=" * 60)
    print("TEST 2: Live Gemini Generation with Retrieved Ground-Truth Chunks")
    print("=" * 60)
    eq = load_ground_truth(language="hi", sample_limit=200)[0]
    print(f"Query: '{eq.query_text}'")

    # Step 1: Retrieve context chunks
    chunks = retrieve(eq.query_text, k=3, collection="chunks_semantic")
    print(f"Retrieved {len(chunks)} chunks from Qdrant.")
    assert len(chunks) > 0, "Retrieval returned 0 chunks"

    # Step 2: Call Stage 5 generate
    timings_2: list[StageTiming] = []
    t0 = time.perf_counter()
    ans_2 = generate(eq.query_text, chunks, timings=timings_2)
    dt_2 = (time.perf_counter() - t0) * 1000

    print(f"\n-> Generated Answer Text:\n'{ans_2.text}'")
    print(f"\n-> Extracted Citations ({len(ans_2.citations)}): {ans_2.citations}")
    print(f"-> Generation Latency: {dt_2:.2f} ms")
    print(f"-> Recorded Timings: {timings_2}")

    assert len(ans_2.text) > 0, "Generated answer text should not be empty"
    assert isinstance(ans_2.citations, list), "Answer citations should be a list"

    print("\n" + "=" * 60)
    print("SUMMARY: STAGE 5 GEMINI GENERATION TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
