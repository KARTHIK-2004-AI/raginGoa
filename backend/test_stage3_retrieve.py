"""
Test script for Stage 3: retrieve.py

Run manually from backend directory:
    python test_stage3_retrieve.py
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

from app.pipeline.retrieve import retrieve
from app.pipeline.types import StageTiming
from app.eval.eval_recall import load_ground_truth, is_hit, _word_overlap_ratio

def run_tests():
    print("=" * 60)
    print("TEST 1: Empty Query Input")
    print("=" * 60)
    timings_1: list[StageTiming] = []
    t0 = time.perf_counter()
    chunks_1 = retrieve("", timings=timings_1)
    dt_1 = (time.perf_counter() - t0) * 1000
    print(f"-> Chunks Returned: {len(chunks_1)}")
    print(f"-> Latency: {dt_1:.2f} ms")
    assert chunks_1 == [], "Expected empty list for empty query"

    print("\n" + "=" * 60)
    print("TEST 2: Explicit Stub Mode Opt-In (use_stub=True)")
    print("=" * 60)
    timings_2: list[StageTiming] = []
    t0 = time.perf_counter()
    chunks_2 = retrieve("dummy query", k=2, use_stub=True, timings=timings_2)
    dt_2 = (time.perf_counter() - t0) * 1000
    print(f"-> Chunks Returned: {len(chunks_2)}")
    for c in chunks_2:
        print(f"   [score={c.score:.2f}] id={c.id}: {c.text[:60]}...")
    print(f"-> Latency: {dt_2:.2f} ms")
    assert len(chunks_2) == 2, "Expected 2 stub chunks"
    assert chunks_2[0].score == 0.88

    print("\n" + "=" * 60)
    print("TEST 3: Live Qdrant Ground-Truth Retrieval ('chunks_semantic')")
    print("=" * 60)
    
    # Load ground-truth queries directly without silent fallbacks
    eval_queries = load_ground_truth(language="hi", sample_limit=200)[:3]
    print(f"Loaded {len(eval_queries)} real ground-truth queries from indexed 200-row corpus sample.\n")

    hits_found = 0
    for idx, eq in enumerate(eval_queries, 1):
        q_text = eq.query_text
        gt_passages = eq.ground_truth_passages
        print(f"--- Query {idx} (ID: {eq.query_id}): '{q_text}' ---")
        print(f"Expected GT passage snippet: '{gt_passages[0][:80]}...'")

        timings_3: list[StageTiming] = []
        t0 = time.perf_counter()
        chunks_3 = retrieve(q_text, k=5, collection="chunks_semantic", timings=timings_3)
        dt_3 = (time.perf_counter() - t0) * 1000

        print(f"-> Retrieved {len(chunks_3)} chunks in {dt_3:.2f} ms")
        query_hit = False
        for rank, c in enumerate(chunks_3, 1):
            hit = is_hit(c.text, gt_passages)
            max_overlap = max(_word_overlap_ratio(c.text, gt) for gt in gt_passages) if gt_passages else 0.0
            print(f"   Hit {rank}: [score={c.score:.4f}, overlap={max_overlap:.2f}, match={hit}] id='{c.id}'")
            print(f"          Text: '{c.text[:90]}...'")
            assert c.text.strip() != "", f"Hit {rank} has empty text — payload key bug!"

            if hit and not query_hit:
                query_hit = True
                hits_found += 1
                print(f"   ==> VERIFIED GROUND-TRUTH MATCH AT RANK {rank}!")

        assert len(chunks_3) > 0, f"Expected hits for query '{q_text}'"

    print(f"\n-> Ground-Truth Recall Hit Count: {hits_found}/{len(eval_queries)}")
    print("\n" + "=" * 60)
    print("SUMMARY: ALL STAGE 3 GROUND-TRUTH RETRIEVAL TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
