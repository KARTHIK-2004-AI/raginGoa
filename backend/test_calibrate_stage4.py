"""
Calibration script for Stage 4: Grounding Gate threshold calibration.

Measures top-1 vector similarity scores against Qdrant ('chunks_semantic') for:
1. Grounded Queries: Queries whose matching passages ARE in the indexed corpus.
2. Ungrounded / Out-of-Corpus Queries: Well-formed queries whose topics are NOT in the indexed corpus.

Run from backend directory:
    python test_calibrate_stage4.py
"""
import sys
import logging
from pathlib import Path

backend_dir = Path(r"c:\Users\YS TECH CENTER\Downloads\raginGoa\raginGoa\backend")
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(level=logging.WARNING)

from app.eval.eval_recall import load_ground_truth
from app.pipeline.retrieve import retrieve

UNGROUNDED_QUERIES = [
    "How to bake a chocolate chip cookie recipe at home?",
    "What is the capital of France?",
    "How to install Python and configure a virtual environment?",
    "what are the rules of basketball?",
    "how does quantum computing work?",
]

def calibrate():
    print("=" * 65)
    print("STAGE 4 CALIBRATION: GROUNDED VS UNGROUNDED SCORE SPREAD")
    print("=" * 65)

    # 1. Grounded Queries (from indexed 200-row corpus)
    eval_queries = load_ground_truth(language="hi", sample_limit=200)[:5]
    grounded_scores = []
    print("\n--- GROUNDED QUERIES (Passages present in index) ---")
    for idx, eq in enumerate(eval_queries, 1):
        chunks = retrieve(eq.query_text, k=1, collection="chunks_semantic")
        score = chunks[0].score if chunks else 0.0
        grounded_scores.append(score)
        print(f"  [{idx}] Top-1 Score: {score:.4f} | Query: '{eq.query_text[:60]}...'")

    # 2. Ungrounded Queries (Out of corpus)
    ungrounded_scores = []
    print("\n--- UNGROUNDED QUERIES (Topics not in index) ---")
    for idx, q in enumerate(UNGROUNDED_QUERIES, 1):
        chunks = retrieve(q, k=1, collection="chunks_semantic")
        score = chunks[0].score if chunks else 0.0
        ungrounded_scores.append(score)
        print(f"  [{idx}] Top-1 Score: {score:.4f} | Query: '{q}'")

    print("\n" + "=" * 65)
    print("SUMMARY SCORE DISTRIBUTION")
    print("=" * 65)
    g_min = min(grounded_scores)
    g_max = max(grounded_scores)
    g_avg = sum(grounded_scores) / len(grounded_scores)

    u_min = min(ungrounded_scores)
    u_max = max(ungrounded_scores)
    u_avg = sum(ungrounded_scores) / len(ungrounded_scores)

    print(f"GROUNDED QUERIES  : Min={g_min:.4f}, Max={g_max:.4f}, Mean={g_avg:.4f}")
    print(f"UNGROUNDED QUERIES: Min={u_min:.4f}, Max={u_max:.4f}, Mean={u_avg:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    calibrate()
