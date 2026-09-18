import csv
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.pipeline.embed import get_embedder, get_qdrant_client
from app.indexing.build_index import (
    COLLECTIONS,
    load_language_dataset,
    iter_parquet_rows,
    ensure_collection,
    embed_and_upsert,
    SAMPLE_HINDI_DATA,
)
from app.indexing.chunk_fixed import chunk_fixed
from app.indexing.chunk_semantic import chunk_semantic
from app.indexing.chunk_structured import chunk_structured
from scripts.eval_retrieval import evaluate_collection
from scripts.bench import percentile

def main():
    print("=== Step 1: Connecting to Qdrant and checking index status ===")
    client = get_qdrant_client()
    embedder = get_embedder()

    # Check collections
    counts = {}
    for col in COLLECTIONS:
        ensure_collection(client, col)
        info = client.get_collection(col)
        counts[col] = info.points_count
        print(f"Collection '{col}': {info.points_count} points currently.")

    # Target 30 rows for evaluation
    TARGET_ROWS = 30
    eval_pairs = []
    queries = []

    print(f"\n=== Step 2: Loading dataset and ensuring at least {TARGET_ROWS} rows are indexed ===")
    local_path = load_language_dataset("hi", split="train")
    print(f"Using cached parquet: {local_path}")

    # We read TARGET_ROWS from parquet
    rows = list(iter_parquet_rows(local_path, limit=TARGET_ROWS))
    print(f"Loaded {len(rows)} rows from dataset.")

    # Check if we need to index rows
    min_points = min(counts.values()) if counts else 0
    if min_points < TARGET_ROWS:
        print(f"Indexing {len(rows)} rows into Qdrant collections...")
        for i, row in enumerate(rows):
            doc_id = str(row.get("query_id", i))
            passages_field = row.get("passages") or {}
            translated = passages_field.get("Translated_passages", [])
            text = " ".join(p for p in translated if p and p.strip())

            fixed_chunks = chunk_fixed(text, doc_id=doc_id)
            semantic_chunks = chunk_semantic(text, doc_id=doc_id, embedder=embedder)
            structured_chunks = chunk_structured(row)

            embed_and_upsert(client, "chunks_fixed", fixed_chunks, embedder)
            embed_and_upsert(client, "chunks_semantic", semantic_chunks, embedder)
            embed_and_upsert(client, "chunks_structured", structured_chunks, embedder)

            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                print(f"Indexed {i + 1}/{len(rows)} rows...")

    for i, row in enumerate(rows):
        q_text = row.get("query", "").strip()
        doc_id = str(row.get("query_id", i))
        if q_text:
            eval_pairs.append((q_text, doc_id))
            queries.append(q_text)

    # Save eval_queries.csv
    eval_csv_path = backend_dir / "eval_queries.csv"
    with open(eval_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "expected_doc_id"])
        for q, d in eval_pairs:
            writer.writerow([q, d])
    print(f"Saved {len(eval_pairs)} query-doc pairs to {eval_csv_path}")

    # Save sample_queries.txt
    queries_txt_path = backend_dir / "sample_queries.txt"
    queries_txt_path.write_text("\n".join(queries), encoding="utf-8")
    print(f"Saved {len(queries)} queries to {queries_txt_path}")

    print("\n=== Step 3: Running eval_retrieval.py metrics across all 3 collections (k=5) ===")
    print(f"{'Collection':<22} {'Recall@5':<12} {'MRR':<12} {'N Queries'}")
    print("-" * 55)
    results = {}
    for col in COLLECTIONS:
        res = evaluate_collection(client, embedder, col, eval_pairs, k=5)
        results[col] = res
        print(f"{res['collection']:<22} {res['recall_at_k']:<12.3f} {res['mrr']:<12.3f} {res['n_queries']}")

    print("\n=== Step 4: Running bench.py retrieval-only latency benchmark ===")
    from app.pipeline.retrieve import retrieve
    latencies = []
    bench_rows = []
    for q in queries:
        timings = []
        t0 = time.perf_counter()
        retrieve(q, k=5, timings=timings)
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)
        bench_rows.append({"query": q, "latency_ms": round(dt_ms, 2)})

    bench_csv_path = backend_dir / "bench_results.csv"
    with open(bench_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "latency_ms"])
        writer.writeheader()
        writer.writerows(bench_rows)

    p50 = percentile(latencies, 50)
    p70 = percentile(latencies, 70)
    p100 = percentile(latencies, 100)
    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"Latency over {len(latencies)} queries:")
    print(f"P50:  {p50:.1f} ms")
    print(f"P70:  {p70:.1f} ms")
    print(f"P100: {p100:.1f} ms")
    print(f"Mean: {mean_lat:.1f} ms")
    print(f"Results written to {bench_csv_path}")

    # Write summary JSON
    summary = {
        "retrieval_evaluation": results,
        "latency_ms": {
            "p50": round(p50, 2),
            "p70": round(p70, 2),
            "p100": round(p100, 2),
            "mean": round(mean_lat, 2),
            "n": len(latencies),
        },
    }
    summary_path = backend_dir / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBenchmark summary written to {summary_path}")

if __name__ == "__main__":
    main()
