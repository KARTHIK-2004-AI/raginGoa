"""
Phase 3 deliverable: Latency benchmarking across 50-100 real MSMARCO Hindi queries.
Measures retrieval-only vs full-pipeline latency (P50, P70, P100) in a single warm process.

Usage:
    python -m app.eval.eval_latency
"""
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from fastapi.testclient import TestClient


backend_dir = Path(__file__).resolve().parent.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.eval.eval_recall import load_ground_truth
from app.main import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("ragingoa.eval_latency")


def run_latency_benchmark(sample_limit: int = 200, query_limit: int = 100) -> dict:
    print("=" * 70)
    print("Phase 3 Latency Benchmark: Ground-Truth Hindi Query Dataset")
    print("=" * 70)

    print(f"Loading ground truth sample (limit={sample_limit})...")
    eval_queries = load_ground_truth(language="hi", sample_limit=sample_limit)
    total_valid_queries = len(eval_queries)
    print(f"Loaded {total_valid_queries} queries with valid ground-truth passages.")

    if total_valid_queries == 0:
        print("ERROR: No valid queries loaded.")
        return {}

    # Extract query strings up to query_limit
    selected_eval_queries = eval_queries[:query_limit]
    total_attempted = len(selected_eval_queries)
    print(f"Selected {total_attempted} queries for benchmarking (target max: {query_limit}).")

    # Initialize TestClient against FastAPI app within the current process
    client = TestClient(app)

    # Cold-Start Handling: Query #1
    first_query = selected_eval_queries[0].query_text
    print(f"\n[1/{total_attempted}] Running Cold-Start Query #1...")
    res_cold = client.post("/ask_text", json={"query": first_query})
    if res_cold.status_code != 200:
        raise RuntimeError(f"Cold start query failed with status {res_cold.status_code}: {res_cold.text}")

    cold_body = res_cold.json()
    cold_total_ms = cold_body.get("latency_ms", {}).get("total", 0.0)
    cold_retrieve_ms = cold_body.get("latency_ms", {}).get("stages", {}).get("retrieve", 0.0)
    print(f"-> Cold Start Total Latency: {cold_total_ms:.2f} ms (Retrieve Stage: {cold_retrieve_ms:.2f} ms)")

    # Warm Benchmark Loop: Queries #2..N
    warm_queries = selected_eval_queries[1:]
    warm_count = len(warm_queries)
    print(f"\nRunning {warm_count} warm queries in single process loop...")

    retrieval_times: list[float] = []
    full_pipeline_all_times: list[float] = []
    full_pipeline_completed_times: list[float] = []
    short_circuit_counts: dict[str, int] = {}

    for idx, eq in enumerate(warm_queries, start=2):
        query_text = eq.query_text
        res = client.post("/ask_text", json={"query": query_text})
        if res.status_code != 200:
            logger.warning("Query #%d failed (HTTP %d): %s", idx, res.status_code, res.text)
            continue

        body = res.json()
        latency_info = body.get("latency_ms", {})
        stages_info = latency_info.get("stages", {})
        stopped_at = body.get("stopped_at", "unknown")

        r_ms = stages_info.get("retrieve", 0.0)
        tot_ms = latency_info.get("total", 0.0)

        retrieval_times.append(r_ms)
        full_pipeline_all_times.append(tot_ms)

        if stopped_at == "verify":
            full_pipeline_completed_times.append(tot_ms)
        else:
            short_circuit_counts[stopped_at] = short_circuit_counts.get(stopped_at, 0) + 1

        if idx % 10 == 0 or idx == total_attempted:
            print(f"  Processed {idx}/{total_attempted} queries (Latest warm query retrieve: {r_ms:.2f} ms, total: {tot_ms:.2f} ms)...")

    # Compute Percentiles using numpy
    retrieval_p50, retrieval_p70, retrieval_p100 = np.percentile(retrieval_times, [50, 70, 100])
    
    if full_pipeline_completed_times:
        comp_p50, comp_p70, comp_p100 = np.percentile(full_pipeline_completed_times, [50, 70, 100])
    else:
        comp_p50 = comp_p70 = comp_p100 = 0.0

    all_p50, all_p70, all_p100 = np.percentile(full_pipeline_all_times, [50, 70, 100])

    results = {
        "dataset_sample_limit": sample_limit,
        "dataset_valid_queries_found": total_valid_queries,
        "queries_attempted": total_attempted,
        "cold_start_query": {
            "query_text": first_query,
            "cold_start_first_request_total_ms": round(cold_total_ms, 2),
            "cold_start_first_request_retrieve_ms": round(cold_retrieve_ms, 2),
            "status": "excluded_from_percentiles",
        },
        "warm_benchmark_sample_size": len(retrieval_times),
        "retrieval_only_latency_ms": {
            "P50": round(float(retrieval_p50), 2),
            "P70": round(float(retrieval_p70), 2),
            "P100": round(float(retrieval_p100), 2),
        },
        "full_pipeline_completed_only_latency_ms": {
            "sample_size": len(full_pipeline_completed_times),
            "P50": round(float(comp_p50), 2),
            "P70": round(float(comp_p70), 2),
            "P100": round(float(comp_p100), 2),
        },
        "full_pipeline_incl_short_circuits_latency_ms": {
            "sample_size": len(full_pipeline_all_times),
            "short_circuit_breakdown": short_circuit_counts,
            "P50": round(float(all_p50), 2),
            "P70": round(float(all_p70), 2),
            "P100": round(float(all_p100), 2),
        },
    }

    # Print Summary Report
    print("\n" + "=" * 70)
    print("PHASE 3 LATENCY BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Cold start (first request, excluded from stats): {cold_total_ms:.2f} ms (Retrieve stage: {cold_retrieve_ms:.2f} ms)")
    print(f"Queries benchmarked: {len(retrieval_times)} warm queries (out of {total_attempted} attempted, 1 cold-start query excluded)")
    print(f"  - Dataset sample: {total_valid_queries} valid queries loaded from {sample_limit}-row MSMARCO sample")
    print(f"  - Completed all 6 stages (stopped_at='verify'): {len(full_pipeline_completed_times)}")
    print(f"  - Short-circuited queries: {sum(short_circuit_counts.values())} (breakdown: {short_circuit_counts})")
    print("-" * 70)
    print(f"Retrieval-only latency:                       P50={results['retrieval_only_latency_ms']['P50']:<7.2f} ms  P70={results['retrieval_only_latency_ms']['P70']:<7.2f} ms  P100={results['retrieval_only_latency_ms']['P100']:<7.2f} ms")
    print(f"Full pipeline (completed only, n={len(full_pipeline_completed_times):<3}):       P50={results['full_pipeline_completed_only_latency_ms']['P50']:<7.2f} ms  P70={results['full_pipeline_completed_only_latency_ms']['P70']:<7.2f} ms  P100={results['full_pipeline_completed_only_latency_ms']['P100']:<7.2f} ms")
    print(f"Full pipeline (incl. short-circuits, n={len(full_pipeline_all_times):<3}): P50={results['full_pipeline_incl_short_circuits_latency_ms']['P50']:<7.2f} ms  P70={results['full_pipeline_incl_short_circuits_latency_ms']['P70']:<7.2f} ms  P100={results['full_pipeline_incl_short_circuits_latency_ms']['P100']:<7.2f} ms")
    print("=" * 70)

    # Save to phase3_latency_results.json
    out_path = backend_dir / "phase3_latency_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Detailed benchmark results saved to: {out_path}")

    # Also save to app/eval/ for convenience
    eval_out_path = Path(__file__).resolve().parent / "phase3_latency_results.json"
    eval_out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    return results


if __name__ == "__main__":
    run_latency_benchmark(sample_limit=200, query_limit=100)
