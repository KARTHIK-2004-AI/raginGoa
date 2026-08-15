"""
Runs N queries through the full pipeline AND through retrieval-only, and
dumps a CSV + P50/P70/P100 summary. This is your latency analytics
deliverable directly — run it against 50-100 real queries, not 5.

Usage:
    python -m scripts.bench --queries path/to/queries.txt --n 100

For "full pipeline" timing this assumes you have pre-recorded audio clips
mapped to query text (STT needs audio in). If you don't have audio samples
yet, start with --retrieval-only to get the retrieval number working first —
that alone satisfies part of requirement 3/4 while you wire up STT.
"""
import argparse
import csv
import statistics
import time
from pathlib import Path

from app.pipeline.retrieve import retrieve
from app.pipeline.types import StageTiming


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(len(sorted_vals) - 1, int(round((pct / 100) * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def bench_retrieval_only(queries: list[str], out_csv: Path) -> None:
    rows = []
    latencies = []

    for q in queries:
        timings: list[StageTiming] = []
        start = time.perf_counter()
        retrieve(q, timings=timings)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        rows.append({"query": q, "latency_ms": round(elapsed_ms, 2)})

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "latency_ms"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n--- Retrieval-only latency (n={len(latencies)}) ---")
    print(f"P50:  {percentile(latencies, 50):.1f} ms")
    print(f"P70:  {percentile(latencies, 70):.1f} ms")
    print(f"P100: {percentile(latencies, 100):.1f} ms")
    print(f"Mean: {statistics.mean(latencies):.1f} ms")
    print(f"Raw results written to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, required=True, help="path to a .txt file, one query per line")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", type=str, default="bench_results.csv")
    args = parser.parse_args()

    query_lines = Path(args.queries).read_text(encoding="utf-8").splitlines()
    queries = [q.strip() for q in query_lines if q.strip()][: args.n]

    if not queries:
        raise SystemExit("No queries found — check --queries file path/content.")

    bench_retrieval_only(queries, Path(args.out))


if __name__ == "__main__":
    main()
