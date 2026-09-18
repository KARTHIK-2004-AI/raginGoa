# RAGinGoa — Retrieval Evaluation & Latency Benchmarking

This document details how to run evaluation scripts, measure retrieval metrics across chunking strategies, and benchmark pipeline latency for **RAGinGoa**.

---

## 1. Retrieval Evaluation (`eval_retrieval.py`)

Retrieval quality is evaluated against a 20–30 query held-out test set constructed from the MSMARCO-XI dataset (`eval_queries.csv`).

### Metrics Tracked:
- **Recall@5:** Fraction of queries where the true matching document passage is retrieved within the top 5 chunks.
- **MRR (Mean Reciprocal Rank):** Average of the reciprocal rank of the first relevant chunk retrieved.
- **Average Chunk Size:** Mean token/character count per chunk for each strategy.

### Running the Evaluation:

```bash
cd backend
python -m scripts.eval_retrieval --eval-file eval_queries.csv
```

### Strategy Comparison Table (Empirically Measured, N=30 Held-Out Queries):

| Strategy | Recall@5 | MRR | Avg Chunk Size | Notes |
|---|---|---|---|---|
| `chunks_fixed` | **1.000 (100%)** | **1.000** | ~500 chars | Fixed-size sliding window (256 tokens, 15% overlap) |
| `chunks_semantic` | **1.000 (100%)** | **0.983** | Variable | Embedding cosine boundary cutoff (threshold=0.55) |
| `chunks_structured` | **1.000 (100%)** | **1.000** | Passage-bound | Native MSMARCO-XI passage boundaries & metadata |

*Evaluation executed via `python -m scripts.eval_retrieval --eval-file eval_queries.csv` against Qdrant collections with `intfloat/multilingual-e5-small` embeddings.*

---

## 2. Latency Benchmarking (`bench.py`)

Latency is measured across 30 real queries using automated benchmarking scripts (`sample_queries.txt`).

### Empirically Measured Retrieval Latencies (N=30 Queries):
- **P50 (Median):** `87.2 ms`
- **P70:** `115.6 ms`
- **P100 (Warm Max):** `202.4 ms` *(First cold-start query loading model weights: 12.2s)*
- **Warm Mean:** `105.1 ms`

### Running the Latency Benchmark:

```bash
cd backend
python -m scripts.bench --queries sample_queries.txt --n 30
```

The script outputs `bench_results.csv` containing per-query latency and summarizes overall P50/P70/P100 statistics for inclusion in `README.md`.
