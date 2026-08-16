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

### Strategy Comparison Table:

| Strategy | Recall@5 | MRR | Avg Chunk Size | Notes |
|---|---|---|---|---|
| `fixed` | TBD | TBD | ~500 chars | Fixed-size with 50-char overlap |
| `semantic` | TBD | TBD | Variable | Embedding distance cutoff |
| `structured` | TBD | TBD | Passage-bound | Passage metadata & boundaries |

*Winner strategy is promoted to `ACTIVE_COLLECTION` in production.*

---

## 2. Latency Benchmarking (`bench.py`)

Latency is measured across 30–50 real queries using automated benchmarking scripts.

### Target Latencies:
- **P50 (Median):** `< 900 ms`
- **P70:** `< 1200 ms`
- **P100 (Max):** `< 2500 ms`

### Running the Latency Benchmark:

```bash
cd backend
python -m scripts.bench --queries sample_queries.txt --n 50
```

The script outputs a CSV containing per-query latency broken down by stage (`transcribe`, `classify`, `retrieve`, `check_grounding`, `generate`, `verify`) and summarizes overall P50/P70/P100 statistics for inclusion in `README.md`.
