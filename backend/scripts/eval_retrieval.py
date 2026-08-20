"""
Compares recall@k and MRR across the 3 Qdrant collections (fixed / semantic /
structured) against a held-out set of (query, expected_doc_id) pairs.

You need a small labeled eval set for this — MSMARCO-XI likely gives you
query -> relevant passage pairs directly (check the schema). Build
eval_queries.csv with columns: query,expected_doc_id

Usage:
    python -m scripts.eval_retrieval --eval-file eval_queries.csv
"""
import argparse
import csv

from app.config import settings
from app.pipeline.embed import get_embedder, get_qdrant_client

COLLECTIONS = ["chunks_fixed", "chunks_semantic", "chunks_structured"]


def load_eval_set(path: str) -> list[tuple[str, str]]:
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["query"], row["expected_doc_id"]))
    return pairs


def evaluate_collection(
    client: QdrantClient,
    embedder: SentenceTransformer,
    collection: str,
    eval_pairs: list[tuple[str, str]],
    k: int = 5,
) -> dict:
    hits_at_k = 0
    reciprocal_ranks = []

    for query, expected_doc_id in eval_pairs:
        vector = embedder.encode(f"query: {query}", normalize_embeddings=True).tolist()
        results = client.search(collection_name=collection, query_vector=vector, limit=k)

        doc_ids = [r.payload.get("doc_id") for r in results]
        if expected_doc_id in doc_ids:
            hits_at_k += 1
            rank = doc_ids.index(expected_doc_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n = len(eval_pairs)
    return {
        "collection": collection,
        "recall_at_k": round(hits_at_k / n, 3) if n else 0.0,
        "mrr": round(sum(reciprocal_ranks) / n, 3) if n else 0.0,
        "n_queries": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    eval_pairs = load_eval_set(args.eval_file)
    embedder = get_embedder()
    client = get_qdrant_client()

    print(f"Evaluating {len(eval_pairs)} queries across {len(COLLECTIONS)} collections at k={args.k}\n")
    print(f"{'Collection':<20} {'Recall@k':<10} {'MRR':<10} {'N'}")
    for collection in COLLECTIONS:
        result = evaluate_collection(client, embedder, collection, eval_pairs, k=args.k)
        print(f"{result['collection']:<20} {result['recall_at_k']:<10} {result['mrr']:<10} {result['n_queries']}")


if __name__ == "__main__":
    main()
