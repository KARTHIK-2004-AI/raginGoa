"""
Phase 1 deliverable: recall@k evaluation comparing chunks_fixed vs.
chunks_semantic vs. chunks_structured.

    python -m app.eval.eval_recall

WHAT "GROUND TRUTH" MEANS HERE (a deliberate design choice, not the only
valid one): for each query, MSMARCO-XI's `is_selected` flag marks which
passage(s) actually answer that query. A retrieval is counted as a HIT only
if the chunk it returns overlaps that specific is_selected passage's text —
not just "any chunk belonging to the same query_id". The looser doc_id-only
check was considered and rejected: chunks_structured indexes ~10 passages
per query (mostly irrelevant), so doc_id-matching would make it look
artificially strong without actually testing whether retrieval found the
*right* passage. Overlap-based matching is the standard recall@k definition
used in IR literature and is what should hold up if this report's
methodology gets questioned.

OVERLAP METRIC: chunks (especially fixed-size ones) can be sub-spans of a
passage, not exact string matches, so we can't require exact text equality.
Instead we use word-level overlap: split both texts into whitespace tokens,
compute |intersection| / min(len(chunk_tokens), len(gt_tokens)), and count
it as a hit if that ratio >= OVERLAP_THRESHOLD. This tolerates a chunk being
a partial slice of the ground-truth passage (common with fixed chunking)
"""
from dataclasses import dataclass
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from app.pipeline.embed import get_embedder, get_qdrant_client

from app.config import settings
from app.indexing.build_index import LANG_CODE_MAP, load_language_dataset, iter_parquet_rows

COLLECTIONS = ["chunks_fixed", "chunks_semantic", "chunks_structured"]
K_VALUES = [1, 3, 5]
OVERLAP_THRESHOLD = 0.5  # fraction of the smaller text's words that must overlap to count as a hit


@dataclass
class EvalQuery:
    query_id: str
    query_text: str
    ground_truth_passages: list[str]  # the is_selected==1 Translated_passages for this query


def load_ground_truth(language: str = "hi", sample_limit: int = 200) -> list[EvalQuery]:
    """
    Rebuilds the same 200-row sample used by build_index.py and extracts,
    for each query, the passage(s) where is_selected == 1. Rows with no
    selected passage are skipped (nothing to evaluate retrieval against).
    """
    local_path = load_language_dataset(language, split="train")
    eval_queries: list[EvalQuery] = []

    for row in iter_parquet_rows(local_path, limit=sample_limit):
        query_id = str(row.get("query_id", ""))
        query_text = row.get("query", "")
        passages_field = row.get("passages") or {}
        translated = passages_field.get("Translated_passages", [])
        is_selected = passages_field.get("is_selected", [])

        gt_passages = [
            translated[i].strip()
            for i in range(len(translated))
            if i < len(is_selected) and is_selected[i] == 1 and translated[i] and translated[i].strip()
        ]

        if not query_text.strip() or not gt_passages:
            continue

        eval_queries.append(
            EvalQuery(query_id=query_id, query_text=query_text.strip(), ground_truth_passages=gt_passages)
        )

    return eval_queries


def _word_overlap_ratio(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    return intersection / min(len(a_tokens), len(b_tokens))


def is_hit(retrieved_text: str, ground_truth_passages: list[str]) -> bool:
    return any(
        _word_overlap_ratio(retrieved_text, gt) >= OVERLAP_THRESHOLD
        for gt in ground_truth_passages
    )


def evaluate_collection(
    client: QdrantClient,
    collection: str,
    embedder: SentenceTransformer,
    eval_queries: list[EvalQuery],
    k_values: list[int],
) -> dict[int, float]:
    """
    Returns {k: recall@k} for one collection, averaged across all eval_queries.
    Runs one search at max(k_values) per query and slices for each k, instead
    of a separate search per k, to keep this cheap.
    """
    max_k = max(k_values)
    hits_at_k = {k: 0 for k in k_values}

    for eq in eval_queries:
        # multilingual-e5 convention: "query: " / "passage: " prefixes must
        # match what embed_and_upsert used when indexing, or similarity
        # scores are not meaningfully comparable.
        query_vector = embedder.encode(f"query: {eq.query_text}", normalize_embeddings=True)

        results = client.query_points(
            collection_name=collection,
            query=query_vector.tolist(),
            limit=max_k,
        ).points

        retrieved_texts = [r.payload.get("text", "") for r in results]

        for k in k_values:
            top_k_texts = retrieved_texts[:k]
            if any(is_hit(text, eq.ground_truth_passages) for text in top_k_texts):
                hits_at_k[k] += 1

    total = len(eval_queries)
    return {k: (hits_at_k[k] / total if total else 0.0) for k in k_values}


def main(language: str = "hi", sample_limit: int = 200) -> None:
    print(f"Loading ground truth from the same {sample_limit}-row sample used for indexing...")
    eval_queries = load_ground_truth(language=language, sample_limit=sample_limit)
    print(f"{len(eval_queries)} queries have a ground-truth passage (out of {sample_limit} rows).")

    if not eval_queries:
        print("No eval queries found — nothing to evaluate. Check is_selected data.")
        return

    embedder = get_embedder()
    client = get_qdrant_client()

    print()
    print(f"{'Collection':<20} " + " ".join(f"recall@{k:<6}" for k in K_VALUES))
    print("-" * 60)

    results_by_collection = {}
    for collection in COLLECTIONS:
        recall = evaluate_collection(client, collection, embedder, eval_queries, K_VALUES)
        results_by_collection[collection] = recall
        row = f"{collection:<20} " + " ".join(f"{recall[k]:<9.3f}" for k in K_VALUES)
        print(row)

    print()
    print("Higher is better. recall@k = fraction of queries where a chunk")
    print("overlapping the correct (is_selected=1) passage appeared in the")
    print(f"top-k results (word-overlap threshold: {OVERLAP_THRESHOLD}).")


if __name__ == "__main__":
    main(language="hi", sample_limit=200)