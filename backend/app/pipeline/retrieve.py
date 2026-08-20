"""
Stage 3: retrieve(query, k, collection) -> list[Chunk]

Embeds query locally using shared multilingual-e5-small model and searches top-k vectors in Qdrant.
Raises RetrievalError on Qdrant/embedding failure without silently masking errors with fake stubs.
Supports an explicit opt-in `use_stub=True` flag for dev work without Qdrant running.
"""
import logging

from app.config import settings
from app.pipeline.embed import get_embedder, get_qdrant_client
from app.pipeline.types import Chunk, timed_stage

logger = logging.getLogger("ragingoa.retrieve")


class RetrievalError(Exception):
    """Raised when vector retrieval fails unexpectedly."""
    pass


def _get_stub_chunks(query: str, k: int) -> list[Chunk]:
    """Explicit dev-mode stub chunks when use_stub=True is requested."""
    logger.info("Using explicit dev-mode stub chunks for query: '%s'", query)
    return [
        Chunk(
            id="passage-1",
            text="Goa is a state on the southwestern coast of India within the Konkan region. Panaji is the state capital.",
            score=0.88,
            metadata={"doc_id": "msmarco-001", "source": "msmarco-xi"},
        ),
        Chunk(
            id="passage-2",
            text="Goa was liberated from Portuguese rule on December 19, 1961 by Operation Vijay.",
            score=0.82,
            metadata={"doc_id": "msmarco-002", "source": "msmarco-xi"},
        ),
    ][:k]


@timed_stage("retrieve")
def retrieve(
    query: str,
    k: int | None = None,
    collection: str | None = None,
    use_stub: bool = False,
) -> list[Chunk]:
    """
    Retrieves top-k matching Chunk objects from Qdrant vector database.
    - query: User search string
    - k: Top-k hit count (defaults to settings.retrieval_top_k)
    - collection: Target Qdrant collection (defaults to settings.active_collection)
    - use_stub: Explicit opt-in flag for offline dev testing
    """
    top_k = k if k is not None else settings.retrieval_top_k
    target_collection = collection or settings.active_collection

    cleaned_query = query.strip()
    if not cleaned_query:
        logger.warning("retrieve received empty query string. Returning empty chunk list.")
        return []

    if use_stub or getattr(settings, "use_stub_retrieval", False):
        return _get_stub_chunks(cleaned_query, top_k)

    try:
        embedder = get_embedder()
        client = get_qdrant_client()

        query_vector = embedder.encode(f"query: {cleaned_query}", normalize_embeddings=True).tolist()

        hits = client.search(
            collection_name=target_collection,
            query_vector=query_vector,
            limit=top_k,
        )

        chunks = [
            Chunk(
                id=str(hit.id),
                text=hit.payload.get("text", ""),
                score=float(hit.score),
                metadata={k_: v for k_, v in hit.payload.items() if k_ != "text"},
            )
            for hit in hits
        ]

        top_score = chunks[0].score if chunks else 0.0
        logger.info(
            "Qdrant retrieve succeeded: collection='%s', hits=%d, top_score=%.4f, query='%s'",
            target_collection,
            len(chunks),
            top_score,
            cleaned_query,
        )
        return chunks

    except Exception as e:
        logger.error(
            "Qdrant retrieve failed on collection '%s' for query '%s': %s",
            target_collection,
            cleaned_query,
            e,
        )
        raise RetrievalError(f"Vector search failed on collection '{target_collection}': {e}") from e

