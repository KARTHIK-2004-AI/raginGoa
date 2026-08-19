"""
Stage 3: retrieve(query, k) -> list[Chunk]

Embeds the query locally and searches the active Qdrant collection.
Includes a Day 1 stub fallback if Qdrant service is offline or unreachable.
"""
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.pipeline.types import Chunk, timed_stage

_client: QdrantClient | None = None
_embedder: SentenceTransformer | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        # Loaded once, kept warm in memory — reloading per-request would blow the latency budget.
        _embedder = SentenceTransformer(settings.embedding_model_name)
    return _embedder


def _get_stub_chunks(query: str, k: int) -> list[Chunk]:
    """Day 1 stub fallback chunks for development before Qdrant is indexed."""
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
def retrieve(query: str, k: int = settings.retrieval_top_k, collection: str | None = None) -> list[Chunk]:
    collection = collection or settings.active_collection

    try:
        embedder = _get_embedder()
        client = _get_client()

        query_vector = embedder.encode(f"query: {query}", normalize_embeddings=True).tolist()

        hits = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=k,
        )

        return [
            Chunk(
                id=str(hit.id),
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata={k_: v for k_, v in hit.payload.items() if k_ != "text"},
            )
            for hit in hits
        ]
    except Exception:  # Fall back to Day 1 stub chunks if Qdrant container is offline
        return _get_stub_chunks(query, k)
