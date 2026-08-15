"""
Stage 3: retrieve(query, k) -> list[Chunk]

Embeds the query locally (no network round trip — this is the part that has
to hit the 200ms bar) and searches the active Qdrant collection.
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


@timed_stage("retrieve")
def retrieve(query: str, k: int = settings.retrieval_top_k, collection: str | None = None) -> list[Chunk]:
    collection = collection or settings.active_collection
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
