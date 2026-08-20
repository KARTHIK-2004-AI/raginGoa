"""
Shared embedding model and Qdrant client singleton loader.

Keeps one SentenceTransformer model warm in memory across all pipeline stages
(classify, retrieve, etc.) to avoid duplicate model loads and memory bloat.
"""
import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import settings

_embedder: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None

# Force PyTorch single-threaded execution to optimize memory usage on constrained containers
try:
    torch.set_num_threads(1)
except Exception:
    pass


def get_embedder() -> SentenceTransformer:
    """Returns the shared, warm SentenceTransformer model instance."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.embedding_model_name)
    return _embedder



def get_qdrant_client() -> QdrantClient:
    """Returns the shared QdrantClient instance."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return _qdrant_client
