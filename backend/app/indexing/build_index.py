"""
Builds all three Qdrant collections (fixed / semantic / structured) from
MSMARCO-XI. Run once (and again any time chunking logic changes):

    python -m app.indexing.build_index

STEP 1 before running for real: load the dataset and print one row to see
the actual schema, then adjust chunk_structured.py's field names to match.
"""
import uuid

from datasets import load_dataset
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.indexing.chunk_fixed import RawChunk, chunk_fixed
from app.indexing.chunk_semantic import chunk_semantic
from app.indexing.chunk_structured import chunk_structured

COLLECTIONS = ["chunks_fixed", "chunks_semantic", "chunks_structured"]
EMBEDDING_DIM = 384  # matches multilingual-e5-small; change if you swap embedders


def ensure_collection(client: QdrantClient, name: str) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def embed_and_upsert(client: QdrantClient, collection: str, chunks: list[RawChunk], embedder: SentenceTransformer) -> None:
    if not chunks:
        return
    texts = [f"passage: {c.text}" for c in chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector.tolist(),
            payload={"text": chunk.text, **chunk.metadata},
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=collection, points=points)


def main(sample_limit: int | None = None) -> None:
    print("Loading dataset...")
    dataset = load_dataset("ai4bharat/MSMARCO-XI", split="train")
    if sample_limit:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))

    print(f"Loaded {len(dataset)} rows. First row keys: {list(dataset[0].keys())}")
    print("^ Confirm these match what chunk_structured.py expects before trusting output.")

    embedder = SentenceTransformer(settings.embedding_model_name)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    for name in COLLECTIONS:
        ensure_collection(client, name)

    for i, row in enumerate(dataset):
        doc_id = str(row.get("query_id", i))
        text = row.get("passage_text") or row.get("text") or ""
        if isinstance(text, list):
            text = " ".join(text)

        fixed_chunks = chunk_fixed(text, doc_id=doc_id)
        semantic_chunks = chunk_semantic(text, doc_id=doc_id, embedder=embedder)
        structured_chunks = chunk_structured(row)

        embed_and_upsert(client, "chunks_fixed", fixed_chunks, embedder)
        embed_and_upsert(client, "chunks_semantic", semantic_chunks, embedder)
        embed_and_upsert(client, "chunks_structured", structured_chunks, embedder)

        if i % 50 == 0:
            print(f"Indexed {i} rows...")

    print("Done.")


if __name__ == "__main__":
    # Start with a small sample while you're still debugging schema/logic —
    # bump sample_limit=None once build_index runs clean end to end.
    main(sample_limit=200)
