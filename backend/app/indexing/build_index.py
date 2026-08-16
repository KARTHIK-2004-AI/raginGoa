"""
Builds all three Qdrant collections (fixed / semantic / structured) from
MSMARCO-XI. Run once (and again any time chunking logic changes):

    python -m app.indexing.build_index

Confirmed schema (ai4bharat/MSMARCO-XI dataset card):
  - MUST load with a language subset, e.g. load_dataset(..., "hi", split="train")
    There is no default config — pick a language your STT plan covers.
  - `passages` is a dict of parallel lists: is_selected / English_passages /
    Translated_passages — NOT a flat list of passage rows.
  - train split alone is 10.1M rows — always pass sample_limit while testing,
    only remove it once the full pipeline runs clean.
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


def main(language: str = "hi", sample_limit: int | None = 200) -> None:
    """
    language: one of as/bn/gu/hi/kn/ml/mr/ne/or/pa/sa/ta/te/ur — must match
    a language your STT choice (Sarvam) actually supports. Confirm against
    Sarvam's docs before committing to one for the whole team.
    """
    print(f"Loading dataset (language={language})...")
    dataset = load_dataset("ai4bharat/MSMARCO-XI", language, split="train")
    if sample_limit:
        dataset = dataset.select(range(min(sample_limit, len(dataset))))

    print(f"Loaded {len(dataset)} rows. First row keys: {list(dataset[0].keys())}")

    embedder = SentenceTransformer(settings.embedding_model_name)
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)

    for name in COLLECTIONS:
        ensure_collection(client, name)

    for i, row in enumerate(dataset):
        doc_id = str(row.get("query_id", i))

        # For fixed/semantic strategies, flatten this row's translated
        # passages into one text blob to re-chunk. Structured strategy
        # chunks the passages dict directly (see chunk_structured.py).
        passages_field = row.get("passages") or {}
        translated = passages_field.get("Translated_passages", [])
        text = " ".join(p for p in translated if p and p.strip())

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
    # Start with a small sample + a single language while you're still
    # debugging — bump sample_limit=None once this runs clean end to end.
    # Language MUST be one Sarvam actually supports; confirm before Day 2.
    main(language="hi", sample_limit=200)