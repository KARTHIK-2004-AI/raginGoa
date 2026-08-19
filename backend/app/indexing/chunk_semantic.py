"""
Chunking strategy 2: semantic chunking. Walk sentences, keep grouping into
the current chunk while consecutive-sentence similarity stays high; start a
new chunk when similarity drops below threshold (a topic boundary).

Needs a sentence splitter + the same embedder used elsewhere so vectors are
comparable across strategies. Kept dependency-light (regex sentence split) —
swap for a proper sentence tokenizer (e.g. nltk/spacy) if quality matters
more than setup time.

IMPORTANT: the split regex must include Hindi sentence-ending punctuation
(danda । U+0964, double danda ॥ U+0965) alongside .!? — Hindi text does not
use periods to end sentences. Without this, an entire Hindi passage is seen
as a single "sentence" and chunk_semantic effectively degenerates into
"one chunk per row" (confirmed: 200 rows -> only 244 chunks before this fix).
"""
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from app.indexing.chunk_fixed import RawChunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥])\s+")


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences


def chunk_semantic(
    text: str,
    doc_id: str,
    embedder: SentenceTransformer,
    similarity_threshold: float = 0.55,
    max_sentences_per_chunk: int = 12,
) -> list[RawChunk]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    embeddings = embedder.encode(sentences, normalize_embeddings=True)

    chunks: list[RawChunk] = []
    current_sentences = [sentences[0]]
    position = 0

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i], embeddings[i - 1]))  # cosine, since normalized
        boundary = sim < similarity_threshold or len(current_sentences) >= max_sentences_per_chunk

        if boundary:
            chunks.append(
                RawChunk(
                    text=" ".join(current_sentences),
                    metadata={"doc_id": doc_id, "position": position, "strategy": "semantic"},
                )
            )
            position += 1
            current_sentences = [sentences[i]]
        else:
            current_sentences.append(sentences[i])

    if current_sentences:
        chunks.append(
            RawChunk(
                text=" ".join(current_sentences),
                metadata={"doc_id": doc_id, "position": position, "strategy": "semantic"},
            )
        )

    return chunks