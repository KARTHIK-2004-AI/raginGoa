"""
Chunking strategy 1: fixed-size with overlap. Baseline/control for comparison
against semantic and structure-aware chunking. Token count approximated by
whitespace split (swap for a real tokenizer if you want exact counts).
"""
from dataclasses import dataclass


@dataclass
class RawChunk:
    text: str
    metadata: dict


def chunk_fixed(text: str, doc_id: str, window_tokens: int = 256, overlap_ratio: float = 0.15) -> list[RawChunk]:
    words = text.split()
    if not words:
        return []

    step = max(1, int(window_tokens * (1 - overlap_ratio)))
    chunks: list[RawChunk] = []

    i = 0
    position = 0
    while i < len(words):
        window = words[i : i + window_tokens]
        chunk_text = " ".join(window)
        chunks.append(
            RawChunk(
                text=chunk_text,
                metadata={"doc_id": doc_id, "position": position, "strategy": "fixed"},
            )
        )
        position += 1
        i += step

    return chunks
