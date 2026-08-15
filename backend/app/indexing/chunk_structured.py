"""
Chunking strategy 3: structure-aware. MSMARCO-XI already has passage-level
structure (each row is roughly a query + a set of candidate passages).
Instead of re-splitting that text, chunk AT the existing passage boundary
and attach rich metadata so retrieval can filter/boost by it later.

IMPORTANT: inspect the actual dataset schema before trusting the field
names below — `load_dataset("ai4bharat/MSMARCO-XI")` and print one row
first. This assumes a `passage_text`, `doc_id`, `language` shape common to
MS MARCO-style datasets; adjust field names to match what you actually see.
"""
from app.indexing.chunk_fixed import RawChunk


def chunk_structured(row: dict) -> list[RawChunk]:
    """
    One dataset row -> one or more RawChunks, one per passage in that row.
    Adjust the field names in row.get(...) once you've confirmed the real
    schema (see docstring above).
    """
    passages = row.get("passages", row.get("passage_text", []))
    if isinstance(passages, str):
        passages = [passages]

    doc_id = str(row.get("doc_id", row.get("query_id", "unknown")))
    language = row.get("language", "unknown")

    chunks: list[RawChunk] = []
    for position, passage_text in enumerate(passages):
        if not passage_text or not passage_text.strip():
            continue
        chunks.append(
            RawChunk(
                text=passage_text.strip(),
                metadata={
                    "doc_id": doc_id,
                    "position": position,
                    "language": language,
                    "strategy": "structured",
                },
            )
        )
    return chunks
