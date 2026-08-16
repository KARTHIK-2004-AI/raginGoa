"""
Chunking strategy 3: structure-aware. Confirmed real schema from the
ai4bharat/MSMARCO-XI dataset card (https://huggingface.co/datasets/ai4bharat/MSMARCO-XI):

  row = {
    "query": str, "Answer": str, "query_id": int, "query_type": str,
    "passages": {
        "is_selected": [1, 0, 0, ...],
        "English_passages": [str, ...],
        "Translated_passages": [str, ...],
    },
    "Eng_Query": str, "Eng_Answer": str,
    "source_lang": str, "target_lang": str, "meta": {...},
  }

`passages` is a DICT of parallel lists, not a list of rows — index into
`Translated_passages` (or `English_passages` if you want an English-only
pipeline). `is_selected` marks which passage MS MARCO's original annotators
flagged as actually answering the query — useful as a metadata boost/filter
at retrieval time (e.g. rank is_selected=1 passages higher, or use it to
build your eval_queries.csv ground truth: query -> the doc_id of its
is_selected=1 passage).
"""
from app.indexing.chunk_fixed import RawChunk


def chunk_structured(row: dict, use_translated: bool = True) -> list[RawChunk]:
    """
    One dataset row -> one RawChunk per passage in that row's `passages` dict.
    Set use_translated=False to index the English_passages instead (useful
    if your STT/answer language plan is English-only).
    """
    passages_field = row.get("passages") or {}
    key = "Translated_passages" if use_translated else "English_passages"
    passage_texts = passages_field.get(key, [])
    is_selected = passages_field.get("is_selected", [])

    doc_id = str(row.get("query_id", "unknown"))
    query_type = row.get("query_type", "unknown")

    chunks: list[RawChunk] = []
    for position, passage_text in enumerate(passage_texts):
        if not passage_text or not passage_text.strip():
            continue
        selected_flag = is_selected[position] if position < len(is_selected) else 0
        chunks.append(
            RawChunk(
                text=passage_text.strip(),
                metadata={
                    "doc_id": doc_id,
                    "position": position,
                    "query_type": query_type,
                    "is_selected": int(selected_flag),
                    "strategy": "structured",
                },
            )
        )
    return chunks