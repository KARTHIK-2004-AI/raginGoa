"""
Builds all three Qdrant collections (fixed / semantic / structured) from
MSMARCO-XI. Run once (and again any time chunking logic changes):

    python -m app.indexing.build_index

Confirmed schema and loading (verified directly against the repo's file
list via huggingface_hub.list_repo_files — the dataset card's example code
using load_dataset(..., "hi", split="train") does NOT work, since this repo
only exposes a single 'default' HF config via its custom loading script):
  - Real files live at train/{code}train.parquet and validation/{code}val.parquet,
    using 3-letter codes (hin, ben, tam...), not ISO 2-letter codes (hi, bn, ta...).
  - Telugu ("tel") only exists in validation/, there is no train/teltrain.parquet.
  - Must load via the 'parquet' builder pointed at the exact hf:// path — see
    LANG_CODE_MAP and load_language_dataset() below.
  - Each language's train file is multiple GB — always use streaming=True and
    sample_limit while testing; only remove sample_limit for a full run.

use_mock: a small hardcoded Hindi sample (3 rows) for testing the pipeline
(chunking -> embedding -> Qdrant upsert) instantly without waiting on the
3.7GB download. Set use_mock=False to run against the real dataset.
"""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.indexing.chunk_fixed import RawChunk, chunk_fixed
from app.indexing.chunk_semantic import chunk_semantic
from app.indexing.chunk_structured import chunk_structured

COLLECTIONS = ["chunks_fixed", "chunks_semantic", "chunks_structured"]
EMBEDDING_DIM = 384  # matches multilingual-e5-small; change if you swap embedders

# ISO 2-letter code -> repo's actual 3-letter file prefix. Confirmed against
# the real file list (huggingface_hub.list_repo_files), NOT the dataset card.
# Only languages with a train/ file are usable for indexing; "tel" (Telugu)
# only exists under validation/, so it is deliberately excluded here.
LANG_CODE_MAP = {
    "as": "asm",
    "bn": "ben",
    "gu": "guj",
    "hi": "hin",
    "kn": "kan",
    "ml": "mal",
    "mr": "mar",
    "ne": "nep",
    "or": "ori",
    "pa": "pan",
    "sa": "san",
    "ta": "tam",
    "ur": "urd",
}


def load_language_dataset(language: str, split: str = "train", streaming: bool = True):
    """
    language: ISO 2-letter code, e.g. "hi". Must be a key in LANG_CODE_MAP
    (i.e. must have a train/ file in the repo — "te"/Telugu does not, and
    will raise a clear error here instead of a confusing HF config error).

    Uses hf_hub_download() (NOT datasets' load_dataset with streaming=True
    against an hf:// path) to resolve the local cached file path — the
    hf:// streaming path in `datasets` does not reliably reuse the
    huggingface_hub cache and can silently re-download over the network
    even when the file is already on disk. hf_hub_download() always checks
    the cache first and returns instantly if the file is already present.
    """
    from huggingface_hub import hf_hub_download

    if language not in LANG_CODE_MAP:
        raise ValueError(
            f"Language '{language}' has no train/ file in ai4bharat/MSMARCO-XI. "
            f"Available: {sorted(LANG_CODE_MAP.keys())}"
        )
    code = LANG_CODE_MAP[language]
    folder = "train" if split == "train" else "validation"
    suffix = "train" if split == "train" else "val"
    filename = f"{folder}/{code}{suffix}.parquet"

    # Instant if already cached (checks huggingface_hub's cache directly,
    # no ambiguity about streaming vs. local reuse).
    local_path = hf_hub_download(
        "ai4bharat/MSMARCO-XI", filename, repo_type="dataset"
    )
    return local_path


def iter_parquet_rows(local_path: str, limit: int | None = None):
    """
    Reads a local parquet file in row-group batches via pyarrow, yielding
    plain dicts — never materializes the whole file into memory (avoids the
    ArrowMemoryError we hit earlier) and never touches the network (avoids
    the silent re-download/hang we hit with datasets' hf:// streaming path).
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(local_path)
    yielded = 0
    for batch in pf.iter_batches(batch_size=50):
        for row in batch.to_pylist():
            if limit is not None and yielded >= limit:
                return
            yield row
            yielded += 1


def ensure_collection(client: QdrantClient, name: str) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def embed_and_upsert(
    client: QdrantClient,
    collection: str,
    chunks: list[RawChunk],
    embedder: SentenceTransformer,
    batch_size: int = 100,
) -> None:
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
    for b in range(0, len(points), batch_size):
        client.upsert(collection_name=collection, points=points[b : b + batch_size])


SAMPLE_HINDI_DATA = [
    {
        "query_id": 101,
        "query_type": "description",
        "query": "गोवा का सबसे प्रसिद्ध समुद्र तट कौन सा है?",
        "Answer": "बागा बीच और कलंगूट बीच गोवा के सबसे प्रसिद्ध समुद्र तटों में से हैं।",
        "Eng_Query": "Which is the most famous beach in Goa?",
        "Eng_Answer": "Baga Beach and Calangute Beach are among the most famous beaches in Goa.",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Baga Beach is one of the most famous beaches in North Goa known for water sports and nightlife.",
                "Goa has a long coastline spanning over 100 kilometers with many beautiful sandy beaches."
            ],
            "Translated_passages": [
                "बागा बीच उत्तरी गोवा के सबसे प्रसिद्ध समुद्र तटों में से एक है जो जल खेलों और रात्रिजीवन के लिए जाना जाता है।",
                "गोवा में 100 किलोमीटर से अधिक लंबी तटरेखा है जिसमें कई सुंदर रेतीले समुद्र तट हैं।"
            ]
        }
    },
    {
        "query_id": 102,
        "query_type": "description",
        "query": "गोवा घूमने का सबसे अच्छा समय कौन सा है?",
        "Answer": "गोवा घूमने का सबसे अच्छा समय नवंबर से फरवरी के बीच होता है जब मौसम सुहाना होता है।",
        "Eng_Query": "What is the best time to visit Goa?",
        "Eng_Answer": "The best time to visit Goa is between November and February when the weather is pleasant.",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "The peak tourist season in Goa is from November to February when temperatures range from 20C to 30C.",
                "Monsoon season in Goa runs from June to September bringing heavy rainfall across the state."
            ],
            "Translated_passages": [
                "गोवा में पीक पर्यटन सीजन नवंबर से फरवरी तक होता है जब तापमान 20 डिग्री से 30 डिग्री सेल्सियस के बीच रहता है।",
                "गोवा में मानसून का मौसम जून से सितंबर तक रहता है जिससे पूरे राज्य में भारी बारिश होती है।"
            ]
        }
    },
    {
        "query_id": 103,
        "query_type": "description",
        "query": "पणजी किस भारतीय राज्य की राजधानी है?",
        "Answer": "पणजी गोवा राज्य की राजधानी है।",
        "Eng_Query": "Panaji is the capital of which Indian state?",
        "Eng_Answer": "Panaji is the capital of the Indian state of Goa.",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Panaji, also known as Panjim, is the capital of Goa located on the banks of the Mandovi River.",
                "Goa is India's smallest state by area located in southwestern India."
            ],
            "Translated_passages": [
                "पणजी, जिसे पणजीम भी कहा जाता है, मांडवी नदी के तट पर स्थित गोवा की राजधानी है।",
                "गोवा क्षेत्रफल के हिसाब से भारत का सबसे छोटा राज्य है जो दक्षिण पश्चिम भारत में स्थित है।"
            ]
        }
    }
]


def main(language: str = "hi", sample_limit: int | None = 200, use_mock: bool = True) -> None:
    """
    language: ISO 2-letter code — must be a key in LANG_CODE_MAP (must have a
    train/ file in the repo). Confirm it also matches a language your STT
    choice (Sarvam) actually supports before committing to one for the team.

    use_mock: Set to True to test indexing instantly without downloading the 3.7GB dataset.
    """
    if use_mock:
        print("Using instant sample dataset (mock data)...")
        dataset = SAMPLE_HINDI_DATA
    else:
        print(f"Resolving local cached file for language={language}...")
        local_path = load_language_dataset(language, split="train")
        print(f"Reading from local file: {local_path}")
        dataset = iter_parquet_rows(local_path, limit=sample_limit)
        if sample_limit:
            print(f"Reading top {sample_limit} rows from local parquet file.")
        else:
            print("Reading full local parquet file (this will take a while).")

    embedder = SentenceTransformer(settings.embedding_model_name)
    client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=60.0,
    )

    for name in COLLECTIONS:
        ensure_collection(client, name)

    indexed_count = 0
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

        indexed_count = i + 1
        if indexed_count % 10 == 0:
            print(f"Indexed {indexed_count} rows...")

    print(f"Done indexing! Total rows processed: {indexed_count}")


if __name__ == "__main__":
    # Set use_mock=True to test the full indexing pipeline instantly without waiting for dataset download.
    # Set use_mock=False when you want to load from HuggingFace MSMARCO-XI.
    main(language="hi", sample_limit=200, use_mock=False)