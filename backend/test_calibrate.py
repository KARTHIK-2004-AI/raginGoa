"""
Calibration script for Stage 2 off_topic_similarity_threshold.

Embeds and queries top-1 cosine similarity for:
- 5 In-scope Hindi / Goa queries
- 5 Nonsense / Out-of-scope / Random queries

Prints raw scores side-by-side to find the empirical threshold gap.
"""
import sys
import logging
from pathlib import Path

backend_dir = Path(r"c:\Users\YS TECH CENTER\Downloads\raginGoa\raginGoa\backend")
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

logging.basicConfig(level=logging.WARNING)

from app.pipeline.embed import get_embedder, get_qdrant_client

IN_SCOPE_QUERIES = [
    "गोवा का सबसे प्रसिद्ध समुद्र तट कौन सा है?",
    "पणजी किस भारतीय राज्य की राजधानी है?",
    "गोवा में घूमने का सबसे अच्छा समय कौन सा है?",
    "बागा बीच उत्तरी गोवा में स्थित है।",
    "गोवा की मुख्य भाषा कौन सी है?",
]

NONSENSE_QUERIES = [
    "asdkfj qwoeiru zzzxx 991122",
    "qwertyuiop asdfghjkl zxcvbnm",
    "1234567890 !@#$%^&*()",
    "what model are you running on?",
    "what is the live weather in Panaji right now 2026",
]

def calibrate():
    embedder = get_embedder()
    client = get_qdrant_client()
    collection_name = "chunks_semantic"

    print("=" * 70)
    print("CALIBRATING E5 EMBEDDING SIMILARITY DISTRIBUTION (chunks_semantic)")
    print("=" * 70)

    in_scope_scores = []
    print("\n--- IN-SCOPE QUERIES ---")
    for q in IN_SCOPE_QUERIES:
        vec = embedder.encode(f"query: {q}", normalize_embeddings=True).tolist()
        hits = client.search(collection_name=collection_name, query_vector=vec, limit=1)
        score = float(hits[0].score) if hits else 0.0
        in_scope_scores.append(score)
        print(f"[{score:.4f}] Query: '{q}'")

    nonsense_scores = []
    print("\n--- NONSENSE / OUT-OF-SCOPE QUERIES ---")
    for q in NONSENSE_QUERIES:
        vec = embedder.encode(f"query: {q}", normalize_embeddings=True).tolist()
        hits = client.search(collection_name=collection_name, query_vector=vec, limit=1)
        score = float(hits[0].score) if hits else 0.0
        nonsense_scores.append(score)
        print(f"[{score:.4f}] Query: '{q}'")

    print("\n" + "=" * 70)
    print("EMPIRICAL SCORE DISTRIBUTION SUMMARY")
    print("=" * 70)
    print(f"In-Scope Min Score : {min(in_scope_scores):.4f}")
    print(f"In-Scope Max Score : {max(in_scope_scores):.4f}")
    print(f"In-Scope Mean Score: {sum(in_scope_scores)/len(in_scope_scores):.4f}")
    print("-" * 70)
    print(f"Nonsense Min Score : {min(nonsense_scores):.4f}")
    print(f"Nonsense Max Score : {max(nonsense_scores):.4f}")
    print(f"Nonsense Mean Score: {sum(nonsense_scores)/len(nonsense_scores):.4f}")
    print("=" * 70)

if __name__ == "__main__":
    calibrate()
