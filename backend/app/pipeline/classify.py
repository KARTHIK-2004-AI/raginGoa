"""
Stage 2: classify_query(transcript) -> QueryIntent

Cheap, fast checks BEFORE we spend a retrieval + LLM call:
1. Unsafe input filter (keyword/pattern based — swap for a real moderation
   endpoint if you have time budget left in the last days).
2. Off-topic detection via embedding similarity against the corpus centroid.
   (The actual embedding call is stubbed here — wire it to the same embedder
   used in indexing/build_index.py so vectors are comparable.)
"""
from app.config import settings
from app.pipeline.types import QueryIntent, QueryVerdict, timed_stage

# Minimal unsafe-content keyword list — placeholder. Replace with a real
# classifier or moderation API call if time allows; this is intentionally
# crude so the guardrail *exists* on day 1.
_UNSAFE_MARKERS = {"kill", "bomb", "weapon", "suicide", "self-harm"}


def _looks_unsafe(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _UNSAFE_MARKERS)


def _corpus_similarity(text: str) -> float:
    """
    Placeholder: embed `text` and compare to the corpus's centroid/reference
    vector to estimate topical relevance. Wire this to your embedder +
    a precomputed corpus centroid (compute once during indexing, cache it).
    Returning 1.0 for now so nothing is blocked until this is implemented.
    """
    # TODO: replace with real embedding similarity
    return 1.0


@timed_stage("classify_query")
def classify_query(transcript_text: str) -> QueryIntent:
    if not transcript_text.strip():
        return QueryIntent(verdict=QueryVerdict.OFF_TOPIC, reason="empty transcript")

    if _looks_unsafe(transcript_text):
        return QueryIntent(verdict=QueryVerdict.UNSAFE, reason="matched unsafe marker")

    sim = _corpus_similarity(transcript_text)
    if sim < settings.off_topic_similarity_threshold:
        return QueryIntent(verdict=QueryVerdict.OFF_TOPIC, reason=f"low corpus similarity ({sim:.2f})")

    return QueryIntent(verdict=QueryVerdict.IN_SCOPE, reason="passed checks")
