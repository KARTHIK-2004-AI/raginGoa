# RAGinGoa — System Architecture

This document describes the end-to-end technical architecture of **RAGinGoa**, a voice-enabled Retrieval-Augmented Generation (RAG) system built for Task 2 of HH Goa 2026.

---

## 1. High-Level Pipeline Architecture

The system operates as a 6-stage sequential pipeline managed by a central FastAPI harness (`orchestrator.py`). Every request records per-stage timing metrics to ensure complete visibility into system latency.

```
 [User Audio / WAV]
        │
        ▼
 ┌───────────────┐
 │ Stage 1: STT  │  ──> Sarvam STT API (Transcribes audio to query string)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Stage 2: Gate │  ──> Query Classifier (Guards against off-topic / unsafe input)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Stage 3: RAG  │  ──> Qdrant Vector Search (Retrieves top-k Chunk objects)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Stage 4: Check│  ──> Grounding Gate (Verifies relevance score threshold)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Stage 5: LLM  │  ──> Gemini 2.5 Flash API (Grounded generation with forced citations)
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │ Stage 6: Verify│ ──> Claim Verification (Entity/number heuristic match check)
 └───────┬───────┘
         │
         ▼
 [JSON Response + Per-Stage Latency Breakdown]
```

---

## 2. Shared Data Contract

Person A (Data & Retrieval) and Person B (Voice, Pipeline & Frontend) interface via a clean python contract:

```python
from dataclasses import dataclass

@dataclass
class Chunk:
    id: str
    text: str
    score: float
    source_id: str
    strategy: str   # "fixed" | "semantic" | "structured"
    metadata: dict

def retrieve(query: str, k: int = 5) -> list[Chunk]:
    """Retrieves top-k chunks from the active Qdrant vector collection."""
    ...
```

---

## 3. Qdrant Retrieval & Chunking Strategies

The system benchmarks 3 distinct chunking strategies over the MSMARCO-XI dataset:

1. **Fixed-Size Chunking (`chunks_fixed`):**
   - Standard fixed token/character length with sliding window overlap (e.g., 500 chars, 50 char overlap).
2. **Semantic Chunking (`chunks_semantic`):**
   - Sentence boundary split using embedding similarity distance to determine chunk boundaries.
3. **Structured Chunking (`chunks_structured`):**
   - Passage-boundary chunking attaching doc ID, language, and structural position metadata.

---

## 4. Generation & Grounding Engine

- **Model:** `gemini-2.5-flash` via the `google-genai` SDK.
- **Citation Format:** Forced bracketed chunk citations (`[chunk-id]`) tied directly to context passages.
- **Heuristic Claim Verification:** Checks that numbers and key named entities in the generated answer exist in retrieved chunks without requiring a second expensive LLM call.

---

## 5. API Response Schema

```json
{
  "answer": "Goa was liberated from Portuguese rule on December 19, 1961 [passage-42].",
  "citations": ["passage-42"],
  "is_fully_grounded": true,
  "flagged_claims": [],
  "stopped_at": "verify",
  "latency_ms": {
    "total": 842.1,
    "stages": {
      "transcribe": 310.2,
      "classify_query": 4.1,
      "retrieve": 38.7,
      "check_grounding": 0.3,
      "generate": 480.0,
      "verify": 8.8
    }
  }
}
```
