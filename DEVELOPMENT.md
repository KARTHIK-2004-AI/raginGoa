# RAGinGoa — Developer & Setup Guide

This guide covers local environment setup, secret management, git branch workflow, and the stub-to-real retrieval swap procedure for **RAGinGoa**.

---

## 1. Prerequisites & Environment Setup

### Required Services & Tools:
- **Python:** 3.10+
- **Docker:** (for running Qdrant vector database locally)
- **Node.js & npm:** (for running Vite/React frontend)

### Environment Variables setup:

```bash
cd backend
cp .env.example .env
```

Fill in the required keys in `.env`:
- `SARVAM_API_KEY`: Sarvam Speech-to-Text API key
- `GEMINI_API_KEY`: Google Gemini API key
- `QDRANT_URL`: `http://localhost:6333` (or production Qdrant Cloud URL)
- `QDRANT_API_KEY`: (Optional for local Qdrant, required for Qdrant Cloud)
- `ACTIVE_COLLECTION`: `chunks_structured` (or winning collection after evaluation)

---

## 2. Running Locally

### Step 1: Start Qdrant
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Step 2: Install Backend Dependencies
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Build Qdrant Collections
```bash
python -m app.indexing.build_index
```

### Step 4: Run FastAPI Server
```bash
uvicorn app.main:app --reload --port 8000
```

---

## 3. Stub Swap-In Instructions (Day 2 Pinch Point)

During Day 1 and early Day 2, Person B develops the voice, guardrail, and generation pipeline against a stub implementation:

```python
# app/pipeline/retrieve.py (Stub state - Day 1)
def retrieve(query: str, k: int = 5) -> list[Chunk]:
    return [
        Chunk(id="fake-1", text="Sample retrieved passage text...", score=0.92, source_id="doc-101", strategy="fixed", metadata={})
    ]
```

On **Day 2 Evening**, Person A and Person B swap the stub implementation for the real Qdrant search function:

```python
# app/pipeline/retrieve.py (Real state - Day 2 Evening)
def retrieve(query: str, k: int = 5) -> list[Chunk]:
    return qdrant_search(query, collection_name=settings.active_collection, top_k=k)
```

---

## 4. Git Workflow & Branching Strategy

- **Repository:** Shared GitHub repository.
- **Branch Naming:**
  - `feat/chunking-*` for Person A's retrieval, indexing, and eval code.
  - `feat/pipeline-*` for Person B's FastAPI harness, voice, guardrails, and frontend.
- **Merge Policy:** Daily pull requests merged into `main` after integration smoke tests. Avoid single end-of-project big merges.
