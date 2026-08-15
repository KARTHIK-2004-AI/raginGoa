# RAGinGoa — Voice-Enabled RAG (HH Goa 2026, Task 2)

## Repo layout

```
raginGoa/
  backend/          <- pipeline owner works here
    app/
      main.py               FastAPI app, exposes POST /ask
      config.py              env-driven settings
      pipeline/
        types.py             shared dataclasses + timing decorator (the harness backbone)
        transcribe.py         Stage 1: Sarvam STT
        classify.py           Stage 2: off-topic/unsafe check
        retrieve.py            Stage 3: Qdrant search
        guardrails.py          Stage 4: grounding check
        generate.py            Stage 5: Claude answer generation
        verify.py               Stage 6: post-hoc claim verification
        orchestrator.py         ties all 6 stages together
      indexing/
        chunk_fixed.py         strategy 1: fixed-size + overlap
        chunk_semantic.py       strategy 2: embedding-similarity boundaries
        chunk_structured.py      strategy 3: dataset passage boundaries
        build_index.py           builds all 3 Qdrant collections
    scripts/
      bench.py                latency benchmark -> P50/P70/P100 CSV
      eval_retrieval.py        recall@k / MRR per chunking strategy
  frontend/          <- frontend owner works here (Vite/React, separate setup)
```

## API contract (frontend builds against this)

`POST /ask`, multipart form field `audio` (wav bytes) →

```json
{
  "answer": "string",
  "citations": ["chunk-id-1", "chunk-id-2"],
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

`stopped_at` tells you which stage the pipeline exited at — if it's not
`"verify"`, the answer is a guardrail message (off-topic/unsafe/not
grounded/STT failure), not a generated answer. Frontend can use this to
style those responses differently (e.g. a muted "couldn't answer" style).

## Backend setup

```bash
cd backend
cp .env.example .env   # fill in SARVAM_API_KEY, ANTHROPIC_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run Qdrant locally (needs Docker):
docker run -p 6333:6333 qdrant/qdrant

# build the index (do this once, and again after chunking changes):
python -m app.indexing.build_index

# run the API:
uvicorn app.main:app --reload --port 8000
```

## TODO before this is real (do these first, in order)

1. `python -c "from datasets import load_dataset; d=load_dataset('ai4bharat/MSMARCO-XI', split='train'); print(d[0])"`
   — confirm actual field names, fix `chunk_structured.py` and
   `build_index.py`'s field lookups to match.
2. Verify Sarvam's actual STT request/response shape against their docs —
   `transcribe.py`'s `_call_sarvam` is a best-guess shape.
3. Wire real embedding similarity into `classify.py`'s `_corpus_similarity`
   (currently a stub that always passes).
4. Build `eval_queries.csv` (query, expected_doc_id pairs) from the dataset
   for `eval_retrieval.py`.

## Latency & chunking numbers

Run and paste results here before submission:

```bash
python -m scripts.bench --queries sample_queries.txt --n 100
python -m scripts.eval_retrieval --eval-file eval_queries.csv
```

| Strategy   | Recall@5 | MRR | Avg chunk size | Notes |
|------------|----------|-----|----------------|-------|
| fixed      |          |     |                |       |
| semantic   |          |     |                |       |
| structured |          |     |                |       |
