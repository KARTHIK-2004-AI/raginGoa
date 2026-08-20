"""
FastAPI entrypoint. Single endpoint: POST /ask, multipart audio upload in,
JSON out. The response shape below is the CONTRACT — your frontend teammate
builds against this shape and shouldn't need to touch this file.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.pipeline.embed import get_embedder, get_qdrant_client
from app.pipeline.orchestrator import run_pipeline

app = FastAPI(title="RAGinGoa")

@app.on_event("startup")
def startup_event():
    """Pre-warms SentenceTransformer embedder and Qdrant client at server startup."""
    get_embedder()
    get_qdrant_client()

# Wide open for hackathon dev — tighten before the live demo if time allows.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio upload")

    result = run_pipeline(audio_bytes)
    return result.to_dict()
