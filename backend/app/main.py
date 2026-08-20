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
from app.pipeline.orchestrator import run_pipeline, run_pipeline_text

app = FastAPI(title="RAGinGoa")

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


@app.post("/ask_text")
async def ask_text(payload: dict):
    query_text = payload.get("query", "")
    if not query_text or not query_text.strip():
        raise HTTPException(status_code=400, detail="empty query")

    result = run_pipeline_text(query_text)
    return result.to_dict()

