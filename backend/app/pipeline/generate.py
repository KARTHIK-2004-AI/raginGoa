"""
Stage 5: generate(query, chunks) -> Answer

Forces the model to answer ONLY from provided chunks, and to cite chunk ids.
Retries on transient API failure.
"""
import time

from google import genai

from app.config import settings
from app.pipeline.types import Answer, Chunk, timed_stage

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _build_prompt(query: str, chunks: list[Chunk]) -> str:
    context_block = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say so explicitly — "
        "do not use outside knowledge. Cite the chunk id(s) you used in "
        "brackets at the end of each sentence that relies on them.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\nAnswer:"
    )


@timed_stage("generate")
def generate(query: str, chunks: list[Chunk]) -> Answer:
    client = _get_client()
    prompt = _build_prompt(query, chunks)

    last_err: Exception | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            text = response.text or ""
            cited_ids = [c.id for c in chunks if f"[{c.id}]" in text]
            return Answer(text=text.strip(), citations=cited_ids)
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < settings.llm_max_retries:
                time.sleep(0.3 * (attempt + 1))
            continue

    return Answer(text="Sorry, I couldn't generate an answer right now — please try again.", citations=[])
