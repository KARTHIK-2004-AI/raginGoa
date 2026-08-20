"""
Central config. Load everything from env so nobody hardcodes API keys.
Copy .env.example to .env and fill in real values.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    sarvam_model: str = os.getenv("SARVAM_MODEL", "saaras:v3")
    sarvam_mode: str = os.getenv("SARVAM_MODE", "transcribe")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    embedding_model_name: str = "intfloat/multilingual-e5-small"

    # Which Qdrant collection is "in production" for the live demo.
    # Updated to 'chunks_semantic' based on Phase 1 high recall evaluation results.
    active_collection: str = os.getenv("ACTIVE_COLLECTION", "chunks_semantic")

    # Guardrail thresholds — empirically calibrated against Qdrant chunks_semantic
    grounding_similarity_threshold: float = 0.83
    off_topic_similarity_threshold: float = 0.40

    retrieval_top_k: int = 5

    # Retry/timeout policy for external calls
    stt_timeout_seconds: float = 8.0
    stt_max_retries: int = 2
    llm_timeout_seconds: float = 10.0
    llm_max_retries: int = 2


settings = Settings()
