"""
Central config. Load everything from env so nobody hardcodes API keys.
Copy .env.example to .env and fill in real values.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")

    embedding_model_name: str = "intfloat/multilingual-e5-small"

    # Which Qdrant collection is "in production" for the live demo.
    # Swap this once you've picked a winning chunking strategy (see eval_retrieval.py results).
    active_collection: str = os.getenv("ACTIVE_COLLECTION", "chunks_structured")

    # Guardrail thresholds — tune these once you see real retrieval scores
    grounding_similarity_threshold: float = 0.55
    off_topic_similarity_threshold: float = 0.40

    retrieval_top_k: int = 5

    # Retry/timeout policy for external calls
    stt_timeout_seconds: float = 8.0
    stt_max_retries: int = 2
    llm_timeout_seconds: float = 10.0
    llm_max_retries: int = 2


settings = Settings()
