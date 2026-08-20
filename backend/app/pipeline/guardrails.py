"""
Stage 4: check_grounding(chunks) -> GroundingDecision

Fast, zero-network Grounding Gate (Stage 4):
Evaluates whether top-1 vector retrieval score clears the calibrated grounding_similarity_threshold (0.83).
If score is insufficient or chunks list is empty, skips the downstream LLM generation stage (Stage 5)
to prevent hallucinations and save latency.
"""
import logging

from app.config import settings
from app.pipeline.types import Chunk, GroundingDecision, timed_stage

logger = logging.getLogger("ragingoa.guardrails")


@timed_stage("check_grounding")
def check_grounding(chunks: list[Chunk]) -> GroundingDecision:
    """
    Evaluates grounding quality of retrieved chunks:
    - Empty list -> GroundingDecision(can_answer=False, reason="no chunks retrieved", top_score=0.0)
    - top_score < threshold -> GroundingDecision(can_answer=False, ...)
    - top_score >= threshold -> GroundingDecision(can_answer=True, ...)
    """
    if not chunks:
        logger.info("check_grounding received empty chunk list. Skipping generation.")
        return GroundingDecision(can_answer=False, reason="no chunks retrieved", top_score=0.0)

    top_score = float(chunks[0].score)
    threshold = float(settings.grounding_similarity_threshold)

    if top_score < threshold:
        reason = f"top retrieval score {top_score:.4f} below grounding threshold {threshold:.2f}"
        logger.info("check_grounding verdict=CANNOT_ANSWER (%s)", reason)
        return GroundingDecision(can_answer=False, reason=reason, top_score=top_score)

    reason = f"sufficient corpus grounding (top_score {top_score:.4f} >= threshold {threshold:.2f})"
    logger.info("check_grounding verdict=CAN_ANSWER (%s)", reason)
    return GroundingDecision(can_answer=True, reason=reason, top_score=top_score)

