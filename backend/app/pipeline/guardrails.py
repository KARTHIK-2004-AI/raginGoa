"""
Stage 4: check_grounding(chunks) -> GroundingDecision

If nothing retrieved clears the similarity bar, we skip the LLM call
entirely. This is both a guardrail (don't hallucinate) and a latency win
(no LLM call on a doomed query).
"""
from app.config import settings
from app.pipeline.types import Chunk, GroundingDecision, timed_stage


@timed_stage("check_grounding")
def check_grounding(chunks: list[Chunk]) -> GroundingDecision:
    if not chunks:
        return GroundingDecision(can_answer=False, reason="no chunks retrieved", top_score=0.0)

    top_score = max(c.score for c in chunks)
    if top_score < settings.grounding_similarity_threshold:
        return GroundingDecision(
            can_answer=False,
            reason=f"top retrieval score {top_score:.2f} below threshold "
                   f"{settings.grounding_similarity_threshold}",
            top_score=top_score,
        )

    return GroundingDecision(can_answer=True, reason="sufficient grounding", top_score=top_score)
