"""
Stage 6: verify(answer, chunks) -> VerifiedAnswer

Cheap heuristic first (numbers/named-entity check against source text);
escalate to a second lightweight LLM call only if you have latency budget
left — heuristic-only is a legitimate, faster choice, note this tradeoff
in the README.
"""
import re

from app.pipeline.types import Answer, Chunk, VerifiedAnswer, timed_stage

_NUMBER_RE = re.compile(r"\b\d{2,}\b")  # crude: 2+ digit numbers worth checking


def _numbers_in(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


@timed_stage("verify")
def verify(answer: Answer, chunks: list[Chunk]) -> VerifiedAnswer:
    source_text = " ".join(c.text for c in chunks)
    source_numbers = _numbers_in(source_text)
    answer_numbers = _numbers_in(answer.text)

    unsupported_numbers = answer_numbers - source_numbers
    flagged = [f"number '{n}' not found in retrieved context" for n in unsupported_numbers]

    # If the model cited nothing at all despite chunks being available, flag it too.
    if chunks and not answer.citations:
        flagged.append("answer contains no chunk citations despite retrieved context")

    return VerifiedAnswer(
        text=answer.text,
        citations=answer.citations,
        flagged_claims=flagged,
        is_fully_grounded=len(flagged) == 0,
    )
