"""
The harness: runs all 6 stages in order, collects timing for each,
and short-circuits early when guardrails say to (off-topic / unsafe /
not grounded) — this is both correctness AND a latency optimization.
"""
from dataclasses import dataclass, field
from typing import Any

from app.pipeline.classify import classify_query
from app.pipeline.generate import generate
from app.pipeline.guardrails import check_grounding
from app.pipeline.retrieve import retrieve
from app.pipeline.transcribe import TranscriptionError, transcribe
from app.pipeline.types import QueryVerdict, StageTiming
from app.pipeline.verify import verify


@dataclass
class PipelineResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    is_fully_grounded: bool = True
    flagged_claims: list[Any] = field(default_factory=list)
    stopped_at: str = "verify"  # which stage the pipeline ended at
    timings: list[StageTiming] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return sum(t.ms for t in self.timings)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "is_fully_grounded": self.is_fully_grounded,
            "flagged_claims": [
                fc.__dict__ if hasattr(fc, "__dict__") else str(fc)
                for fc in self.flagged_claims
            ],
            "stopped_at": self.stopped_at,
            "latency_ms": {
                "total": round(self.total_ms, 2),
                "stages": {t.name: round(t.ms, 2) for t in self.timings},
            },
        }


def run_pipeline(audio_bytes: bytes) -> PipelineResult:
    timings: list[StageTiming] = []

    # Stage 1: transcribe
    try:
        transcript = transcribe(audio_bytes, timings=timings)
    except TranscriptionError:
        return PipelineResult(
            answer="Sorry, I didn't catch that — could you try again?",
            stopped_at="transcribe",
            timings=timings,
        )

    if not transcript.text:
        return PipelineResult(
            answer="I couldn't hear any clear speech in the audio file — please try recording again.",
            stopped_at="transcribe",
            timings=timings,
        )

    # Stage 2: classify
    intent = classify_query(transcript.text, timings=timings)
    if intent.verdict == QueryVerdict.UNSAFE:
        return PipelineResult(
            answer="I can't help with that request.",
            stopped_at="classify_query",
            timings=timings,
        )
    if intent.verdict == QueryVerdict.OFF_TOPIC:
        return PipelineResult(
            answer="That's outside what this dataset covers, so I don't have a grounded answer for it.",
            stopped_at="classify_query",
            timings=timings,
        )

    # Stage 3: retrieve
    chunks = retrieve(transcript.text, timings=timings)

    # Stage 4: grounding guardrail
    grounding = check_grounding(chunks, timings=timings)
    if not grounding.can_answer:
        return PipelineResult(
            answer="I don't have enough information in the dataset to answer that confidently.",
            stopped_at="check_grounding",
            timings=timings,
        )

    # Stage 5: generate
    answer = generate(transcript.text, chunks, timings=timings)

    # Stage 6: verify
    verified = verify(answer, chunks, timings=timings)

    return PipelineResult(
        answer=verified.text,
        citations=verified.citations,
        is_fully_grounded=verified.is_fully_grounded,
        flagged_claims=verified.flagged_claims,
        stopped_at="verify",
        timings=timings,
    )


def run_pipeline_text(query_text: str) -> PipelineResult:
    """
    Runs stages 2-6 for direct text input, skipping stage 1 (transcribe).
    Uses the exact same stage functions, embedder, vector retrieval, guardrails,
    generator, and verifier as run_pipeline(audio_bytes).
    """
    timings: list[StageTiming] = []

    text = query_text.strip()
    if not text:
        return PipelineResult(
            answer="Empty query text provided.",
            stopped_at="classify_query",
            timings=timings,
        )

    # Stage 2: classify
    intent = classify_query(text, timings=timings)
    if intent.verdict == QueryVerdict.UNSAFE:
        return PipelineResult(
            answer="I can't help with that request.",
            stopped_at="classify_query",
            timings=timings,
        )
    if intent.verdict == QueryVerdict.OFF_TOPIC:
        return PipelineResult(
            answer="That's outside what this dataset covers, so I don't have a grounded answer for it.",
            stopped_at="classify_query",
            timings=timings,
        )

    # Stage 3: retrieve (uses existing retrieve() function which applies 'query: ' prefix)
    chunks = retrieve(text, timings=timings)

    # Stage 4: grounding guardrail
    grounding = check_grounding(chunks, timings=timings)
    if not grounding.can_answer:
        return PipelineResult(
            answer="I don't have enough information in the dataset to answer that confidently.",
            stopped_at="check_grounding",
            timings=timings,
        )

    # Stage 5: generate
    answer = generate(text, chunks, timings=timings)

    # Stage 6: verify
    verified = verify(answer, chunks, timings=timings)

    return PipelineResult(
        answer=verified.text,
        citations=verified.citations,
        is_fully_grounded=verified.is_fully_grounded,
        flagged_claims=verified.flagged_claims,
        stopped_at="verify",
        timings=timings,
    )

