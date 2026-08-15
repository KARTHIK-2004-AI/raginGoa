"""
Typed data structures passed between pipeline stages, plus a timing decorator.

Why this file matters for the "harness" requirement: every stage takes a typed
input and returns a typed output, and every stage's duration is captured
automatically. This is what turns 6 functions into an orchestrated pipeline
instead of "a script that calls things."
"""
from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar


# ---------- Stage I/O types ----------

@dataclass
class Transcript:
    text: str
    language: str | None = None
    confidence: float | None = None


class QueryVerdict(str, Enum):
    IN_SCOPE = "in_scope"
    OFF_TOPIC = "off_topic"
    UNSAFE = "unsafe"


@dataclass
class QueryIntent:
    verdict: QueryVerdict
    reason: str = ""


@dataclass
class Chunk:
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class GroundingDecision:
    can_answer: bool
    reason: str
    top_score: float


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)


@dataclass
class VerifiedAnswer:
    text: str
    citations: list[str]
    flagged_claims: list[str] = field(default_factory=list)
    is_fully_grounded: bool = True


# ---------- Timing ----------

@dataclass
class StageTiming:
    name: str
    ms: float
    ok: bool
    error: str | None = None


F = TypeVar("F", bound=Callable[..., Any])


def timed_stage(name: str):
    """
    Decorator: wraps a pipeline stage function, records wall-clock time,
    and attaches a StageTiming to a `timings` list passed via kwarg.

    Usage:
        @timed_stage("transcribe")
        def transcribe(audio_bytes: bytes) -> Transcript: ...

        result = transcribe(audio_bytes, timings=timings_list)
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, timings: list[StageTiming] | None = None, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if timings is not None:
                    timings.append(StageTiming(name=name, ms=elapsed_ms, ok=True))
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if timings is not None:
                    timings.append(StageTiming(name=name, ms=elapsed_ms, ok=False, error=str(e)))
                raise
        return wrapper  # type: ignore
    return decorator
