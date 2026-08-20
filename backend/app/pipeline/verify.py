"""
Stage 6: verify(answer, chunks) -> VerifiedAnswer

Deterministic, regex-only factual grounding check (no LLM/embedding calls).
Extracts numeric/date claims (integers, decimals, percentages, years) from generated text,
normalizes Devanagari digits to Latin digits, maps inline [N] citation tags to retrieved chunks,
and verifies substring presence of numbers in the cited source chunks.

Returns VerifiedAnswer(text, citations, flagged_claims, is_fully_grounded).
"""
import logging
import re
from typing import Any

from app.pipeline.types import Answer, Chunk, FlaggedClaim, VerifiedAnswer, timed_stage

logger = logging.getLogger("ragingoa.verify")

# Translation table to map Devanagari digits (०-९) to Latin digits (0-9)
DEVANAGARI_TO_LATIN = str.maketrans("०१२३४५६७८९", "0123456789")

# Regex to match numeric claims: integers, decimals, percentages, years
NUMERIC_CLAIM_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?%?\b")

# Regex to extract citation markers like [1], [2]
CITATION_TAG_RE = re.compile(r"\[(\d+)\]")

# Sentence enders: Hindi Purna Viram (।), period (.), question mark (?), exclamation (!), newline (\n)
SENTENCE_SPLIT_RE = re.compile(r"[।\.\?!\n]+")


def normalize_digits(text: str) -> str:
    """Normalizes Devanagari numerals (०-९) to standard Latin digits (0-9)."""
    return text.translate(DEVANAGARI_TO_LATIN)


def _get_chunk_id(chunk: Any) -> str:
    """Duck-type accessor for chunk ID."""
    if hasattr(chunk, "id"):
        return str(chunk.id)
    if hasattr(chunk, "chunk_id"):
        return str(chunk.chunk_id)
    if isinstance(chunk, dict):
        return str(chunk.get("id", chunk.get("chunk_id", "")))
    return ""


def _get_chunk_text(chunk: Any) -> str:
    """Duck-type accessor for chunk text content."""
    if hasattr(chunk, "text"):
        return str(chunk.text)
    if hasattr(chunk, "content"):
        return str(chunk.content)
    if isinstance(chunk, dict):
        return str(chunk.get("text", chunk.get("content", "")))
    return ""


@timed_stage("verify")
def verify(
    answer: Answer | str,
    chunks: list[Chunk] | list[Any] | None = None,
    citations: list[str] | None = None,
) -> VerifiedAnswer:
    """
    Stage 6 Deterministic Grounding Verification:
    - Extracts numeric claims ( Latin and Devanagari digits ) sentence by sentence.
    - Associates claims with inline bracket citations ([1], [2], etc.).
    - Validates presence of numbers in the cited chunk's text.
    - Returns VerifiedAnswer containing flagged claims and boolean grounding verdict.
    """
    # Disambiguate arguments flexibly across all calling conventions:
    # 1) verify(answer: Answer, chunks: list[Chunk])
    # 2) verify(answer_text: str, citations: list[str], chunks: list[Chunk])
    # 3) verify(answer_text: str, chunks: list[Chunk], citations: list[str])
    if isinstance(answer, Answer):
        answer_text = answer.text
        ans_citations = answer.citations
        target_chunks = chunks or []
    elif isinstance(chunks, list) and len(chunks) > 0 and isinstance(chunks[0], str):
        answer_text = str(answer or "")
        ans_citations = chunks
        target_chunks = citations or []
    else:
        answer_text = str(answer or "")
        target_chunks = chunks or []
        ans_citations = citations or []

    cleaned_text = answer_text.strip()
    if not cleaned_text:
        return VerifiedAnswer(
            text=answer_text,
            citations=ans_citations,
            flagged_claims=[],
            is_fully_grounded=True,
        )

    # Pre-normalize chunk texts with digit translation
    normalized_chunks_text = []
    for c in target_chunks:
        raw_t = _get_chunk_text(c)
        normalized_chunks_text.append(normalize_digits(raw_t))

    flagged_claims: list[FlaggedClaim] = []

    # Split text into sentences to associate citations with claims in the same sentence
    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(cleaned_text) if s.strip()]

    for sentence in sentences:
        norm_sentence = normalize_digits(sentence)

        # Find citation tags like [1], [2] in sentence first
        citation_matches = CITATION_TAG_RE.findall(sentence)
        citation_indices = [int(idx_str) for idx_str in citation_matches]

        # Strip citation tags ([1], [2], etc.) before scanning for numeric claims
        text_without_citations = CITATION_TAG_RE.sub("", norm_sentence)
        claims = NUMERIC_CLAIM_RE.findall(text_without_citations)
        if not claims:
            continue

        if citation_indices:
            # Check numeric claims against each cited chunk
            for claim in set(claims):
                for cit_idx in set(citation_indices):
                    if cit_idx < 1 or cit_idx > len(target_chunks):
                        flagged_claims.append(
                            FlaggedClaim(
                                claim_text=claim,
                                citation_index=cit_idx,
                                chunk_id=None,
                                reason=f"citation index [{cit_idx}] has no matching retrieved chunk",
                            )
                        )
                    else:
                        target_chunk = target_chunks[cit_idx - 1]
                        chunk_id = _get_chunk_id(target_chunk)
                        chunk_text_norm = normalized_chunks_text[cit_idx - 1]

                        if claim not in chunk_text_norm:
                            flagged_claims.append(
                                FlaggedClaim(
                                    claim_text=claim,
                                    citation_index=cit_idx,
                                    chunk_id=chunk_id,
                                    reason=f"numeric claim '{claim}' not found in cited chunk [{cit_idx}] ({chunk_id})",
                                )
                            )
        else:
            # Uncited numeric claims: check if number exists in ANY retrieved chunk
            for claim in set(claims):
                found_in_any = any(claim in chunk_norm for chunk_norm in normalized_chunks_text)
                if not found_in_any:
                    flagged_claims.append(
                        FlaggedClaim(
                            claim_text=claim,
                            citation_index=None,
                            chunk_id=None,
                            reason=f"uncited numeric claim '{claim}' not found in any retrieved chunk",
                        )
                    )

    is_fully_grounded = len(flagged_claims) == 0

    if not is_fully_grounded:
        logger.warning("Verify stage flagged %d ungrounded claim(s)", len(flagged_claims))

    return VerifiedAnswer(
        text=answer_text,
        citations=ans_citations,
        flagged_claims=flagged_claims,
        is_fully_grounded=is_fully_grounded,
    )
