"""
Stage 5: generate(query, chunks) -> Answer

Generates grounded Hindi answers using Google Gemini LLM API based ONLY on retrieved chunks.
Forces inline bracket citations ([1], [2]) and maps them back to actual chunk UUIDs.
Returns Answer(text, citations). Returns Answer(text="", citations=[]) on API failure.
"""
import logging
import re
import time

try:
    from google import genai  # type: ignore
    _USE_NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai  # type: ignore
        _USE_NEW_SDK = False
    except ImportError:
        genai = None  # type: ignore
        _USE_NEW_SDK = False

from app.config import settings
from app.pipeline.types import Answer, Chunk, timed_stage

logger = logging.getLogger("ragingoa.generate")


def _call_gemini_api(prompt: str, model_name: str) -> str:
    """Invokes Gemini API with fallback between google-genai and google-generativeai SDKs."""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing in configuration.")

    if genai is None:
        raise ImportError("Neither 'google-genai' nor 'google-generativeai' is installed in virtualenv.")

    # Clean model name for Gemini API compatibility
    target_model = model_name.replace("models/", "")

    if _USE_NEW_SDK:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = client.models.generate_content(
            model=target_model,
            contents=prompt,
        )
        return (response.text or "").strip()
    else:
        genai.configure(api_key=settings.gemini_api_key)
        model_obj = genai.GenerativeModel(model_name=target_model)
        response = model_obj.generate_content(prompt)
        return (response.text or "").strip()


def _build_prompt(query: str, chunks: list[Chunk]) -> tuple[str, dict[str, str]]:
    """
    Builds structured prompt with short numeric citation tags ([1], [2], etc.).
    Returns (prompt_text, tag_to_chunk_id_map).
    """
    context_lines = []
    tag_to_id = {}
    for idx, c in enumerate(chunks, 1):
        tag = f"[{idx}]"
        tag_to_id[tag] = c.id
        context_lines.append(f"{tag} {c.text}")

    context_block = "\n\n".join(context_lines)
    prompt = (
        "आप एक सहायक RAG AI हैं। केवल नीचे दिए गए संदर्भ (Context) का उपयोग करके प्रश्न का उत्तर हिंदी में दें।\n"
        "नियम:\n"
        "1. प्रत्येक मुख्य वाक्य या दावे के अंत में संबंधित संदर्भ नंबर ([1], [2], आदि) का उपयोग करके उद्धरण दें।\n"
        "2. केवल संदर्भ में दी गई जानकारी का उपयोग करें। यदि संदर्भ उत्तर प्रदान नहीं करता है, तो स्पष्ट रूप से बताएं कि उत्तर उपलब्ध नहीं है। मनगढ़ंत जानकारी न जोड़ें।\n\n"
        f"संदर्भ (Context):\n{context_block}\n\n"
        f"प्रश्न (Question): {query}\n\n"
        "उत्तर (Answer):"
    )
    return prompt, tag_to_id


_DISCOVERED_MODELS_CACHE: list[str] | None = None


def _get_available_gemini_models() -> list[str]:
    """Dynamically queries Gemini API for available text-generation models, with caching and strict modality filtering."""
    global _DISCOVERED_MODELS_CACHE
    if _DISCOVERED_MODELS_CACHE is not None:
        return _DISCOVERED_MODELS_CACHE

    if not settings.gemini_api_key or genai is None:
        _DISCOVERED_MODELS_CACHE = []
        return _DISCOVERED_MODELS_CACHE

    exclude_keywords = [
        "tts",
        "image",
        "imagen",
        "robotics",
        "computer-use",
        "customtools",
        "embed",
        "embedding",
        "audio",
        "bison",
        "preview",
        "omni",
        "exp",
    ]

    try:
        valid = []
        if _USE_NEW_SDK:
            client = genai.Client(api_key=settings.gemini_api_key)
            models_list = list(client.models.list())
            for m in models_list:
                name_raw = getattr(m, "name", "").replace("models/", "")
                name_lower = name_raw.lower()
                if any(kw in name_lower for kw in exclude_keywords):
                    continue
                methods = getattr(m, "supported_generation_methods", []) or getattr(m, "supported_actions", [])
                if not methods or any("generateContent" in str(x) or "generate_content" in str(x) for x in methods):
                    if "gemini" in name_lower:
                        valid.append(name_raw)
        else:
            genai.configure(api_key=settings.gemini_api_key)
            models_list = list(genai.list_models())
            for m in models_list:
                name_raw = getattr(m, "name", "").replace("models/", "")
                name_lower = name_raw.lower()
                if any(kw in name_lower for kw in exclude_keywords):
                    continue
                methods = getattr(m, "supported_generation_methods", [])
                if any("generateContent" in str(x) for x in methods):
                    valid.append(name_raw)

        _DISCOVERED_MODELS_CACHE = valid
        return _DISCOVERED_MODELS_CACHE
    except Exception as e:
        logger.warning("Failed to list models dynamically from Gemini API: %s", e)
        _DISCOVERED_MODELS_CACHE = []
        return _DISCOVERED_MODELS_CACHE


def _build_candidate_models(primary_model: str) -> list[str]:
    """
    Builds candidate list ONLY from live models returned by models.list().
    Ranks candidates by performance/reliability preference:
      1. Primary configured model (if present in live models)
      2. gemini-flash-lite-latest
      3. gemini-flash-latest
      4. gemini-1.5-flash-latest
      5. gemini-2.5-flash
      6. Other live discovered text models
    """
    discovered = _get_available_gemini_models()

    priority_order = [
        primary_model,
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-1.5-flash-latest",
        "gemini-2.5-flash",
    ]

    candidates = []
    # Add prioritized models ONLY if present in live discovered list
    for p in priority_order:
        if p and p in discovered and p not in candidates:
            candidates.append(p)

    # Append any other surviving discovered models
    for d in discovered:
        if d not in candidates:
            candidates.append(d)

    # Fallback safety net only if models.list() call failed entirely
    if not candidates:
        fallback = [primary_model, "gemini-flash-lite-latest", "gemini-flash-latest"]
        for f in fallback:
            if f and f not in candidates:
                candidates.append(f)

    return candidates


def _clean_text_formatting(text: str) -> str:
    """Strips leftover citation tags and stray commas/spaces before sentence enders."""
    # 1. Remove bracket citation patterns like [1], [1, 2], [1], [2]
    cleaned = re.sub(r"\[[\d,\s]+\]", "", text)
    # 2. Remove orphan commas and spaces before punctuation or at end of text
    cleaned = re.sub(r"[\s,]+(?=[।\.\?!]|$)", "", cleaned)
    # 3. Collapse multiple consecutive commas or spaces
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_citations_and_clean_text(raw_text: str, chunks: list[Chunk]) -> tuple[str, list[str]]:
    """
    Extracts citation chunk IDs from both single tags ([1]) and grouped tags ([1, 2, 3]).
    If response indicates no answer is available, strips leftover citation tags and returns empty citations.
    Otherwise, cleans orphaned bracket tags and returns (cleaned_text, cited_chunk_ids).
    """
    tag_matches = re.findall(r"\[([\d,\s]+)\]", raw_text)
    cited_ids = []

    for match in tag_matches:
        parts = [p.strip() for p in match.split(",") if p.strip().isdigit()]
        for p in parts:
            idx = int(p)
            if 1 <= idx <= len(chunks):
                chunk_id = chunks[idx - 1].id
                if chunk_id not in cited_ids:
                    cited_ids.append(chunk_id)

    refusal_keywords = [
        "उत्तर उपलब्ध नहीं है",
        "जानकारी नहीं है",
        "संदर्भ में उत्तर नहीं है",
        "उल्लेख नहीं है",
        "नहीं दिया गया",
        "स्पष्ट नहीं है",
        "नहीं है",
        "no answer available",
        "not provided in the context",
        "not mentioned",
        "cannot be answered",
    ]
    is_refusal = any(kw in raw_text.lower() for kw in refusal_keywords)

    # Defensive cleanup: Gemini occasionally leaves reference-list syntax ([N], [1, 2], or bare commas)
    # in refusal/empty-citation answers even when no valid citations are extracted.
    if is_refusal or not cited_ids:
        cleaned_text = _clean_text_formatting(raw_text)
        return cleaned_text, []

    return raw_text, cited_ids


@timed_stage("generate")
def generate(query: str, chunks: list[Chunk]) -> Answer:
    """
    Stage 5 LLM Generation:
    - If chunks is empty, returns empty Answer(text="", citations=[]).
    - Constructs prompt with [1], [2] citation tags.
    - Calls Gemini API with retries over live candidate models.
    - Parses cited tags back to real chunk IDs.
    - Returns Answer(text, citations).
    """
    cleaned_query = query.strip()
    if not chunks or not cleaned_query:
        logger.warning("generate called with empty chunks or empty query. Returning empty Answer.")
        return Answer(text="", citations=[])

    prompt, tag_to_id = _build_prompt(cleaned_query, chunks)
    primary_model = getattr(settings, "gemini_model", "gemini-flash-lite-latest")
    models_to_try = _build_candidate_models(primary_model)
    logger.info("Candidate Gemini models (filtered & prioritized): %s", models_to_try)

    # Limit retries per candidate model to 2 attempts for faster failover
    max_retries_per_model = min(settings.llm_max_retries, 1)

    last_err: Exception | None = None
    for model_name in models_to_try:
        for attempt in range(max_retries_per_model + 1):
            try:
                logger.info(
                    "Calling Gemini LLM (model=%s, attempt=%d/%d)...",
                    model_name,
                    attempt + 1,
                    max_retries_per_model + 1,
                )
                raw_text = _call_gemini_api(prompt, model_name)
                if not raw_text:
                    logger.warning("Gemini API returned empty text response.")
                    return Answer(text="", citations=[])

                cleaned_text, cited_ids = _extract_citations_and_clean_text(raw_text, chunks)

                logger.info(
                    "Gemini generation succeeded with model '%s': text_len=%d, citations_count=%d",
                    model_name,
                    len(cleaned_text),
                    len(cited_ids),
                )
                return Answer(text=cleaned_text, citations=cited_ids)

            except Exception as e:
                last_err = e
                err_str = str(e)
                if "404" in err_str or "NOT_FOUND" in err_str or "not found" in err_str.lower():
                    logger.warning("Gemini model '%s' not found (404). Trying next candidate model...", model_name)
                    break

                if attempt < max_retries_per_model:
                    delay = 0.2 * (2 ** attempt)
                    logger.warning(
                        "Gemini call attempt %d failed for model '%s': %s. Retrying in %.2fs...",
                        attempt + 1,
                        model_name,
                        e,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Gemini LLM call failed for model '%s' after %d attempts: %s",
                        model_name,
                        max_retries_per_model + 1,
                        e,
                    )

    logger.error("All Gemini model candidates failed. Last error: %s", last_err)
    return Answer(text="", citations=[])


