"""
Stage 2: classify_query(transcript_text) -> QueryIntent

Fast, deterministic Safety & Input Quality Gate (Stage 2):
1. Early return for empty input.
2. Safety check: Fast keyword/pattern filter for unsafe content (returns QueryVerdict.UNSAFE).
3. Lexical quality check: Heuristic detection of keyboard mash, symbol spam, and gibberish (returns QueryVerdict.OFF_TOPIC).
4. Well-formed queries pass cleanly to Stage 4 (Grounding Gate) for retrieval-based grounding verification.
"""
import logging
import re

from app.pipeline.types import QueryIntent, QueryVerdict, timed_stage

logger = logging.getLogger("ragingoa.classify")

# Short, sane list of unsafe keywords (English & Hindi)
_UNSAFE_PATTERNS = [
    r"\bbomb\b",
    r"\bweapon(s)?\b",
    r"\bkill\b",
    r"\bsuicide\b",
    r"\bself-harm\b",
    r"\bexplosive(s)?\b",
    r"\bहत्या\b",
    r"\bबम\b",
    r"\bहथियार\b",
    r"\bआत्महत्या\b",
]

_UNSAFE_REGEX = re.compile("|".join(_UNSAFE_PATTERNS), re.IGNORECASE)

# Direct prompt injection & role manipulation heuristics (keyword/regex first-pass filter)
_INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+(?:instructions|rules)\b",
    r"\bdisregard\s+(?:all\s+)?(?:previous|prior)\s+(?:guidance|rules)\b",
    r"\bforget\s+everything\s+(?:you\s+were\s+told)?\b",
    r"\bshow\s+(?:me\s+)?(?:your\s+)?system\s+prompt\b",
    r"\bprint\s+(?:your\s+)?system\s+prompt\b",
    r"\bsecret\s+(?:developer\s+)?key\b",
    r"\bdeveloper\s+mode\b",
    r"\byou\s+are\s+now\s+dan\b",
    r"\bact\s+as\s+(?:an?\s+)?unrestricted\b",
    r"\bpretend\s+(?:that\s+)?restrictions?\s+don'?t\s+exist\b",
    r"\[\s*SYSTEM\s*:",
    r"\[\s*ASSISTANT\s*:",
    r"<\|im_start\|>",
    r"<\/system>",
]

_INJECTION_REGEX = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Lexical gibberish & garbage input heuristics
_REPEATED_CHARS_REGEX = re.compile(r"(.)\1{4,}")  # 5+ repeated characters (e.g., "aaaaa", "zzzzzz", "ककककक")
_LONG_CONSONANTS_REGEX = re.compile(r"[bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZ]{5,}")  # 5+ consecutive English consonants (excluding y/Y)
_SYMBOL_SPAM_REGEX = re.compile(r"^[^\w\s]+$")  # Pure punctuation/symbol string

# Devanagari-specific gibberish heuristics
_DEVANAGARI_VIRAMA_SPAM = re.compile(r"[\u094D\u0900-\u0903\u093A-\u094C]{3,}")  # 3+ repeated halants/matras without consonants
_DEVANAGARI_BARE_CONSONANTS = re.compile(r"[\u0915-\u0939\u0958-\u095F]{6,}")  # 6+ consecutive bare Devanagari consonants with no matras


# Domain anchor texts for corpus centroid calculation (Goa domain exemplar embeddings)
_GOA_DOMAIN_ANCHORS = [
    "गोवा के पर्यटन स्थल और प्रसिद्ध समुद्र तट",
    "Which is the most famous beach in Goa?",
    "गोवा की राजधानी पणजी और इतिहास",
    "Goa geography, climate, tourism, beaches, culture, and history",
    "गोवा घूमने का सबसे अच्छा समय और मौसम",
    "Baga Beach, Calangute Beach, Panaji, Mandovi River in Goa",
]

_DOMAIN_CENTROID = None


def _get_domain_centroid():
    """Lazily computes and caches normalized mean centroid embedding vector of Goa domain anchors."""
    global _DOMAIN_CENTROID
    if _DOMAIN_CENTROID is None:
        try:
            import numpy as np
            from app.pipeline.embed import get_embedder

            embedder = get_embedder()
            texts = [f"passage: {t}" for t in _GOA_DOMAIN_ANCHORS]
            vectors = embedder.encode(texts, normalize_embeddings=True)
            centroid = np.mean(vectors, axis=0)
            _DOMAIN_CENTROID = centroid / np.linalg.norm(centroid)
        except Exception as e:
            logger.warning("Failed to compute domain centroid: %s", e)
            _DOMAIN_CENTROID = None
    return _DOMAIN_CENTROID


def _check_safety(text: str) -> str | None:
    match = _UNSAFE_REGEX.search(text)
    if match:
        return match.group(0)
    return None


def _check_injection(text: str) -> str | None:
    match = _INJECTION_REGEX.search(text)
    if match:
        return match.group(0)
    return None


def _check_corpus_similarity(text: str) -> tuple[float, bool]:
    """
    Computes cosine similarity of query embedding against domain corpus centroid.
    Returns (similarity_score, is_off_topic).
    """
    centroid = _get_domain_centroid()
    if centroid is None:
        return 1.0, False  # Fallback to passing if centroid fails

    try:
        import numpy as np
        from app.config import settings
        from app.pipeline.embed import get_embedder

        embedder = get_embedder()
        query_vec = embedder.encode(f"query: {text}", normalize_embeddings=True)
        sim = float(np.dot(query_vec, centroid))
        threshold = float(getattr(settings, "off_topic_similarity_threshold", 0.83))
        is_off_topic = sim < threshold
        return sim, is_off_topic
    except Exception as e:
        logger.warning("Corpus similarity check failed: %s", e)
        return 1.0, False


def _check_gibberish(text: str) -> str | None:
    """
    Fast lexical heuristic for garbage/gibberish input (English + Devanagari + Script-agnostic):
    - Pure symbol/punctuation spam
    - 5+ repeated consecutive characters (any script)
    - Devanagari 3+ consecutive unattached halants/matras (e.g., ्््)
    - Devanagari 6+ consecutive bare consonants without matras (e.g., कखगघचछ)
    - 4+ consecutive English consonants (keyboard mash)
    - Low vowel ratio in English strings
    - High single-character dominance in any script (> 45% identical chars)
    """
    if _SYMBOL_SPAM_REGEX.match(text):
        return "pure symbol/punctuation spam"

    match_repeated = _REPEATED_CHARS_REGEX.search(text)
    if match_repeated:
        return f"repeated character sequence '{match_repeated.group(0)}'"

    # Devanagari-specific heuristics
    match_dev_virama = _DEVANAGARI_VIRAMA_SPAM.search(text)
    if match_dev_virama:
        return f"Devanagari halant/matra spam '{match_dev_virama.group(0)}'"

    match_dev_consonants = _DEVANAGARI_BARE_CONSONANTS.search(text)
    if match_dev_consonants:
        return f"Devanagari bare consonant sequence '{match_dev_consonants.group(0)}'"

    # English consonant sequence heuristic
    match_consonants = _LONG_CONSONANTS_REGEX.search(text)
    if match_consonants:
        return f"keyboard mash consonant sequence '{match_consonants.group(0)}'"

    # Script-agnostic character dominance ratio check
    non_space_chars = [c for c in text if not c.isspace()]
    if len(non_space_chars) >= 6:
        char_counts = {}
        for c in non_space_chars:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_char_count = max(char_counts.values())
        if max_char_count / len(non_space_chars) > 0.45:
            return f"high character dominance ratio ({(max_char_count / len(non_space_chars)):.2f})"

    # Check vowel ratio for English alphabetic text longer than 6 characters
    alpha_chars = [c.lower() for c in text if c.isalpha() and ord(c) < 128]
    if len(alpha_chars) >= 6:
        vowels = set("aeiou")
        vowel_count = sum(1 for c in alpha_chars if c in vowels)
        vowel_ratio = vowel_count / len(alpha_chars)
        if vowel_ratio < 0.22:
            return f"low English vowel ratio ({vowel_ratio:.2f}) in text"

    # Devanagari matra density check for pure Devanagari text >= 6 chars
    dev_chars = [c for c in text if '\u0900' <= c <= '\u097F']
    if len(dev_chars) >= 6:
        dev_vowels_matras = [
            c for c in dev_chars
            if ('\u0904' <= c <= '\u0914') or ('\u0900' <= c <= '\u0903') or ('\u093A' <= c <= '\u094F')
        ]
        dev_vowel_ratio = len(dev_vowels_matras) / len(dev_chars)
        if dev_vowel_ratio < 0.10:  # Less than 10% vowels/matras in Hindi text
            return f"low Devanagari matra/vowel ratio ({dev_vowel_ratio:.2f}) in Hindi text"

    return None


@timed_stage("classify_query")
def classify_query(transcript_text: str) -> QueryIntent:
    """
    Classifies user transcript text (Stage 2 Gate):
    1. Empty input -> OFF_TOPIC
    2. Unsafe keywords or Direct Prompt Injections -> UNSAFE
    3. Gibberish/garbage input -> OFF_TOPIC (Fast lexical check)
    4. Off-topic domain similarity check -> OFF_TOPIC (Embedding distance < threshold)
    5. Well-formed input -> IN_SCOPE (passes to Stage 4 Grounding Gate)
    """
    cleaned_text = transcript_text.strip()
    if not cleaned_text:
        logger.warning("classify_query received empty transcript.")
        return QueryIntent(verdict=QueryVerdict.OFF_TOPIC, reason="empty transcript input")

    # Step 1: Safety Filter (Dangerous / Harmful content)
    matched_unsafe = _check_safety(cleaned_text)
    if matched_unsafe:
        reason = f"matched safety keyword '{matched_unsafe}'"
        logger.warning("classify_query verdict=UNSAFE (%s)", reason)
        return QueryIntent(verdict=QueryVerdict.UNSAFE, reason=reason)

    # Step 2: Direct Prompt Injection Filter
    matched_injection = _check_injection(cleaned_text)
    if matched_injection:
        reason = f"detected prompt injection pattern '{matched_injection}'"
        logger.warning("classify_query verdict=UNSAFE (%s)", reason)
        return QueryIntent(verdict=QueryVerdict.UNSAFE, reason=reason)

    # Step 3: Lexical Gibberish & Input Quality Filter
    gibberish_reason = _check_gibberish(cleaned_text)
    if gibberish_reason:
        reason = f"detected gibberish/nonsense input ({gibberish_reason})"
        logger.info("classify_query verdict=OFF_TOPIC (%s)", reason)
        return QueryIntent(verdict=QueryVerdict.OFF_TOPIC, reason=reason)

    # Step 4: Embedding-based Corpus Domain Similarity Check
    sim_score, is_off_topic = _check_corpus_similarity(cleaned_text)
    if is_off_topic:
        reason = f"domain cosine similarity {sim_score:.4f} below threshold"
        logger.info("classify_query verdict=OFF_TOPIC (%s)", reason)
        return QueryIntent(verdict=QueryVerdict.OFF_TOPIC, reason=reason)

    # Step 5: Well-formed input passes cleanly
    reason = f"passed safety, injection, and domain checks (similarity={sim_score:.4f})"
    logger.info("classify_query verdict=IN_SCOPE (%s)", reason)
    return QueryIntent(verdict=QueryVerdict.IN_SCOPE, reason=reason)



