"""
Text processing utilities.
"""

import re
import unicodedata
from typing import Dict, List, Optional
from urllib.parse import urlparse


def clean_url(url: str) -> str:
    """
    Clean and standardize a URL.

    Args:
        url: Raw URL string

    Returns:
        Cleaned URL in format: <url>: (domain/path)
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Extract meaningful path (first 1-2 segments)
        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            important_path = "/".join(path_parts[:2])
            return f"<url>: ({domain}/{important_path})"
        else:
            return f"<url>: ({domain})"
    except Exception:
        return "<url>: (unknown)"


def extract_domain(url: str) -> Optional[str]:
    """
    Extract domain from URL.

    Args:
        url: URL string

    Returns:
        Domain name or None if invalid
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def remove_special_characters(text: str, keep_chars: str = "") -> str:
    """
    Remove special characters from text.

    Args:
        text: Input text
        keep_chars: Characters to keep (e.g., ".,!?")

    Returns:
        Cleaned text
    """
    pattern = f"[^a-zA-Z0-9\s{re.escape(keep_chars)}]"
    return re.sub(pattern, "", text)


def truncate_text(text: str, max_length: int = 512, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.

    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode characters.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    return unicodedata.normalize("NFKC", text)


def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from text.

    Args:
        text: Input text

    Returns:
        List of URLs
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    return re.findall(url_pattern, text)


def count_tokens_approximate(text: str) -> int:
    """
    Approximate token count (simple word-based).

    Args:
        text: Input text

    Returns:
        Approximate token count
    """
    # Simple approximation: split by whitespace and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text)
    return len(tokens)


def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    """
    Batch texts into groups.

    Args:
        texts: List of texts
        batch_size: Size of each batch

    Returns:
        List of batches
    """
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


def calculate_text_stats(texts: List[str]) -> Dict[str, float]:
    """
    Calculate statistics for a list of texts.

    Args:
        texts: List of texts

    Returns:
        Dictionary of statistics
    """
    lengths = [len(text) for text in texts]
    token_counts = [count_tokens_approximate(text) for text in texts]

    return {
        "count": len(texts),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
    }
