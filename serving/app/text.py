"""Text cleaning — the *same* ``clean_text`` as preprocess.ipynb, so text arriving
over HTTP is normalised exactly like the training cohort was.

Sprint 3 used a regex to strip tags. Sprint 4's pipeline tests compared it with the
notebook's BeautifulSoup version on all 4,968 raw Kaggle texts and found 13 that
came out differently (malformed entities such as ``&ndash`` without a semicolon,
and stray ``<15>``-style text that BeautifulSoup keeps but the regex dropped).
Small, but it is training/serving skew, so the API now uses the identical
BeautifulSoup call. Plain text with no ``<`` or ``&`` skips the parser entirely —
the result is the same and it costs nothing.
"""
from __future__ import annotations

import hashlib
import re

from bs4 import BeautifulSoup

_NON_PRINTABLE_RE = re.compile(r"[^\x20-\x7E]")
_WS_RE = re.compile(r"\s+")
_LETTER_RE = re.compile(r"[A-Za-z]")


def strip_html(text: str) -> str:
    if "<" not in text and "&" not in text:  # nothing for the parser to do
        return text
    return BeautifulSoup(text, "html.parser").get_text(separator=" ")


def clean_text(text: str) -> str:
    """Strip HTML, drop non-printable/non-ASCII chars, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    text = _NON_PRINTABLE_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def letter_count(text: str) -> int:
    """Used to reject inputs like '!!!@@@###' that pass a length check but carry no words."""
    return len(_LETTER_RE.findall(text))


def text_key(text: str) -> str:
    """Stable cache key for a cleaned text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
