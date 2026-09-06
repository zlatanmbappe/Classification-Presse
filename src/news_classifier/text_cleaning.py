"""Cleaning and checksum functions for French news articles."""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from bs4 import BeautifulSoup

SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "lxml")
    for tag in soup(["script", "style", "noscript", "template", "svg", "form"]):
        tag.decompose()
    return soup.get_text(" ")


def clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = unicodedata.normalize("NFC", text)
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    text = re.sub(r"https?://\S+", " ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def checksum(text: str) -> str:
    return hashlib.sha256(clean_text(text).encode("utf-8", errors="ignore")).hexdigest()
