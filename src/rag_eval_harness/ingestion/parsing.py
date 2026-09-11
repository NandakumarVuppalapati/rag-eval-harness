"""Extract clean narrative text from real SEC iXBRL filing HTML.

SEC filings since ~2019 are filed as Inline XBRL (iXBRL): the
human-readable HTML has machine-readable XBRL tagging woven directly
into it, plus a large hidden block (``ix:header`` / ``ix:hidden``) that
carries raw XBRL facts, contexts, and units never meant to be displayed.
Naively stripping HTML tags pulls all of that hidden tag soup into the
"visible" text -- in one filing we measured, that hidden block was over
90% of the naively-extracted text. This module removes it properly so
the corpus is genuine narrative/table content, not XBRL metadata noise.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_HEADER_OR_HIDDEN_TAG = re.compile(r"(^|:)(header|hidden)$")


def extract_clean_text(html_path: Path) -> str:
    raw = html_path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "lxml")

    for tag in soup(["script", "style"]):
        tag.decompose()

    # iXBRL hidden metadata: <ix:header> wraps <ix:hidden>/<ix:references>/
    # <ix:resources> -- raw facts, contexts, and units, not narrative text.
    for tag in soup.find_all(_HEADER_OR_HIDDEN_TAG):
        tag.decompose()

    # Anything else explicitly hidden via inline style or EDGAR's convention.
    for tag in soup.select('[style*="display:none"], [style*="display: none"], .hidden'):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()
