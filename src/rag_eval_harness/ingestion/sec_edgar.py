"""Client for pulling real filings and structured facts from SEC EDGAR.

SEC EDGAR (https://www.sec.gov/edgar) is the public system every US
public company's filings pass through. It exposes two things we care
about for this project:

1. Filing documents (10-K annual reports, 10-Q quarterly reports) as
   real HTML/text -- the messy, unstructured corpus our RAG pipeline
   retrieves from and generates answers over.
2. The XBRL "company facts" API -- the same numbers, but structured
   and machine-readable, tagged by concept (Revenues, NetIncomeLoss,
   etc.) and fiscal period. This is SEC-verified ground truth we use to
   generate numeric golden questions with provably correct answers,
   instead of hand-transcribing figures out of a 200-page PDF.

All requests must set a descriptive User-Agent identifying the caller,
per SEC's fair-access policy (https://www.sec.gov/os/webmaster-faq#developers).
Requests are rate-limited client-side to stay well under SEC's published
limit of 10 requests/second.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

USER_AGENT = "rag-eval-harness research vuppalapatinandakumar@gmail.com"
_HEADERS = {"User-Agent": USER_AGENT}
_MIN_REQUEST_INTERVAL_SECONDS = 0.15  # stays well under SEC's 10 req/s limit
_last_request_time = 0.0


def _throttled_get(url: str, **kwargs) -> requests.Response:
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
        time.sleep(_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
    response = requests.get(url, headers=_HEADERS, timeout=30, **kwargs)
    _last_request_time = time.monotonic()
    response.raise_for_status()
    return response


@dataclass(frozen=True)
class FilingRef:
    ticker: str
    cik: str  # zero-padded to 10 digits
    form: str  # "10-K" or "10-Q"
    accession_number: str  # as returned by EDGAR, dashes included
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def accession_no_dashes(self) -> str:
        return self.accession_number.replace("-", "")

    @property
    def document_url(self) -> str:
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{self.accession_no_dashes}/{self.primary_document}"
        )

    @property
    def local_filename(self) -> str:
        safe_period = self.report_date.replace("-", "")
        return f"{self.form.replace('-', '')}_{safe_period}.htm"


def load_ticker_to_cik(tickers_json_path: Path, tickers: list[str]) -> dict[str, str]:
    """Resolve tickers to zero-padded 10-digit CIKs using SEC's own registry file."""
    data = json.loads(tickers_json_path.read_text())
    by_ticker = {v["ticker"]: v for v in data.values()}
    result = {}
    for ticker in tickers:
        entry = by_ticker.get(ticker)
        if entry is None:
            raise KeyError(f"Ticker {ticker!r} not found in SEC ticker registry")
        result[ticker] = str(entry["cik_str"]).zfill(10)
    return result


def get_submissions(cik: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return _throttled_get(url).json()


def get_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    return _throttled_get(url).json()


def select_filings(
    ticker: str,
    cik: str,
    submissions: dict,
    max_10k: int = 1,
    max_10q: int = 4,
) -> list[FilingRef]:
    """Pick the most recent N 10-Ks and 10-Qs from a company's filing history.

    EDGAR's `submissions` payload keeps parallel arrays (not a list of
    records), most-recent-first, under `filings.recent`.
    """
    recent = submissions["filings"]["recent"]
    n = len(recent["form"])
    candidates: list[FilingRef] = []
    for i in range(n):
        form = recent["form"][i]
        if form not in ("10-K", "10-Q"):
            continue
        candidates.append(
            FilingRef(
                ticker=ticker,
                cik=cik,
                form=form,
                accession_number=recent["accessionNumber"][i],
                filing_date=recent["filingDate"][i],
                report_date=recent["reportDate"][i],
                primary_document=recent["primaryDocument"][i],
            )
        )

    selected: list[FilingRef] = []
    selected += [f for f in candidates if f.form == "10-K"][:max_10k]
    selected += [f for f in candidates if f.form == "10-Q"][:max_10q]
    return selected


def download_filing(filing: FilingRef, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filing.local_filename
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return dest_path
    response = _throttled_get(filing.document_url)
    dest_path.write_bytes(response.content)
    return dest_path
