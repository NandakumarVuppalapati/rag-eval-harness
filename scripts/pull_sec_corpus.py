"""Pull a real, small-but-diverse corpus of SEC filings + XBRL facts.

Ten companies across five sectors, most recent 10-K plus trailing four
10-Qs each. Run from the repo root:

    python scripts/pull_sec_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_eval_harness.ingestion.sec_edgar import (  # noqa: E402
    download_filing,
    get_company_facts,
    get_submissions,
    load_ticker_to_cik,
    select_filings,
)

COMPANIES = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "WMT": "Retail",
    "TGT": "Retail",
    "JPM": "Finance",
    "GS": "Finance",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "CAT": "Industrials",
    "BA": "Industrials",
}

CORPUS_DIR = REPO_ROOT / "data" / "corpus"
XBRL_DIR = REPO_ROOT / "data" / "xbrl"
META_DIR = REPO_ROOT / "data" / "raw_meta"


def main() -> None:
    tickers_json = META_DIR / "company_tickers.json"
    ticker_to_cik = load_ticker_to_cik(tickers_json, list(COMPANIES))

    filing_index = []

    for ticker, sector in COMPANIES.items():
        cik = ticker_to_cik[ticker]
        print(f"[{ticker}] cik={cik} sector={sector}")

        submissions = get_submissions(cik)
        company_name = submissions.get("name", ticker)
        filings = select_filings(ticker, cik, submissions, max_10k=1, max_10q=4)
        print(f"[{ticker}] selected {len(filings)} filings: "
              f"{[(f.form, f.report_date) for f in filings]}")

        dest_dir = CORPUS_DIR / ticker
        for filing in filings:
            path = download_filing(filing, dest_dir)
            size_kb = path.stat().st_size / 1024
            print(f"[{ticker}]   -> {path.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")
            filing_index.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "sector": sector,
                    "cik": cik,
                    "form": filing.form,
                    "filing_date": filing.filing_date,
                    "report_date": filing.report_date,
                    "accession_number": filing.accession_number,
                    "source_url": filing.document_url,
                    "local_path": str(path.relative_to(REPO_ROOT)),
                }
            )

        facts = get_company_facts(cik)
        XBRL_DIR.mkdir(parents=True, exist_ok=True)
        (XBRL_DIR / f"{ticker}.json").write_text(json.dumps(facts))
        print(f"[{ticker}]   -> data/xbrl/{ticker}.json "
              f"({len(json.dumps(facts)) / 1024:.0f} KB)")

    META_DIR.mkdir(parents=True, exist_ok=True)
    (META_DIR / "filings_index.json").write_text(json.dumps(filing_index, indent=2))
    print(f"\nDone. {len(filing_index)} filings indexed across {len(COMPANIES)} companies.")
    print(f"Index written to {META_DIR / 'filings_index.json'}")


if __name__ == "__main__":
    main()
