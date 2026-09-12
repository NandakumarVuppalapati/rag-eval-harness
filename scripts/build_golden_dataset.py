"""Generate XBRL-grounded numeric golden questions.

Ground truth for these questions comes directly from SEC's structured
XBRL company-facts data, filtered to only facts whose accession number
matches a filing actually in this project's corpus -- so every answer
is both provably correct (it's SEC's own reported figure) and
retrievable (the source filing is in our Pinecone indexes). Each
generated question is also spot-checked against the filing's own
chunked text to confirm the number actually appears verbatim.

    python scripts/build_golden_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_eval_harness.ingestion.parsing import extract_clean_text  # noqa: E402

FILINGS_INDEX = REPO_ROOT / "data" / "raw_meta" / "filings_index.json"
XBRL_DIR = REPO_ROOT / "data" / "xbrl"
OUT_PATH = REPO_ROOT / "data" / "golden_dataset" / "numeric_questions.json"

# concept -> (human label, question template)
CONCEPTS = {
    "Revenues": ("total revenue", "What was {company}'s total revenue for the period ended {end}, as reported in its {form} filed {filed}?"),
    "RevenueFromContractWithCustomerExcludingAssessedTax": ("total revenue", "What was {company}'s total revenue for the period ended {end}, as reported in its {form} filed {filed}?"),
    "NetIncomeLoss": ("net income", "What was {company}'s net income for the period ended {end}, as reported in its {form} filed {filed}?"),
    "OperatingIncomeLoss": ("operating income", "What was {company}'s operating income for the period ended {end}, as reported in its {form} filed {filed}?"),
    "EarningsPerShareDiluted": ("diluted earnings per share", "What was {company}'s diluted earnings per share for the period ended {end}, as reported in its {form} filed {filed}?"),
    "Assets": ("total assets", "What were {company}'s total assets as of {end}, as reported in its {form} filed {filed}?"),
}

MAX_PER_COMPANY = 4


def format_value(concept: str, val: float) -> str:
    if concept == "EarningsPerShareDiluted":
        return f"${val:.2f}"
    return f"${val:,.0f}"


def main() -> None:
    filings = json.loads(FILINGS_INDEX.read_text())
    accn_to_filing = {f["accession_number"]: f for f in filings}
    tickers = sorted({f["ticker"] for f in filings})

    questions = []
    qid = 0

    for ticker in tickers:
        xbrl_path = XBRL_DIR / f"{ticker}.json"
        if not xbrl_path.exists():
            continue
        facts = json.loads(xbrl_path.read_text())["facts"].get("us-gaap", {})
        company_filings = [f for f in filings if f["ticker"] == ticker]
        company_name = company_filings[0]["company_name"]

        picked_for_company = 0
        used_concepts_this_company = set()

        for concept, (label, template) in CONCEPTS.items():
            if picked_for_company >= MAX_PER_COMPANY:
                break
            if label in used_concepts_this_company:
                continue  # revenue has two possible tag names; don't double up
            if concept not in facts:
                continue
            datapoints = facts[concept].get("units", {}).get("USD", []) or facts[concept].get(
                "units", {}
            ).get("USD/shares", [])
            # Prefer one 10-K (annual) fact and one 10-Q (quarterly) fact per concept
            # A single filing's XBRL facts include prior-year comparative
            # periods alongside the current one (10-Ks typically show 2-3
            # years of income-statement history). Only keep the datapoint
            # whose period actually matches the filing's own primary report
            # date -- otherwise we'd generate a question that names the
            # filing but quotes a different, older period's figure.
            by_form = {"10-K": None, "10-Q": None}
            for dp in datapoints:
                filing = accn_to_filing.get(dp["accn"])
                if filing is None:
                    continue
                if dp["end"] != filing["report_date"]:
                    continue
                if filing["form"] in by_form and by_form[filing["form"]] is None:
                    by_form[filing["form"]] = (dp, filing)

            for form_key in ("10-K", "10-Q"):
                if picked_for_company >= MAX_PER_COMPANY:
                    break
                entry = by_form[form_key]
                if entry is None:
                    continue
                dp, filing = entry

                question_text = template.format(
                    company=company_name, end=dp["end"], form=filing["form"], filed=dp["filed"]
                )
                answer_text = format_value(concept, dp["val"])

                # Spot-check: does this number actually appear in the filing's own text?
                filing_text = extract_clean_text(REPO_ROOT / filing["local_path"])
                # Filings print negatives in accounting-parentheses style, e.g.
                # "($2,222)" rather than "-2,222", so verification checks the
                # magnitude, matching either a plain or millions-scaled form.
                magnitude = abs(dp["val"])
                numeric_str = f"{magnitude:,.0f}" if concept != "EarningsPerShareDiluted" else f"{magnitude:.2f}"
                millions_str = f"{magnitude / 1_000_000:,.0f}" if magnitude >= 1_000_000 else None
                found = (numeric_str in filing_text) or (millions_str and millions_str in filing_text)

                questions.append(
                    {
                        "id": f"numeric_{qid:03d}",
                        "category": "numeric_xbrl",
                        "question": question_text,
                        "reference_answer": answer_text,
                        "reference_answer_raw_value": dp["val"],
                        "ticker": ticker,
                        "company_name": company_name,
                        "concept": concept,
                        "period_end": dp["end"],
                        "expected_form": filing["form"],
                        "expected_report_date": filing["report_date"],
                        "expected_source_url": filing["source_url"],
                        "verbatim_in_filing_text": found,
                    }
                )
                qid += 1
                picked_for_company += 1
                used_concepts_this_company.add(label)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(questions, indent=2))

    unverified = [q for q in questions if not q["verbatim_in_filing_text"]]
    print(f"Generated {len(questions)} numeric golden questions -> {OUT_PATH}")
    print(f"{len(questions) - len(unverified)}/{len(questions)} verified verbatim in their source filing text")
    if unverified:
        print("NOT verbatim-verified (needs manual review):")
        for q in unverified:
            print(f"  {q['id']}: {q['ticker']} {q['concept']} {q['period_end']} = {q['reference_answer']}")


if __name__ == "__main__":
    main()
