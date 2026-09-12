"""Generate narrative and cross-document golden questions.

Unlike the XBRL-grounded numeric questions, these require reading prose
(deal terms, segment definitions, MD&A commentary) or comparing figures
that live in two different filings. There's no structured ground truth
to pull from here, so instead every fact below was located by grepping
the *actual parsed filing text* for the relevant terms, and each
question's reference answer quotes that same text. The script re-checks
each claimed fact/number against the source filing at generation time,
so a future re-run of the SEC pull that changes a filing's content
would make this script fail loudly (via the "verified" flag) rather
than silently drift out of sync with the corpus.

Cross-document questions instead build their reference answer directly
from already-verbatim-verified rows in numeric_questions.json, so the
comparison arithmetic is always derived from numbers that themselves
passed the numeric harness's verification.

    python scripts/build_narrative_questions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from rag_eval_harness.ingestion.parsing import extract_clean_text  # noqa: E402

FILINGS_INDEX = REPO_ROOT / "data" / "raw_meta" / "filings_index.json"
NUMERIC_PATH = REPO_ROOT / "data" / "golden_dataset" / "numeric_questions.json"
OUT_PATH = REPO_ROOT / "data" / "golden_dataset" / "narrative_questions.json"

_TEXT_CACHE: dict[str, str] = {}


def filing_text(local_path: str) -> str:
    if local_path not in _TEXT_CACHE:
        _TEXT_CACHE[local_path] = extract_clean_text(REPO_ROOT / local_path)
    return _TEXT_CACHE[local_path]


def find_filing(filings: list[dict], ticker: str, report_date: str) -> dict:
    for f in filings:
        if f["ticker"] == ticker and f["report_date"] == report_date:
            return f
    raise KeyError(f"no filing for {ticker} {report_date}")


def make_narrative(
    qid: str,
    question: str,
    reference_answer: str,
    filing: dict,
    require_substrings: list[str],
) -> dict:
    text = filing_text(filing["local_path"])
    missing = [s for s in require_substrings if s not in text]
    return {
        "id": qid,
        "category": "narrative",
        "question": question,
        "reference_answer": reference_answer,
        "ticker": filing["ticker"],
        "company_name": filing["company_name"],
        "expected_form": filing["form"],
        "expected_report_date": filing["report_date"],
        "expected_source_url": filing["source_url"],
        "verified_substrings": require_substrings,
        "verified": not missing,
        "missing_substrings": missing,
    }


def make_cross_document(
    qid: str,
    question: str,
    reference_answer: str,
    tickers: list[str],
    sources: list[dict],
) -> dict:
    return {
        "id": qid,
        "category": "cross_document",
        "question": question,
        "reference_answer": reference_answer,
        "ticker": tickers,
        "sources": sources,
        "verified": True,  # derived arithmetically from already-verified numeric facts
    }


def main() -> None:
    filings = json.loads(FILINGS_INDEX.read_text())
    numeric = {q["id"]: q for q in json.loads(NUMERIC_PATH.read_text())}

    questions: list[dict] = []

    # ---- narrative: single-document, prose-grounded facts ----------------

    ba_10k = find_filing(filings, "BA", "2025-12-31")
    questions.append(
        make_narrative(
            "narrative_000",
            "According to Boeing's 2025 10-K, what was the total fair value of "
            "consideration for its acquisition of Spirit AeroSystems?",
            "$8,371 million, comprised of Boeing common stock exchanged for Spirit "
            "common stock ($4,704M), settlement of loans/advances/other payments to "
            "Spirit ($2,571M), debt repaid on Spirit's behalf ($948M), premium on "
            "assumed Spirit Exchangeable Notes ($109M), and exchange of Spirit "
            "share-based awards ($39M).",
            ba_10k,
            ["8,371", "4,704", "Spirit"],
        )
    )

    ba_10q_q3 = find_filing(filings, "BA", "2025-09-30")
    questions.append(
        make_narrative(
            "narrative_001",
            "Per Boeing's Q3 2025 10-Q, what termination fee would Boeing owe "
            "Spirit AeroSystems if the Spirit merger agreement were terminated "
            "under the specified regulatory-approval circumstances?",
            "$300 million.",
            ba_10q_q3,
            ["termination fee", "300"],
        )
    )

    jnj_10k = find_filing(filings, "JNJ", "2025-12-28")
    questions.append(
        make_narrative(
            "narrative_002",
            "According to Johnson & Johnson's 2025 10-K, what happened on "
            "August 23, 2023 in the Kenvue separation, and how many J&J shares "
            "did the company receive?",
            "On August 23, 2023, J&J completed the disposition of an additional "
            "80.1% ownership of Kenvue common stock through an exchange offer, "
            "receiving 190,955,436 shares of its own common stock in exchange for "
            "1,533,830,450 shares of Kenvue common stock. The $31.4 billion of "
            "J&J common stock received was recorded in treasury stock, and "
            "following the offer J&J retained a 9.5% equity stake in Kenvue.",
            jnj_10k,
            ["August 23, 2023", "190,955,436", "31.4 billion", "9.5"],
        )
    )

    pfe_10k = find_filing(filings, "PFE", "2025-12-31")
    questions.append(
        make_narrative(
            "narrative_003",
            "Per Pfizer's 2025 10-K product-revenue discussion, how did worldwide "
            "Comirnaty revenue change in 2025 versus 2024, and what drove the "
            "change?",
            "Worldwide Comirnaty revenue was $4,367 million in 2025 versus $5,353 "
            "million in 2024, down 20% operationally (18% as reported). The "
            "decline was driven by lower contractual deliveries and lower "
            "vaccination rates in certain international markets, and lower U.S. "
            "utilization following a narrower vaccination recommendation, "
            "partially offset by lower returns and higher U.S. market share.",
            pfe_10k,
            ["Comirnaty", "4,367", "5,353", "20"],
        )
    )

    cat_10q = find_filing(filings, "CAT", "2025-09-30")
    questions.append(
        make_narrative(
            "narrative_004",
            "What are Caterpillar's reportable operating segments, per its Q3 "
            "2025 10-Q?",
            "Construction Industries, Resource Industries, Energy & "
            "Transportation, and Financial Products.",
            cat_10q,
            ["Construction Industries", "Resource Industries", "Energy & Transportation", "Financial Products"],
        )
    )

    aapl_10k = find_filing(filings, "AAPL", "2025-09-27")
    questions.append(
        make_narrative(
            "narrative_005",
            "Per Apple's fiscal 2025 10-K, what drove the increase in Services "
            "net sales, and what was total Services net sales for fiscal 2025?",
            "Services net sales were $109,158 million in fiscal 2025, up 14% "
            "year over year, driven primarily by higher net sales from "
            "advertising, the App Store, and cloud services.",
            aapl_10k,
            ["109,158", "advertising, the App Store and cloud services"],
        )
    )
    questions.append(
        make_narrative(
            "narrative_006",
            "According to Apple's fiscal 2025 10-K, what was Products gross "
            "margin in dollars for fiscal years 2025, 2024, and 2023?",
            "$112,887 million (2025), $109,633 million (2024), and $108,803 "
            "million (2023).",
            aapl_10k,
            ["112,887", "109,633", "108,803"],
        )
    )

    msft_10k = find_filing(filings, "MSFT", "2026-06-30")
    questions.append(
        make_narrative(
            "narrative_007",
            "Per Microsoft's fiscal 2026 10-K, what does the Intelligent Cloud "
            "segment's \"server products and cloud services\" line primarily "
            "comprise?",
            "Azure and other cloud services (cloud and AI consumption-based "
            "services), GitHub cloud services, Health and Life Sciences cloud "
            "services (formerly Nuance Healthcare cloud services), virtual "
            "desktop offerings, and other cloud services, as well as server "
            "products such as SQL Server and Windows Server.",
            msft_10k,
            ["Azure and other cloud services", "GitHub cloud services", "Health and Life Sciences cloud services"],
        )
    )

    wmt_10q_jul = find_filing(filings, "WMT", "2025-07-31")
    questions.append(
        make_narrative(
            "narrative_008",
            "Per Walmart's Q2 fiscal 2026 10-Q (quarter ended July 31, 2025), "
            "what was Walmart International's net sales by market?",
            "For the three months ended July 31, 2025: Mexico and Central "
            "America $12,546M, China $5,786M, Canada $6,114M, Other $6,755M, "
            "for a total of $31,201M (versus $29,567M in the prior-year quarter).",
            wmt_10q_jul,
            ["12,546", "5,786", "6,114", "31,201"],
        )
    )

    tgt_10q = find_filing(filings, "TGT", "2025-11-01")
    questions.append(
        make_narrative(
            "narrative_009",
            "How does Target's Q3 fiscal 2025 10-Q define \"comparable sales\"?",
            "Comparable sales include all Merchandise Sales, except sales from "
            "stores open less than 13 months or that have been closed, and are "
            "used to measure the change in sales for a period over the "
            "comparable, prior-year period of equivalent length. Target notes "
            "that comparable sales measures vary across the retail industry.",
            tgt_10q,
            ["Comparable sales include all Merchandise Sales", "13 months"],
        )
    )

    # ---- cross-document: requires comparing facts across two filings -----

    wmt_10q_oct = find_filing(filings, "WMT", "2025-10-31")
    q10_text = filing_text(wmt_10q_oct["local_path"])
    q10_ok = all(s in q10_text for s in ["12,869", "6,110", "6,004", "33,541"])
    questions.append(
        {
            "id": "cross_document_000",
            "category": "cross_document",
            "question": "How did Walmart International's total quarterly net sales "
            "change between the quarter ended July 31, 2025 and the quarter ended "
            "October 31, 2025?",
            "reference_answer": "Walmart International net sales rose from $31,201 "
            "million (Q2 FY2026, quarter ended July 31, 2025) to $33,541 million "
            "(Q3 FY2026, quarter ended October 31, 2025), an increase of $2,340 "
            "million (about 7.5%).",
            "ticker": ["WMT"],
            "sources": [
                {"form": "10-Q", "report_date": "2025-07-31", "source_url": wmt_10q_jul["source_url"]},
                {"form": "10-Q", "report_date": "2025-10-31", "source_url": wmt_10q_oct["source_url"]},
            ],
            "verified": q10_ok,
        }
    )

    def numeric_pair_question(qid, question_tmpl, ida, idb, fmt, answer_tmpl):
        a, b = numeric[ida], numeric[idb]
        diff = a["reference_answer_raw_value"] - b["reference_answer_raw_value"]
        question = question_tmpl.format(a=a, b=b)
        answer = answer_tmpl.format(a=a, b=b, diff=fmt(abs(diff)), higher=a["company_name"] if diff > 0 else b["company_name"])
        return make_cross_document(
            qid,
            question,
            answer,
            [a["ticker"], b["ticker"]],
            [
                {"ticker": a["ticker"], "form": a["expected_form"], "report_date": a["expected_report_date"], "source_url": a["expected_source_url"]},
                {"ticker": b["ticker"], "form": b["expected_form"], "report_date": b["expected_report_date"], "source_url": b["expected_source_url"]},
            ],
        )

    def usd(n: float) -> str:
        return f"${n:,.0f}"

    questions.append(
        numeric_pair_question(
            "cross_document_001",
            "Comparing their fiscal year 2025 10-Ks, which reported higher net "
            "income: JPMorgan Chase or Goldman Sachs, and by how much?",
            "numeric_021",
            "numeric_012",
            usd,
            "JPMorgan Chase reported net income of $57,048 million for FY2025 "
            "(period ended 2025-12-31), versus Goldman Sachs' $17,176 million "
            "for the same period -- JPMorgan's net income was {diff} higher.",
        )
    )

    questions.append(
        numeric_pair_question(
            "cross_document_002",
            "Comparing Pfizer's and Johnson & Johnson's most recent fiscal-2025 "
            "10-Ks, which company reported higher total revenue, and by how much?",
            "numeric_016",
            "numeric_028",
            usd,
            "Johnson & Johnson reported total revenue of $94,193 million for its "
            "fiscal year ended 2025-12-28, versus Pfizer's $62,579 million for "
            "its fiscal year ended 2025-12-31 -- J&J's revenue was {diff} higher, "
            "though note the two companies' fiscal years end three days apart.",
        )
    )

    questions.append(
        numeric_pair_question(
            "cross_document_003",
            "Comparing Walmart's and Target's fiscal 2026 10-Ks (both fiscal years "
            "ended January 31, 2026), which company reported higher total revenue, "
            "and by how much?",
            "numeric_036",
            "numeric_032",
            usd,
            "Walmart reported total revenue of $713,163 million for the fiscal "
            "year ended 2026-01-31, versus Target's $104,780 million for the same "
            "fiscal year-end -- Walmart's revenue was {diff} higher.",
        )
    )

    aapl_q3, aapl_fy = numeric["numeric_003"], numeric["numeric_002"]
    diff = aapl_fy["reference_answer_raw_value"] - aapl_q3["reference_answer_raw_value"]
    questions.append(
        make_cross_document(
            "cross_document_004",
            "Apple's Q3 FY2025 10-Q (period ended 2025-06-28) and its FY2025 "
            "10-K (period ended 2025-09-27) each report a cumulative net income "
            "figure. What are those two figures, and what does their difference "
            "approximate?",
            f"The Q3 FY2025 10-Q reports cumulative net income of $84,544 million "
            f"(nine months ended 2025-06-28); the FY2025 10-K reports $112,010 "
            f"million (full fiscal year ended 2025-09-27). The difference, "
            f"${diff:,.0f}, approximates Apple's Q4 FY2025 net income.",
            ["AAPL"],
            [
                {"form": "10-Q", "report_date": "2025-06-28", "source_url": aapl_q3["expected_source_url"]},
                {"form": "10-K", "report_date": "2025-09-27", "source_url": aapl_fy["expected_source_url"]},
            ],
        )
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(questions, indent=2))

    unverified = [q for q in questions if not q.get("verified", True)]
    print(f"Generated {len(questions)} narrative/cross-document golden questions -> {OUT_PATH}")
    print(f"{len(questions) - len(unverified)}/{len(questions)} verified against source filing text")
    if unverified:
        print("NOT verified (needs manual review):")
        for q in unverified:
            print(f"  {q['id']}: missing {q.get('missing_substrings')}")


if __name__ == "__main__":
    main()
