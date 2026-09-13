# Test fixtures

These are trimmed excerpts of a real filing, not synthetic data, to keep the
ingestion tests honest about what the parser actually has to handle.

## `aapl_10q_excerpt.htm`

A ~4KB slice of Apple's real Q1 FY2026 10-Q (`data/corpus/AAPL/10Q_20251227.htm`,
gitignored since the full corpus is downloaded at run time, not checked in).
It keeps the file's real preamble and XML namespaces, its real
`<ix:header><ix:hidden>...</ix:hidden></ix:header>` block (actual XBRL facts:
`AmendmentFlag`, `DocumentFiscalYearFocus`, etc., byte-for-byte from the
filing, just with the tail of that block cut off and the tags closed
manually so it stays parseable), and one real visible `<ix:nonNumeric>`
paragraph (the filing's actual "Basis of Presentation and Preparation" note).
Nothing in it was hand-written except the closing tags needed to make the
truncation valid HTML/XML.

## `aapl_10q_real_narrative.txt`

The real, already-cleaned text `extract_clean_text()` produces from a larger
(~40KB) slice of the same filing -- three real footnotes (Basis of
Presentation, Note 2 Revenue, Note 3 Earnings Per Share), including real
dollar figures and share counts. Used to test `chunk_filing_text()` against
text with the length and structure (long run-on paragraphs, no `\n\n`
breaks, since `get_text()` already collapsed those) it actually receives
in production, rather than lorem-ipsum.
