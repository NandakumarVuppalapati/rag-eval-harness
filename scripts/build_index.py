"""Chunk the pulled SEC corpus, embed it two ways, and upsert to Pinecone.

Resumable at the batch level: tracks which (filing, batch) pairs have
already been embedded+upserted in data/raw_meta/index_progress.json, so
re-running after an interruption -- including mid-way through a large
filing like a bank's 900+ chunk 10-K -- picks up exactly where it left
off instead of redoing already-completed batches. Upserts are also
idempotent by chunk_id regardless, so this is a performance optimization,
not a correctness requirement.

    python scripts/build_index.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from rag_eval_harness.ingestion.chunking import chunk_filing_text  # noqa: E402
from rag_eval_harness.ingestion.embed_and_index import (  # noqa: E402
    _BATCH_SIZE,
    ensure_indexes,
    index_chunk_batch,
    make_clients,
)
from rag_eval_harness.ingestion.parsing import extract_clean_text  # noqa: E402

FILINGS_INDEX = REPO_ROOT / "data" / "raw_meta" / "filings_index.json"
PROGRESS_FILE = REPO_ROOT / "data" / "raw_meta" / "index_progress.json"


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_filings": [], "filing_batches_done": {}, "total_chunks_indexed": 0}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def main() -> None:
    filings = json.loads(FILINGS_INDEX.read_text())
    progress = load_progress()
    progress.setdefault("filing_batches_done", {})
    completed = set(progress["completed_filings"])

    pc, voyage_client, openai_client = make_clients()
    ensure_indexes(pc)
    voyage_index = pc.Index("rag-eval-harness-voyage")
    openai_index = pc.Index("rag-eval-harness-openai")

    start_time = time.monotonic()
    for entry in filings:
        key = f"{entry['ticker']}_{entry['form']}_{entry['report_date']}"
        if key in completed:
            print(f"[skip] {key} already indexed")
            continue

        text = extract_clean_text(REPO_ROOT / entry["local_path"])
        chunks = chunk_filing_text(text, entry)
        batches = [chunks[i : i + _BATCH_SIZE] for i in range(0, len(chunks), _BATCH_SIZE)]
        done_batches = set(progress["filing_batches_done"].get(key, []))

        if not done_batches:
            print(f"[{key}] {len(chunks)} chunks in {len(batches)} batches -> embedding + upserting...")

        for batch_num, batch in enumerate(batches):
            if batch_num in done_batches:
                continue
            index_chunk_batch(
                batch, voyage_index, openai_index, voyage_client, openai_client
            )
            done_batches.add(batch_num)
            progress["filing_batches_done"][key] = sorted(done_batches)
            save_progress(progress)
            print(f"[{key}]   batch {batch_num}/{len(batches)-1} done "
                  f"({len(batch)} chunks: {batch[0].chunk_id} .. {batch[-1].chunk_id})")

        completed.add(key)
        progress["completed_filings"] = sorted(completed)
        progress["filing_batches_done"].pop(key, None)
        progress["total_chunks_indexed"] += len(chunks)
        save_progress(progress)
        elapsed = time.monotonic() - start_time
        print(f"[{key}] DONE. {len(completed)}/{len(filings)} filings complete. "
              f"elapsed={elapsed:.0f}s")

    print(f"\nAll done. {progress['total_chunks_indexed']} total chunks indexed "
          f"across {len(completed)} filings.")


if __name__ == "__main__":
    main()
