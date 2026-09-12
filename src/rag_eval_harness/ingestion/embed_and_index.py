"""Embed chunks with two independent embedding models and upsert to Pinecone.

Two full, separate indexes are built from the same chunk set:

- rag-eval-harness-voyage  (voyage-finance-2, 1024-dim)  -- the primary,
  domain-tuned embedding model this project is built around.
- rag-eval-harness-openai  (text-embedding-3-small, 1536-dim) -- a
  general-purpose baseline, so the retrieval-quality delta between
  domain-tuned and general-purpose embeddings can be measured directly
  instead of assumed (see docs/adr/0002-embedding-model-choice.md).

Pinecone indexes must have a fixed dimension, so two models with
different output dimensions inherently need two indexes -- this is not
an accident of the ablation design, it's a Pinecone constraint.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable

import openai
import voyageai
from pinecone import Pinecone, ServerlessSpec

from rag_eval_harness.ingestion.chunking import Chunk

VOYAGE_INDEX_NAME = "rag-eval-harness-voyage"
OPENAI_INDEX_NAME = "rag-eval-harness-openai"
VOYAGE_MODEL = "voyage-finance-2"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
VOYAGE_DIM = 1024
OPENAI_DIM = 1536

_BATCH_SIZE = 96
_UPSERT_BATCH_SIZE = 100


def _batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ensure_indexes(pc: Pinecone) -> None:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if VOYAGE_INDEX_NAME not in existing:
        pc.create_index(
            name=VOYAGE_INDEX_NAME,
            dimension=VOYAGE_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    if OPENAI_INDEX_NAME not in existing:
        pc.create_index(
            name=OPENAI_INDEX_NAME,
            dimension=OPENAI_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def embed_voyage(voyage_client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    result = voyage_client.embed(texts, model=VOYAGE_MODEL, input_type="document")
    # See the matching comment in retrieval/retriever.py._embed_query: voyageai's
    # stubs allow int8-quantized output (list[int]), which this project never
    # requests, so this normalizes to the float type the Pinecone index (and
    # this function's own signature) actually expects.
    return [[float(x) for x in embedding] for embedding in result.embeddings]


def embed_openai(openai_client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    result = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in result.data]


def _with_retry(fn, *args, progress_log=print, **kwargs):
    for attempt in range(4):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise
            progress_log(f"  retry {attempt+1} for {fn.__name__}: {exc}")
            time.sleep(2**attempt)


def index_chunk_batch(
    batch: list[Chunk],
    voyage_index,
    openai_index,
    voyage_client: voyageai.Client,
    openai_client: openai.OpenAI,
    progress_log=print,
) -> None:
    """Embed one batch of chunks with both models and upsert to both indexes."""
    texts = [c.text for c in batch]

    voyage_vecs = _with_retry(embed_voyage, voyage_client, texts, progress_log=progress_log)
    openai_vecs = _with_retry(embed_openai, openai_client, texts, progress_log=progress_log)

    voyage_upserts = [
        (c.chunk_id, vec, c.to_pinecone_metadata())
        for c, vec in zip(batch, voyage_vecs, strict=True)
    ]
    openai_upserts = [
        (c.chunk_id, vec, c.to_pinecone_metadata())
        for c, vec in zip(batch, openai_vecs, strict=True)
    ]

    for sub in _batched(voyage_upserts, _UPSERT_BATCH_SIZE):
        voyage_index.upsert(vectors=sub)
    for sub in _batched(openai_upserts, _UPSERT_BATCH_SIZE):
        openai_index.upsert(vectors=sub)


def index_chunks(
    chunks: list[Chunk],
    pc: Pinecone,
    voyage_client: voyageai.Client,
    openai_client: openai.OpenAI,
    progress_log=print,
) -> None:
    """Embed+upsert a full filing's chunks in one call (non-checkpointed)."""
    voyage_index = pc.Index(VOYAGE_INDEX_NAME)
    openai_index = pc.Index(OPENAI_INDEX_NAME)
    for batch_num, batch in enumerate(_batched(chunks, _BATCH_SIZE)):
        index_chunk_batch(batch, voyage_index, openai_index, voyage_client, openai_client, progress_log)
        progress_log(
            f"  batch {batch_num}: embedded+upserted {len(batch)} chunks "
            f"(chunk_ids {batch[0].chunk_id} .. {batch[-1].chunk_id})"
        )


def make_clients() -> tuple[Pinecone, voyageai.Client, openai.OpenAI]:
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return pc, voyage_client, openai_client
