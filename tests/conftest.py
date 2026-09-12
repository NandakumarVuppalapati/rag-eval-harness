"""Shared test fixtures.

The API's lifespan (src/rag_eval_harness/api/main.py) constructs real
Pinecone/Voyage/OpenAI/Anthropic clients on startup. Recent pinecone-client
versions authenticate eagerly at construction time -- so even a harmless
placeholder API key makes a real network call and fails (in CI, offline, or
anywhere without a working key), independent of whether the test itself
ever needed real retrieval or generation.

This fixture replaces all four client constructors with mocks so the API's
lifespan never touches the network, regardless of what credentials (real,
fake, or absent) happen to be set. It is deliberately *not* autouse:
tests/test_compat.py's own test depends on `rag_eval_harness.api.main` (and
therefore `rag_eval_harness` itself) not being imported as a side effect of
unrelated fixture setup, since that import is the exact thing under test.
tests/test_api.py's `client` fixture requests this fixture by name instead.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# The lifespan reads these via os.environ[...] (not .get()), so they must
# exist even though the client construction itself is mocked below.
os.environ.setdefault("OPENAI_API_KEY", "test-not-a-real-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-not-a-real-key")
os.environ.setdefault("VOYAGE_API_KEY", "test-not-a-real-key")
os.environ.setdefault("PINECONE_API_KEY", "test-not-a-real-key")
os.environ.setdefault("PINECONE_ENVIRONMENT", "us-east-1")


@pytest.fixture
def mock_external_api_clients():
    with (
        patch("rag_eval_harness.api.main.Pinecone", return_value=MagicMock()),
        patch("rag_eval_harness.api.main.voyageai.Client", return_value=MagicMock()),
        patch("rag_eval_harness.api.main.openai.OpenAI", return_value=MagicMock()),
        patch("rag_eval_harness.api.main.anthropic.Anthropic", return_value=MagicMock()),
    ):
        yield
