"""Tests for the ragas/langchain-community import compatibility shim."""

import sys

import pytest


def test_patch_makes_ragas_importable():
    """Importing rag_eval_harness (which runs the patch) must make ragas importable."""
    import rag_eval_harness  # noqa: F401 - import side effect is the point

    import ragas  # should not raise ModuleNotFoundError

    assert ragas.__version__


def test_patch_is_idempotent():
    from rag_eval_harness._compat import patch_ragas_vertexai_import

    patch_ragas_vertexai_import()
    patch_ragas_vertexai_import()  # calling twice must not raise or duplicate anything

    assert "langchain_community.chat_models.vertexai" in sys.modules


def test_stub_chat_vertexai_is_not_a_real_implementation():
    from rag_eval_harness._compat import patch_ragas_vertexai_import

    patch_ragas_vertexai_import()
    stub_module = sys.modules["langchain_community.chat_models.vertexai"]

    with pytest.raises(NotImplementedError):
        stub_module.ChatVertexAI()
