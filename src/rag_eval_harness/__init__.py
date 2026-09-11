"""RAG Evaluation & Observability Harness."""

from rag_eval_harness._compat import patch_ragas_vertexai_import

patch_ragas_vertexai_import()

__version__ = "0.1.0"
