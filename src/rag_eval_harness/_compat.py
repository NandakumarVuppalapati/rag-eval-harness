"""Compatibility shims for third-party packaging issues.

ragas 0.3.x-0.4.x unconditionally import
``langchain_community.chat_models.vertexai.ChatVertexAI`` at module load
time (see ``ragas/llms/base.py``), even though Vertex AI support is
optional and unrelated to this project. Recent versions of
``langchain-community`` (>=0.3) removed that deprecated submodule
entirely, so importing ``ragas`` raises ``ModuleNotFoundError`` before
any of our code even runs.

The two conventional fixes -- installing ``langchain-google-vertexai``
(pulls in the full Google Cloud SDK, ~150MB of unrelated dependencies)
or pinning ``langchain-community`` to a pre-0.3 release (which conflicts
with the ``langchain`` 1.x / ``langchain-core`` versions everything else
in this project depends on) -- are both worse than just satisfying the
import ourselves. Neither this project nor ragas's own default code path
actually touches Vertex AI, so a minimal stub module is sufficient and
keeps the dependency tree honest.

See docs/adr/0001-ragas-vertexai-import-shim.md for the full writeup.
"""

from __future__ import annotations

import sys
import types


def patch_ragas_vertexai_import() -> None:
    """Insert a stub for the missing langchain_community.chat_models.vertexai module.

    Must run before the first `import ragas` anywhere in the process.
    Idempotent -- safe to call more than once.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: D401 - intentionally minimal stub
        """Placeholder. Real Vertex AI support is not used by this project."""

        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError(
                "ChatVertexAI is a compatibility stub inserted by "
                "rag_eval_harness._compat and is not a real implementation. "
                "This project does not use Google Vertex AI."
            )

    # types.ModuleType has no static attributes by design -- this is a
    # deliberate runtime monkeypatch (that's the whole point of this
    # module), which no type checker can verify statically.
    stub.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[module_name] = stub
