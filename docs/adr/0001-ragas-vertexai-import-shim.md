# ADR 0001: Work around a ragas / langchain-community import incompatibility

## Status

Accepted

## Context

`ragas` (evaluated at 0.4.3, also reproduced on 0.3.9) imports
`langchain_community.chat_models.vertexai.ChatVertexAI` unconditionally
at module load time, in `ragas/llms/base.py`, purely to support Google
Vertex AI as one of several optional LLM backends. This project does
not use Vertex AI anywhere.

Current `langchain-community` releases (0.3.x and 0.4.x) removed that
deprecated submodule from the package entirely. The result: `import
ragas` raises `ModuleNotFoundError` out of the box, with no code of
ours involved, on a perfectly normal `pip install`.

## Options considered

1. **Install `langchain-google-vertexai`.** Satisfies the import, but
   pulls in the full Google Cloud SDK (`google-cloud-aiplatform`,
   `google-cloud-bigquery`, `grpcio`, etc.) — over a dozen packages and
   tens of MB — for a feature this project never calls. Rejected: pure
   dependency bloat with a larger attack surface for no functional gain.
2. **Pin `langchain-community` to a pre-0.3 release that still has the
   submodule.** Rejected: conflicts with the `langchain` 1.x /
   `langchain-core` versions the rest of the stack (langchain-anthropic,
   langchain-voyageai, langchain_openai) depends on.
3. **Stub the missing module ourselves.** A ~30-line shim
   (`rag_eval_harness._compat.patch_ragas_vertexai_import`) inserts a
   placeholder module into `sys.modules` before ragas is imported. The
   stub's `ChatVertexAI` class raises `NotImplementedError` if anyone
   ever actually tries to instantiate it — which nothing in this
   project does. Accepted.

## Decision

Use the local shim (option 3), invoked automatically from
`rag_eval_harness/__init__.py` so it's transparent to every module in
this project — nobody importing `rag_eval_harness` needs to know this
happened.

## Consequences

- Dependency tree stays minimal and honest: no unused Google Cloud
  packages installed.
- If `ragas` fixes this upstream (making the Vertex AI import lazy, as
  it already is in `langchain_community.llms`), the shim becomes a
  no-op the moment `langchain_community.chat_models.vertexai` exists
  again — `patch_ragas_vertexai_import` checks `sys.modules` first and
  the real module would already satisfy any subsequent import, though
  removing the shim entirely at that point would be the cleaner move.
- Anyone reading this project's code and wondering why `_compat.py`
  exists has this document to explain it.
