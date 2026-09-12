# RAG pipeline API -- the service the evaluation harness exercises.
#
# Build context is the repo root (see docker-compose.yml: `context: ..`),
# because the package needs both pyproject.toml and src/ to `pip install .`.
FROM python:3.11-slim

WORKDIR /app

# Only copy what the dependency install needs first, so code-only changes
# don't invalidate the pip layer on every rebuild.
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Everything else (scripts/, data/golden_dataset, etc.) -- the API itself
# only needs src/, but later stages of this project (loading eval results,
# reading the golden dataset for ad-hoc checks) benefit from having the
# rest of the repo available too.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "rag_eval_harness.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
