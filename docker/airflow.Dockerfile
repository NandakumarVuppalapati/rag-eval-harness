# Airflow image with this project's own package installed, so DAG tasks
# can `import rag_eval_harness` directly instead of manipulating sys.path
# inside the container.
#
# Build context is the repo root (see docker-compose.yml: `context: ..`).
FROM apache/airflow:2.10.4-python3.10

USER airflow

COPY pyproject.toml README.md /opt/rag-eval-harness/
COPY src /opt/rag-eval-harness/src

# Airflow pins a lot of its own transitive dependencies; installing the
# project's [orchestration] extra here (rather than in the base image)
# keeps the API image lean and free of the Airflow package entirely.
RUN pip install --no-cache-dir "/opt/rag-eval-harness"
