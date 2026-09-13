# Airflow image with this project's own package installed, so DAG tasks
# can `import rag_eval_harness` directly instead of manipulating sys.path
# inside the container.
#
# Build context is the repo root (see docker-compose.yml: `context: ..`).
FROM apache/airflow:2.10.4-python3.10

USER airflow

# --chown matters: USER airflow above only sets the user RUN executes as --
# COPY still defaults to root:root ownership regardless of it. Without
# --chown, pip's build step (it needs to write *.egg-info into the source
# tree) hits Permission denied as the non-root airflow user. apache/airflow
# images use UID 50000 with GID 0 (root group) by convention, hence
# airflow:0 rather than airflow:airflow (which doesn't exist as a group).
COPY --chown=airflow:0 pyproject.toml README.md /opt/rag-eval-harness/
COPY --chown=airflow:0 src /opt/rag-eval-harness/src

# Airflow pins a lot of its own transitive dependencies; installing the
# project's [orchestration] extra here (rather than in the base image)
# keeps the API image lean and free of the Airflow package entirely.
RUN pip install --no-cache-dir "/opt/rag-eval-harness"
