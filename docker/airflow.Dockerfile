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
#
# --constraint matters as much as --chown did: our pyproject.toml pins
# sqlalchemy>=2.0 with no upper bound, so a bare `pip install` happily
# resolves the newest 2.x release. Airflow 2.10.4's own ORM models
# (airflow/models/taskinstance.py) use legacy, pre-"Mapped[]" type
# annotations that a too-new SQLAlchemy's stricter declarative mapping
# rejects outright -- the webserver and scheduler both crashed at import
# time with "MappedAnnotationError: Type annotation for
# TaskInstance.dag_model can't be correctly interpreted", found by
# actually running the stack, not just building the image (the build
# succeeds either way; only Airflow's own runtime import breaks). Airflow
# publishes a constraints file per version+Python combination precisely so
# that installing extra packages into its image doesn't upgrade a
# dependency Airflow itself is pinned against; pointing pip at it keeps
# sqlalchemy (and anything else Airflow already ships) at the version
# 2.10.4 was actually tested with, while still installing this project's
# other dependencies normally.
RUN pip install --no-cache-dir "/opt/rag-eval-harness" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.4/constraints-3.10.txt"
