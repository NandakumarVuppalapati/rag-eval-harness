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
# Our pyproject.toml originally pinned sqlalchemy>=2.0 with no upper
# bound, so a bare `pip install` happily resolved the newest 2.x release.
# Airflow 2.10.4's own ORM models (airflow/models/taskinstance.py) use
# legacy, pre-"Mapped[]" type annotations that a too-new SQLAlchemy's
# stricter declarative mapping rejects outright -- the webserver and
# scheduler both crashed at import time with "MappedAnnotationError: Type
# annotation for TaskInstance.dag_model can't be correctly interpreted",
# found by actually running the stack, not just building the image (the
# build succeeds either way; only Airflow's own runtime import breaks).
#
# Airflow publishes a constraints file per version+Python combination for
# exactly this reason, but pointing pip at the *full* file backfired: it
# pins hundreds of packages for Airflow's optional extras we don't use at
# all (this project runs LocalExecutor with no provider packages -- see
# docker-compose.yml -- so e.g. its Celery/Redis-provider pins are pure
# dead weight here), and several of those collided with our own, much
# newer langchain/openai/anthropic stack for no protective benefit (e.g.
# the full file pins async-timeout==5.0.1, but langchain-classic needs
# async-timeout<5.0.0 on Python <3.11 -- a real conflict over a package
# this project never imports). What actually has to match what Airflow
# 2.10.4 was tested with is the SQLAlchemy family its webserver/scheduler
# import directly in-process, so we download the official file and keep
# only those lines, letting everything else resolve normally.
# python3 (not curl) fetches the file -- apache/airflow images obviously
# ship Python, so this doesn't add a dependency on curl being present.
RUN python3 -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/apache/airflow/constraints-2.10.4/constraints-3.10.txt', '/tmp/airflow-constraints-full.txt')" \
    && grep -iE '^(SQLAlchemy|Flask-SQLAlchemy|SQLAlchemy-JSONField|SQLAlchemy-Utils|alembic|marshmallow|marshmallow-sqlalchemy|marshmallow-oneofschema)==' \
        /tmp/airflow-constraints-full.txt > /tmp/airflow-constraints-sqlalchemy.txt \
    && cat /tmp/airflow-constraints-sqlalchemy.txt \
    && pip install --no-cache-dir "/opt/rag-eval-harness" \
        --constraint /tmp/airflow-constraints-sqlalchemy.txt
