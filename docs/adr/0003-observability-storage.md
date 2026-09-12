# ADR 0003: Backend-agnostic schema for observability storage

## Status

Accepted

## Context

The observability layer needs to persist every nightly evaluation run
(aggregate Ragas scores, per-question scores, refusal rate, cost) so the
regression-detection logic has a history to compare against, and so a
Grafana dashboard has something to query. Production runs this against
Postgres, brought up by `docker-compose.yml` alongside the API service,
Prometheus, and Grafana.

This project's development and CI environment does not have a Postgres
server available, and a portfolio project's tests should not depend on
Docker being installed and running just to verify that a SQL insert and a
`SELECT ... ORDER BY ... LIMIT` work.

## Decision

`src/rag_eval_harness/observability/storage.py` is written against
SQLAlchemy Core using only types with well-behaved SQLite equivalents
(`Integer`, `String`, `Float`, `Boolean`, `DateTime`, `Text` -- no
Postgres-specific `JSONB`, `ARRAY`, or `UUID` columns). `tests/test_storage.py`
exercises the exact same `load_run` / `get_recent_runs` / `get_latest_run`
functions against `sqlite:///:memory:`. Production points
`DATABASE_URL` at the Postgres instance from `docker-compose.yml` instead;
no code path differs between the two, only the connection string.

Per-question Ragas scores and the answer text are stored in a normalized
`eval_question_results` table (one row per question per run) rather than
inline JSON blobs, so a dashboard or a future notebook can slice by
category or ticker with plain SQL instead of unpacking JSON in every query.

## Consequences

- Regression-detection logic (`observability/regression.py`) and the
  storage layer are both fully unit-tested without any external service.
- The one path that genuinely cannot be exercised outside Docker --
  connecting to a *real* Postgres server, with its own auth and network
  behavior -- is not covered by these tests. That gap is acceptable here:
  the SQL this project issues (a handful of `INSERT`s and an `ORDER BY
  ... LIMIT`) has no Postgres-specific behavior to go wrong, and
  `docker-compose up` is the integration test for that connection.
- If a future migration wants Postgres-specific features (e.g. `JSONB`
  for flexible per-question metadata, or `ON CONFLICT` upserts), the
  SQLite-compatibility constraint would need to be revisited or the
  tests would need a real Postgres fixture (e.g. `testcontainers`).
