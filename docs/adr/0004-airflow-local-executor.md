# ADR 0004: LocalExecutor for Airflow, no Celery/Redis

## Status

Accepted

## Context

`dags/nightly_evaluation_dag.py` has one DAG with two independent branches
(one per embedding model), each two tasks, running once a day. Airflow's
default "full" reference deployment (the `docker-compose.yaml` in Airflow's
own quick-start docs) uses `CeleryExecutor`, which additionally requires a
message broker (Redis), one or more separate worker containers, and Flower
if you want visibility into the queue.

## Decision

`docker/docker-compose.yml` runs Airflow with `LocalExecutor` instead:
`airflow-scheduler` forks task processes directly on the same container
that already exists, against the same Postgres instance used for
`AIRFLOW__DATABASE__SQL_ALCHEMY_CONN`. No Redis, no Celery worker
containers, no Flower.

## Consequences

- Four fewer moving parts (Redis, one-or-more Celery workers, Flower, the
  Celery result backend config) to run, monitor, and keep patched, for a
  workload that is two branches a day, not a fleet of concurrent workers.
- `LocalExecutor` runs tasks on the scheduler's own machine, so it does not
  horizontally scale across multiple hosts. That is a real limitation --
  but this project's DAG will not outgrow a single scheduler container
  before the evaluation workload itself changes shape (e.g. sharding the
  golden dataset across many parallel workers), and that point is a
  natural time to revisit this decision, not before it.
- If a future DAG needs to run many more tasks concurrently than one
  scheduler process comfortably forks, or needs workers on separate
  machines, switching to `CeleryExecutor` (or `KubernetesExecutor`, which
  avoids the standing Celery/Redis footprint entirely by launching one pod
  per task) is a configuration change, not a rewrite of the DAG itself --
  Airflow's executor is an infrastructure choice orthogonal to DAG code.
