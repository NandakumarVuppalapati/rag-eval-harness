-- POSTGRES_DB (rag_eval_harness) is created automatically by the postgres
-- image on first boot. Airflow's own metadata lives in a separate
-- database on the same instance, created here, so `docker compose down -v`
-- and volume inspection cleanly separate "our app data" from "Airflow's
-- internal bookkeeping" -- and so a future switch to a managed Postgres
-- for one and not the other is a one-line connection-string change.
CREATE DATABASE airflow;
