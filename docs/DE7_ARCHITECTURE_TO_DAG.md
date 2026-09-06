# DE7 — End-to-End Airflow Orchestration

## Objective

Automate the complete telecom network-intelligence batch pipeline using one Airflow trigger.

## Architecture

Daily Telecom CSV
        |
        v
data/landing/
        |
        v
DE7: ingest
        |
        v
DE7: validate
        |
        v
data/raw/
        |
        v
DE7: spark_process
        |
        +--> data/processed/
        |
        +--> data/analytics/
        |
        v
DE7: load_warehouse
        |
        v
warehouse/network_analytics.db
        |
        v
DE7: quality_check
        |
        +--> logs/pipeline_status/latest_pipeline_status.json
        |
        v
DE7: notify

## Task Mapping

| Airflow Task | Existing Component | Responsibility |
|---|---|---|
| ingest | ingestion/ingestion.py | Detect, validate and route incoming files |
| validate | ingestion audit output | Confirm ingestion result |
| spark_process | spark/telecom_pipeline.py | Execute existing Spark processing |
| load_warehouse | warehouse/de6_load.py | Populate DE6 warehouse |
| quality_check | SQLite + status writer | Validate warehouse and publish machine-readable status |
| notify | Airflow task | Report successful completion |

## Reusability

DE7 does not reimplement Spark cleaning, aggregation or business logic.

It invokes the reusable components created in earlier phases.

## Quality Status

The quality_check task writes:

logs/pipeline_status/latest_pipeline_status.json

The record contains:

- run_id
- run_timestamp
- per-task status
- rows_in
- rows_rejected
- nulls_handled
- rows_published
- AS_OF
- dimension counts
- fact count
- orphan-key validation

## Failure Behaviour

Airflow task dependencies use the default failure propagation behaviour.

If a task fails:

- downstream tasks do not execute
- the DAG run becomes failed
- the failure is visible in Airflow
- troubleshooting begins in the failed component

## Troubleshooting Map

| Failure | Investigate |
|---|---|
| File not ingested | ingestion/ingestion.py |
| Validation failure | ingestion audit log |
| Spark failure | spark/telecom_pipeline.py |
| Warehouse failure | warehouse/de6_load.py |
| Quality failure | quality_check task / warehouse |
| Notification failure | DE7 DAG notify task |

## Safe Rerun

Existing ingestion logic handles duplicate files explicitly.

The warehouse loader rebuilds the DE6 SQLite warehouse from the analytics source, preventing duplicate accumulation during a complete rebuild.

## DE7 Status

IMPLEMENTED — pending Airflow execution validation.
