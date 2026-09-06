from datetime import datetime
from pathlib import Path
import os

from airflow import DAG
from airflow.operators.bash import BashOperator


BASE_DIR = Path(os.environ.get(
    "NETWORK_INTELLIGENCE_HOME",
    "/home/jaideep/network-intelligence"
))

RAW_DIR = Path(os.environ.get(
    "DE3_RAW_DIR",
    BASE_DIR / "data" / "raw"
))

PROCESSED_DIR = Path(os.environ.get(
    "DE3_PROCESSED_DIR",
    BASE_DIR / "data" / "processed"
))

ANALYTICS_DIR = Path(os.environ.get(
    "DE3_ANALYTICS_DIR",
    BASE_DIR / "data" / "analytics"
))

GEOJSON_PATH = Path(os.environ.get(
    "MILANO_GEOJSON_PATH",
    BASE_DIR / "data" / "reference" / "milano-grid.geojson"
))

SPARK_PIPELINE = BASE_DIR / "spark" / "telecom_pipeline.py"


with DAG(
    dag_id="de3_spark_orchestration",
    description="DE3 - Orchestrate the existing telecom Spark pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["DE3", "spark", "telecom"],
) as dag:

    check_ingestion = BashOperator(
        task_id="check_ingestion_success",
        bash_command=(
            f'echo "Checking ingestion output in {RAW_DIR}" && '
            f'count=$(find "{RAW_DIR}" -maxdepth 1 '
            f'-type f -name "sms-call-internet-mi-*.csv" | wc -l) && '
            'if [ "$count" -eq 0 ]; then '
            f'echo "ERROR: No ingested raw CSV files found in {RAW_DIR}"; '
            'exit 1; '
            'fi; '
            'echo "Ingestion check passed."; '
            'echo "Raw files available: $count"'
        ),
    )

    run_spark = BashOperator(
        task_id="run_de3_spark_pipeline",
        bash_command=(
            f'cd "{BASE_DIR}" && '
            f'python "{SPARK_PIPELINE}" '
            f'--raw-dir "{RAW_DIR}" '
            f'--processed-dir "{PROCESSED_DIR}" '
            f'--analytics-dir "{ANALYTICS_DIR}" '
            f'--geojson "{GEOJSON_PATH}"'
        ),
    )

    verify = BashOperator(
        task_id="verify_spark_outputs",
        bash_command=(
            f'echo "Verifying Spark outputs..." && '
            f'test -e "{PROCESSED_DIR}/activity" || '
            f'(echo "ERROR: Processed activity output missing"; exit 1) && '
            f'test -e "{ANALYTICS_DIR}/hourly_grid_summary" || '
            f'(echo "ERROR: hourly_grid_summary missing"; exit 1) && '
            f'test -e "{ANALYTICS_DIR}/grid_performance" || '
            f'(echo "ERROR: grid_performance missing"; exit 1) && '
            f'test -e "{ANALYTICS_DIR}/temporal_performance" || '
            f'(echo "ERROR: temporal_performance missing"; exit 1) && '
            'echo "DE3 output verification passed."'
        ),
    )

    check_ingestion >> run_spark >> verify
