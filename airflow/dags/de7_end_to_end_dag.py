from datetime import datetime
from pathlib import Path
import os

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


BASE_DIR = Path(
    os.environ.get(
        "NETWORK_INTELLIGENCE_HOME",
        "/home/jaideep/network-intelligence"
    )
)

RAW_DIR = Path(
    os.environ.get(
        "DE3_RAW_DIR",
        str(BASE_DIR / "data" / "raw")
    )
)

PROCESSED_DIR = Path(
    os.environ.get(
        "DE3_PROCESSED_DIR",
        str(BASE_DIR / "data" / "processed")
    )
)

ANALYTICS_DIR = Path(
    os.environ.get(
        "DE3_ANALYTICS_DIR",
        str(BASE_DIR / "data" / "analytics")
    )
)

GEOJSON_PATH = Path(
    os.environ.get(
        "MILANO_GEOJSON_PATH",
        str(BASE_DIR / "data" / "reference" / "milano-grid.geojson")
    )
)

INGESTION_SCRIPT = BASE_DIR / "ingestion" / "ingestion.py"
SPARK_PIPELINE = BASE_DIR / "spark" / "telecom_pipeline.py"
WAREHOUSE_SCRIPT = BASE_DIR / "warehouse" / "de6_load.py"

LOG_FILE = BASE_DIR / "logs" / "ingestion_log.csv"
STATUS_DIR = BASE_DIR / "logs" / "pipeline_status"
STATUS_FILE = STATUS_DIR / "latest_pipeline_status.json"


with DAG(
    dag_id="de7_end_to_end_orchestration",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["DE7", "telecom", "end-to-end"],
) as dag:

    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"""
set -e

echo "========================================"
echo "DE7 - INGEST"
echo "========================================"

cd "{BASE_DIR}"

test -f "{INGESTION_SCRIPT}"

python "{INGESTION_SCRIPT}"

echo
echo "Landing files:"
find "{BASE_DIR}/data/landing" -maxdepth 1 \
    -type f -name 'sms-call-internet-mi-*.csv' \
    -printf '%f\\n' | sort

echo
echo "Raw files:"
find "{RAW_DIR}" -maxdepth 1 \
    -type f -name 'sms-call-internet-mi-*.csv' \
    -printf '%f\\n' | sort

echo
echo "INGEST PASSED"
""",
    )

    validate = BashOperator(
        task_id="validate",
        bash_command=f"""
set -e

echo "========================================"
echo "DE7 - VALIDATE"
echo "========================================"

cd "{BASE_DIR}"

test -s "{LOG_FILE}"

echo
echo "Latest ingestion audit entries:"
tail -n 10 "{LOG_FILE}"

echo
echo "VALIDATE PASSED"
""",
    )

    spark_process = BashOperator(
        task_id="spark_process",
        bash_command=f"""
set -e

echo "========================================"
echo "DE7 - SPARK PROCESS"
echo "========================================"

cd "{BASE_DIR}"

test -f "{SPARK_PIPELINE}"
test -d "{RAW_DIR}"
test -s "{GEOJSON_PATH}"

python "{SPARK_PIPELINE}" \
    --raw-dir "{RAW_DIR}" \
    --processed-dir "{PROCESSED_DIR}" \
    --analytics-dir "{ANALYTICS_DIR}" \
    --geojson "{GEOJSON_PATH}"

echo
echo "SPARK PROCESS PASSED"
""",
    )

    load_warehouse = BashOperator(
        task_id="load_warehouse",
        bash_command=f"""
set -e

echo "========================================"
echo "DE7 - LOAD WAREHOUSE"
echo "========================================"

cd "{BASE_DIR}"

test -f "{WAREHOUSE_SCRIPT}"

python "{WAREHOUSE_SCRIPT}"

test -s "{BASE_DIR}/warehouse/network_analytics.db"

echo
echo "WAREHOUSE LOAD PASSED"
""",
    )

    quality_check = BashOperator(
        task_id="quality_check",
        env={
            "DE7_BASE_DIR": str(BASE_DIR),
            "DE7_STATUS_DIR": str(STATUS_DIR),
            "DE7_STATUS_FILE": str(STATUS_FILE),
            "DE7_LOG_FILE": str(LOG_FILE),
        },
        bash_command="""
set -e

echo "========================================"
echo "DE7 - QUALITY CHECK"
echo "========================================"

cd "$DE7_BASE_DIR"

mkdir -p "$DE7_STATUS_DIR"

/home/jaideep/airflow-venv/bin/python - <<'PY2'
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

base = Path(__import__("os").environ["DE7_BASE_DIR"])
db_path = base / "warehouse" / "network_analytics.db"
status_file = Path(__import__("os").environ["DE7_STATUS_FILE"])
ingestion_log = Path(__import__("os").environ["DE7_LOG_FILE"])

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM dim_time")
dim_time_rows = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM dim_grid")
dim_grid_rows = cur.fetchone()[0]

cur.execute("SELECT COUNT(*) FROM fact_network_activity")
fact_rows = cur.fetchone()[0]

cur.execute(
    "SELECT COUNT(*) "
    "FROM fact_network_activity f "
    "LEFT JOIN dim_time t ON f.time_id = t.time_id "
    "WHERE t.time_id IS NULL"
)
orphan_time_keys = cur.fetchone()[0]

cur.execute(
    "SELECT COUNT(*) "
    "FROM fact_network_activity f "
    "LEFT JOIN dim_grid g ON f.grid_id = g.grid_id "
    "WHERE g.grid_id IS NULL"
)
orphan_grid_keys = cur.fetchone()[0]

cur.execute(
    "SELECT MAX(t.timestamp) "
    "FROM fact_network_activity f "
    "JOIN dim_time t ON f.time_id = t.time_id"
)
as_of = cur.fetchone()[0]

conn.close()

rows_rejected = 0

if ingestion_log.exists():
    with ingestion_log.open("r", encoding="utf-8") as fh:
        for line in fh:
            if ",REJECTED," in line or ",REJECTED_DUPLICATE," in line:
                rows_rejected += 1

status = {
    "run_id": "DE7_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "run_timestamp": datetime.now(timezone.utc).isoformat(),
    "status": "SUCCESS",
    "tasks": {
        "ingest": "SUCCESS",
        "validate": "SUCCESS",
        "spark_process": "SUCCESS",
        "load_warehouse": "SUCCESS",
        "quality_check": "SUCCESS",
        "notify": "PENDING"
    },
    "rows_in": fact_rows,
    "rows_rejected": rows_rejected,
    "nulls_handled": None,
    "rows_published": fact_rows,
    "as_of": as_of,
    "dim_time_rows": dim_time_rows,
    "dim_grid_rows": dim_grid_rows,
    "fact_rows": fact_rows,
    "orphan_time_keys": orphan_time_keys,
    "orphan_grid_keys": orphan_grid_keys
}

status_file.write_text(
    json.dumps(status, indent=2),
    encoding="utf-8"
)

print(status_file)
print(json.dumps(status, indent=2))
PY2

echo
echo "QUALITY CHECK PASSED"
""",
    )

    notify = BashOperator(
        task_id="notify",
        env={
            "DE7_STATUS_FILE": str(STATUS_FILE),
        },
        bash_command="""
set -e

echo "========================================"
echo "DE7 - NOTIFY"
echo "========================================"

test -s "$DE7_STATUS_FILE"

echo
echo "Final pipeline status:"
cat "$DE7_STATUS_FILE"

echo
echo "DE7 END-TO-END PIPELINE COMPLETED SUCCESSFULLY"
""",
    )

    ingest >> validate >> spark_process >> load_warehouse >> quality_check >> notify
