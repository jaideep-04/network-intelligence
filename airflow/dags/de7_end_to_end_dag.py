from datetime import datetime
from pathlib import Path
import os

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


# ============================================================
# BASE PATHS
# ============================================================

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


# ============================================================
# PIPELINE SCRIPTS
# ============================================================

INGESTION_SCRIPT = BASE_DIR / "ingestion" / "ingestion.py"
SPARK_PIPELINE = BASE_DIR / "spark" / "telecom_pipeline.py"
WAREHOUSE_SCRIPT = BASE_DIR / "warehouse" / "de6_load.py"

FEATURE_SCRIPT = BASE_DIR / "ml" / "features.py"
ML6_SCRIPT = BASE_DIR / "ml" / "ml6_batch_score.py"


# ============================================================
# LOG / STATUS PATHS
# ============================================================

LOG_FILE = BASE_DIR / "logs" / "ingestion_log.csv"
STATUS_DIR = BASE_DIR / "logs" / "pipeline_status"
STATUS_FILE = STATUS_DIR / "latest_pipeline_status.json"

DB_PATH = BASE_DIR / "warehouse" / "network_analytics.db"

# Use the same Python environment already used by the quality check.
PYTHON_BIN = os.environ.get(
    "DE7_PYTHON_BIN",
    "/home/jaideep/airflow-venv/bin/python"
)


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="de7_end_to_end_orchestration",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["DE7", "telecom", "end-to-end", "ML6"],
) as dag:

    # ========================================================
    # 1. INGEST
    # ========================================================

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


    # ========================================================
    # 2. VALIDATE
    # ========================================================

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


    # ========================================================
    # 3. SPARK PROCESS
    # ========================================================

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


    # ========================================================
    # 4. LOAD WAREHOUSE
    # ========================================================

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

test -s "{DB_PATH}"

echo
echo "WAREHOUSE LOAD PASSED"
""",
    )


    # ========================================================
    # 5. FEATURE GENERATION
    # ========================================================

    feature_generation = BashOperator(
        task_id="feature_generation",
        bash_command=f"""
set -e

echo "========================================"
echo "DE7 - FEATURE GENERATION"
echo "========================================"

cd "{BASE_DIR}"

test -f "{FEATURE_SCRIPT}"
test -s "{DB_PATH}"

"{PYTHON_BIN}" "{FEATURE_SCRIPT}"

echo
echo "Checking network_features table..."

"{PYTHON_BIN}" - <<'PY'
import sqlite3
from pathlib import Path

db_path = Path(r"{DB_PATH}")

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()

cur.execute(
    "SELECT name FROM sqlite_master "
    "WHERE type='table' AND name='network_features'"
)

if cur.fetchone() is None:
    conn.close()
    raise SystemExit("network_features table was not created")

cur.execute("SELECT COUNT(*) FROM network_features")
rows = cur.fetchone()[0]

conn.close()

if rows <= 0:
    raise SystemExit("network_features table is empty")

print(f"network_features rows: {{rows}}")
print("FEATURE GENERATION VALIDATION PASSED")
PY

echo
echo "FEATURE GENERATION PASSED"
""",
    )


    # ========================================================
    # 6. ML6 BATCH SCORE
    # ========================================================

    ml6_batch_score = BashOperator(
        task_id="ml6_batch_score",
        bash_command=f"""
set -e

echo "========================================"
echo "DE7 - ML6 BATCH SCORE"
echo "========================================"

cd "{BASE_DIR}"

test -f "{ML6_SCRIPT}"
test -s "{DB_PATH}"

echo
echo "Running ML6 batch scoring..."

"{PYTHON_BIN}" "{ML6_SCRIPT}"

echo
echo "Checking network_risk_scores table..."

"{PYTHON_BIN}" - <<'PY'
import sqlite3
from pathlib import Path

db_path = Path(r"{DB_PATH}")

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()

cur.execute(
    "SELECT name FROM sqlite_master "
    "WHERE type='table' AND name='network_risk_scores'"
)

if cur.fetchone() is None:
    conn.close()
    raise SystemExit("network_risk_scores table was not created")

cur.execute("SELECT COUNT(*) FROM network_risk_scores")
rows = cur.fetchone()[0]

cur.execute(
    "SELECT COUNT(*) "
    "FROM network_risk_scores "
    "WHERE model_version IS NULL OR TRIM(model_version) = ''"
)

missing_model_version = cur.fetchone()[0]

conn.close()

if rows <= 0:
    raise SystemExit("network_risk_scores table is empty")

if missing_model_version > 0:
    raise SystemExit(
        f"network_risk_scores contains {{missing_model_version}} "
        "rows without model_version"
    )

print(f"network_risk_scores rows: {{rows}}")
print("ML6 PUBLICATION VALIDATION PASSED")
PY

echo
echo "ML6 BATCH SCORE PASSED"
""",
    )


    # ========================================================
    # 7. QUALITY CHECK
    # ========================================================

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
import os

base = Path(os.environ["DE7_BASE_DIR"])
db_path = base / "warehouse" / "network_analytics.db"
status_file = Path(os.environ["DE7_STATUS_FILE"])
ingestion_log = Path(os.environ["DE7_LOG_FILE"])

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()


# ------------------------------------------------------------
# Warehouse checks
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# ML feature checks
# ------------------------------------------------------------

cur.execute(
    "SELECT name FROM sqlite_master "
    "WHERE type='table' AND name='network_features'"
)

network_features_exists = cur.fetchone() is not None

if network_features_exists:
    cur.execute("SELECT COUNT(*) FROM network_features")
    network_features_rows = cur.fetchone()[0]
else:
    network_features_rows = 0


# ------------------------------------------------------------
# ML risk-score checks
# ------------------------------------------------------------

cur.execute(
    "SELECT name FROM sqlite_master "
    "WHERE type='table' AND name='network_risk_scores'"
)

network_risk_scores_exists = cur.fetchone() is not None

if network_risk_scores_exists:
    cur.execute("SELECT COUNT(*) FROM network_risk_scores")
    network_risk_scores_rows = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) "
        "FROM network_risk_scores "
        "WHERE model_version IS NULL "
        "OR TRIM(model_version) = ''"
    )
    risk_scores_missing_model_version = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) "
        "FROM network_risk_scores "
        "WHERE risk_score < 0 OR risk_score > 1"
    )
    invalid_risk_scores = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) "
        "FROM ("
        "SELECT grid_id, feature_timestamp, COUNT(*) AS c "
        "FROM network_risk_scores "
        "GROUP BY grid_id, feature_timestamp "
        "HAVING c > 1"
        ")"
    )
    risk_score_duplicates = cur.fetchone()[0]

else:
    network_risk_scores_rows = 0
    risk_scores_missing_model_version = 0
    invalid_risk_scores = 0
    risk_score_duplicates = 0


conn.close()


# ------------------------------------------------------------
# Ingestion rejection count
# ------------------------------------------------------------

rows_rejected = 0

if ingestion_log.exists():
    with ingestion_log.open("r", encoding="utf-8") as fh:
        for line in fh:
            if ",REJECTED," in line or ",REJECTED_DUPLICATE," in line:
                rows_rejected += 1


# ------------------------------------------------------------
# Overall quality status
# ------------------------------------------------------------

quality_passed = (
    dim_time_rows > 0
    and dim_grid_rows > 0
    and fact_rows > 0
    and orphan_time_keys == 0
    and orphan_grid_keys == 0
    and network_features_exists
    and network_features_rows > 0
    and network_risk_scores_exists
    and network_risk_scores_rows > 0
    and risk_scores_missing_model_version == 0
    and invalid_risk_scores == 0
    and risk_score_duplicates == 0
)


pipeline_status = "SUCCESS" if quality_passed else "FAILED"


status = {
    "run_id": "DE7_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "run_timestamp": datetime.now(timezone.utc).isoformat(),
    "status": pipeline_status,

    "tasks": {
        "ingest": "SUCCESS",
        "validate": "SUCCESS",
        "spark_process": "SUCCESS",
        "load_warehouse": "SUCCESS",
        "feature_generation": "SUCCESS",
        "ml6_batch_score": "SUCCESS",
        "quality_check": pipeline_status,
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
    "orphan_grid_keys": orphan_grid_keys,

    "network_features_rows": network_features_rows,
    "network_risk_scores_rows": network_risk_scores_rows,
    "risk_scores_missing_model_version": risk_scores_missing_model_version,
    "invalid_risk_scores": invalid_risk_scores,
    "risk_score_duplicates": risk_score_duplicates
}


status_file.write_text(
    json.dumps(status, indent=2),
    encoding="utf-8"
)


print(status_file)
print(json.dumps(status, indent=2))


if not quality_passed:
    raise SystemExit("QUALITY CHECK FAILED")

PY2

echo
echo "QUALITY CHECK PASSED"
""",
    )


    # ========================================================
    # 8. NOTIFY
    # ========================================================

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


    # ========================================================
    # DEPENDENCIES
    # ========================================================

    ingest >> validate >> spark_process >> load_warehouse
    load_warehouse >> feature_generation
    feature_generation >> ml6_batch_score
    ml6_batch_score >> quality_check
    quality_check >> notify