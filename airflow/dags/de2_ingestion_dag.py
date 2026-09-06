import sys
import json
from datetime import datetime
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path("/home/jaideep/network-intelligence")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# AIRFLOW
# ============================================================

from airflow.sdk import dag, task


# ============================================================
# INGESTION FUNCTIONS
# ============================================================

from ingestion.ingestion import (
    detect_files,
    validate_schema,
    validate_minimum_quality,
    route_file,
)


# ============================================================
# RESULT FILE
# ============================================================

RESULT_FILE = PROJECT_ROOT / "logs" / "validation_results.json"


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="de2_telecom_ingestion",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["phase3", "de2", "ingestion"],
)
def de2_telecom_ingestion():

    # ========================================================
    # TASK 1: DETECT
    # ========================================================

    @task
    def detect():

        files = detect_files()

        print(f"Detected files: {len(files)}")

        for file_path in files:
            print(f"  - {file_path}")

        # Do not return Path objects.
        # Return nothing so Airflow does not need to push
        # the detected file list into XCom.
        print("Detection completed.")


    # ========================================================
    # TASK 2: VALIDATE
    # ========================================================

    @task
    def validate():

        files = detect_files()

        print(f"Validating {len(files)} files")

        results = []

        for file_path in files:

            print(f"\nValidating: {file_path}")

            # ------------------------------------------------
            # SCHEMA VALIDATION
            # ------------------------------------------------

            schema_valid, schema_reason = validate_schema(
                file_path
            )

            if not schema_valid:

                print(f"[REJECT] {schema_reason}")

                results.append(
                    {
                        "file_path": str(file_path),
                        "valid": False,
                        "row_count": 0,
                        "reason": str(schema_reason),
                    }
                )

                continue


            # ------------------------------------------------
            # MINIMUM QUALITY VALIDATION
            # ------------------------------------------------

            quality_valid, row_count, quality_reason = (
                validate_minimum_quality(file_path)
            )

            if not quality_valid:

                print(f"[REJECT] {quality_reason}")

                results.append(
                    {
                        "file_path": str(file_path),
                        "valid": False,
                        "row_count": int(row_count),
                        "reason": str(quality_reason),
                    }
                )

                continue


            # ------------------------------------------------
            # VALID FILE
            # ------------------------------------------------

            print(f"[ACCEPT] rows={row_count}")

            results.append(
                {
                    "file_path": str(file_path),
                    "valid": True,
                    "row_count": int(row_count),
                    "reason": "Validation passed",
                }
            )


        # ----------------------------------------------------
        # SAVE RESULTS TO FILE
        # ----------------------------------------------------

        RESULT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            RESULT_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                results,
                f,
                indent=2
            )


        print(
            f"\nValidation completed for {len(results)} files"
        )

        print(
            f"Validation results saved to: {RESULT_FILE}"
        )

        # IMPORTANT:
        # No return statement.
        # This prevents Airflow from pushing the results
        # into XCom.


    # ========================================================
    # TASK 3: ROUTE
    # ========================================================

    @task
    def route():

        # Read validation results from the file.
        if not RESULT_FILE.exists():

            raise FileNotFoundError(
                f"Validation result file not found: {RESULT_FILE}"
            )


        with open(
            RESULT_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            results = json.load(f)


        print(f"Routing {len(results)} files")

        for result in results:

            status = route_file(
                result["file_path"],
                result["valid"],
                result["reason"],
                result["row_count"],
            )

            print(
                f"{Path(result['file_path']).name}: "
                f"{status}"
            )


        print("Routing completed.")


    # ========================================================
    # TASK 4: LOG
    # ========================================================

    @task
    def log():

        print("Ingestion workflow completed.")

        if RESULT_FILE.exists():

            with open(
                RESULT_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                results = json.load(f)


            for result in results:

                print(
                    f"File: "
                    f"{Path(result['file_path']).name} | "
                    f"Valid: {result['valid']} | "
                    f"Rows: {result['row_count']}"
                )

        else:

            print("No validation results file found.")


    # ========================================================
    # CREATE TASKS
    # ========================================================

    detected = detect()

    validated = validate()

    routed = route()

    logged = log()


    # ========================================================
    # DEPENDENCIES
    # ========================================================

    detected >> validated
    validated >> routed
    routed >> logged


# ============================================================
# CREATE DAG
# ============================================================

de2_telecom_ingestion()
