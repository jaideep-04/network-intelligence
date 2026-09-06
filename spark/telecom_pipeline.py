import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = BASE_DIR / "logs" / "spark_pipeline_log.csv"


def write_log(status: str, message: str):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not exists:
            writer.writerow(["processed_at", "status", "message"])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            status,
            message
        ])


def parse_args():
    parser = argparse.ArgumentParser(
        description="DE3 - Telecom Spark Processing Entry Point"
    )

    parser.add_argument(
        "--raw-dir",
        required=True
    )

    parser.add_argument(
        "--processed-dir",
        required=True
    )

    parser.add_argument(
        "--analytics-dir",
        required=True
    )

    parser.add_argument(
        "--geojson",
        required=True
    )

    return parser.parse_args()


def validate_inputs(args):
    raw_dir = Path(args.raw_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    analytics_dir = Path(args.analytics_dir).resolve()
    geojson = Path(args.geojson).resolve()

    if not raw_dir.exists():
        raise RuntimeError(f"Raw directory does not exist: {raw_dir}")

    raw_files = list(raw_dir.glob("sms-call-internet-mi-*.csv"))

    if not raw_files:
        raise RuntimeError(
            f"Empty input: no telecom CSV files found in {raw_dir}"
        )

    if not geojson.exists():
        raise RuntimeError(
            f"GeoJSON reference does not exist: {geojson}"
        )

    if geojson.stat().st_size == 0:
        raise RuntimeError(
            f"GeoJSON reference is empty: {geojson}"
        )

    processed_dir.mkdir(parents=True, exist_ok=True)
    analytics_dir.mkdir(parents=True, exist_ok=True)

    return raw_dir, processed_dir, analytics_dir, geojson, raw_files


def run():
    args = parse_args()

    write_log(
        "START",
        "DE3 Spark processing job started"
    )

    try:
        (
            raw_dir,
            processed_dir,
            analytics_dir,
            geojson,
            raw_files
        ) = validate_inputs(args)

        print("=" * 75)
        print("DE3 - TELECOM SPARK PROCESSING JOB")
        print("=" * 75)
        print(f"Raw directory       : {raw_dir}")
        print(f"Processed directory : {processed_dir}")
        print(f"Analytics directory : {analytics_dir}")
        print(f"GeoJSON reference   : {geojson}")
        print(f"Input CSV files     : {len(raw_files)}")
        print("=" * 75)

        write_log(
            "RUNNING",
            f"Launching SP7 Spark pipeline with {len(raw_files)} input files"
        )

        env = os.environ.copy()

        # Pass the DE3 configuration to the existing Spark pipeline.
        env["DE3_RAW_DIR"] = str(raw_dir)
        env["DE3_PROCESSED_DIR"] = str(processed_dir)
        env["DE3_ANALYTICS_DIR"] = str(analytics_dir)
        env["MILANO_GEOJSON_PATH"] = str(geojson)

        sp7_script = BASE_DIR / "phase2" / "sp7_pipeline.py"

        if not sp7_script.exists():
            raise RuntimeError(
                f"SP7 pipeline not found: {sp7_script}"
            )

        print()
        print("Launching existing SP7 end-to-end Spark pipeline...")
        print()

        result = subprocess.run(
            [sys.executable, str(sp7_script)],
            cwd=str(BASE_DIR),
            env=env,
            check=False
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"SP7 Spark pipeline failed with exit code "
                f"{result.returncode}"
            )

        write_log(
            "SUCCESS",
            "DE3 Spark processing completed successfully"
        )

        print()
        print("=" * 75)
        print("DE3 SPARK PROCESSING SUCCESS")
        print("=" * 75)
        print(f"Processed output : {processed_dir}")
        print(f"Analytics output : {analytics_dir}")
        print("=" * 75)

        return 0

    except Exception as exc:
        write_log(
            "FAILED",
            f"DE3 Spark processing failed: {exc}"
        )

        print()
        print("=" * 75)
        print("DE3 SPARK PROCESSING FAILED")
        print("=" * 75)
        print(f"Reason : {exc}")
        print("=" * 75)

        return 1


if __name__ == "__main__":
    sys.exit(run())
