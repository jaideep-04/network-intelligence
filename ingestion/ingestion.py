from pathlib import Path
from datetime import datetime
import csv
import shutil


# ============================================================
# DE2 - LANDING TO RAW INGESTION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

LANDING_DIR = BASE_DIR / "data" / "landing"
RAW_DIR = BASE_DIR / "data" / "raw"
REJECTED_DIR = BASE_DIR / "data" / "rejected"
LOG_DIR = BASE_DIR / "logs"

LOG_FILE = LOG_DIR / "ingestion_log.csv"

FILE_PATTERN = "sms-call-internet-mi-*.csv"

EXPECTED_COLUMNS = [
    "datetime",
    "CellID",
    "countrycode",
    "smsin",
    "smsout",
    "callin",
    "callout",
    "internet",
]

ACTIVITY_COLUMNS = [
    "smsin",
    "smsout",
    "callin",
    "callout",
    "internet",
]


# ============================================================
# DIRECTORY SETUP
# ============================================================

def setup_directories():
    LANDING_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DE2 - detect_files()
# ============================================================

def detect_files():
    """
    Detect only daily telecom activity CSV files
    matching sms-call-internet-mi-*.csv.
    """

    setup_directories()

    files = sorted(LANDING_DIR.glob(FILE_PATTERN))

    return files


# ============================================================
# DE2 - validate_schema()
# ============================================================

def validate_schema(file_path):
    """
    Validate that the CSV contains exactly the required
    telecom activity columns.
    """

    try:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)

            header = next(reader, None)

        if header is None:
            return False, "File is empty"

        if header != EXPECTED_COLUMNS:
            missing = [
                col for col in EXPECTED_COLUMNS
                if col not in header
            ]

            unexpected = [
                col for col in header
                if col not in EXPECTED_COLUMNS
            ]

            reasons = []

            if missing:
                reasons.append(
                    f"missing columns: {missing}"
                )

            if unexpected:
                reasons.append(
                    f"unexpected columns: {unexpected}"
                )

            if not missing and not unexpected:
                reasons.append(
                    "columns are in an unexpected order"
                )

            return False, "; ".join(reasons)

        return True, "Schema valid"

    except Exception as exc:
        return False, f"Schema validation error: {exc}"


# ============================================================
# DE2 - validate_minimum_quality()
# ============================================================

def validate_minimum_quality(file_path):
    """
    Perform lightweight quality checks suitable for ingestion.

    Important:
    Blank activity values are allowed because the source dataset
    contains legitimate blanks.
    """

    row_count = 0

    try:
        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:

                row_count += 1

                # ------------------------------------------------
                # Validate datetime
                # ------------------------------------------------
                datetime_value = row.get("datetime", "").strip()

                if not datetime_value:
                    return False, row_count, (
                        f"missing datetime at row {row_count}"
                    )

                try:
                    datetime.strptime(
                        datetime_value,
                        "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    return False, row_count, (
                        f"invalid datetime at row {row_count}: "
                        f"{datetime_value}"
                    )

                # ------------------------------------------------
                # Validate CellID
                # ------------------------------------------------
                cell_id = row.get("CellID", "").strip()

                if not cell_id:
                    return False, row_count, (
                        f"missing CellID at row {row_count}"
                    )

                try:
                    int(cell_id)
                except ValueError:
                    return False, row_count, (
                        f"invalid CellID at row {row_count}: "
                        f"{cell_id}"
                    )

                # ------------------------------------------------
                # Validate countrycode if present
                # ------------------------------------------------
                countrycode = row.get("countrycode", "").strip()

                if countrycode:
                    try:
                        int(countrycode)
                    except ValueError:
                        return False, row_count, (
                            f"invalid countrycode at row {row_count}: "
                            f"{countrycode}"
                        )

                # ------------------------------------------------
                # Activity values
                #
                # Blank values are allowed.
                # If present, they must be numeric and non-negative.
                # ------------------------------------------------
                for column in ACTIVITY_COLUMNS:

                    value = row.get(column, "").strip()

                    if value == "":
                        continue

                    try:
                        numeric_value = float(value)
                    except ValueError:
                        return False, row_count, (
                            f"non-numeric {column} at row "
                            f"{row_count}: {value}"
                        )

                    if numeric_value < 0:
                        return False, row_count, (
                            f"negative {column} at row "
                            f"{row_count}: {numeric_value}"
                        )

        if row_count == 0:
            return False, 0, "File contains header but no data rows"

        return True, row_count, "Minimum quality checks passed"

    except Exception as exc:
        return False, row_count, (
            f"Quality validation error: {exc}"
        )


# ============================================================
# DE2 - route_file()
# ============================================================

def route_file(file_path, valid, reason, row_count):
    """
    Route the file to RAW or REJECTED.
    """

    processed_at = datetime.now().isoformat(timespec="seconds")

    if valid:

        destination = RAW_DIR / file_path.name

        # Avoid accidental overwrite during reruns.
        if destination.exists():
            status = "SKIPPED_DUPLICATE"
            final_reason = (
                "File already exists in raw; "
                "not overwritten"
            )
        else:
            shutil.copy2(file_path, destination)
            status = "ACCEPTED"
            final_reason = reason

    else:

        destination = REJECTED_DIR / file_path.name

        if destination.exists():
            status = "REJECTED_DUPLICATE"
            final_reason = reason
        else:
            shutil.copy2(file_path, destination)
            status = "REJECTED"
            final_reason = reason

    log_ingestion(
        filename=file_path.name,
        status=status,
        row_count=row_count,
        reason=final_reason,
        processed_at=processed_at,
    )

    return status, destination


# ============================================================
# INGESTION LOG
# ============================================================

def log_ingestion(
    filename,
    status,
    row_count,
    reason,
    processed_at,
):
    """
    Write machine-readable ingestion metadata.
    """

    setup_directories()

    file_exists = LOG_FILE.exists()

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "filename",
                "status",
                "row_count",
                "reason",
                "processed_at",
            ])

        writer.writerow([
            filename,
            status,
            row_count,
            reason,
            processed_at,
        ])


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path):

    print("=" * 70)
    print(f"Processing: {file_path.name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Schema validation
    # --------------------------------------------------------

    schema_valid, schema_reason = validate_schema(file_path)

    if not schema_valid:

        print(f"[REJECT] {schema_reason}")

        status, destination = route_file(
            file_path=file_path,
            valid=False,
            reason=schema_reason,
            row_count=0,
        )

        return status

    print("[PASS] Schema validation")

    # --------------------------------------------------------
    # Minimum quality validation
    # --------------------------------------------------------

    quality_valid, row_count, quality_reason = (
        validate_minimum_quality(file_path)
    )

    if not quality_valid:

        print(f"[REJECT] {quality_reason}")

        status, destination = route_file(
            file_path=file_path,
            valid=False,
            reason=quality_reason,
            row_count=row_count,
        )

        return status

    print(f"[PASS] Minimum quality validation")
    print(f"[INFO] Rows: {row_count}")

    # --------------------------------------------------------
    # Route
    # --------------------------------------------------------

    status, destination = route_file(
        file_path=file_path,
        valid=True,
        reason=quality_reason,
        row_count=row_count,
    )

    print(f"[ROUTE] {status}")
    print(f"[DESTINATION] {destination}")

    return status


# ============================================================
# MAIN
# ============================================================

def main():

    setup_directories()

    print("=" * 70)
    print("DE2 - TELECOM LANDING TO RAW INGESTION")
    print("=" * 70)

    files = detect_files()

    if not files:
        print("No matching files found in landing.")
        return

    print(f"Detected files: {len(files)}")
    print()

    for file_path in files:
        process_file(file_path)

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)

    print(f"Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
