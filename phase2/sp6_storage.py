"""
============================================================================
SP6 - STORAGE & DATA ORGANIZATION
============================================================================

Purpose
-------
SP6 is the storage layer for the Phase 2 PySpark pipeline.

SP6 responsibilities:

1. Validate project storage directories.
2. Read the processed SP4 Parquet output.
3. Read SP5 performance outputs safely.
4. Store the processed activity data as Parquet.
5. Store the SP5 analytics outputs as curated Parquet.
6. Preserve the project storage-zone design.
7. Validate Parquet schemas, row counts, nulls and duplicates.
8. Verify that the stored datasets can be read back successfully.

IMPORTANT
---------
SP5 produces multiple analytical datasets:

    Grid performance:
        10,000 rows

    Temporal performance:
        168 rows

The SP5 directory may therefore contain nested Parquet datasets.

DO NOT blindly execute:

    spark.read.parquet(SP5_PATH)

because the root directory can contain metadata/files/directories rather
than one directly readable Parquet dataset.

This version recursively discovers actual *.parquet files and reads the
appropriate dataset.

Windows / PySpark:
    PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are explicitly configured.
============================================================================
"""

import os
import sys
import shutil
import time
from pathlib import Path


# ============================================================================
# WINDOWS PYTHON WORKER CONFIGURATION
# ============================================================================

PYTHON_EXE = sys.executable

os.environ["PYSPARK_PYTHON"] = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE


# ============================================================================
# PYSPARK IMPORTS
# ============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    LongType,
    DoubleType,
    TimestampType,
)


# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

PROCESSED_DIR = os.path.join(
    DATA_DIR,
    "processed"
)

ANALYTICS_DIR = os.path.join(
    DATA_DIR,
    "analytics"
)

REFERENCE_DIR = os.path.join(
    DATA_DIR,
    "reference"
)


# ============================================================================
# INPUT LOCATIONS
# ============================================================================

SP4_PATH = os.path.join(
    PROCESSED_DIR,
    "sp4_geo_enriched"
)

SP5_PATH = os.path.join(
    PROCESSED_DIR,
    "sp5_performance"
)


# ============================================================================
# OUTPUT LOCATIONS
# ============================================================================

ACTIVITY_OUTPUT_PATH = os.path.join(
    PROCESSED_DIR,
    "activity"
)

HOURLY_GRID_SUMMARY_OUTPUT_PATH = os.path.join(
    ANALYTICS_DIR,
    "hourly_grid_summary"
)

GRID_PERFORMANCE_OUTPUT_PATH = os.path.join(
    ANALYTICS_DIR,
    "grid_performance"
)

TEMPORAL_PERFORMANCE_OUTPUT_PATH = os.path.join(
    ANALYTICS_DIR,
    "temporal_performance"
)


# ============================================================================
# EXPECTED COUNTS
# ============================================================================

EXPECTED_SP4_ROWS = 1_679_994

EXPECTED_GRID_PERFORMANCE_ROWS = 10_000

EXPECTED_TEMPORAL_PERFORMANCE_ROWS = 168

EXPECTED_SP5_TOTAL_ROWS = (
    EXPECTED_GRID_PERFORMANCE_ROWS
    + EXPECTED_TEMPORAL_PERFORMANCE_ROWS
)


# ============================================================================
# REQUIRED SP4 COLUMNS
# ============================================================================

REQUIRED_SP4_COLUMNS = [
    "timestamp",
    "grid_id",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_activity",
    "total_activity",
    "geometry",
]


# ============================================================================
# EXPECTED SP5 GRID PERFORMANCE COLUMNS
# ============================================================================

REQUIRED_GRID_PERFORMANCE_COLUMNS = [
    "grid_id",
    "window_sms_in",
    "window_sms_out",
    "window_call_in",
    "window_call_out",
    "window_internet_activity",
    "window_total_activity",
]


# ============================================================================
# EXPECTED SP5 TEMPORAL PERFORMANCE COLUMNS
# ============================================================================

REQUIRED_TEMPORAL_PERFORMANCE_COLUMNS = [
    "timestamp",
    "timestamp_total_activity",
]


# ============================================================================
# OUTPUT SP4 ACTIVITY COLUMNS
# ============================================================================

ACTIVITY_COLUMNS = [
    "timestamp",
    "grid_id",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_activity",
    "total_activity",
    "geometry",
]


# ============================================================================
# OUTPUT HOURLY GRID SUMMARY COLUMNS
# ============================================================================

HOURLY_GRID_SUMMARY_COLUMNS = [
    "timestamp",
    "grid_id",
    "sms_in",
    "sms_out",
    "call_in",
    "call_out",
    "internet_activity",
    "total_activity",
    "geometry",
]


# ============================================================================
# SPARK SESSION
# ============================================================================

def create_spark_session():

    print()
    print("=" * 75)
    print("[1] CREATING SPARK SESSION")
    print("=" * 75)

    spark = (
        SparkSession.builder
        .appName(
            "SP6_Storage_Data_Organization"
        )
        .config(
            "spark.sql.shuffle.partitions",
            "8"
        )
        .config(
            "spark.sql.adaptive.enabled",
            "true"
        )
        .config(
            "spark.sql.debug.maxToStringFields",
            "200"
        )
        .config(
            "spark.pyspark.python",
            PYTHON_EXE
        )
        .config(
            "spark.pyspark.driver.python",
            PYTHON_EXE
        )
        .config(
            "spark.python.worker.reuse",
            "true"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel(
        "WARN"
    )

    print(
        "Spark session created successfully."
    )

    print(
        f"Spark version             : "
        f"{spark.version}"
    )

    print(
        "Adaptive Query Execution : "
        f"{spark.conf.get('spark.sql.adaptive.enabled')}"
    )

    print(
        "Shuffle partitions        : "
        f"{spark.conf.get('spark.sql.shuffle.partitions')}"
    )

    print(
        f"Python interpreter        : "
        f"{PYTHON_EXE}"
    )

    return spark


# ============================================================================
# STORAGE DIRECTORIES
# ============================================================================

def prepare_storage_directories():

    print()
    print("=" * 75)
    print("[2] PREPARING SP6 STORAGE DIRECTORIES")
    print("=" * 75)

    directories = [
        PROCESSED_DIR,
        ANALYTICS_DIR,
        ACTIVITY_OUTPUT_PATH,
        HOURLY_GRID_SUMMARY_OUTPUT_PATH,
        GRID_PERFORMANCE_OUTPUT_PATH,
        TEMPORAL_PERFORMANCE_OUTPUT_PATH,
        REFERENCE_DIR,
    ]

    for directory in directories:

        os.makedirs(
            directory,
            exist_ok=True
        )

        print(
            f"Ready: {directory}"
        )

    print()
    print(
        "Storage directories: PASS"
    )


# ============================================================================
# RECURSIVE PARQUET DISCOVERY
# ============================================================================

def find_parquet_files(root_path):

    """
    Recursively find real Parquet files.

    This is the important SP6 fix.

    Spark can fail with:

        [UNABLE_TO_INFER_SCHEMA]
        Unable to infer schema for Parquet.

    when a directory passed to spark.read.parquet() does not itself
    contain directly readable Parquet files.

    Therefore SP6 explicitly discovers *.parquet files.
    """

    root = Path(root_path)

    if not root.exists():

        raise FileNotFoundError(
            f"Path does not exist:\n{root_path}"
        )

    if not root.is_dir():

        raise ValueError(
            f"Expected directory but received:\n{root_path}"
        )

    parquet_files = []

    for path in root.rglob("*.parquet"):

        if path.is_file():

            parquet_files.append(
                str(path)
            )

    parquet_files.sort()

    return parquet_files


# ============================================================================
# PRINT DISCOVERED PARQUET FILES
# ============================================================================

def show_parquet_files(
    label,
    root_path
):

    print()
    print(
        f"{label} Parquet discovery:"
    )

    print("-" * 75)

    parquet_files = find_parquet_files(
        root_path
    )

    print(
        f"Root location : {root_path}"
    )

    print(
        f"Parquet files : {len(parquet_files)}"
    )

    for index, path in enumerate(
        parquet_files,
        start=1
    ):

        print(
            f"  {index:02d}. {path}"
        )

    return parquet_files


# ============================================================================
# VALIDATE SP4 LOCATION
# ============================================================================

def validate_sp4_location():

    print()
    print("=" * 75)
    print("[3] VALIDATING SP4 INPUT LOCATION")
    print("=" * 75)

    print(
        "SP4 input location:"
    )

    print(
        SP4_PATH
    )

    if not os.path.exists(
        SP4_PATH
    ):

        raise FileNotFoundError(
            f"SP4 output does not exist:\n{SP4_PATH}"
        )

    parquet_files = find_parquet_files(
        SP4_PATH
    )

    print(
        f"SP4 Parquet files found: "
        f"{len(parquet_files)}"
    )

    if len(parquet_files) == 0:

        raise FileNotFoundError(
            "No Parquet files found in SP4 output."
        )

    print(
        "SP4 input location: PASS"
    )

    return parquet_files


# ============================================================================
# VALIDATE SP5 LOCATION
# ============================================================================

def validate_sp5_location():

    print()
    print("=" * 75)
    print("[4] VALIDATING SP5 INPUT LOCATION")
    print("=" * 75)

    print(
        "SP5 input location:"
    )

    print(
        SP5_PATH
    )

    if not os.path.exists(
        SP5_PATH
    ):

        raise FileNotFoundError(
            f"SP5 output does not exist:\n{SP5_PATH}"
        )

    parquet_files = find_parquet_files(
        SP5_PATH
    )

    print(
        f"SP5 Parquet files found: "
        f"{len(parquet_files)}"
    )

    if len(parquet_files) == 0:

        raise FileNotFoundError(
            "No Parquet files found in SP5 output."
        )

    print(
        "SP5 input location: PASS"
    )

    return parquet_files


# ============================================================================
# READ PARQUET FILE LIST SAFELY
# ============================================================================

def read_parquet_files(
    spark,
    parquet_files,
    label
):

    if not parquet_files:

        raise ValueError(
            f"No Parquet files available for {label}."
        )

    print()
    print(
        f"Reading {label}..."
    )

    print(
        f"Files supplied to Spark: "
        f"{len(parquet_files)}"
    )

    df = (
        spark.read
        .parquet(*parquet_files)
    )

    return df


# ============================================================================
# LOAD SP4
# ============================================================================

def load_sp4(
    spark,
    parquet_files
):

    print()
    print("=" * 75)
    print("[5] READING SP4 PARQUET OUTPUT")
    print("=" * 75)

    print(
        f"Reading:\n{SP4_PATH}"
    )

    sp4_df = read_parquet_files(
        spark,
        parquet_files,
        "SP4"
    )

    print()
    print(
        f"SP4 columns: {sp4_df.columns}"
    )

    return sp4_df


# ============================================================================
# LOAD SP5 ROOT DATASET
# ============================================================================

def load_sp5_files(
    spark,
    parquet_files
):

    print()
    print("=" * 75)
    print("[6] READING SP5 PARQUET OUTPUT")
    print("=" * 75)

    print(
        f"Reading recursively from:"
    )

    print(
        SP5_PATH
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "SP5 contains multiple analytical datasets."
    )

    print(
        "SP6 therefore reads discovered Parquet files "
        "instead of assuming the SP5 root is one dataset."
    )

    sp5_df = read_parquet_files(
        spark,
        parquet_files,
        "SP5"
    )

    print()
    print(
        f"SP5 combined columns: "
        f"{sp5_df.columns}"
    )

    return sp5_df


# ============================================================================
# VALIDATE SP4 SCHEMA
# ============================================================================

def validate_sp4_schema(df):

    print()
    print("=" * 75)
    print("[7] VALIDATING SP4 SCHEMA")
    print("=" * 75)

    missing = [
        column
        for column in REQUIRED_SP4_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "SP4 missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print(
        "SP4 schema: PASS"
    )

    df.printSchema()


# ============================================================================
# VALIDATE SP4 DATA
# ============================================================================

def validate_sp4_data(df):

    print()
    print("=" * 75)
    print("[8] VALIDATING SP4 DATA")
    print("=" * 75)

    row_count = df.count()

    print(
        f"SP4 rows: {row_count:,}"
    )

    if row_count == 0:

        raise ValueError(
            "SP4 contains zero rows."
        )

    if row_count != EXPECTED_SP4_ROWS:

        raise ValueError(
            "Unexpected SP4 row count. "
            f"Expected {EXPECTED_SP4_ROWS:,}, "
            f"received {row_count:,}."
        )

    print(
        "SP4 row count: PASS"
    )

    # ------------------------------------------------------------------------
    # NULL VALIDATION
    # ------------------------------------------------------------------------

    null_condition = None

    for column in ACTIVITY_COLUMNS:

        condition = F.col(column).isNull()

        if null_condition is None:

            null_condition = condition

        else:

            null_condition = (
                null_condition
                | condition
            )

    null_rows = (
        df
        .filter(null_condition)
        .count()
    )

    print(
        f"Rows containing NULLs: "
        f"{null_rows:,}"
    )

    if null_rows != 0:

        raise ValueError(
            "SP4 contains NULL values "
            "in required output columns."
        )

    print(
        "SP4 NULL validation: PASS"
    )

    # ------------------------------------------------------------------------
    # NEGATIVE ACTIVITY VALIDATION
    # ------------------------------------------------------------------------

    negative_condition = (
        (F.col("sms_in") < 0)
        | (F.col("sms_out") < 0)
        | (F.col("call_in") < 0)
        | (F.col("call_out") < 0)
        | (F.col("internet_activity") < 0)
        | (F.col("total_activity") < 0)
    )

    negative_rows = (
        df
        .filter(negative_condition)
        .count()
    )

    print(
        f"Rows containing negative activity: "
        f"{negative_rows:,}"
    )

    if negative_rows != 0:

        raise ValueError(
            "SP4 contains negative activity values."
        )

    print(
        "SP4 negative validation: PASS"
    )

    # ------------------------------------------------------------------------
    # GRAIN VALIDATION
    # ------------------------------------------------------------------------

    duplicate_groups = (
        df
        .groupBy(
            "timestamp",
            "grid_id"
        )
        .count()
        .filter(
            F.col("count") > 1
        )
        .count()
    )

    print(
        f"Duplicate timestamp + grid_id groups: "
        f"{duplicate_groups}"
    )

    if duplicate_groups != 0:

        raise ValueError(
            "SP4 contains duplicate "
            "timestamp + grid_id groups."
        )

    print(
        "SP4 grain validation: PASS"
    )


# ============================================================================
# PREPARE ACTIVITY DATASET
# ============================================================================

def prepare_activity_dataset(
    sp4_df
):

    print()
    print("=" * 75)
    print("[9] PREPARING PROCESSED ACTIVITY DATA")
    print("=" * 75)

    activity_df = (
        sp4_df
        .select(
            *ACTIVITY_COLUMNS
        )
    )

    print(
        "Activity columns:"
    )

    for column in activity_df.columns:

        print(
            f"  - {column}"
        )

    return activity_df


# ============================================================================
# PREPARE HOURLY GRID SUMMARY
# ============================================================================

def prepare_hourly_grid_summary(
    sp4_df
):

    print()
    print("=" * 75)
    print("[10] PREPARING HOURLY GRID SUMMARY")
    print("=" * 75)

    print(
        "Source: SP4 geo-enriched activity data"
    )

    print(
        "Aggregation grain:"
    )

    print(
        "timestamp + grid_id"
    )

    hourly_grid_summary = (
        sp4_df
        .groupBy(
            "timestamp",
            "grid_id"
        )
        .agg(
            F.round(
                F.sum("sms_in"),
                4
            ).alias(
                "sms_in"
            ),

            F.round(
                F.sum("sms_out"),
                4
            ).alias(
                "sms_out"
            ),

            F.round(
                F.sum("call_in"),
                4
            ).alias(
                "call_in"
            ),

            F.round(
                F.sum("call_out"),
                4
            ).alias(
                "call_out"
            ),

            F.round(
                F.sum("internet_activity"),
                4
            ).alias(
                "internet_activity"
            ),

            F.round(
                F.sum("total_activity"),
                4
            ).alias(
                "total_activity"
            ),

            F.first(
                "geometry",
                ignorenulls=True
            ).alias(
                "geometry"
            )
        )
        .select(
            *HOURLY_GRID_SUMMARY_COLUMNS
        )
    )

    return hourly_grid_summary


# ============================================================================
# VALIDATE HOURLY GRID SUMMARY
# ============================================================================

def validate_hourly_grid_summary(
    df,
    expected_rows=EXPECTED_SP4_ROWS
):

    print()
    print("=" * 75)
    print("[11] VALIDATING HOURLY GRID SUMMARY")
    print("=" * 75)

    row_count = df.count()

    print(
        f"Hourly grid summary rows: "
        f"{row_count:,}"
    )

    if row_count != expected_rows:

        raise ValueError(
            "Unexpected hourly grid summary row count. "
            f"Expected {expected_rows:,}, "
            f"received {row_count:,}."
        )

    duplicate_groups = (
        df
        .groupBy(
            "timestamp",
            "grid_id"
        )
        .count()
        .filter(
            F.col("count") > 1
        )
        .count()
    )

    print(
        f"Duplicate timestamp + grid_id groups: "
        f"{duplicate_groups}"
    )

    if duplicate_groups != 0:

        raise ValueError(
            "Hourly grid summary contains duplicates."
        )

    print(
        "Duplicate validation: PASS"
    )

    null_geometry = (
        df
        .filter(
            F.col("geometry").isNull()
        )
        .count()
    )

    print(
        f"Rows with NULL geometry: "
        f"{null_geometry:,}"
    )

    if null_geometry != 0:

        raise ValueError(
            "Hourly grid summary contains "
            "NULL geometry."
        )

    print(
        "Geometry validation: PASS"
    )

    negative_rows = (
        df
        .filter(
            (F.col("sms_in") < 0)
            | (F.col("sms_out") < 0)
            | (F.col("call_in") < 0)
            | (F.col("call_out") < 0)
            | (F.col("internet_activity") < 0)
            | (F.col("total_activity") < 0)
        )
        .count()
    )

    print(
        f"Rows with negative values: "
        f"{negative_rows:,}"
    )

    if negative_rows != 0:

        raise ValueError(
            "Hourly grid summary contains "
            "negative activity."
        )

    print(
        "Negative validation: PASS"
    )


# ============================================================================
# DETECT SP5 DATASETS
# ============================================================================

def classify_sp5_schema(df):

    """
    Identify whether a DataFrame is:

        1. Grid performance
        2. Temporal performance

    based on the actual SP5 schema.
    """

    columns = set(
        df.columns
    )

    grid_required = set(
        REQUIRED_GRID_PERFORMANCE_COLUMNS
    )

    temporal_required = set(
        REQUIRED_TEMPORAL_PERFORMANCE_COLUMNS
    )

    if grid_required.issubset(columns):

        return "grid"

    if temporal_required.issubset(columns):

        return "temporal"

    return "unknown"


# ============================================================================
# LOAD SP5 DATASET BY DIRECTORY
# ============================================================================

def try_read_parquet_directory(
    spark,
    path
):

    if not os.path.exists(path):

        return None

    files = find_parquet_files(
        path
    )

    if not files:

        return None

    try:

        return (
            spark.read
            .parquet(*files)
        )

    except Exception:

        return None


# ============================================================================
# DISCOVER SP5 DATASETS
# ============================================================================

def discover_sp5_datasets(
    spark,
    sp5_files
):

    print()
    print("=" * 75)
    print("[12] IDENTIFYING SP5 ANALYTICS DATASETS")
    print("=" * 75)

    grid_candidates = [
        os.path.join(
            SP5_PATH,
            "grid_performance"
        ),
        os.path.join(
            SP5_PATH,
            "grid"
        ),
        os.path.join(
            SP5_PATH,
            "grid_aggregation"
        ),
        os.path.join(
            SP5_PATH,
            "grid_metrics"
        ),
    ]

    temporal_candidates = [
        os.path.join(
            SP5_PATH,
            "temporal_performance"
        ),
        os.path.join(
            SP5_PATH,
            "temporal"
        ),
        os.path.join(
            SP5_PATH,
            "temporal_aggregation"
        ),
        os.path.join(
            SP5_PATH,
            "temporal_metrics"
        ),
    ]

    grid_df = None
    temporal_df = None

    # ------------------------------------------------------------------------
    # FIRST: LOOK FOR EXPECTED SUBDIRECTORIES
    # ------------------------------------------------------------------------

    for path in grid_candidates:

        candidate = (
            try_read_parquet_directory(
                spark,
                path
            )
        )

        if candidate is not None:

            classification = classify_sp5_schema(
                candidate
            )

            if classification == "grid":

                grid_df = candidate

                print(
                    f"Grid performance dataset found:\n"
                    f"{path}"
                )

                break

    for path in temporal_candidates:

        candidate = (
            try_read_parquet_directory(
                spark,
                path
            )
        )

        if candidate is not None:

            classification = classify_sp5_schema(
                candidate
            )

            if classification == "temporal":

                temporal_df = candidate

                print(
                    f"Temporal performance dataset found:\n"
                    f"{path}"
                )

                break

    # ------------------------------------------------------------------------
    # SECOND: FALLBACK
    #
    # If SP5 wrote both datasets into the same root, inspect discovered
    # Parquet files individually.
    # ------------------------------------------------------------------------

    if grid_df is None or temporal_df is None:

        print()
        print(
            "Searching individual SP5 Parquet files..."
        )

        for parquet_file in sp5_files:

            try:

                candidate = (
                    spark.read
                    .parquet(
                        parquet_file
                    )
                )

                classification = (
                    classify_sp5_schema(
                        candidate
                    )
                )

                if classification == "grid":

                    if grid_df is None:

                        grid_df = candidate

                        print(
                            "Grid performance dataset "
                            f"detected from:\n"
                            f"{parquet_file}"
                        )

                elif classification == "temporal":

                    if temporal_df is None:

                        temporal_df = candidate

                        print(
                            "Temporal performance dataset "
                            f"detected from:\n"
                            f"{parquet_file}"
                        )

            except Exception:

                continue

            if (
                grid_df is not None
                and
                temporal_df is not None
            ):

                break

    # ------------------------------------------------------------------------
    # VALIDATE DETECTION
    # ------------------------------------------------------------------------

    if grid_df is None:

        raise ValueError(
            "Could not identify the SP5 grid performance dataset."
        )

    if temporal_df is None:

        raise ValueError(
            "Could not identify the SP5 temporal performance dataset."
        )

    print()
    print(
        "SP5 dataset detection: PASS"
    )

    return (
        grid_df,
        temporal_df
    )


# ============================================================================
# VALIDATE GRID PERFORMANCE
# ============================================================================

def validate_grid_performance(
    df
):

    print()
    print("=" * 75)
    print("[13] VALIDATING GRID PERFORMANCE DATA")
    print("=" * 75)

    missing = [
        column
        for column in REQUIRED_GRID_PERFORMANCE_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Grid performance missing columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    row_count = df.count()

    print(
        f"Grid performance rows: "
        f"{row_count:,}"
    )

    if row_count != EXPECTED_GRID_PERFORMANCE_ROWS:

        raise ValueError(
            "Unexpected grid performance row count. "
            f"Expected {EXPECTED_GRID_PERFORMANCE_ROWS:,}, "
            f"received {row_count:,}."
        )

    duplicate_count = (
        df
        .groupBy(
            "grid_id"
        )
        .count()
        .filter(
            F.col("count") > 1
        )
        .count()
    )

    print(
        f"Duplicate grid_id values: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:

        raise ValueError(
            "Grid performance contains duplicate grid_id values."
        )

    null_rows = (
        df
        .filter(
            F.col("grid_id").isNull()
            |
            F.col("window_total_activity").isNull()
        )
        .count()
    )

    print(
        f"Rows with critical NULL values: "
        f"{null_rows:,}"
    )

    if null_rows != 0:

        raise ValueError(
            "Grid performance contains critical NULL values."
        )

    negative_rows = (
        df
        .filter(
            (F.col("window_sms_in") < 0)
            |
            (F.col("window_sms_out") < 0)
            |
            (F.col("window_call_in") < 0)
            |
            (F.col("window_call_out") < 0)
            |
            (F.col("window_internet_activity") < 0)
            |
            (F.col("window_total_activity") < 0)
        )
        .count()
    )

    print(
        f"Rows with negative metrics: "
        f"{negative_rows:,}"
    )

    if negative_rows != 0:

        raise ValueError(
            "Grid performance contains negative metrics."
        )

    print(
        "Grid performance validation: PASS"
    )

    df.printSchema()


# ============================================================================
# VALIDATE TEMPORAL PERFORMANCE
# ============================================================================

def validate_temporal_performance(
    df
):

    print()
    print("=" * 75)
    print("[14] VALIDATING TEMPORAL PERFORMANCE DATA")
    print("=" * 75)

    missing = [
        column
        for column in REQUIRED_TEMPORAL_PERFORMANCE_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Temporal performance missing columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    row_count = df.count()

    print(
        f"Temporal performance rows: "
        f"{row_count:,}"
    )

    if row_count != EXPECTED_TEMPORAL_PERFORMANCE_ROWS:

        raise ValueError(
            "Unexpected temporal performance row count. "
            f"Expected {EXPECTED_TEMPORAL_PERFORMANCE_ROWS:,}, "
            f"received {row_count:,}."
        )

    duplicate_count = (
        df
        .groupBy(
            "timestamp"
        )
        .count()
        .filter(
            F.col("count") > 1
        )
        .count()
    )

    print(
        f"Duplicate timestamps: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:

        raise ValueError(
            "Temporal performance contains duplicate timestamps."
        )

    null_rows = (
        df
        .filter(
            F.col("timestamp").isNull()
            |
            F.col("timestamp_total_activity").isNull()
        )
        .count()
    )

    print(
        f"Rows with critical NULL values: "
        f"{null_rows:,}"
    )

    if null_rows != 0:

        raise ValueError(
            "Temporal performance contains critical NULL values."
        )

    negative_rows = (
        df
        .filter(
            F.col("timestamp_total_activity") < 0
        )
        .count()
    )

    print(
        f"Rows with negative activity: "
        f"{negative_rows:,}"
    )

    if negative_rows != 0:

        raise ValueError(
            "Temporal performance contains negative activity."
        )

    print(
        "Temporal performance validation: PASS"
    )

    df.printSchema()


# ============================================================================
# WRITE PARQUET SAFELY
# ============================================================================

def write_parquet(
    df,
    output_path,
    label,
    mode="overwrite"
):

    print()
    print(
        f"Writing {label}..."
    )

    print(
        f"Output: {output_path}"
    )

    start = time.perf_counter()

    (
        df
        .write
        .mode(mode)
        .option(
            "compression",
            "snappy"
        )
        .parquet(
            output_path
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"{label} write completed "
        f"in {elapsed:.3f} seconds."
    )

    return elapsed


# ============================================================================
# VERIFY PARQUET OUTPUT
# ============================================================================

def verify_parquet_output(
    spark,
    output_path,
    expected_rows,
    label,
    required_columns
):

    print()
    print(
        f"Verifying {label} Parquet output..."
    )

    if not os.path.exists(
        output_path
    ):

        raise FileNotFoundError(
            f"{label} output does not exist:\n"
            f"{output_path}"
        )

    parquet_files = find_parquet_files(
        output_path
    )

    print(
        f"{label} Parquet files found: "
        f"{len(parquet_files)}"
    )

    if len(parquet_files) == 0:

        raise ValueError(
            f"No Parquet files found for {label}."
        )

    df = (
        spark.read
        .parquet(
            *parquet_files
        )
    )

    row_count = df.count()

    print(
        f"{label} rows verified: "
        f"{row_count:,}"
    )

    if row_count != expected_rows:

        raise ValueError(
            f"{label} row-count verification failed. "
            f"Expected {expected_rows:,}, "
            f"received {row_count:,}."
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{label} missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print()
    print(
        f"{label} verified schema:"
    )

    df.printSchema()

    print(
        f"{label} Parquet verification: PASS"
    )

    return df


# ============================================================================
# COPY / STORE REFERENCE FILE
# ============================================================================

def validate_reference_storage():

    print()
    print("=" * 75)
    print("[15] VALIDATING REFERENCE DATA LOCATION")
    print("=" * 75)

    print(
        f"Reference directory:\n"
        f"{REFERENCE_DIR}"
    )

    if not os.path.exists(
        REFERENCE_DIR
    ):

        raise FileNotFoundError(
            "Reference directory does not exist."
        )

    reference_files = list(
        Path(
            REFERENCE_DIR
        ).iterdir()
    )

    print(
        f"Reference files found: "
        f"{len(reference_files)}"
    )

    geojson_files = [
        path
        for path in reference_files
        if path.is_file()
        and path.suffix.lower()
        == ".geojson"
    ]

    if not geojson_files:

        print(
            "WARNING: No GeoJSON file detected."
        )

    else:

        for path in geojson_files:

            print(
                f"Reference: {path}"
            )

    print(
        "Reference storage validation: PASS"
    )


# ============================================================================
# WRITE SP6 DATASETS
# ============================================================================

def write_sp6_datasets(
    activity_df,
    hourly_grid_summary_df,
    grid_performance_df,
    temporal_performance_df
):

    print()
    print("=" * 75)
    print("[16] WRITING SP6 DATASETS")
    print("=" * 75)

    activity_time = write_parquet(
        activity_df,
        ACTIVITY_OUTPUT_PATH,
        "processed activity"
    )

    hourly_time = write_parquet(
        hourly_grid_summary_df,
        HOURLY_GRID_SUMMARY_OUTPUT_PATH,
        "hourly grid summary"
    )

    grid_time = write_parquet(
        grid_performance_df,
        GRID_PERFORMANCE_OUTPUT_PATH,
        "grid performance"
    )

    temporal_time = write_parquet(
        temporal_performance_df,
        TEMPORAL_PERFORMANCE_OUTPUT_PATH,
        "temporal performance"
    )

    return {
        "activity": activity_time,
        "hourly_grid_summary": hourly_time,
        "grid_performance": grid_time,
        "temporal_performance": temporal_time,
    }


# ============================================================================
# FINAL SP6 DATA QUALITY VALIDATION
# ============================================================================

def final_data_quality_validation(
    activity_df,
    hourly_grid_summary_df,
    grid_performance_df,
    temporal_performance_df
):

    print()
    print("=" * 75)
    print("[17] FINAL SP6 DATA QUALITY VALIDATION")
    print("=" * 75)

    activity_rows = activity_df.count()

    hourly_rows = (
        hourly_grid_summary_df.count()
    )

    grid_rows = (
        grid_performance_df.count()
    )

    temporal_rows = (
        temporal_performance_df.count()
    )

    print(
        f"Activity rows             : "
        f"{activity_rows:,}"
    )

    print(
        f"Hourly grid summary rows  : "
        f"{hourly_rows:,}"
    )

    print(
        f"Grid performance rows     : "
        f"{grid_rows:,}"
    )

    print(
        f"Temporal performance rows : "
        f"{temporal_rows:,}"
    )

    # ------------------------------------------------------------------------
    # ACTIVITY
    # ------------------------------------------------------------------------

    if activity_rows != EXPECTED_SP4_ROWS:

        raise ValueError(
            "Activity output row count is incorrect."
        )

    # ------------------------------------------------------------------------
    # HOURLY GRID SUMMARY
    # ------------------------------------------------------------------------

    if hourly_rows != EXPECTED_SP4_ROWS:

        raise ValueError(
            "Hourly grid summary row count is incorrect."
        )

    # ------------------------------------------------------------------------
    # SP5 ANALYTICS
    # ------------------------------------------------------------------------

    if grid_rows != EXPECTED_GRID_PERFORMANCE_ROWS:

        raise ValueError(
            "Grid performance row count is incorrect."
        )

    if temporal_rows != EXPECTED_TEMPORAL_PERFORMANCE_ROWS:

        raise ValueError(
            "Temporal performance row count is incorrect."
        )

    # ------------------------------------------------------------------------
    # GRID ID RANGE
    # ------------------------------------------------------------------------

    min_grid = (
        grid_performance_df
        .agg(
            F.min("grid_id")
        )
        .collect()[0][0]
    )

    max_grid = (
        grid_performance_df
        .agg(
            F.max("grid_id")
        )
        .collect()[0][0]
    )

    print(
        f"Grid performance minimum grid_id: "
        f"{min_grid}"
    )

    print(
        f"Grid performance maximum grid_id: "
        f"{max_grid}"
    )

    if min_grid != 1 or max_grid != 10000:

        raise ValueError(
            "Grid performance grid_id range "
            "is not 1-10000."
        )

    print(
        "Grid ID range validation: PASS"
    )

    # ------------------------------------------------------------------------
    # TEMPORAL COUNT
    # ------------------------------------------------------------------------

    temporal_timestamps = (
        temporal_performance_df
        .select("timestamp")
        .distinct()
        .count()
    )

    print(
        f"Distinct temporal timestamps: "
        f"{temporal_timestamps}"
    )

    if temporal_timestamps != EXPECTED_TEMPORAL_PERFORMANCE_ROWS:

        raise ValueError(
            "Temporal timestamp count is not 168."
        )

    print(
        "Temporal coverage validation: PASS"
    )

    print()
    print(
        "SP6 data quality validation: PASS"
    )


# ============================================================================
# SHOW STORAGE SUMMARY
# ============================================================================

def show_storage_summary():

    print()
    print("=" * 75)
    print("[18] SP6 STORAGE SUMMARY")
    print("=" * 75)

    print()
    print(
        "PROCESSED ZONE"
    )

    print(
        f"  Activity:"
    )

    print(
        f"  {ACTIVITY_OUTPUT_PATH}"
    )

    print()
    print(
        "ANALYTICS ZONE"
    )

    print(
        "  Hourly grid summary:"
    )

    print(
        f"  {HOURLY_GRID_SUMMARY_OUTPUT_PATH}"
    )

    print()
    print(
        "  Grid performance:"
    )

    print(
        f"  {GRID_PERFORMANCE_OUTPUT_PATH}"
    )

    print()
    print(
        "  Temporal performance:"
    )

    print(
        f"  {TEMPORAL_PERFORMANCE_OUTPUT_PATH}"
    )

    print()
    print(
        "REFERENCE ZONE"
    )

    print(
        f"  {REFERENCE_DIR}"
    )


# ============================================================================
# FINAL PARQUET VERIFICATION
# ============================================================================

def final_parquet_verification(
    spark
):

    print()
    print("=" * 75)
    print("[19] VERIFYING SP6 PARQUET OUTPUTS")
    print("=" * 75)

    activity_df = verify_parquet_output(
        spark,
        ACTIVITY_OUTPUT_PATH,
        EXPECTED_SP4_ROWS,
        "Processed activity",
        ACTIVITY_COLUMNS
    )

    hourly_df = verify_parquet_output(
        spark,
        HOURLY_GRID_SUMMARY_OUTPUT_PATH,
        EXPECTED_SP4_ROWS,
        "Hourly grid summary",
        HOURLY_GRID_SUMMARY_COLUMNS
    )

    grid_df = verify_parquet_output(
        spark,
        GRID_PERFORMANCE_OUTPUT_PATH,
        EXPECTED_GRID_PERFORMANCE_ROWS,
        "Grid performance",
        REQUIRED_GRID_PERFORMANCE_COLUMNS
    )

    temporal_df = verify_parquet_output(
        spark,
        TEMPORAL_PERFORMANCE_OUTPUT_PATH,
        EXPECTED_TEMPORAL_PERFORMANCE_ROWS,
        "Temporal performance",
        REQUIRED_TEMPORAL_PERFORMANCE_COLUMNS
    )

    print()
    print(
        "SP6 Parquet verification: PASS"
    )

    return (
        activity_df,
        hourly_df,
        grid_df,
        temporal_df
    )


# ============================================================================
# MAIN SP6
# ============================================================================

def run_sp6():

    spark = None

    try:

        # ====================================================================
        # SPARK
        # ====================================================================

        spark = create_spark_session()

        print()
        print("=" * 75)
        print(
            "SP6 - WRITE PROCESSED & ANALYTICS DATA"
        )
        print("=" * 75)

        # ====================================================================
        # DIRECTORIES
        # ====================================================================

        prepare_storage_directories()

        # ====================================================================
        # INPUT LOCATIONS
        # ====================================================================

        sp4_files = validate_sp4_location()

        sp5_files = validate_sp5_location()

        # ====================================================================
        # READ SP4
        # ====================================================================

        sp4_df = load_sp4(
            spark,
            sp4_files
        )

        validate_sp4_schema(
            sp4_df
        )

        validate_sp4_data(
            sp4_df
        )

        # ====================================================================
        # READ SP5
        # ====================================================================

        sp5_root_df = load_sp5_files(
            spark,
            sp5_files
        )

        # ====================================================================
        # IDENTIFY SP5 DATASETS
        # ====================================================================

        (
            grid_performance_df,
            temporal_performance_df
        ) = discover_sp5_datasets(
            spark,
            sp5_files
        )

        # ====================================================================
        # SP5 VALIDATION
        # ====================================================================

        validate_grid_performance(
            grid_performance_df
        )

        validate_temporal_performance(
            temporal_performance_df
        )

        # ====================================================================
        # PREPARE PROCESSED DATA
        # ====================================================================

        activity_df = (
            prepare_activity_dataset(
                sp4_df
            )
        )

        # ====================================================================
        # PREPARE HOURLY GRID SUMMARY
        # ====================================================================

        hourly_grid_summary_df = (
            prepare_hourly_grid_summary(
                sp4_df
            )
        )

        validate_hourly_grid_summary(
            hourly_grid_summary_df
        )

        # ====================================================================
        # REFERENCE
        # ====================================================================

        validate_reference_storage()

        # ====================================================================
        # WRITE OUTPUTS
        # ====================================================================

        write_times = (
            write_sp6_datasets(
                activity_df,
                hourly_grid_summary_df,
                grid_performance_df,
                temporal_performance_df
            )
        )

        # ====================================================================
        # READ-BACK VERIFICATION
        # ====================================================================

        (
            stored_activity_df,
            stored_hourly_df,
            stored_grid_df,
            stored_temporal_df
        ) = final_parquet_verification(
            spark
        )

        # ====================================================================
        # FINAL QUALITY
        # ====================================================================

        final_data_quality_validation(
            stored_activity_df,
            stored_hourly_df,
            stored_grid_df,
            stored_temporal_df
        )

        # ====================================================================
        # STORAGE SUMMARY
        # ====================================================================

        show_storage_summary()

        # ====================================================================
        # SUCCESS
        # ====================================================================

        print()
        print("=" * 75)
        print(
            "SP6 STORAGE COMPLETED SUCCESSFULLY"
        )
        print("=" * 75)

        print()
        print(
            "SP4 input rows              : "
            f"{EXPECTED_SP4_ROWS:,}"
        )

        print(
            "Processed activity rows     : "
            f"{stored_activity_df.count():,}"
        )

        print(
            "Hourly grid summary rows    : "
            f"{stored_hourly_df.count():,}"
        )

        print(
            "Grid performance rows       : "
            f"{stored_grid_df.count():,}"
        )

        print(
            "Temporal performance rows   : "
            f"{stored_temporal_df.count():,}"
        )

        print()
        print(
            "Processed storage           : Parquet"
        )

        print(
            "Analytics storage           : Parquet"
        )

        print(
            "Compression                 : Snappy"
        )

        print(
            "SP5 grid analytics          : 10,000 rows"
        )

        print(
            "SP5 temporal analytics      : 168 rows"
        )

        print(
            "SP4 activity preservation   : PASS"
        )

        print(
            "SP5 analytics preservation  : PASS"
        )

        print(
            "Schema validation           : PASS"
        )

        print(
            "NULL validation             : PASS"
        )

        print(
            "Negative validation         : PASS"
        )

        print(
            "Parquet read-back           : PASS"
        )

        print(
            "Storage validation          : PASS"
        )

        print(
            "Status                      : SUCCESS"
        )

        print("=" * 75)

        return {
            "activity": stored_activity_df,
            "hourly_grid_summary": stored_hourly_df,
            "grid_performance": stored_grid_df,
            "temporal_performance": stored_temporal_df,
        }

    except Exception as exc:

        print()
        print("=" * 75)
        print(
            "SP6 STORAGE FAILED"
        )
        print("=" * 75)

        print(
            f"Error type : "
            f"{type(exc).__name__}"
        )

        print(
            f"Error      : "
            f"{exc}"
        )

        raise

    finally:

        if spark is not None:

            try:

                spark.stop()

            except Exception:

                pass


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    run_sp6()